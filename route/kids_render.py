from __future__ import annotations

import os
import base64
import uuid
import traceback
from pathlib import Path
from typing import Dict, Optional, Any

import anyio
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from openai import OpenAI
from tripo3d import TripoClient
from tripo3d.models import TaskStatus

from service.nano_banana import render_one_image

router = APIRouter(prefix="/api/v1/kids", tags=["kids"])


# -----------------------------
# Config / Paths
# -----------------------------
def _is_truthy(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


DEBUG = _is_truthy(os.environ.get("DEBUG", "false"))


def _find_project_root(start: Path) -> Path:
    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent

    markers = ("pyproject.toml", "requirements.txt", ".git")
    for p in [cur] + list(cur.parents):
        for m in markers:
            if (p / m).exists():
                return p
    return cur


PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "")).expanduser()
if str(PROJECT_ROOT).strip():
    PROJECT_ROOT = PROJECT_ROOT.resolve()
else:
    PROJECT_ROOT = _find_project_root(Path(__file__))

PUBLIC_DIR = Path(os.environ.get("PUBLIC_DIR", PROJECT_ROOT / "public")).resolve()
GENERATED_DIR = Path(os.environ.get("GENERATED_DIR", PUBLIC_DIR / "generated")).resolve()
GENERATED_DIR.mkdir(parents=True, exist_ok=True)


def _mime_to_ext(mime: str) -> str:
    m = (mime or "").lower()
    if "png" in m:
        return ".png"
    if "jpeg" in m or "jpg" in m:
        return ".jpg"
    if "webp" in m:
        return ".webp"
    return ".png"


def _write_error_log(out_dir: Path, text: str) -> None:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "error.log").write_text(text, encoding="utf-8")
    except Exception:
        pass


# ✅ URL prefix는 /api/generated 로 통일
STATIC_PREFIX = os.environ.get("GENERATED_URL_PREFIX", "/api/generated").rstrip("/")


def _to_generated_url(p: Path, out_dir: Path) -> str:
    """
    GENERATED_DIR 아래 파일이면: /api/generated/... 로 변환
    GENERATED_DIR 밖 파일이면: out_dir로 복사 후 /api/generated/... 로 변환
    """
    p = Path(p).resolve()
    gen = GENERATED_DIR.resolve()

    try:
        rel = p.relative_to(gen)
        return f"{STATIC_PREFIX}/" + rel.as_posix()
    except ValueError:
        out_dir = out_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / p.name
        if p != dst:
            dst.write_bytes(p.read_bytes())
        rel = dst.relative_to(gen)
        return f"{STATIC_PREFIX}/" + rel.as_posix()


def _pick_model_url(files_url: Dict[str, str]) -> str:
    for k, u in files_url.items():
        if "glb" in k.lower() or u.lower().endswith(".glb"):
            return u
    for k, u in files_url.items():
        if "pbr" in k.lower():
            return u
    return next(iter(files_url.values()))


async def _download_http_to_file(url: str, dst: Path) -> Path:
    import httpx
    dst.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=180.0, follow_redirects=True) as client:
        r = await client.get(url)
        r.raise_for_status()
        dst.write_bytes(r.content)
    return dst


def _parse_bool(v: str | bool | None, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


# -----------------------------
# OpenAI client
# -----------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_INSTRUCTIONS = """You are an expert prompt engineer for 3D modeling from a reference photo.
Goal: output a single-paragraph English prompt for creating a SIMPLE mesh suitable for LEGO building instructions.

Rules:
- Output ONLY the prompt text (one paragraph). No bullets, no numbering, no extra commentary.
- Low-poly hard-surface, blocky primitives, clean planar surfaces.
- Keep geometry simple: large primitives, straight edges, shallow insets only.
- Remove micro details: logos, text, tiny vents, patterns, textures, stickers, decals.
- Prefer symmetry when possible.
"""


def _extract_text(resp) -> str:
    t = getattr(resp, "output_text", None)
    if isinstance(t, str) and t.strip():
        return t.strip()

    out = getattr(resp, "output", []) or []
    texts: list[str] = []
    for item in out:
        for c in getattr(item, "content", []) or []:
            if getattr(c, "type", "") in ("output_text", "text"):
                val = getattr(c, "text", "") or ""
                if val.strip():
                    texts.append(val.strip())
    return "\n".join(texts).strip()


def build_lego_prompt_sync(image_bytes: bytes, mime: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    resp = openai_client.responses.create(
        model=OPENAI_MODEL,
        instructions=SYSTEM_INSTRUCTIONS,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Analyze the image and produce the modeling prompt following the rules."},
                    {"type": "input_image", "image_url": f"data:{mime};base64,{b64}", "detail": "low"},
                ],
            }
        ],
        max_output_tokens=10000,
    )

    prompt = _extract_text(resp)
    if not prompt:
        raise RuntimeError("Empty prompt from OpenAI")
    return prompt


# -----------------------------
# Brickify helpers
# -----------------------------
AGE_TO_BUDGET = {"4-5": 20, "6-7": 60, "8-10": 120}


def _sanitize_glb_url(u: str) -> str:
    u = (u or "").strip()
    # 프론트 버그로 /api/api/generated 같은게 오면 복구
    if u.startswith("/api/api/"):
        u = u.replace("/api/api/", "/api/", 1)
    return u


def _local_generated_path_from_url(u: str) -> Optional[Path]:
    """
    /api/generated/xxx.glb -> GENERATED_DIR/xxx.glb 로 매핑
    """
    u = _sanitize_glb_url(u)
    if u.startswith("/api/generated/"):
        rel = u[len("/api/generated/"):]
        return (GENERATED_DIR / rel).resolve()
    if u.startswith("/generated/"):
        rel = u[len("/generated/"):]
        return (GENERATED_DIR / rel).resolve()
    return None


def _load_engine_convert():
    """
    brick-engine 폴더명이 하이픈이라 import가 안 되니까 파일경로로 모듈 로드.
    """
    import importlib.util

    engine_path = (PROJECT_ROOT / "brick-engine" / "glb_to_ldr_embedded.py").resolve()
    if not engine_path.exists():
        raise RuntimeError(f"engine file missing: {engine_path}")

    spec = importlib.util.spec_from_file_location("glb_to_ldr_embedded", str(engine_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load spec for glb_to_ldr_embedded")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    if not hasattr(mod, "convert_glb_to_ldr"):
        raise RuntimeError("convert_glb_to_ldr not found in engine module")

    return mod.convert_glb_to_ldr


_CONVERT_FN = None


def _budget_to_start_target(eff_budget: int) -> int:
    if eff_budget <= 25:
        return 24
    if eff_budget <= 70:
        return 45
    return 60


def _find_glb_in_dir(out_dir: Path) -> Optional[Path]:
    glbs = [p for p in out_dir.rglob("*.glb") if p.is_file() and p.stat().st_size > 0]
    return glbs[0] if glbs else None


def _pick_glb_from_downloaded(downloaded: Dict[str, str], out_dir: Path) -> Optional[Path]:
    # downloaded dict 안에 .glb가 있으면 그거 우선
    for _, v in (downloaded or {}).items():
        p = Path(v)
        if p.suffix.lower() == ".glb" and p.exists() and p.stat().st_size > 0:
            return p
    return _find_glb_in_dir(out_dir)


# -----------------------------
# Response schema
# -----------------------------
class ProcessResp(BaseModel):
    ok: bool
    reqId: str

    correctedUrl: str
    prompt: str

    taskId: str
    modelUrl: str
    files: Dict[str, str]

    ldrUrl: str
    ldrData: Optional[str] = None

    parts: int
    finalTarget: int


# -----------------------------
# ✅ 단일 API
# -----------------------------
@router.post("/process-all", response_model=ProcessResp)
async def process(
    file: UploadFile = File(...),
    age: str = Form("6-7"),
    budget: int | None = Form(default=None),
    # 프론트가 prompt를 보내면 그대로 사용(없으면 생성)
    prompt: str | None = Form(default=None),
    # 필요할 때만 LDR을 base64로 같이 보냄
    returnLdrData: str | bool | None = Form(default="true"),
):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="image only")

    TRIPO_API_KEY = os.environ.get("TRIPO_API_KEY", "")
    if not TRIPO_API_KEY:
        raise HTTPException(status_code=500, detail="TRIPO_API_KEY is not set")

    req_id = uuid.uuid4().hex
    out_req_dir = GENERATED_DIR / f"req_{req_id}"
    out_tripo_dir = GENERATED_DIR / f"tripo_{req_id}"
    out_brick_dir = GENERATED_DIR / f"brickify_{req_id}"
    out_req_dir.mkdir(parents=True, exist_ok=True)
    out_tripo_dir.mkdir(parents=True, exist_ok=True)
    out_brick_dir.mkdir(parents=True, exist_ok=True)

    try:
        # -----------------
        # 0) 입력 저장
        # -----------------
        img_bytes = await file.read()
        raw_ext = _mime_to_ext(file.content_type or "image/png")
        raw_path = out_req_dir / f"raw{raw_ext}"
        raw_path.write_bytes(img_bytes)

        # -----------------
        # 1) 보정
        # -----------------
        corrected_bytes = await render_one_image(img_bytes, file.content_type or "image/png")
        corrected_path = out_req_dir / "corrected.png"
        corrected_path.write_bytes(corrected_bytes)
        corrected_url = _to_generated_url(corrected_path, out_dir=out_req_dir)

        # -----------------
        # 2) Prompt
        # -----------------
        if prompt and prompt.strip():
            prompt_text = prompt.strip()
        else:
            prompt_text = await anyio.to_thread.run_sync(
                build_lego_prompt_sync, corrected_bytes, "image/png"
            )

        # -----------------
        # 3) Tripo 3D (생성 → 대기 → 다운로드)
        # -----------------
        negative_prompt = (
            "low quality, blurry, noisy, over-detailed, high poly, "
            "thin parts, tiny features, text, logos, patterns, textures"
        )

        async with TripoClient(api_key=TRIPO_API_KEY) as client:
            task_id = await client.text_to_model(prompt=prompt_text, negative_prompt=negative_prompt)
            task = await client.wait_for_task(task_id, verbose=DEBUG)
            if task.status != TaskStatus.SUCCESS:
                raise RuntimeError(f"Tripo task failed: status={task.status}")
            downloaded = await client.download_task_models(task, str(out_tripo_dir))

        # -----------------
        # 3-1) downloaded 정규화 (URL이면 다시 받아서 파일로)
        # -----------------
        fixed_downloaded: Dict[str, str] = {}
        for model_type, path_or_url in (downloaded or {}).items():
            if not path_or_url:
                continue
            s = str(path_or_url)
            if s.startswith(("http://", "https://")):
                ext_guess = ".glb" if ".glb" in s.lower() else ".bin"
                dst = out_tripo_dir / f"{model_type}{ext_guess}"
                await _download_http_to_file(s, dst)
                fixed_downloaded[model_type] = str(dst)
            else:
                fixed_downloaded[model_type] = s

        # 파일 존재/사이즈 체크
        missing = []
        for k, v in fixed_downloaded.items():
            pv = Path(v)
            if not pv.exists():
                missing.append((k, v, "NOT_EXISTS"))
            elif pv.stat().st_size == 0:
                missing.append((k, v, "ZERO_SIZE"))
        if missing:
            raise RuntimeError(f"Downloaded files missing: {missing}")

        # -----------------
        # 3-2) Tripo 파일 URL 맵 만들기
        # -----------------
        files_url: Dict[str, str] = {}
        for model_type, path_str in fixed_downloaded.items():
            files_url[model_type] = _to_generated_url(Path(path_str), out_dir=out_tripo_dir)

        # fallback: out_tripo_dir 내 glb라도 넣기
        if not any(u.lower().endswith(".glb") for u in files_url.values()):
            glb_fallback = _find_glb_in_dir(out_tripo_dir)
            if glb_fallback:
                files_url["glb"] = _to_generated_url(glb_fallback, out_dir=out_tripo_dir)

        if not files_url:
            raise RuntimeError("No downloadable model files found in out_tripo_dir")

        model_url = _pick_model_url(files_url)

        # -----------------
        # 4) Brickify 입력 GLB 확보
        # -----------------
        glb_path = _pick_glb_from_downloaded(fixed_downloaded, out_tripo_dir)

        # 그래도 없으면 model_url로부터 로컬 매핑/다운로드
        if glb_path is None:
            local = _local_generated_path_from_url(model_url)
            if local and local.exists() and local.stat().st_size > 0:
                glb_path = local
            else:
                if not model_url.startswith(("http://", "https://")):
                    raise RuntimeError(f"Cannot resolve glb for brickify: {model_url}")
                glb_path = out_brick_dir / "input.glb"
                await _download_http_to_file(model_url, glb_path)

        if not glb_path.exists() or glb_path.stat().st_size == 0:
            raise RuntimeError(f"GLB missing/empty: {glb_path}")

        # -----------------
        # 5) Brickify 실행
        # -----------------
        eff_budget = int(budget) if budget is not None else int(AGE_TO_BUDGET.get(age.strip(), 60))
        start_target = _budget_to_start_target(eff_budget)

        global _CONVERT_FN
        if _CONVERT_FN is None:
            _CONVERT_FN = _load_engine_convert()

        out_ldr = out_brick_dir / "result.ldr"

        # ✅ 너가 쓰던 핵심 파라미터 풀세트 유지
        result: Dict[str, Any] = await anyio.to_thread.run_sync(
            lambda: _CONVERT_FN(  # type: ignore
                str(glb_path),
                str(out_ldr),
                budget=int(eff_budget),
                target=int(start_target),
                min_target=5,
                shrink=0.85,
                search_iters=6,
                kind="brick",
                plates_per_voxel=3,
                interlock=True,
                max_area=20,
                solid_color=4,
                use_mesh_color=True,
                invert_y=False,
                smart_fix=True,
                # 아래 인자들은 엔진에서 "호환용으로 받아서 무시" 가능
                span=4,
                max_new_voxels=12000,
                refine_iters=8,
                ensure_connected=True,
                min_embed=2,
                erosion_iters=1,
                fast_search=True,
                step_order="bottomup",
                extend_catalog=True,
                max_len=8,
            )
        )

        if not out_ldr.exists() or out_ldr.stat().st_size == 0:
            raise RuntimeError("LDR output missing/empty")

        ldr_url = _to_generated_url(out_ldr, out_dir=out_brick_dir)

        ldr_data_uri: Optional[str] = None
        if _parse_bool(returnLdrData, default=True):
            b64_str = base64.b64encode(out_ldr.read_bytes()).decode("utf-8")
            ldr_data_uri = f"data:text/plain;base64,{b64_str}"

        return {
            "ok": True,
            "reqId": req_id,
            "correctedUrl": corrected_url,
            "prompt": prompt_text,
            "taskId": str(task_id),
            "modelUrl": model_url,
            "files": files_url,
            "ldrUrl": ldr_url,
            "ldrData": ldr_data_uri,
            "parts": int(result.get("parts", 0)),
            "finalTarget": int(result.get("final_target", 0)),
        }

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[kids.process] ERROR req_id={req_id}\n{tb}")
        _write_error_log(out_req_dir, tb)
        _write_error_log(out_tripo_dir, tb)
        _write_error_log(out_brick_dir, tb)

        if DEBUG:
            raise HTTPException(status_code=500, detail={"reqId": req_id, "error": str(e)})

        raise HTTPException(status_code=500, detail="process failed")