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

# -----------------------------
# Router
# -----------------------------
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
        max_output_tokens=220,
    )

    prompt = _extract_text(resp)
    if not prompt:
        raise RuntimeError("Empty prompt from OpenAI")
    return prompt


# -----------------------------
# 1) 2D 이미지 생성
# -----------------------------
@router.post("/render-image")
async def render_image(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="image only")

    try:
        img_bytes = await file.read()
        out_bytes = await render_one_image(img_bytes, file.content_type or "image/png")
        return Response(content=out_bytes, media_type="image/png")
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception:
        tb = traceback.format_exc()
        print("[kids.render_image] ERROR\n", tb)
        raise HTTPException(status_code=500, detail="Internal Server Error during generation")


# -----------------------------
# 2) 프롬프트만 생성
# -----------------------------
class PromptResp(BaseModel):
    prompt: str


@router.post("/generate-prompt", response_model=PromptResp)
async def generate_prompt(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="image only")

    try:
        img_bytes = await file.read()
        prompt = await anyio.to_thread.run_sync(
            build_lego_prompt_sync, img_bytes, file.content_type or "image/png"
        )
        return {"prompt": prompt}
    except Exception as e:
        tb = traceback.format_exc()
        print("[kids.generate_prompt] ERROR\n", tb)
        if DEBUG:
            raise HTTPException(status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail="Internal Server Error during prompt generation")


# -----------------------------
# 3) 이미지 -> prompt -> Tripo -> 다운로드 -> URL 반환
# -----------------------------
class Generate3DResp(BaseModel):
    prompt: str
    taskId: str
    modelUrl: str
    files: Dict[str, str]


@router.post("/generate-3d", response_model=Generate3DResp)
async def generate_3d(
    file: UploadFile = File(...),
    referenceLdr: str | None = Form(default=None),
    # ✅ 프론트가 prompt를 보내니까 지원해줌(있으면 그대로 쓰고 없으면 생성)
    prompt: str | None = Form(default=None),
):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="image only")

    TRIPO_API_KEY = os.environ.get("TRIPO_API_KEY", "")
    if not TRIPO_API_KEY:
        raise HTTPException(status_code=500, detail="TRIPO_API_KEY is not set")

    req_id = uuid.uuid4().hex
    ext = _mime_to_ext(file.content_type or "image/png")

    raw_path = GENERATED_DIR / f"{req_id}_raw{ext}"
    out_dir = GENERATED_DIR / f"tripo_{req_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        img_bytes = await file.read()
        raw_path.write_bytes(img_bytes)

        # (1) prompt: 프론트가 준 prompt가 있으면 그거 사용
        if prompt and prompt.strip():
            prompt = prompt.strip()
        else:
            prompt = await anyio.to_thread.run_sync(
                build_lego_prompt_sync, img_bytes, file.content_type or "image/png"
            )

        # (2) Tripo 3D
        negative_prompt = (
            "low quality, blurry, noisy, over-detailed, high poly, "
            "thin parts, tiny features, text, logos, patterns, textures"
        )

        async with TripoClient(api_key=TRIPO_API_KEY) as client:
            task_id = await client.text_to_model(prompt=prompt, negative_prompt=negative_prompt)
            task = await client.wait_for_task(task_id, verbose=DEBUG)
            if task.status != TaskStatus.SUCCESS:
                raise RuntimeError(f"Tripo task failed: status={task.status}")
            downloaded = await client.download_task_models(task, str(out_dir))

        # URL/None 대비
        fixed_downloaded: Dict[str, str] = {}
        for model_type, path_or_url in (downloaded or {}).items():
            if not path_or_url:
                continue
            s = str(path_or_url)
            if s.startswith(("http://", "https://")):
                ext_guess = ".glb" if ".glb" in s.lower() else ".bin"
                dst = out_dir / f"{model_type}{ext_guess}"
                await _download_http_to_file(s, dst)
                fixed_downloaded[model_type] = str(dst)
            else:
                fixed_downloaded[model_type] = s

        # 존재 체크
        missing = []
        for k, v in fixed_downloaded.items():
            pv = Path(v)
            if not pv.exists():
                missing.append((k, v, "NOT_EXISTS"))
            elif pv.stat().st_size == 0:
                missing.append((k, v, "ZERO_SIZE"))
        if missing:
            raise RuntimeError(f"Downloaded files missing: {missing}")

        files_url: Dict[str, str] = {}
        for model_type, path_str in fixed_downloaded.items():
            url = _to_generated_url(Path(path_str), out_dir=out_dir)
            files_url[model_type] = url

        if not files_url:
            glbs = [p for p in out_dir.rglob("*.glb") if p.is_file()]
            if glbs:
                files_url["glb"] = _to_generated_url(glbs[0], out_dir=out_dir)

        if not files_url:
            raise RuntimeError("No downloadable model files found in out_dir")

        model_url = _pick_model_url(files_url)

        return {
            "prompt": prompt,
            "taskId": str(task_id),
            "modelUrl": model_url,
            "files": files_url,
        }

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[kids.generate_3d] ERROR req_id={req_id}\n{tb}")
        _write_error_log(out_dir, tb)

        if DEBUG:
            raise HTTPException(status_code=500, detail={"reqId": req_id, "error": str(e)})

        raise HTTPException(status_code=500, detail="Internal Server Error during 3D generation")


# -----------------------------
# ✅ 4) GLB -> LDR (brickify) 추가
# -----------------------------
class BrickifyResp(BaseModel):
    ldrUrl: str
    parts: int
    finalTarget: int


AGE_TO_BUDGET = {"4-5": 20, "6-7": 60, "8-10": 120}


def _sanitize_glb_url(u: str) -> str:
    # 프론트 버그로 /api/api/generated 같은게 오면 복구
    u = (u or "").strip()
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
    brick-engine 폴더명이 하이픈이라 import가 안 되니까,
    파일경로로 모듈 로드.
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



@router.post("/brickify", response_model=BrickifyResp)
async def brickify(
    glbUrl: str = Form(...),
    age: str | None = Form(default=None),
    budget: int | None = Form(default=None),
):
    global _CONVERT_FN
    glbUrl = _sanitize_glb_url(glbUrl)

    # 예산 결정
    eff_budget = budget if budget is not None else AGE_TO_BUDGET.get((age or "").strip(), 60)

    # 작업 폴더
    req_id = uuid.uuid4().hex
    out_dir = GENERATED_DIR / f"brickify_{req_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    glb_path: Optional[Path] = None
    try:
        # 1) GLB 확보 (로컬 매핑 우선)
        local = _local_generated_path_from_url(glbUrl)
        if local and local.exists():
            glb_path = local
        else:
            if not glbUrl.startswith(("http://", "https://")):
                raise RuntimeError(f"Cannot resolve glbUrl: {glbUrl}")
            glb_path = out_dir / "input.glb"
            await _download_http_to_file(glbUrl, glb_path)

        if not glb_path.exists() or glb_path.stat().st_size == 0:
            raise RuntimeError(f"GLB missing/empty: {glb_path}")

        # 2) 엔진 로드 (캐시)
        if _CONVERT_FN is None:
            _CONVERT_FN = _load_engine_convert()

        # 3) start_target 프리셋
        if int(eff_budget) <= 25:
            start_target = 24
        elif int(eff_budget) <= 70:
            start_target = 45
        else:
            start_target = 60

        out_ldr = out_dir / "result.ldr"

        # ✅ 정상 호출 (딱 한 번)
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
                # 아래 인자들은 엔진에서 "호환용으로 받아서 무시"하게 해둠
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

    except Exception as e:
        tb = traceback.format_exc()
        _write_error_log(out_dir, tb)
        print("[kids.brickify] ERROR\n", tb)
        if DEBUG:
            raise HTTPException(status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail="brickify failed")

    # 결과 URL
    ldr_url = _to_generated_url(out_ldr, out_dir=out_dir)
    return BrickifyResp(
        ldrUrl=ldr_url,
        parts=int(result.get("parts", 0)),
        finalTarget=int(result.get("final_target", 0)),
    )
