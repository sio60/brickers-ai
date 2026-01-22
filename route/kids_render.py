from __future__ import annotations

import os
import base64
import uuid
import traceback
from pathlib import Path
from typing import Dict

import anyio
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from openai import OpenAI
from tripo3d import TripoClient
from tripo3d.models import TaskStatus

# ✅ 기존 2D 생성 로직 유지
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
    """
    cwd가 바뀌어도 경로가 안정적이도록,
    __file__ 기준으로 위로 올라가면서 "프로젝트 루트"를 추정.
    """
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


# ✅ URL은 /api/generated 로 통일 (프론트 로그가 /api/generated 로 요청하니까)
STATIC_PREFIX = os.environ.get("GENERATED_URL_PREFIX", "/api/generated").rstrip("/")


def _to_generated_url(p: Path, out_dir: Path) -> str:
    """
    downloaded 파일이 GENERATED_DIR 밖에 있을 수도 있으니,
    밖이면 out_dir로 복사해서 확실히 GENERATED_DIR 하위로 통일.
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
    # glb 확률 높은 것 우선
    for k, u in files_url.items():
        if "glb" in k.lower() or u.lower().endswith(".glb"):
            return u
    # pbr 우선
    for k, u in files_url.items():
        if "pbr" in k.lower():
            return u
    return next(iter(files_url.values()))


async def _download_http_to_file(url: str, dst: Path) -> Path:
    """
    tripo SDK가 로컬 파일 경로 대신 URL을 주는 경우를 대비한 백업 다운로드.
    anyio 기반으로 requests 없이 처리.
    """
    import httpx

    dst.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
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
                    {
                        "type": "input_text",
                        "text": "Analyze the image and produce the modeling prompt following the rules.",
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime};base64,{b64}",
                        "detail": "low",
                    },
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
# 1) 기존: 2D 이미지 생성 (유지)
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
# 2) 프롬프트만 생성 (유지)
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
        if DEBUG:
            print("[kids.generate_3d] req_id =", req_id)
            print("[kids.generate_3d] CWD =", Path.cwd())
            print("[kids.generate_3d] PROJECT_ROOT =", PROJECT_ROOT)
            print("[kids.generate_3d] PUBLIC_DIR =", PUBLIC_DIR)
            print("[kids.generate_3d] GENERATED_DIR =", GENERATED_DIR)
            print("[kids.generate_3d] raw_path =", raw_path.resolve())
            print("[kids.generate_3d] out_dir =", out_dir.resolve())
            print("[kids.generate_3d] STATIC_PREFIX =", STATIC_PREFIX)

        # (0) read + save raw
        img_bytes = await file.read()
        raw_path.write_bytes(img_bytes)

        if DEBUG:
            print("[kids.generate_3d] raw saved exists =", raw_path.exists(), "size =", raw_path.stat().st_size)

        # (1) ChatGPT로 프롬프트 생성
        prompt = await anyio.to_thread.run_sync(
            build_lego_prompt_sync, img_bytes, file.content_type or "image/png"
        )

        # (2) Tripo로 3D 생성 + 다운로드
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

        # ✅ 다운로드 결과 검증/로그
        created_files = [p for p in out_dir.rglob("*") if p.is_file()]
        if DEBUG:
            print("[kids.generate_3d] downloaded(raw) =", downloaded)
            print("[kids.generate_3d] out_dir files =", [p.as_posix() for p in created_files])

        # downloaded가 URL/None을 줄 때를 대비한 백업 처리
        fixed_downloaded: Dict[str, str] = {}
        for model_type, path_or_url in (downloaded or {}).items():
            if not path_or_url:
                continue

            s = str(path_or_url)
            if s.startswith(("http://", "https://")):
                # URL이면 직접 받아서 out_dir에 저장
                # 확장자 대충 추정: url에 .glb 있으면 glb, 아니면 bin
                ext_guess = ".glb" if ".glb" in s.lower() else ".bin"
                dst = out_dir / f"{model_type}{ext_guess}"
                await _download_http_to_file(s, dst)
                fixed_downloaded[model_type] = str(dst)
            else:
                fixed_downloaded[model_type] = s

        # 실제 파일 존재 체크
        missing = []
        for k, v in fixed_downloaded.items():
            pv = Path(v)
            if not pv.exists():
                missing.append((k, v, "NOT_EXISTS"))
            elif pv.stat().st_size == 0:
                missing.append((k, v, "ZERO_SIZE"))

        if missing:
            raise RuntimeError(f"Downloaded files missing: {missing}")

        # (3) 파일들을 URL로 변환해서 반환
        files_url: Dict[str, str] = {}
        for model_type, path_str in fixed_downloaded.items():
            url = _to_generated_url(Path(path_str), out_dir=out_dir)
            files_url[model_type] = url

        if not files_url:
            # out_dir에 파일이 있는데 fixed_downloaded가 비었을 수도 있어 fallback
            # (rglob로 glb라도 있으면 그걸로 만들어줌)
            glbs = [p for p in out_dir.rglob("*.glb") if p.is_file()]
            if glbs:
                files_url["glb"] = _to_generated_url(glbs[0], out_dir=out_dir)

        if not files_url:
            raise RuntimeError("No downloadable model files found in out_dir")

        model_url = _pick_model_url(files_url)

        if DEBUG:
            print("[kids.generate_3d] files_url =", files_url)
            print("[kids.generate_3d] model_url =", model_url)

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


"""
⚠️ 중요: /api/generated 로 파일을 열려면 main(app) 쪽에 Static mount가 필요합니다.

예) main.py/app.py

from fastapi.staticfiles import StaticFiles
from route.kids_render import GENERATED_DIR   # 이 파일 경로에 맞게 import
app.mount("/api/generated", StaticFiles(directory=str(GENERATED_DIR)), name="api_generated")

(선택) 디버깅용으로 /generated 도 같이 열고 싶으면:
app.mount("/generated", StaticFiles(directory=str(GENERATED_DIR)), name="generated")
"""