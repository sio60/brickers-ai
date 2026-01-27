# brickers-ai/route/kids_render.py
from __future__ import annotations

import os
import base64
import uuid
import traceback
from pathlib import Path
from typing import Dict, Optional, Any

import anyio
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

import httpx

# ---- OpenAI ----
from openai import OpenAI

# ---- Tripo ----
from tripo3d import TripoClient
from tripo3d.models import TaskStatus

# ---- Gemini (Nano Banana) ----
from google import genai
from google.genai import types as genai_types


router = APIRouter(prefix="/api/v1/kids", tags=["kids"])

# -----------------------------
# Helpers / Config
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
PROJECT_ROOT = PROJECT_ROOT.resolve() if str(PROJECT_ROOT).strip() else _find_project_root(Path(__file__))

PUBLIC_DIR = Path(os.environ.get("PUBLIC_DIR", PROJECT_ROOT / "public")).resolve()
GENERATED_DIR = Path(os.environ.get("GENERATED_DIR", PUBLIC_DIR / "generated")).resolve()
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

# ✅ URL prefix는 /api/generated 로 통일
STATIC_PREFIX = os.environ.get("GENERATED_URL_PREFIX", "/api/generated").rstrip("/")

# 타임아웃(필요하면 env로 조절)
KIDS_TOTAL_TIMEOUT_SEC = int(os.environ.get("KIDS_TOTAL_TIMEOUT_SEC", "1800"))     # 전체 30분
TRIPO_WAIT_TIMEOUT_SEC = int(os.environ.get("TRIPO_WAIT_TIMEOUT_SEC", "900"))     # 트리포 대기 15분
DOWNLOAD_TIMEOUT_SEC = float(os.environ.get("KIDS_DOWNLOAD_TIMEOUT_SEC", "180.0"))

# -----------------------------
# ✅ S3 업로드 옵션 (AWS_* 시크릿 기반)
# -----------------------------
# boto3 lazy import
try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None  # type: ignore
    ClientError = Exception  # type: ignore

AI_PUBLIC_BASE_URL = os.environ.get("AI_PUBLIC_BASE_URL", "").strip().rstrip("/")

AWS_REGION = (
    os.environ.get("AWS_REGION", "").strip()
    or os.environ.get("AWS_DEFAULT_REGION", "").strip()
)

S3_BUCKET = os.environ.get("AWS_S3_BUCKET", "").strip()

# 버킷 URL을 직접 지정하고 싶으면 이것도 시크릿으로 추가 가능(선택)
# 예) https://my-bucket.s3.ap-northeast-2.amazonaws.com  또는 CloudFront 도메인
S3_PUBLIC_BASE_URL = os.environ.get("S3_PUBLIC_BASE_URL", "").strip().rstrip("/")

# S3 켤지 여부: 기본은 "버킷 있으면 켜짐"
USE_S3 = _is_truthy(os.environ.get("USE_S3", "true" if S3_BUCKET else "false"))

# prefix (선택)
S3_PREFIX = os.environ.get("S3_PREFIX", "kids").strip().strip("/")

# 공개 URL이 안 되면 presign으로 내려주기(선택)
S3_PRESIGN = _is_truthy(os.environ.get("S3_PRESIGN", "false"))
S3_PRESIGN_EXPIRES = int(os.environ.get("S3_PRESIGN_EXPIRES", "86400"))

# 다운로드로 강제(ldr/glb)
S3_FORCE_ATTACHMENT = _is_truthy(os.environ.get("S3_FORCE_ATTACHMENT", "true"))

# 버킷이 ACL public-read 허용일 때만(보통은 false 유지 추천)
S3_USE_ACL_PUBLIC_READ = _is_truthy(os.environ.get("S3_USE_ACL_PUBLIC_READ", "false"))

_S3_CLIENT = None

def _require_s3_ready() -> None:
    if not USE_S3:
        return
    if boto3 is None:
        raise RuntimeError("boto3 is not installed (pip install boto3)")
    if not S3_BUCKET:
        raise RuntimeError("AWS_S3_BUCKET is not set")
    if not AWS_REGION:
        raise RuntimeError("AWS_REGION is not set")

def _get_s3_client():
    global _S3_CLIENT
    if _S3_CLIENT is not None:
        return _S3_CLIENT
    _require_s3_ready()

    # boto3는 아래 env를 자동으로 읽음:
    # AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION
    _S3_CLIENT = boto3.client("s3", region_name=AWS_REGION)
    return _S3_CLIENT

def _public_s3_url(key: str) -> str:
    # 1) 명시한 base url 있으면 그걸 사용 (CloudFront면 여기 넣는 게 베스트)
    if S3_PUBLIC_BASE_URL:
        return f"{S3_PUBLIC_BASE_URL}/{key}"

    # 2) 없으면 region 기반 기본 URL 생성 (버킷이 public/또는 CloudFront 없으면 접근 안 될 수 있음)
    if AWS_REGION == "us-east-1":
        return f"https://{S3_BUCKET}.s3.amazonaws.com/{key}"
    return f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"

def _presigned_get_url(key: str) -> str:
    client = _get_s3_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=S3_PRESIGN_EXPIRES,
    )

def _s3_url_for_key(key: str) -> str:
    # public 접근이 안 되면 presign 켜
    if S3_PRESIGN:
        return _presigned_get_url(key)
    return _public_s3_url(key)

def _upload_to_s3(local_path: Path, key: str, content_type: str | None = None) -> str:
    """
    로컬 파일을 S3에 업로드하고 public URL 반환.
    USE_S3=false면 아무것도 안 함.
    """
    if not USE_S3:
        return ""
    
    client = _get_s3_client()
    
    extra_args: dict = {}
    if content_type:
        extra_args["ContentType"] = content_type
    
    # ldr/glb는 다운로드 강제
    if S3_FORCE_ATTACHMENT and local_path.suffix.lower() in (".ldr", ".glb"):
        extra_args["ContentDisposition"] = f'attachment; filename="{local_path.name}"'
    
    if S3_USE_ACL_PUBLIC_READ:
        extra_args["ACL"] = "public-read"
    
    client.upload_file(str(local_path), S3_BUCKET, key, ExtraArgs=extra_args if extra_args else None)
    
    return _s3_url_for_key(key)

def _guess_content_type(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".glb": "application/octet-stream",
        ".gltf": "model/gltf+json",
        ".ldr": "text/plain",
        ".json": "application/json",
    }.get(ext, "application/octet-stream")

def _mime_to_ext(mime: str) -> str:
    m = (mime or "").lower()
    if "png" in m:
        return ".png"
    if "jpeg" in m or "jpg" in m:
        return ".jpg"
    if "webp" in m:
        return ".webp"
    return ".png"

async def _write_bytes_async(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    await anyio.to_thread.run_sync(path.write_bytes, data)

async def _read_bytes_async(path: Path) -> bytes:
    return await anyio.to_thread.run_sync(path.read_bytes)

def _write_error_log(out_dir: Path, text: str) -> None:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "error.log").write_text(text, encoding="utf-8")
    except Exception:
        pass

def _to_generated_url(p: Path, out_dir: Path) -> str:
    """
    USE_S3=true면: S3에 업로드 후 S3 URL 반환
    USE_S3=false면: GENERATED_DIR 기준 /api/generated/... URL 반환
    GENERATED_DIR 밖 파일이면: out_dir로 복사 후 처리
    """
    p = Path(p).resolve()
    gen = GENERATED_DIR.resolve()

    # GENERATED_DIR 밖 파일이면 out_dir로 복사
    try:
        rel = p.relative_to(gen)
    except ValueError:
        out_dir = out_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / p.name
        if p != dst:
            dst.write_bytes(p.read_bytes())
        p = dst
        rel = dst.relative_to(gen)

    # ✅ S3 업로드 모드
    if USE_S3:
        try:
            s3_key = f"{S3_PREFIX}/{rel.as_posix()}" if S3_PREFIX else rel.as_posix()
            content_type = _guess_content_type(p)
            s3_url = _upload_to_s3(p, s3_key, content_type)
            if s3_url:
                return s3_url
        except Exception as e:
            print(f"[S3 upload failed, fallback to local] {e}")

    # ✅ 로컬 fallback
    url = f"{STATIC_PREFIX}/" + rel.as_posix()
    if AI_PUBLIC_BASE_URL:
        return f"{AI_PUBLIC_BASE_URL}{url}"
    return url

async def _download_http_to_file(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT_SEC, follow_redirects=True) as client:
        r = await client.get(url)
        r.raise_for_status()
        await _write_bytes_async(dst, r.content)
    return dst

def _parse_bool(v: str | bool | None, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

def _pick_model_url(files_url: Dict[str, str]) -> str:
    for k, u in files_url.items():
        if "glb" in k.lower() or u.lower().endswith(".glb"):
            return u
    for k, u in files_url.items():
        if "pbr" in k.lower():
            return u
    return next(iter(files_url.values()))

def _sanitize_glb_url(u: str) -> str:
    u = (u or "").strip()
    if u.startswith("/api/api/"):
        u = u.replace("/api/api/", "/api/", 1)
    return u

def _local_generated_path_from_url(u: str) -> Optional[Path]:
    u = _sanitize_glb_url(u)
    if u.startswith("/api/generated/"):
        rel = u[len("/api/generated/"):]
        return (GENERATED_DIR / rel).resolve()
    if u.startswith("/generated/"):
        rel = u[len("/generated/"):]
        return (GENERATED_DIR / rel).resolve()
    return None

# -----------------------------
# OpenAI prompt builder (sync -> thread)
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
        max_output_tokens=800,  # ✅ 10000은 너무 큼(속도/비용/안정성)
    )

    prompt = _extract_text(resp)
    if not prompt:
        raise RuntimeError("Empty prompt from OpenAI")
    return prompt

# -----------------------------
# Nano Banana (Gemini) - sync -> thread
# -----------------------------
PROMPT_NANO_BANANA = """
Create ONE SINGLE IMAGE that is a 2x2 grid collage (four panels inside one image).
Do NOT output separate images. Do NOT output a single-view image.

Layout (must be visible as 4 distinct panels with clear spacing/borders):
- Top-left: front view
- Top-right: left 3/4 view
- Bottom-left: right 3/4 view
- Bottom-right: back view

Rules:
- The result must be ONE composite image containing all four panels.
- Clean white studio background in every panel.
- Soft shadow in every panel.
- Keep the exact same LEGO model/subject consistent across panels.
- Same lighting and scale across panels.
- Add thin dividers or margins between panels so the 2x2 grid is obvious.
"""

def _decode_if_base64_image(data: bytes | str) -> bytes:
    if data is None:
        return b""

    if isinstance(data, str):
        s = data.strip()
    else:
        try:
            s = data.decode("utf-8", errors="strict").strip()
        except Exception:
            return data

    if s.startswith("data:image"):
        try:
            s = s.split(",", 1)[1].strip()
        except Exception:
            pass

    looks_like_base64 = (
        s.startswith("iVBOR") or
        s.startswith("/9j/") or
        all(c.isalnum() or c in "+/=\n\r" for c in s[:200])
    )

    if looks_like_base64:
        try:
            return base64.b64decode(s, validate=False)
        except Exception:
            return data if isinstance(data, (bytes, bytearray)) else s.encode("utf-8")

    return data if isinstance(data, (bytes, bytearray)) else s.encode("utf-8")

def _render_one_image_sync(img_bytes: bytes, mime: str) -> bytes:
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    model = os.environ.get("NANO_BANANA_MODEL", "gemini-2.5-flash-image")
    client = genai.Client(api_key=gemini_key)

    resp = client.models.generate_content(
        model=model,
        contents=[
            {"text": PROMPT_NANO_BANANA},
            {"inline_data": {"mime_type": mime, "data": img_bytes}},
        ],
        config=genai_types.GenerateContentConfig(response_modalities=["Text", "Image"]),
    )

    if not resp.candidates:
        raise ValueError("no candidates from model")

    parts = resp.candidates[0].content.parts if resp.candidates[0].content else []
    out_bytes = None
    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            out_bytes = inline.data
            break

    if out_bytes is None:
        raise ValueError("no image returned from model")

    out_bytes = _decode_if_base64_image(out_bytes)

    # PNG/JPG 매직넘버 체크
    if len(out_bytes) >= 2 and out_bytes[0] == 0xFF and out_bytes[1] == 0xD8:
        return out_bytes
    if len(out_bytes) >= 8 and out_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return out_bytes

    # 남아있는 base64 한 번 더
    try:
        head = out_bytes[:20].decode("utf-8", errors="ignore")
        if head.startswith("iVBOR") or head.startswith("/9j/"):
            out_bytes = base64.b64decode(out_bytes, validate=False)
    except Exception:
        pass

    return out_bytes

async def render_one_image_async(img_bytes: bytes, mime: str) -> bytes:
    # ✅ gemini 호출은 동기라서 thread로 빼야 “완전 async 안전”
    return await anyio.to_thread.run_sync(_render_one_image_sync, img_bytes, mime)

# -----------------------------
# Brickify engine loader
# -----------------------------
AGE_TO_BUDGET = {"4-5": 20, "6-7": 60, "8-10": 120}

def _budget_to_start_target(eff_budget: int) -> int:
    if eff_budget <= 25:
        return 24
    if eff_budget <= 70:
        return 45
    return 60

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

def _find_glb_in_dir(out_dir: Path) -> Optional[Path]:
    glbs = [p for p in out_dir.rglob("*.glb") if p.is_file() and p.stat().st_size > 0]
    return glbs[0] if glbs else None

def _pick_glb_from_downloaded(downloaded: Dict[str, str], out_dir: Path) -> Optional[Path]:
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
    prompt: str | None = Form(default=None),
    returnLdrData: str | bool | None = Form(default="true"),
):
    import time
    total_start = time.time()

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

    print("═" * 70)
    print(f"🚀 [AI-SERVER] 요청 시작 | reqId={req_id}")
    print(f"📁 파일: {file.filename} | size={file.size}bytes | type={file.content_type}")
    print(f"📊 파라미터: age={age} | budget={budget} | prompt={'custom' if prompt else 'auto'}")
    print(f"⚙️  S3 모드: {'✅ ON' if USE_S3 else '❌ OFF'} | bucket={S3_BUCKET or 'N/A'}")
    print("═" * 70)

    try:
        # ✅ 전체 타임아웃 (무한 대기 방지)
        with anyio.fail_after(KIDS_TOTAL_TIMEOUT_SEC):

            # -----------------
            # 0) 입력 저장
            # -----------------
            step_start = time.time()
            print(f"📌 [STEP 0/5] 입력 이미지 저장 중...")
            img_bytes = await file.read()
            raw_ext = _mime_to_ext(file.content_type or "image/png")
            raw_path = out_req_dir / f"raw{raw_ext}"
            await _write_bytes_async(raw_path, img_bytes)
            print(f"✅ [STEP 0/5] 입력 저장 완료 | {len(img_bytes)/1024:.1f}KB | {time.time()-step_start:.2f}s")

            # -----------------
            # 1) 보정 (Gemini) - thread로 안전
            # -----------------
            step_start = time.time()
            print(f"📌 [STEP 1/5] Gemini 이미지 보정 시작...")
            corrected_bytes = await render_one_image_async(img_bytes, file.content_type or "image/png")
            corrected_path = out_req_dir / "corrected.png"
            await _write_bytes_async(corrected_path, corrected_bytes)
            corrected_url = _to_generated_url(corrected_path, out_dir=out_req_dir)
            print(f"✅ [STEP 1/5] Gemini 보정 완료 | {len(corrected_bytes)/1024:.1f}KB | {time.time()-step_start:.2f}s")

            # -----------------
            # 2) Prompt (OpenAI) - thread로 안전
            # -----------------
            step_start = time.time()
            if prompt and prompt.strip():
                prompt_text = prompt.strip()
                print(f"📌 [STEP 2/5] 커스텀 프롬프트 사용 | len={len(prompt_text)}")
            else:
                print(f"📌 [STEP 2/5] OpenAI 프롬프트 생성 시작...")
                prompt_text = await anyio.to_thread.run_sync(
                    build_lego_prompt_sync, corrected_bytes, "image/png"
                )
                print(f"✅ [STEP 2/5] 프롬프트 생성 완료 | len={len(prompt_text)} | {time.time()-step_start:.2f}s")
                print(f"   📝 프롬프트: {prompt_text[:100]}...")

            # -----------------
            # 3) Tripo 3D (생성 → 대기 → 다운로드)
            # -----------------
            step_start = time.time()
            print(f"📌 [STEP 3/5] Tripo 3D 모델 생성 시작... (timeout={TRIPO_WAIT_TIMEOUT_SEC}s)")
            negative_prompt = (
                "low quality, blurry, noisy, over-detailed, high poly, "
                "thin parts, tiny features, text, logos, patterns, textures"
            )

            async with TripoClient(api_key=TRIPO_API_KEY) as client:
                task_id = await client.text_to_model(prompt=prompt_text, negative_prompt=negative_prompt)
                print(f"   🔄 Tripo 작업 생성됨 | taskId={task_id}")

                # ✅ Tripo 대기 타임아웃
                with anyio.fail_after(TRIPO_WAIT_TIMEOUT_SEC):
                    task = await client.wait_for_task(task_id, verbose=DEBUG)

                if task.status != TaskStatus.SUCCESS:
                    raise RuntimeError(f"Tripo task failed: status={task.status}")

                print(f"   ✅ Tripo 작업 완료 | status={task.status}")
                downloaded = await client.download_task_models(task, str(out_tripo_dir))
                print(f"   📥 Tripo 파일 다운로드 완료 | files={list(downloaded.keys()) if downloaded else 'None'}")

            tripo_elapsed = time.time() - step_start
            print(f"✅ [STEP 3/5] Tripo 완료 | {tripo_elapsed:.2f}s")

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

            print(f"   📦 GLB 준비완료 | path={glb_path.name} | size={glb_path.stat().st_size/1024:.1f}KB")

            # -----------------
            # 5) Brickify 실행 (CPU heavy -> thread)
            # -----------------
            step_start = time.time()
            eff_budget = int(budget) if budget is not None else int(AGE_TO_BUDGET.get(age.strip(), 60))
            start_target = _budget_to_start_target(eff_budget)
            print(f"📌 [STEP 4/5] Brickify LDR 변환 시작... | budget={eff_budget} | target={start_target}")

            global _CONVERT_FN
            if _CONVERT_FN is None:
                _CONVERT_FN = _load_engine_convert()

            out_ldr = out_brick_dir / "result.ldr"

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

            brickify_elapsed = time.time() - step_start
            print(f"✅ [STEP 4/5] Brickify 완료 | parts={result.get('parts')} | target={result.get('final_target')} | {brickify_elapsed:.2f}s")

            if not out_ldr.exists() or out_ldr.stat().st_size == 0:
                raise RuntimeError("LDR output missing/empty")

            # -----------------
            # 5) 결과 URL 생성
            # -----------------
            step_start = time.time()
            print(f"📌 [STEP 5/5] 결과 URL 생성 중... (S3={'ON' if USE_S3 else 'OFF'})")
            ldr_url = _to_generated_url(out_ldr, out_dir=out_brick_dir)

            ldr_data_uri: Optional[str] = None
            if _parse_bool(returnLdrData, default=True):
                b = await _read_bytes_async(out_ldr)
                b64_str = base64.b64encode(b).decode("utf-8")
                ldr_data_uri = f"data:text/plain;base64,{b64_str}"

            print(f"✅ [STEP 5/5] URL 생성 완료 | {time.time()-step_start:.2f}s")

            total_elapsed = time.time() - total_start
            print("═" * 70)
            print(f"🎉 [AI-SERVER] 요청 완료! | reqId={req_id}")
            print(f"⏱️  총 소요시간: {total_elapsed:.2f}s ({total_elapsed/60:.1f}분)")
            print(f"   - Tripo 3D: {tripo_elapsed:.2f}s")
            print(f"   - Brickify: {brickify_elapsed:.2f}s")
            print(f"📦 결과: parts={result.get('parts')} | ldrSize={out_ldr.stat().st_size/1024:.1f}KB")
            print("═" * 70)

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
        total_elapsed = time.time() - total_start
        tb = traceback.format_exc()
        print("═" * 70)
        print(f"❌ [AI-SERVER] 요청 실패! | reqId={req_id} | 소요시간={total_elapsed:.2f}s")
        print(f"❌ 에러: {str(e)}")
        print("═" * 70)
        print(tb)
        _write_error_log(out_req_dir, tb)
        _write_error_log(out_tripo_dir, tb)
        _write_error_log(out_brick_dir, tb)

        if DEBUG:
            raise HTTPException(status_code=500, detail={"reqId": req_id, "error": str(e)})

        raise HTTPException(status_code=500, detail="process failed")

