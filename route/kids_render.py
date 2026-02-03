# brickers-ai/route/kids_render.py
from __future__ import annotations

import os
import sys
import base64
import uuid
import traceback
import time
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime

import anyio
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

import httpx


def log(msg: str) -> None:
    """타임스탬프 포함 로그 출력"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}")

# ---- OpenAI ----
# OpenAI import removed - no longer needed

# ---- Tripo ----
from tripo3d import TripoClient
from tripo3d.models import TaskStatus

# ---- Gemini (Nano Banana) ----
from google import genai
from google.genai import types as genai_types


router = APIRouter(prefix="/api/v1/kids", tags=["kids"])

# -----------------------------
# Backend 연동 (Stage 업데이트)
# -----------------------------
BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8080").rstrip("/")

async def update_job_stage(job_id: str, stage: str) -> None:
    """
    Backend에 Job stage 업데이트 요청
    - 실패해도 전체 플로우에 영향 없음 (로그만 출력)
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.patch(
                f"{BACKEND_URL}/api/kids/jobs/{job_id}/stage",
                json={"stage": stage}
            )
        print(f"   ✅ [Stage Update] {stage}")
    except Exception as e:
        print(f"   ⚠️ [Stage Update] 실패 (무시) | stage={stage} | error={str(e)}")

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
S3_PREFIX = os.environ.get("S3_PREFIX", "uploads/ai-generated").strip().strip("/")

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
    from datetime import datetime

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

    # ✅ S3 업로드 모드 (타임스탬프 기반 경로 구조화)
    if USE_S3:
        try:
            now = datetime.now()
            year, month = now.year, now.month
            # uploads/ai-generated/2026/01/req_abc/corrected.png
            s3_key = f"{S3_PREFIX}/{year:04d}/{month:02d}/{rel.as_posix()}" if S3_PREFIX else rel.as_posix()
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

async def _download_from_s3(url: str) -> bytes:
    """S3 URL에서 파일 다운로드"""
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content

def _generate_bom_from_ldr(ldr_path: Path) -> Dict[str, Any]:
    """
    LDR 파일에서 BOM (Bill of Materials) 생성
    Returns: {
        "total_parts": int,
        "parts": [{"part_id": str, "color": str, "quantity": int}, ...]
    }
    """
    from collections import Counter

    if not ldr_path.exists():
        return {"total_parts": 0, "parts": []}

    content = ldr_path.read_text(encoding="utf-8", errors="ignore")
    lines = content.splitlines()

    # LDraw 파츠 라인 추출 (타입 1)
    parts_counter = Counter()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("0"):  # 주석 또는 메타데이터
            continue

        parts = line.split()
        if len(parts) >= 15 and parts[0] == "1":  # 타입 1: 서브파일 참조 (파츠)
            color = parts[1]
            part_id = parts[14] if len(parts) > 14 else "unknown"
            # .dat 확장자 제거
            if part_id.endswith(".dat"):
                part_id = part_id[:-4]

            key = f"{part_id}_{color}"
            parts_counter[key] += 1

    # BOM 생성
    bom_parts = []
    for key, qty in parts_counter.most_common():
        part_id, color = key.rsplit("_", 1)
        bom_parts.append({
            "part_id": part_id,
            "color": color,
            "quantity": qty
        })

    return {
        "total_parts": sum(parts_counter.values()),
        "parts": bom_parts
    }

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
# OpenAI prompt builder - REMOVED (직접 이미지로 Tripo 호출)
# -----------------------------
# ✅ OpenAI 프롬프트 생성 단계 제거됨
# ✅ Nano Banana 이미지를 Tripo에 직접 전달

# -----------------------------
# Nano Banana (Gemini) - sync -> thread
# -----------------------------
PROMPT_NANO_BANANA = """
Create a high-quality, vibrant image suitable for 3D modeling.

Requirements:
- Single view from the best angle to capture the subject's key features
- Clean white studio background
- Soft, even lighting with subtle shadows
- Vivid, saturated colors with clear color separation
- Sharp, well-defined edges and contours
- High contrast between subject and background
- Professional product photography style
- Remove any text, logos, patterns, or decorative details
- Simplify complex textures into solid, uniform colors
- Maintain the overall shape and proportions of the subject

Output a single, polished image optimized for 3D mesh generation.
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
AGE_TO_BUDGET = {"4-5": 150, "6-7": 200, "8-10": 250}

def _budget_to_start_target(eff_budget: int) -> int:
    # Frontend budgets: 150 / 200 / 250
    if eff_budget <= 150:
        return 40
    if eff_budget <= 200:
        return 50
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
    # Ensure module is registered for dataclasses/type resolution
    sys.modules[spec.name] = mod
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
# Request schema
# -----------------------------
class KidsProcessRequest(BaseModel):
    sourceImageUrl: str  # S3 URL (Frontend가 직접 업로드한 URL)
    age: str = "6-7"
    budget: Optional[int] = None
    prompt: Optional[str] = None
    returnLdrData: bool = False  # S3 사용 시 기본값 False

# Response schema
# -----------------------------
class ProcessResp(BaseModel):
    ok: bool
    reqId: str

    correctedUrl: str

    taskId: str
    modelUrl: str
    files: Dict[str, str]

    ldrUrl: str
    ldrData: Optional[str] = None
    bomUrl: str  # ✅ BOM 파일 URL 추가

    parts: int
    finalTarget: int

# -----------------------------
# ✅ 내부 함수 (SQS Consumer에서 호출)
# -----------------------------
async def process_kids_request_internal(
    job_id: str,
    source_image_url: str,
    age: str,
    budget: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Kids 렌더링 내부 로직 (SQS Consumer에서 호출)
    - S3에서 이미지 다운로드
    - Gemini 보정
    - Tripo 3D 생성
    - Brickify LDR 변환
    - BOM 생성

    Args:
        job_id: Job ID (MongoDB)
        source_image_url: S3 업로드된 원본 이미지 URL
        age: 나이 범위 (4-5, 6-7, 8-10)
        budget: 예산 (None이면 age 기반 자동 설정)

    Returns:
        {
            "correctedUrl": str,
            "modelUrl": str,
            "ldrUrl": str,
            "bomUrl": str,
            "parts": int,
            "finalTarget": int,
        }

    Raises:
        RuntimeError: 처리 실패 시
    """
    import time
    total_start = time.time()

    TRIPO_API_KEY = os.environ.get("TRIPO_API_KEY", "")
    if not TRIPO_API_KEY:
        raise RuntimeError("TRIPO_API_KEY is not set")
    
    # ✅ Co-Scientist Agent 관련 Import
    # brick_engine.agent 패키지 경로 확보를 위해 sys.path 설정이 필요할 수 있음
    # (이미 _load_engine_convert 에서 경로 트릭을 쓰고 있지만, 여기서는 정석대로 import 시도)
    try:
        from brick_engine.agent.llm_regeneration_agent import regeneration_loop
        from brick_engine.agent.llm_clients import GeminiClient
    except ImportError:
        # 경로 문제 시 fallback: sys.path에 추가 후 import
        import sys
        agent_dir = PROJECT_ROOT / "brick-engine"
        if str(agent_dir) not in sys.path:
            sys.path.insert(0, str(agent_dir))
        from agent.llm_regeneration_agent import regeneration_loop
        from agent.llm_clients import GeminiClient

    req_id = job_id  # Job ID를 req_id로 사용
    out_req_dir = GENERATED_DIR / f"req_{req_id}"
    out_tripo_dir = GENERATED_DIR / f"tripo_{req_id}"
    out_brick_dir = GENERATED_DIR / f"brickify_{req_id}"
    out_req_dir.mkdir(parents=True, exist_ok=True)
    out_tripo_dir.mkdir(parents=True, exist_ok=True)
    out_brick_dir.mkdir(parents=True, exist_ok=True)

    log("═" * 70)
    log(f"🚀 [AI-SERVER] 요청 시작 (Co-Scientist Agent) | jobId={job_id}")
    log(f"📁 원본 이미지 URL: {source_image_url}")
    log(f"📊 파라미터: age={age} | budget={budget}")
    log(f"⚙️  S3 모드: {'✅ ON' if USE_S3 else '❌ OFF'} | bucket={S3_BUCKET or 'N/A'}")
    log("═" * 70)

    try:
        # ✅ 전체 타임아웃 (Co-Scientist Agent는 오래 걸릴 수 있으므로 넉넉히)
        agent_timeout = KIDS_TOTAL_TIMEOUT_SEC + 600 # 기존 + 10분 여유
        
        with anyio.fail_after(agent_timeout):

            # -----------------
            # 0) S3에서 원본 이미지 다운로드
            # -----------------
            step_start = time.time()
            log(f"📌 [STEP 0/5] S3에서 원본 이미지 다운로드 중...")
            img_bytes = await _download_from_s3(source_image_url)
            raw_path = out_req_dir / "raw.png"
            await _write_bytes_async(raw_path, img_bytes)
            log(f"✅ [STEP 0/5] 다운로드 완료 | {len(img_bytes)/1024:.1f}KB | {time.time()-step_start:.2f}s")

            # -----------------
            # 1) 보정 (Gemini) - thread로 안전
            # -----------------
            step_start = time.time()
            log(f"📌 [STEP 1/5] Gemini 이미지 보정 시작...")
            corrected_bytes = await render_one_image_async(img_bytes, "image/png")
            corrected_path = out_req_dir / "corrected.png"
            await _write_bytes_async(corrected_path, corrected_bytes)
            corrected_url = _to_generated_url(corrected_path, out_dir=out_req_dir)
            log(f"✅ [STEP 1/5] Gemini 보정 완료 | {len(corrected_bytes)/1024:.1f}KB | {time.time()-step_start:.2f}s")

            # -----------------
            # 2) Tripo 3D (이미지 → 3D 모델 생성)
            # -----------------
            step_start = time.time()
            log(f"📌 [STEP 2/4] Tripo 3D 모델 생성 시작 (image-to-model)... (timeout={TRIPO_WAIT_TIMEOUT_SEC}s)")

            # ✅ Backend에 stage 업데이트
            await update_job_stage(job_id, "THREE_D_PREVIEW")

            async with TripoClient(api_key=TRIPO_API_KEY) as client:
                # ✅ image_to_model: Nano Banana 이미지를 직접 Tripo에 전달
                task_id = await client.image_to_model(image=str(corrected_path))
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
            log(f"✅ [STEP 2/4] Tripo 완료 | {tripo_elapsed:.2f}s")

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
            # 3) Brickify 입력 GLB 확보
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
            # 4) Co-Scientist Agent 실행 (Brickify + Regeneration Loop)
            # -----------------
            step_start = time.time()
            log(f"📌 [STEP 3/4] Co-Scientist Agent 시작 (Generate -> Verify -> Fix)...")

            # ✅ Backend에 stage 업데이트
            await update_job_stage(job_id, "MODEL")

            # Agent 설정
            default_budget = int(AGE_TO_BUDGET.get(age.strip(), 60))
            eff_budget = int(budget) if budget is not None else default_budget
            start_target = _budget_to_start_target(eff_budget)
            
            out_ldr = out_brick_dir / "result.ldr"
            
            gemini_key = os.environ.get("GEMINI_API_KEY")
            if not gemini_key:
                # GEMINI_API_KEY가 없으면 에이전트 실행 불가 (또는 Groq 등 다른 키 필요)
                # 여기서는 일단 로그 찍고 진행 시도
                log("⚠️ GEMINI_API_KEY가 없습니다. 에이전트가 정상 작동하지 않을 수 있습니다.")

            # LLM 클라이언트 초기화
            llm_client = GeminiClient(api_key=gemini_key)

            # 에이전트 실행 (스레드로 격리)
            # loop 내에서 convert_glb_to_ldr 호출 시 필요한 기본 파라미터 전달
            # 하지만 regeneration_loop는 내부적으로 params 딕셔너리를 사용함
            # 따라서 초기 params를 agent에 맞게 구성해야 함
            
            # DEFAULT_PARAMS를 가져와서 budget/target만 수정
            # (llm_regeneration_agent.py의 DEFAULT_PARAMS와 싱크 필요)
            agent_params = {
                "target": start_target,
                "budget": eff_budget,
                "min_target": 5,
                "shrink": 0.85, # 에이전트 기본값 0.7이지만 여기서는 0.85로 시작
                "search_iters": 6,
                "kind": "brick",
                "plates_per_voxel": 3,
                "interlock": True,
                "max_area": 20,
                "solid_color": 4,
                "use_mesh_color": True,
                "invert_y": False,
                "smart_fix": True,
                "fill": False, # kids 모드는 속 빈 모델 선호? (원래 코드: fill=False with embedded logic)
                               # embedded 코드에서는 fill=False 처리됨.
                "step_order": "bottomup",
            }

            def run_agent_sync():
                return regeneration_loop(
                    glb_path=str(glb_path),
                    output_ldr_path=str(out_ldr),
                    llm_client=llm_client,
                    max_retries=3, # 재시도 횟수 제한
                    gui=False
                    # acceptable_failure_ratio 등은 기본값 사용
                )

            final_state = await anyio.to_thread.run_sync(run_agent_sync)
            
            # 결과 확인
            agent_report = final_state.get('final_report', {})
            success = agent_report.get('success', False)
            
            # 최종 메트릭
            final_metrics = agent_report.get('final_metrics', {})
            final_parts = final_metrics.get('total_bricks', 0)
            
            # 에이전트가 실패했더라도 LDR이 생성되었으면 진행 (최선)
            if not out_ldr.exists() or out_ldr.stat().st_size == 0:
                raise RuntimeError("Agent failed to generate any LDR file")

            brickify_elapsed = time.time() - step_start
            
            status_msg = "성공" if success else "실패(부분완료)"
            log(f"✅ [STEP 3/4] Co-Scientist 완료 ({status_msg}) | parts={final_parts} | {brickify_elapsed:.2f}s")
            
            # 결과 딕셔너리 구성 (기존 코드 호환)
            # result 딕셔너리 대신 final_state 정보를 활용
            
            # -----------------
            # 5) 결과 URL 생성 및 BOM 파일 생성
            # -----------------
            step_start = time.time()
            log(f"📌 [STEP 4/4] 결과 URL 생성 및 BOM 파일 생성 중... (S3={'ON' if USE_S3 else 'OFF'})")
            ldr_url = _to_generated_url(out_ldr, out_dir=out_brick_dir)

            # ✅ BOM (Bill of Materials) 파일 생성
            print(f"   📋 BOM 파일 생성 중...")
            bom_data = await anyio.to_thread.run_sync(_generate_bom_from_ldr, out_ldr)
            out_bom = out_brick_dir / "bom.json"
            import json
            await _write_bytes_async(out_bom, json.dumps(bom_data, indent=2, ensure_ascii=False).encode("utf-8"))
            bom_url = _to_generated_url(out_bom, out_dir=out_brick_dir)
            print(f"   ✅ BOM 파일 생성 완료 | total_parts={bom_data['total_parts']} | unique={len(bom_data['parts'])}")

            log(f"✅ [STEP 4/4] URL 생성 완료 | {time.time()-step_start:.2f}s")

            total_elapsed = time.time() - total_start
            log("═" * 70)
            log(f"🎉 [AI-SERVER] 요청 완료! | jobId={job_id}")
            log(f"⏱️  총 소요시간: {total_elapsed:.2f}s ({total_elapsed/60:.1f}분)")
            log(f"   - Tripo 3D: {tripo_elapsed:.2f}s")
            log(f"   - Brickify: {brickify_elapsed:.2f}s")
            log(f"📦 결과: parts={result.get('parts')} | ldrSize={out_ldr.stat().st_size/1024:.1f}KB")
            print("═" * 70)

            return {
                "correctedUrl": corrected_url,
                "modelUrl": model_url,
                "ldrUrl": ldr_url,
                "bomUrl": bom_url,
                "parts": int(result.get("parts", 0)),
                "finalTarget": int(result.get("final_target", 0)),
            }

    except Exception as e:
        total_elapsed = time.time() - total_start
        tb = traceback.format_exc()
        log("═" * 70)
        log(f"❌ [AI-SERVER] 요청 실패! | jobId={job_id} | 소요시간={total_elapsed:.2f}s")
        log(f"❌ 에러: {str(e)}")
        log("═" * 70)
        log(tb)
        _write_error_log(out_req_dir, tb)
        _write_error_log(out_tripo_dir, tb)
        _write_error_log(out_brick_dir, tb)

        raise RuntimeError(str(e)) from e


# -----------------------------
# ✅ HTTP API (호환성 유지)
# -----------------------------
@router.post("/process-all", response_model=ProcessResp)
async def process(request: KidsProcessRequest):
    """
    Kids Mode 처리 (HTTP 엔드포인트)
    - 기존 호환성 유지
    - 내부적으로 process_kids_request_internal 호출
    """
    req_id = uuid.uuid4().hex

    try:
        result = await process_kids_request_internal(
            job_id=req_id,
            source_image_url=request.sourceImageUrl,
            age=request.age,
            budget=request.budget,
        )

        # ✅ S3 사용 시에는 ldrData 생략 (불필요한 base64 인코딩 제거)
        ldr_data_uri: Optional[str] = None
        if not USE_S3 and request.returnLdrData:
            # ldrUrl에서 파일 읽기
            ldr_path = _local_generated_path_from_url(result["ldrUrl"])
            if ldr_path and ldr_path.exists():
                b = await _read_bytes_async(ldr_path)
                b64_str = base64.b64encode(b).decode("utf-8")
                ldr_data_uri = f"data:text/plain;base64,{b64_str}"

        return {
            "ok": True,
            "reqId": req_id,
            "correctedUrl": result["correctedUrl"],
            "taskId": req_id,  # task_id는 없지만 호환성 유지
            "modelUrl": result["modelUrl"],
            "files": {},  # files는 내부 함수에서 반환 안 함
            "ldrUrl": result["ldrUrl"],
            "ldrData": ldr_data_uri,
            "bomUrl": result["bomUrl"],
            "parts": result["parts"],
            "finalTarget": result["finalTarget"],
        }

    except HTTPException:
        raise
    except Exception as e:
        if DEBUG:
            raise HTTPException(status_code=500, detail={"reqId": req_id, "error": str(e)})
        raise HTTPException(status_code=500, detail="process failed")

