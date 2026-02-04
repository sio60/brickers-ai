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
    if cur.is_file(): cur = cur.parent
    markers = ("pyproject.toml", "requirements.txt", ".git")
    for p in [cur] + list(cur.parents):
        for m in markers:
            if (p / m).exists(): return p
    return cur

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "")).expanduser()
PROJECT_ROOT = PROJECT_ROOT.resolve() if str(PROJECT_ROOT).strip() else _find_project_root(Path(__file__))

PUBLIC_DIR = Path(os.environ.get("PUBLIC_DIR", PROJECT_ROOT / "public")).resolve()
GENERATED_DIR = Path(os.environ.get("GENERATED_DIR", PUBLIC_DIR / "generated")).resolve()
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

STATIC_PREFIX = os.environ.get("GENERATED_URL_PREFIX", "/api/generated").rstrip("/")
KIDS_TOTAL_TIMEOUT_SEC = int(os.environ.get("KIDS_TOTAL_TIMEOUT_SEC", "1800"))
TRIPO_WAIT_TIMEOUT_SEC = int(os.environ.get("TRIPO_WAIT_TIMEOUT_SEC", "900"))
DOWNLOAD_TIMEOUT_SEC = float(os.environ.get("KIDS_DOWNLOAD_TIMEOUT_SEC", "180.0"))

# -----------------------------
# ✅ S3 Integration
# -----------------------------
try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None
    ClientError = Exception

AI_PUBLIC_BASE_URL = os.environ.get("AI_PUBLIC_BASE_URL", "").strip().rstrip("/")
AWS_REGION = os.environ.get("AWS_REGION", "").strip() or os.environ.get("AWS_DEFAULT_REGION", "").strip()
S3_BUCKET = os.environ.get("AWS_S3_BUCKET", "").strip()
S3_PUBLIC_BASE_URL = os.environ.get("S3_PUBLIC_BASE_URL", "").strip().rstrip("/")
USE_S3 = _is_truthy(os.environ.get("USE_S3", "true" if S3_BUCKET else "false"))
S3_PREFIX = os.environ.get("S3_PREFIX", "uploads/ai-generated").strip().strip("/")
S3_PRESIGN = _is_truthy(os.environ.get("S3_PRESIGN", "false"))
S3_PRESIGN_EXPIRES = int(os.environ.get("S3_PRESIGN_EXPIRES", "86400"))
S3_FORCE_ATTACHMENT = _is_truthy(os.environ.get("S3_FORCE_ATTACHMENT", "true"))
S3_USE_ACL_PUBLIC_READ = _is_truthy(os.environ.get("S3_USE_ACL_PUBLIC_READ", "false"))

_S3_CLIENT = None

def _get_s3_client():
    global _S3_CLIENT
    if _S3_CLIENT is not None: return _S3_CLIENT
    if boto3 is None or not S3_BUCKET: return None
    _S3_CLIENT = boto3.client("s3", region_name=AWS_REGION)
    return _S3_CLIENT

def _s3_url_for_key(key: str) -> str:
    if S3_PRESIGN:
        client = _get_s3_client()
        return client.generate_presigned_url("get_object", Params={"Bucket": S3_BUCKET, "Key": key}, ExpiresIn=S3_PRESIGN_EXPIRES)
    if S3_PUBLIC_BASE_URL: return f"{S3_PUBLIC_BASE_URL}/{key}"
    host = f"{S3_BUCKET}.s3.amazonaws.com" if AWS_REGION == "us-east-1" else f"{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com"
    return f"https://{host}/{key}"

def _upload_to_s3(local_path: Path, key: str, content_type: str | None = None) -> str:
    if not USE_S3: return ""
    client = _get_s3_client()
    if not client: return ""
    extra_args = {}
    if content_type: extra_args["ContentType"] = content_type
    if S3_FORCE_ATTACHMENT and local_path.suffix.lower() in (".ldr", ".glb"):
        extra_args["ContentDisposition"] = f'attachment; filename="{local_path.name}"'
    if S3_USE_ACL_PUBLIC_READ: extra_args["ACL"] = "public-read"
    client.upload_file(str(local_path), S3_BUCKET, key, ExtraArgs=extra_args if extra_args else None)
    return _s3_url_for_key(key)

def _to_generated_url(p: Path, out_dir: Path) -> str:
    p = Path(p).resolve(); gen = GENERATED_DIR.resolve()
    try: rel = p.relative_to(gen)
    except ValueError:
        out_dir = out_dir.resolve(); out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / p.name
        if p != dst: dst.write_bytes(p.read_bytes())
        p = dst; rel = dst.relative_to(gen)

    if USE_S3:
        try:
            now = datetime.now()
            s3_key = f"{S3_PREFIX}/{now.year:04d}/{now.month:02d}/{rel.as_posix()}" if S3_PREFIX else rel.as_posix()
            url = _upload_to_s3(p, s3_key, _guess_content_type(p))
            if url: return url
        except Exception as e: print(f"[S3 upload failed] {e}")

    url = f"{STATIC_PREFIX}/" + rel.as_posix()
    return f"{AI_PUBLIC_BASE_URL}{url}" if AI_PUBLIC_BASE_URL else url

def _guess_content_type(path: Path) -> str:
    return {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".glb": "application/octet-stream", ".ldr": "text/plain", ".json": "application/json"}.get(path.suffix.lower(), "application/octet-stream")

async def _write_bytes_async(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    await anyio.to_thread.run_sync(path.write_bytes, data)

async def _download_from_s3(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        r = await client.get(url); r.raise_for_status(); return r.content

# -----------------------------
# ✅ BOM (Bill of Materials)
# -----------------------------
def _generate_bom_from_ldr(ldr_path: Path) -> Dict[str, Any]:
    from collections import Counter
    if not ldr_path.exists(): return {"total_parts": 0, "parts": []}
    lines = ldr_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    pc = Counter()
    for ln in lines:
        toks = ln.strip().split()
        if len(toks) >= 15 and toks[0] == "1":
            p_id = toks[14][:-4] if toks[14].endswith(".dat") else toks[14]
            pc[f"{p_id}_{toks[1]}"] += 1
    bom_parts = [{"part_id": k.rsplit("_", 1)[0], "color": k.rsplit("_", 1)[1], "quantity": v} for k, v in pc.most_common()]
    return {"total_parts": sum(pc.values()), "parts": bom_parts}

# -----------------------------
# ✅ Nano Banana (Gemini)
# -----------------------------
PROMPT_NANO_BANANA = """Create a high-quality, vibrant image suitable for 3D modeling... (simplified for brevity)"""
def _render_one_image_sync(img_bytes: bytes, mime: str) -> bytes:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key: raise RuntimeError("GEMINI_API_KEY missing")
    model_name = os.environ.get("NANO_BANANA_MODEL", "gemini-2.0-flash-exp")
    client = genai.Client(api_key=key)
    resp = client.models.generate_content(model=model_name, contents=[{"text": PROMPT_NANO_BANANA}, {"inline_data": {"mime_type": mime, "data": img_bytes}}], config=genai_types.GenerateContentConfig(response_modalities=["Text", "Image"]))
    for part in (resp.candidates[0].content.parts if resp.candidates else []):
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            data = inline.data
            if isinstance(data, str) and (data.startswith("iVBOR") or data.startswith("/9j/")): return base64.b64decode(data)
            return data
    raise ValueError("No image from Gemini")

async def render_one_image_async(img_bytes: bytes, mime: str) -> bytes:
    return await anyio.to_thread.run_sync(_render_one_image_sync, img_bytes, mime)

# -----------------------------
# ✅ Engine Loader
# -----------------------------
AGE_TO_BUDGET = {"4-5": 400, "6-7": 450, "8-10": 500}
_CONVERT_FN = None

def _load_engine_convert():
    import importlib.util
    path = (PROJECT_ROOT / "brick-engine" / "glb_to_ldr_embedded.py").resolve()
    spec = importlib.util.spec_from_file_location("glb_to_ldr_embedded", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
    return mod.convert_glb_to_ldr

# -----------------------------
# ✅ Core Process Logic
# -----------------------------
async def process_kids_request_internal(job_id: str, source_image_url: str, age: str, budget: Optional[int] = None) -> Dict[str, Any]:
    req_id = job_id; out_dir = GENERATED_DIR / f"req_{req_id}"; out_dir.mkdir(parents=True, exist_ok=True)
    log(f"🚀 [Kids-Internal] Start | jobId={job_id} | age={age}")
    
    try:
        with anyio.fail_after(KIDS_TOTAL_TIMEOUT_SEC):
            # 1. Image
            img_bytes = await _download_from_s3(source_image_url)
            corrected_bytes = await render_one_image_async(img_bytes, "image/png")
            corrected_path = out_dir / "corrected.png"; await _write_bytes_async(corrected_path, corrected_bytes)
            corrected_url = _to_generated_url(corrected_path, out_dir)

            # 2. Tripo
            await update_job_stage(job_id, "THREE_D_PREVIEW")
            async with TripoClient(api_key=os.environ.get("TRIPO_API_KEY", "")) as client:
                tid = await client.image_to_model(image=str(corrected_path))
                task = await client.wait_for_task(tid)
                downloads = await client.download_task_models(task, str(out_dir))
            
            glb_path = Path(next(v for k,v in downloads.items() if "glb" in k.lower() or str(v).lower().endswith(".glb")))
            model_url = _to_generated_url(glb_path, out_dir)

            # 3. Brickify
            await update_job_stage(job_id, "MODEL")
            eff_budget = budget if budget else AGE_TO_BUDGET.get(age, 450)
            global _CONVERT_FN; 
            if not _CONVERT_FN: _CONVERT_FN = _load_engine_convert()
            
            # [Added] Progress Callback
            def _progress_cb(msg: str):
                log(f"   📣 [BrickGen] {msg}")
                try:
                    with httpx.Client(timeout=1.0) as client:
                        client.post(f"{BACKEND_URL}/api/kids/jobs/{job_id}/logs", json={"message": msg, "level": "INFO"})
                except Exception: pass

            out_ldr = out_dir / "result.ldr"
            result = await anyio.to_thread.run_sync(
                lambda: _CONVERT_FN(str(glb_path), str(out_ldr), budget=int(eff_budget), target=60, smart_fix=True, mode="kids", callback=_progress_cb)
            )
            ldr_url = _to_generated_url(out_ldr, out_dir)

            # 4. BOM
            bom_data = await anyio.to_thread.run_sync(_generate_bom_from_ldr, out_ldr)
            out_bom = out_dir / "bom.json"; import json
            await _write_bytes_async(out_bom, json.dumps(bom_data, indent=2).encode("utf-8"))
            bom_url = _to_generated_url(out_bom, out_dir)

            return {"correctedUrl": corrected_url, "modelUrl": model_url, "ldrUrl": ldr_url, "bomUrl": bom_url, "parts": result["parts"]}
    except Exception as e:
        log(f"❌ Failed: {str(e)}"); traceback.print_exc(); raise RuntimeError(str(e))

class KidsProcessRequest(BaseModel): sourceImageUrl: str; age: str = "6-7"; budget: Optional[int] = None
class ProcessResp(BaseModel): ok: bool; reqId: str; correctedUrl: str; modelUrl: str; ldrUrl: str; bomUrl: str; parts: int

@router.post("/process-all", response_model=ProcessResp)
async def process(request: KidsProcessRequest):
    req_id = uuid.uuid4().hex
    res = await process_kids_request_internal(req_id, request.sourceImageUrl, request.age, request.budget)
    return {"ok": True, "reqId": req_id, **res}
