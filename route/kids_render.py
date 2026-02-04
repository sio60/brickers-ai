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
from openai import AsyncOpenAI

# ---- Tripo ----
from tripo3d import TripoClient
from tripo3d.models import TaskStatus

router = APIRouter(prefix="/api/v1/kids", tags=["kids"])

# -----------------------------
# Backend 연동 (Stage 업데이트)
# -----------------------------
BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8080").rstrip("/")

async def update_job_stage(job_id: str, stage: str) -> None:
    """
    Backend에 Job stage 업데이트 요청
    - SSL Error [WRONG_VERSION_NUMBER] 해결을 위해 verify=False 적용
    """
    try:
        async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
            target_url = f"{BACKEND_URL}/api/kids/jobs/{job_id}/stage"
            await client.patch(
                target_url,
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
# OpenAI GPT Vision Analyzer
# -----------------------------
async def analyze_image_with_gpt_vision(img_bytes: bytes) -> str:
    """GPT-4o Vision을 사용하여 이미지를 분석하고 Tripo용 고품질 텍스트 프롬프트 생성"""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key: raise RuntimeError("OPENAI_API_KEY is not set")
    
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    client = AsyncOpenAI(api_key=api_key)
    base64_image = base64.b64encode(img_bytes).decode('utf-8')
    
    prompt = """Analyze this child's drawing for 3D modeling.
1. Identify the main subject (e.g., an elephant, a sports car).
2. COLOR INFERENCE (CRITICAL): 
   - If the drawing is just a sketch (outlines only, no colors), infer and suggest vibrant, child-friendly representative colors for the object (e.g., a blue whale, a green dinosaur).
   - If it already has colors, maintain those colors but describe them vividly.
3. Style Guide:
   - Describe it as a 'vibrant, high-quality, volumetric 3D plastic toy'.
   - Ensure the style is 'smooth 3D rendering' with clear color separation.
4. Create a CONCISE but VIVID 3D generation prompt for Tripo AI.

Example for a sketch of a cat: 'A cute volumetric 3D plastic toy of a bright ginger cat with white paws, vibrant orange fur, smooth glossy finish, studio lighting'
Just output the prompt text only."""

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            }
        ],
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()

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
            # 1. GPT Vision Analysis (Color Inference included)
            img_bytes = await _download_from_s3(source_image_url)
            log(f"   [GPT] Analyzing drawing & inferring colors...")
            gpt_prompt = await analyze_image_with_gpt_vision(img_bytes)
            log(f"   ✅ [GPT] Generated 3D Prompt: {gpt_prompt}")

            # 2. Tripo (Text-to-3D)
            await update_job_stage(job_id, "THREE_D_PREVIEW")
            tripo_api_key = os.environ.get("TRIPO_API_KEY", "")
            async with TripoClient(api_key=tripo_api_key) as client:
                log(f"   [Tripo] Creating 3D model using Method B (Text-to-Model)...")
                tid = await client.text_to_model(prompt=gpt_prompt)
                log(f"   [Tripo] Task created: {tid}. Waiting for completion...")
                task = await client.wait_for_task(tid)
                
                if task.status != TaskStatus.SUCCESS:
                    raise RuntimeError(f"Tripo task failed with status: {task.status}")
                
                downloads = await client.download_task_models(task, str(out_dir))
            
            glb_path = Path(next(v for k,v in downloads.items() if "glb" in k.lower() or str(v).lower().endswith(".glb")))
            model_url = _to_generated_url(glb_path, out_dir)
            
            # Using original image as correctedUrl
            corrected_url = source_image_url

            # 3. Brickify
            await update_job_stage(job_id, "MODEL")
            eff_budget = budget if budget else AGE_TO_BUDGET.get(age.strip(), 450)
            global _CONVERT_FN; 
            if not _CONVERT_FN: _CONVERT_FN = _load_engine_convert()
            
            out_ldr = out_dir / "result.ldr"
            result_obj = await anyio.to_thread.run_sync(lambda: _CONVERT_FN(str(glb_path), str(out_ldr), budget=int(eff_budget), target=60, smart_fix=True, mode="kids"))
            ldr_url = _to_generated_url(out_ldr, out_dir)
            
            # Extract parts count safely
            parts_count = 0
            if isinstance(result_obj, dict):
                parts_count = result_obj.get("parts", 0)
            else:
                parts_count = getattr(result_obj, "total_bricks", 0)

            # 4. BOM
            bom_data = await anyio.to_thread.run_sync(_generate_bom_from_ldr, out_ldr)
            out_bom = out_dir / "bom.json"; import json
            await _write_bytes_async(out_bom, json.dumps(bom_data, indent=2).encode("utf-8"))
            bom_url = _to_generated_url(out_bom, out_dir)

            return {
                "correctedUrl": corrected_url, 
                "modelUrl": model_url, 
                "ldrUrl": ldr_url, 
                "bomUrl": bom_url, 
                "parts": parts_count, 
                "finalTarget": 60
            }
    except Exception as e:
        log(f"❌ Failed: {str(e)}")
        traceback.print_exc()
        raise RuntimeError(str(e))

class KidsProcessRequest(BaseModel): sourceImageUrl: str; age: str = "6-7"; budget: Optional[int] = None
class ProcessResp(BaseModel): ok: bool; reqId: str; correctedUrl: str; modelUrl: str; ldrUrl: str; bomUrl: str; parts: int; finalTarget: int

@router.post("/process-all", response_model=ProcessResp)
async def process(request: KidsProcessRequest):
    req_id = uuid.uuid4().hex
    res = await process_kids_request_internal(req_id, request.sourceImageUrl, request.age, request.budget)
    return {"ok": True, "reqId": req_id, **res}
