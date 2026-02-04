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
import asyncio
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

import httpx

def log(msg: str) -> None:
    """타임스탬프 포함 로그 출력"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}")

# ---- OpenAI ----
from openai import AsyncOpenAI

# ---- Gemini (Nano Banana) ----
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

router = APIRouter(prefix="/api/v1/kids", tags=["kids"])

# -----------------------------
# Backend 연동 (Stage 업데이트)
# -----------------------------
BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8080").rstrip("/")

async def update_job_stage(job_id: str, stage: str) -> None:
    """Backend에 Job stage 업데이트 요청 (SSL 에러 방지)"""
    try:
        async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
            target_url = f"{BACKEND_URL}/api/kids/jobs/{job_id}/stage"
            await client.patch(target_url, json={"stage": stage})
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

def _upload_to_s3(local_path: Path, key: str, content_type: str | None = None) -> str:
    if not USE_S3 or boto3 is None or not S3_BUCKET: return ""
    client = boto3.client("s3", region_name=AWS_REGION)
    extra_args = {}
    if content_type: extra_args["ContentType"] = content_type
    if S3_FORCE_ATTACHMENT and local_path.suffix.lower() in (".ldr", ".glb"):
        extra_args["ContentDisposition"] = f'attachment; filename="{local_path.name}"'
    if S3_USE_ACL_PUBLIC_READ: extra_args["ACL"] = "public-read"
    client.upload_file(str(local_path), S3_BUCKET, key, ExtraArgs=extra_args if extra_args else None)
    if S3_PRESIGN: return client.generate_presigned_url("get_object", Params={"Bucket": S3_BUCKET, "Key": key}, ExpiresIn=S3_PRESIGN_EXPIRES)
    if S3_PUBLIC_BASE_URL: return f"{S3_PUBLIC_BASE_URL}/{key}"
    host = f"{S3_BUCKET}.s3.amazonaws.com" if AWS_REGION == "us-east-1" else f"{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com"
    return f"https://{host}/{key}"

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
            s3_key = f"{S3_PREFIX}/{datetime.now().strftime('%Y/%m')}/{rel.as_posix()}" if S3_PREFIX else rel.as_posix()
            url = _upload_to_s3(p, s3_key, _guess_content_type(p))
            if url: return url
        except Exception as e: print(f"[S3 fail] {e}")
    url = f"{STATIC_PREFIX}/{rel.as_posix()}"
    return f"{AI_PUBLIC_BASE_URL}{url}" if AI_PUBLIC_BASE_URL else url

def _guess_content_type(path: Path) -> str:
    return {".png": "image/png", ".jpg": "image/jpeg", ".glb": "application/octet-stream", ".ldr": "text/plain"}.get(path.suffix.lower(), "application/octet-stream")

async def _write_bytes_async(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    await anyio.to_thread.run_sync(path.write_bytes, data)

async def _download_from_s3(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        r = await client.get(url); r.raise_for_status(); return r.content

# -----------------------------
# OpenAI GPT Vision Analyzer
# -----------------------------
async def analyze_image_with_gpt_vision(img_bytes: bytes) -> str:
    """GPT-4o Vision을 사용하여 이미지를 분석하고 고품질 텍스트 프롬프트 생성"""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    client = AsyncOpenAI(api_key=api_key)
    base64_image = base64.b64encode(img_bytes).decode('utf-8')
    prompt = """Analyze this child's drawing for 3D modeling.
1. Identify the subject and infer vibrant, representative colors (especially for sketches).
2. Describe it as a 'vibrant, high-quality, volumetric 3D plastic toy, smooth glossy finish, clear color separation, studio lighting'.
3. Output the concise 3D prompt only."""
    response = await client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]}],
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()

# -----------------------------
# Gemini (Nano Banana) Correction
# -----------------------------
PROMPT_VOLUMETRIC = """Transform this 2D drawing into a 3D-volumetric plastic toy. Smooth, glossy finish, vibrant colors, white background, no text/people."""

async def render_one_image_async(img_bytes: bytes, prompt: str) -> bytes:
    """Gemini를 사용하여 이미지를 정제된 3D 토이 느낌으로 변환"""
    if not genai: raise RuntimeError("google-genai not installed")
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
    resp = client.models.generate_content(
        model=os.environ.get("NANO_BANANA_MODEL", "gemini-2.0-flash-exp"),
        contents=[{"text": prompt}, {"inline_data": {"mime_type": "image/png", "data": img_bytes}}],
        config=genai_types.GenerateContentConfig(response_modalities=["Text", "Image"])
    )
    for part in (resp.candidates[0].content.parts if resp.candidates else []):
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            data = inline.data
            return base64.b64decode(data) if isinstance(data, str) and (data.startswith("iVBOR") or data.startswith("/9j/")) else data
    raise ValueError("No image from Gemini")

# -----------------------------
# Tripo API (V3 Refinement)
# -----------------------------
TRIPO_API_BASE = "https://api.tripo3d.ai/v2/openapi"

async def tripo_call(api_key: str, method: str, path: str, payload: Dict = None) -> Dict:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.request(method, f"{TRIPO_API_BASE}{path}", headers=headers, json=payload)
        r.raise_for_status(); data = r.json()
        if data.get("code") != 0: raise RuntimeError(f"Tripo Error: {data.get('message')}")
        return data["data"]

async def tripo_wait(api_key: str, tid: str, timeout: int = 900) -> Dict:
    start = time.time()
    while time.time() - start < timeout:
        task = await tripo_call(api_key, "GET", f"/task/{tid}")
        if task["status"] == "success": return task
        if task["status"] in ("failed", "cancelled"): raise RuntimeError(f"Tripo task {tid} {task['status']}")
        await anyio.sleep(5)
    raise TimeoutError(f"Tripo task {tid} timeout")

# -----------------------------
# Core Process Logic (Hybrid V3)
# -----------------------------
_CONVERT_FN = None
def _load_engine_convert():
    import importlib.util
    path = (PROJECT_ROOT / "brick-engine" / "glb_to_ldr_embedded.py").resolve()
    spec = importlib.util.spec_from_file_location("glb_to_ldr_embedded", str(path)); mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod; spec.loader.exec_module(mod); return mod.convert_glb_to_ldr

async def process_kids_request_internal(job_id: str, source_image_url: str, age: str, budget: Optional[int] = None) -> Dict[str, Any]:
    out_dir = GENERATED_DIR / f"req_{job_id}"; out_dir.mkdir(parents=True, exist_ok=True)
    log(f"🚀 [Kids-V3-Ultimate] Start | job={job_id}")
    
    try:
        with anyio.fail_after(1800):
            img_bytes = await _download_from_s3(source_image_url)
            
            # 1. Hybrid Enhancement (Gemini for Image, GPT for Context)
            log(f"   [AI] Running Hybrid Enhancement (Gemini + GPT)...")
            gpt_prompt, corrected_bytes = await asyncio.gather(
                analyze_image_with_gpt_vision(img_bytes),
                render_one_image_async(img_bytes, PROMPT_VOLUMETRIC)
            )
            corrected_path = out_dir / "corrected.png"; await _write_bytes_async(corrected_path, corrected_bytes)
            corrected_url = _to_generated_url(corrected_path, out_dir)
            log(f"   ✅ [AI] Corrected image & GPT prompt ready.")

            # 2. Tripo V3 Two-Stage
            await update_job_stage(job_id, "THREE_D_PREVIEW")
            tripo_api_key = os.environ.get("TRIPO_API_KEY", "")
            
            # Step 2.1: Initial Mesh (Try Image, Fallback to Text)
            tid = None
            try:
                log(f"   [Tripo] Step 1: image_to_model (Using S3 URL: {corrected_url})...")
                tid = await tripo_create_task_v2(tripo_api_key, {
                    "type": "image_to_model", 
                    "url": corrected_url, 
                    "model_version": "v3.0-20250812"
                })
            except Exception as e:
                log(f"   ⚠️ [Tripo-Image] Failed or Rejected: {str(e)}")
                if "2014" in str(e) or "auditing" in str(e).lower() or "400" in str(e):
                    log(f"   ⚠️ [Fallback] Falling back to text_to_model (GPT prompt)...")
                    tid = await tripo_create_task_v2(tripo_api_key, {
                        "type": "text_to_model", 
                        "prompt": gpt_prompt, 
                        "model_version": "v3.0-20250812"
                    })
                else: raise e
            
            await tripo_wait(tripo_api_key, tid)

            # Step 2.2: Texture Refinement (HD Color Enhancement)
            log(f"   [Tripo] Step 2: texture_model refinement (detailed HD)...")
            refine_tid = await tripo_create_task_v2(tripo_api_key, {
                "type": "texture_model", 
                "original_model_task_id": tid, 
                "texture_quality": "detailed", 
                "texture": True,
                "pbr": True,
                "model_version": "v3.0-20250812"
            })
            final_task = await tripo_wait(tripo_api_key, refine_tid)
            
            async with httpx.AsyncClient() as client:
                r = await client.get(final_task["output"]["model"]); glb_path = out_dir / "model.glb"; glb_path.write_bytes(r.content)
            model_url = _to_generated_url(glb_path, out_dir)

            # 3. Brickify & BOM
            await update_job_stage(job_id, "MODEL")
            global _CONVERT_FN; _CONVERT_FN = _CONVERT_FN or _load_engine_convert()
            out_ldr = out_dir / "result.ldr"
            result_obj = await anyio.to_thread.run_sync(lambda: _CONVERT_FN(str(glb_path), str(out_ldr), budget=int(budget or 450), target=60, smart_fix=True, mode="kids"))
            ldr_url = _to_generated_url(out_ldr, out_dir)
            
            # Extract parts count safely
            parts_count = result_obj.get("parts", 0) if isinstance(result_obj, dict) else getattr(result_obj, "total_bricks", 0)

            # 4. BOM
            from collections import Counter
            lines = out_ldr.read_text(encoding="utf-8", errors="ignore").splitlines()
            pc = Counter()
            for ln in lines:
                toks = ln.strip().split()
                if len(toks) >= 15 and toks[0] == "1": pc[f"{toks[14][:-4] if toks[14].endswith('.dat') else toks[14]}_{toks[1]}"] += 1
            bom_parts = [{"part_id": k.rsplit("_", 1)[0], "color": k.rsplit("_", 1)[1], "quantity": v} for k, v in pc.most_common()]
            out_bom = out_dir / "bom.json"; out_bom.write_text(import_json().dumps({"total_parts": sum(pc.values()), "parts": bom_parts}, indent=2))
            bom_url = _to_generated_url(out_bom, out_dir)

            return {"correctedUrl": corrected_url, "modelUrl": model_url, "ldrUrl": ldr_url, "bomUrl": bom_url, "parts": parts_count, "finalTarget": 60}
    except Exception as e: log(f"❌ Failed: {str(e)}"); traceback.print_exc(); raise RuntimeError(str(e))

# Helper for V2 Task Creation (Internal used)
async def tripo_create_task_v2(api_key: str, payload: Dict) -> str:
    res = await tripo_call(api_key, "POST", "/task", payload); return res["task_id"]

def import_json(): import json; return json

class KidsProcessRequest(BaseModel): sourceImageUrl: str; age: str = "6-7"; budget: Optional[int] = None
class ProcessResp(BaseModel): ok: bool; reqId: str; correctedUrl: str; modelUrl: str; ldrUrl: str; bomUrl: str; parts: int; finalTarget: int


@router.post("/process-all", response_model=ProcessResp)
async def process(request: KidsProcessRequest):
    req_id = uuid.uuid4().hex
    res = await process_kids_request_internal(req_id, request.sourceImageUrl, request.age, request.budget)
    return {"ok": True, "reqId": req_id, **res}
