# brickers-ai/route/kids_render.py
"""Kids Mode 라우터 + 오케스트레이션 (리팩토링 버전)"""
from __future__ import annotations

import os
import json
import re
import uuid
import base64
import time
import traceback
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import asyncio

import anyio
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tripo3d import TripoClient
from tripo3d.models import TaskStatus

# --- Service modules ---
from service.kids_config import (
    GENERATED_DIR,
    KIDS_TOTAL_TIMEOUT_SEC,
    TRIPO_WAIT_TIMEOUT_SEC,
    DOWNLOAD_TIMEOUT_SEC,
    MAX_CONCURRENT_TASKS,
    AGE_TO_BUDGET,
    budget_to_start_target,
    DEBUG,
)
from service.s3_client import USE_S3, S3_BUCKET, to_generated_url, upload_bytes_to_s3
from service.gemini_image import render_one_image_async
from service.backend_client import (
    update_job_stage,
    update_job_suggested_tags,
    make_agent_log_sender,
    send_agent_log,
)
from service.bom_generator import generate_bom_from_ldr
from service.brickify_loader import (
    load_engine_convert,
    load_agent_modules,
    find_glb_in_dir,
    pick_glb_from_downloaded,
)

# PDF Generation
from route.instructions_pdf import parse_ldr_step_boms, generate_pdf_with_images_and_bom
from service.render_client import render_ldr_steps, RENDER_ENABLED

# Re-export for app.py / sqs_consumer.py
__all__ = ["router", "GENERATED_DIR", "process_kids_request_internal"]

router = APIRouter(prefix="/api/v1/kids", tags=["kids"])

# --- Engine cache ---
_CONVERT_FN = None

def log(msg: str, user_email: str = "System") -> None:
    """타임스탬프 및 사용자 정보 포함 로그 출력"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    # 긴 이메일은 앞부분만 출력
    user_tag = f"[{user_email}]" if user_email else "[System]"
    print(f"[{ts}] {user_tag} {msg}")


# --------------- helpers ---------------

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


async def _download_http_to_file(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT_SEC, follow_redirects=True) as client:
        r = await client.get(url)
        r.raise_for_status()
        await _write_bytes_async(dst, r.content)
    return dst


async def _download_from_s3(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content


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



# --------------- Request / Response ---------------

class KidsProcessRequest(BaseModel):
    sourceImageUrl: str
    userEmail: Optional[str] = "unknown@brickers.shop"  # [추가]
    age: str = "6-7"
    budget: Optional[int] = None
    subject: Optional[str] = None
    prompt: Optional[str] = None
    returnLdrData: bool = False


class ProcessResp(BaseModel):
    ok: bool
    reqId: str
    correctedUrl: str
    taskId: str
    modelUrl: str
    files: Dict[str, str]
    ldrUrl: str
    ldrData: Optional[str] = None
    bomUrl: str
    pdfUrl: Optional[str] = None # [New]
    subject: str
    tags: list[str]
    parts: int
    finalTarget: int


# --------------- Core orchestration ---------------

async def process_kids_request_internal(
    job_id: str,
    source_image_url: str,
    age: str,
    budget: Optional[int] = None,
    subject: Optional[str] = None,
    user_email: str = "unknown", # [추가]
) -> Dict[str, Any]:
    """
    Kids 렌더링 내부 로직 (SQS Consumer에서 호출)
    시그니처/리턴 100% 동일 유지
    """
    total_start = time.time()
    
    # 내부 래퍼 로그 (이메일 자동 주입)
    def _log(msg: str):
        log(msg, user_email=user_email)

    TRIPO_API_KEY = os.environ.get("TRIPO_API_KEY", "")
    if not TRIPO_API_KEY:
        raise RuntimeError("TRIPO_API_KEY is not set")

    req_id = job_id
    out_req_dir = GENERATED_DIR / f"req_{req_id}"
    out_tripo_dir = GENERATED_DIR / f"tripo_{req_id}"
    out_brick_dir = GENERATED_DIR / f"brickify_{req_id}"
    out_req_dir.mkdir(parents=True, exist_ok=True)
    out_tripo_dir.mkdir(parents=True, exist_ok=True)
    out_brick_dir.mkdir(parents=True, exist_ok=True)

    # --- SSE 실시간 로그 전송용 (async) ---
    async def _sse(step: str, message: str):
        """파이프라인 단계별 SSE 로그 전송 (async httpx)"""
        await send_agent_log(job_id, step, message)

    _log("\u2550" * 70)
    _log(f"\U0001f680 [AI-SERVER] \uc694\uccad \uc2dc\uc791 | jobId={job_id}")
    _log(f"\U0001f4c1 \uc6d0\ubcf8 \uc774\ubbf8\uc9c0 URL: {source_image_url}")
    _log(f"\U0001f4ca \ud30c\ub77c\ubbf8\ud130: subject={subject} | age={age} | budget={budget}")
    s3_label = "ON" if USE_S3 else "OFF"
    _log(f"\u2699\ufe0f  S3 \ubaa8\ub4dc: {s3_label} | bucket={S3_BUCKET or 'N/A'}")
    _log("\u2550" * 70)

    try:
        with anyio.fail_after(KIDS_TOTAL_TIMEOUT_SEC):

            # 0) S3에서 원본 이미지 다운로드
            step_start = time.time()
            _log(f"\U0001f4cc [STEP 0/5] S3\uc5d0\uc11c \uc6d0\ubcf8 \uc774\ubbf8\uc9c0 \ub2e4\uc6b4\ub85c\ub4dc \uc911...")
            await _sse("download", "그림을 받아서 살펴보고 있어요...")
            img_bytes = await _download_from_s3(source_image_url)
            raw_path = out_req_dir / "raw.png"
            await _write_bytes_async(raw_path, img_bytes)
            _log(f"\u2705 [STEP 0/5] \ub2e4\uc6b4\ub85c\ub4dc \uc644\ub8cc | {len(img_bytes)/1024:.1f}KB | {time.time()-step_start:.2f}s")

            # 1) Gemini 보정
            step_start = time.time()
            _log(f"\U0001f4cc [STEP 1/5] Gemini \uc774\ubbf8\uc9c0 \ubcf4\uc815 \ubc0f \ud0dc\uadf8 \ucd94\ucd9c \uc2dc\uc791...")
            await _sse("gemini", "어떤 모양인지 분석하고 있어요...")
            corrected_bytes, ai_subject, ai_tags = await render_one_image_async(img_bytes, "image/png")

            final_subject = subject or ai_subject

            corrected_path = out_req_dir / "corrected.png"
            await _write_bytes_async(corrected_path, corrected_bytes)
            corrected_url = to_generated_url(corrected_path, out_dir=out_req_dir)
            _log(f"\u2705 [STEP 1/5] Gemini \uc644\ub8cc | Subject: {final_subject} | Tags: {ai_tags} | {time.time()-step_start:.2f}s")

            await update_job_suggested_tags(job_id, ai_tags)

            # 2) Tripo 3D
            step_start = time.time()
            _log(f"\U0001f4cc [STEP 2/4] Tripo 3D \ubaa8\ub378 \uc0dd\uc131 \uc2dc\uc791 (image-to-model)... (timeout={TRIPO_WAIT_TIMEOUT_SEC}s)")
            await _sse("tripo", "입체적인 3D 모델을 상상하고 있어요...")
            await update_job_stage(job_id, "THREE_D_PREVIEW")

            async with TripoClient(api_key=TRIPO_API_KEY) as client:
                task_id = await client.image_to_model(image=str(corrected_path))
                print(f"   \U0001f504 Tripo \uc791\uc5c5 \uc0dd\uc131\ub428 | taskId={task_id}")

                with anyio.fail_after(TRIPO_WAIT_TIMEOUT_SEC):
                    task = await client.wait_for_task(task_id, verbose=DEBUG)

                if task.status != TaskStatus.SUCCESS:
                    raise RuntimeError(f"Tripo task failed: status={task.status}")

                print(f"   \u2705 Tripo \uc791\uc5c5 \uc644\ub8cc | status={task.status}")
                downloaded = await client.download_task_models(task, str(out_tripo_dir))
                print(f"   \U0001f4e5 Tripo \ud30c\uc77c \ub2e4\uc6b4\ub85c\ub4dc \uc644\ub8cc | files={list(downloaded.keys()) if downloaded else 'None'}")

            tripo_elapsed = time.time() - step_start
            _log(f"\u2705 [STEP 2/4] Tripo \uc644\ub8cc | {tripo_elapsed:.2f}s")

            # 3-1) downloaded 정규화
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

            # 3-2) URL map
            files_url: Dict[str, str] = {}
            for model_type, path_str in fixed_downloaded.items():
                files_url[model_type] = to_generated_url(Path(path_str), out_dir=out_tripo_dir)

            if not any(u.lower().endswith(".glb") for u in files_url.values()):
                glb_fallback = find_glb_in_dir(out_tripo_dir)
                if glb_fallback:
                    files_url["glb"] = to_generated_url(glb_fallback, out_dir=out_tripo_dir)

            if not files_url:
                raise RuntimeError("No downloadable model files found in out_tripo_dir")

            model_url = _pick_model_url(files_url)

            # 3) GLB 확보
            glb_path = pick_glb_from_downloaded(fixed_downloaded, out_tripo_dir)

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

            print(f"   \U0001f4e6 GLB \uc900\ube44\uc644\ub8cc | path={glb_path.name} | size={glb_path.stat().st_size/1024:.1f}KB")

            # 4) Brickify (CPU Intensive - Process Pool for True Parallelism)
            step_start = time.time()
            eff_budget = int(budget) if budget is not None else int(AGE_TO_BUDGET.get(age.strip(), 100))
            # PRO 모드(5000개 수준) 시 복셀 제한 상향 (해상도 유지)
            v_limit = 50000 if eff_budget >= 4000 else (20000 if eff_budget >= 1000 else 6000)
            start_target = budget_to_start_target(eff_budget)
            
            _log(f"🚀 [STEP 3/4] Brickify LDR 변환 시작... | budget={eff_budget} | target={start_target}")
            await update_job_stage(job_id, "MODEL")
            await _sse("brickify", "레고 블록으로 변환하고 있어요...")

            out_ldr = out_brick_dir / "result.ldr"

            # Engine 로드 (캐시됨)
            global _CONVERT_FN
            if _CONVERT_FN is None:
                _CONVERT_FN = load_engine_convert()

            # Brickify 실행 (CPU-heavy -> thread로 실행)
            def run_brickify():
                return _CONVERT_FN(
                    str(glb_path),
                    str(out_ldr),
                    target=start_target,
                    budget=eff_budget,
                    min_target=5,
                    shrink=0.85,
                    search_iters=3,        # 6→3 (속도 개선)
                    kind="brick",
                    plates_per_voxel=3,
                    interlock=True,
                    max_area=20,
                    solid_color=4,
                    use_mesh_color=True,
                    invert_y=False,
                    smart_fix=True,
                    span=4,
                    max_new_voxels=v_limit,
                    refine_iters=4,        # 8→4 (속도 개선)
                    ensure_connected=True,
                    min_embed=2,
                    erosion_iters=1,
                    fast_search=True,
                    step_order="bottomup",
                    extend_catalog=True,
                    max_len=8,
                )

            result = await anyio.to_thread.run_sync(run_brickify)

            brickify_elapsed = time.time() - step_start
            _log(f"\u2705 [STEP 3/4] Brickify \uc644\ub8cc | parts={result.get('parts')} | target={result.get('final_target')} | {brickify_elapsed:.2f}s")

            if not out_ldr.exists() or out_ldr.stat().st_size == 0:
                raise RuntimeError(f"LDR output missing/empty: {out_ldr}")

            # 5) 결과 URL + BOM
            step_start = time.time()
            s3_mode = "ON" if USE_S3 else "OFF"
            _log(f"\U0001f4cc [STEP 4/4] \uacb0\uacfc URL \uc0dd\uc131 \ubc0f BOM \ud30c\uc77c \uc0dd\uc131 \uc911... (S3={s3_mode})")
            await _sse("bom", "어떤 부품이 필요한지 정리하고 있어요...")
            ldr_url = to_generated_url(out_ldr, out_dir=out_brick_dir)

            print(f"   \U0001f4cb BOM \ud30c\uc77c \uc0dd\uc131 \uc911...")
            bom_data = await anyio.to_thread.run_sync(generate_bom_from_ldr, out_ldr)
            out_bom = out_brick_dir / "bom.json"
            await _write_bytes_async(out_bom, json.dumps(bom_data, indent=2, ensure_ascii=False).encode("utf-8"))
            bom_url = to_generated_url(out_bom, out_dir=out_brick_dir)
            print(f"   \u2705 BOM \ud30c\uc77c \uc0dd\uc131 \uc644\ub8cc | total_parts={bom_data['total_parts']} | unique={len(bom_data['parts'])}")

            _log(f"\u2705 [STEP 4/4] URL \uc0dd\uc131 \uc644\ub8cc | {time.time()-step_start:.2f}s")

            # 5-2) PDF 생성 (렌더링 서비스 사용)
            pdf_url = None
            if RENDER_ENABLED:
                try:
                    step_start_pdf = time.time()
                    _log(f"📌 [STEP 5/5] PDF 생성 시작 (LDView 로컬 렌더링)")
                    await _sse("pdf", "조립 설명서를 만들고 있어요...")

                    # LDR 텍스트 읽기
                    ldr_text = await anyio.to_thread.run_sync(
                        lambda: out_ldr.read_text(encoding="utf-8")
                    )

                    # 렌더링 서비스에서 스텝별 이미지 받기
                    step_images = await render_ldr_steps(ldr_text)
                    _log(f"   ✅ 렌더링 완료: {len(step_images)} steps")

                    # LDR에서 Step별 BOM 파싱
                    step_boms = parse_ldr_step_boms(ldr_text)

                    # 커버 이미지 (마지막 스텝 첫 번째 뷰)
                    cover_img = None
                    if step_images and step_images[-1] and step_images[-1][0]:
                        cover_img = step_images[-1][0]

                    # PDF 생성
                    pdf_bytes = generate_pdf_with_images_and_bom(
                        model_name=final_subject or "Brickers Model",
                        step_images=step_images,
                        step_boms=step_boms,
                        cover_image=cover_img,
                    )
                    _log(f"   ✅ PDF 생성: {len(pdf_bytes)} bytes")

                    # S3 업로드
                    if USE_S3 and S3_BUCKET:
                        now = datetime.now()
                        safe_name = re.sub(r'[\\/:*?"<>|]+', "_", final_subject or "instructions")
                        s3_key = f"uploads/pdf/{now.year:04d}/{now.month:02d}/{uuid.uuid4().hex[:8]}_{safe_name}.pdf"
                        pdf_url = upload_bytes_to_s3(pdf_bytes, s3_key, "application/pdf")
                        _log(f"   ✅ PDF S3 업로드: {pdf_url}")
                    else:
                        # 로컬 저장
                        pdf_path = out_brick_dir / "instructions.pdf"
                        await _write_bytes_async(pdf_path, pdf_bytes)
                        pdf_url = to_generated_url(pdf_path, out_dir=out_brick_dir)

                    _log(f"✅ [STEP 5/5] PDF 생성 완료 | {time.time()-step_start_pdf:.2f}s")
                except Exception as pdf_err:
                    _log(f"⚠️ [STEP 5/5] PDF 생성 실패 (파이프라인 계속): {pdf_err}")
                    pdf_url = None
            else:
                _log(f"📌 [INFO] LDView 미설치 - PDF 생성 스킵")

            await _sse("complete", "완성! 브릭 모델이 준비됐어요!")

            total_elapsed = time.time() - total_start
            _log("\u2550" * 70)
            _log(f"\U0001f389 [AI-SERVER] \uc694\uccad \uc644\ub8cc! | jobId={job_id}")
            _log(f"\u23f1\ufe0f  \ucd1d \uc18c\uc694\uc2dc\uac04: {total_elapsed:.2f}s ({total_elapsed/60:.1f}\ubd84)")
            _log(f"   - Tripo 3D: {tripo_elapsed:.2f}s")
            _log(f"   - Brickify: {brickify_elapsed:.2f}s")
            _log(f"\U0001f4e6 \uacb0\uacfc: parts={result.get('parts')} | ldrSize={out_ldr.stat().st_size/1024:.1f}KB")
            print("\u2550" * 70)

            return {
                "correctedUrl": corrected_url,
                "modelUrl": model_url,
                "ldrUrl": ldr_url,
                "bomUrl": bom_url,
                "pdfUrl": pdf_url,
                "subject": final_subject,
                "tags": ai_tags,
                "parts": int(result.get("parts", 0)),
                "finalTarget": int(result.get("final_target", 0)),
            }

    except Exception as e:
        total_elapsed = time.time() - total_start
        tb = traceback.format_exc()
        _log("\u2550" * 70)
        _log(f"\u274c [AI-SERVER] \uc694\uccad \uc2e4\ud328! | jobId={job_id} | \uc18c\uc694\uc2dc\uac04={total_elapsed:.2f}s")
        _log(f"\u274c \uc5d0\ub7ec: {str(e)}")
        _log("\u2550" * 70)
        log(tb, user_email=user_email)
        _write_error_log(out_req_dir, tb)
        _write_error_log(out_tripo_dir, tb)
        _write_error_log(out_brick_dir, tb)
        raise RuntimeError(str(e)) from e


# --------------- HTTP endpoint ---------------

@router.post("/process-all", response_model=ProcessResp)
async def process(request: KidsProcessRequest):
    """Kids Mode 처리 (HTTP 엔드포인트) - 호환성 유지"""
    req_id = uuid.uuid4().hex

    try:
        result = await process_kids_request_internal(
            job_id=req_id,
            source_image_url=request.sourceImageUrl,
            age=request.age,
            budget=request.budget,
            subject=request.subject,
            user_email=request.userEmail or "unknown",
        )

        ldr_data_uri: Optional[str] = None
        if not USE_S3 and request.returnLdrData:
            ldr_path = _local_generated_path_from_url(result["ldrUrl"])
            if ldr_path and ldr_path.exists():
                b = await _read_bytes_async(ldr_path)
                b64_str = base64.b64encode(b).decode("utf-8")
                ldr_data_uri = f"data:text/plain;base64,{b64_str}"

        return {
            "ok": True,
            "reqId": req_id,
            "correctedUrl": result["correctedUrl"],
            "taskId": req_id,
            "modelUrl": result["modelUrl"],
            "files": {},
            "ldrUrl": result["ldrUrl"],
            "ldrData": ldr_data_uri,
            "bomUrl": result["bomUrl"],
            "pdfUrl": result.get("pdfUrl"), # [New]
            "subject": result["subject"],
            "tags": result["tags"],
            "parts": result["parts"],
            "finalTarget": result["finalTarget"],
        }

    except HTTPException:
        raise
    except Exception as e:
        if DEBUG:
            raise HTTPException(status_code=500, detail={"reqId": req_id, "error": str(e)})
        raise HTTPException(status_code=500, detail="process failed")
