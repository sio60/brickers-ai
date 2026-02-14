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
import sys
import io
import asyncio
from typing import TextIO

from service.log_context import JobLogContext

# [REMOVED] Local LogCapture Class (Replaced by GlobalLogCapture + JobLogContext)

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
    AGE_TO_BUDGET,
    budget_to_start_target,
    DEBUG,
)
from service.s3_client import USE_S3, S3_BUCKET, to_generated_url, upload_bytes_to_s3
from service.gemini_image import render_one_image_async
from service.background_composer import generate_background_async
from service.backend_client import (
    update_job_stage,
    update_job_suggested_tags,
    make_agent_log_sender,
    send_agent_log,
    send_agent_trace,
)
from service.bom_generator import generate_bom_from_ldr
from service.ldraw_bundler import generate_parts_bundle
from service.brickify_loader import (
    load_engine_convert,
    load_agent_modules,
    find_glb_in_dir,
    pick_glb_from_downloaded,
)

# PDF Generation (SQS로 Blueprint 서버에 위임)
from route.sqs_producer import (
    send_pdf_request_message, 
    send_screenshot_request_message,
    send_background_request_message
)

# from brick_engine.agent.log_analyzer.persistence import archive_job_logs # [DECOUPLED]
from brick_engine.agent.log_analyzer.persistence import archive_job_logs

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
    print(f"[{ts}] {user_tag} {msg}", flush=True)


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
    language: Optional[str] = "en"  # [NEW]
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
    backgroundUrl: Optional[str] = None # [New]
    subject: str
    tags: list[str]
    parts: int
    finalTarget: int
    lmmLatency: Optional[int] = None # [New] AI 생성 시간 (ms)


# --------------- Core orchestration ---------------

async def process_kids_request_internal(
    job_id: str,
    source_image_url: str,
    age: str,
    budget: Optional[int] = None,
    subject: Optional[str] = None,
    user_email: str = "unknown",
    language: str = "en", # [NEW]
    external_log_buffer: Optional[list[str]] = None, # [NEW]
) -> Dict[str, Any]:
    """
    Kids 렌더링 내부 로직 (SQS Consumer에서 호출)
    """
    total_start = time.time()
    
    # 내부 래퍼 로그 (이메일 자동 주입 + Job ID 태깅)
    def _log(msg: str):
        log(f"[{job_id}] {msg}", user_email=user_email)

    TRIPO_API_KEY = os.environ.get("TRIPO_API_KEY", "")
    if not TRIPO_API_KEY:
        raise RuntimeError("TRIPO_API_KEY is not set")
    
    # [NEW] In-Memory Log Buffer (Use external if provided)
    if external_log_buffer is not None:
        job_log_buffer = external_log_buffer
    else:
        job_log_buffer = []

    req_id = job_id
    out_req_dir = GENERATED_DIR / f"req_{req_id}"
    out_tripo_dir = GENERATED_DIR / f"tripo_{req_id}"
    out_brick_dir = GENERATED_DIR / f"brickify_{req_id}"
    out_req_dir.mkdir(parents=True, exist_ok=True)
    out_tripo_dir.mkdir(parents=True, exist_ok=True)
    out_brick_dir.mkdir(parents=True, exist_ok=True)
    background_url: Optional[str] = None
    background_requested = False

    # 내부 래퍼 로그 (이메일 자동 주입 + 버퍼링)
    # [수정] LogCapture가 이미 stdout을 캡쳐하므로, 
    # 여기서는 print()만 찍으면 Docker와 Buffer 양쪽으로 들어감.
    # 단, log() 함수 자체가 print()를 호출하므로 중복 저장을 막기 위해
    # log()는 그대로 두고, LogCapture가 알아서 캡쳐하게 둠.
    def _log(msg: str):
        # 1. 콘솔 출력 -> LogCapture가 가로채서 Buffer에도 넣음
        log(f"[{job_id}] {msg}", user_email=user_email)
        
    # --- SSE 실시간 로그 전송용 (async) ---
    async def _sse(step: str, message: str = ""):
        """파이프라인 단계별 SSE 로그 전송 (async httpx)"""
        await send_agent_log(job_id, step, message)

    async def _trace(node: str, status: str, message: str = "", input_data: dict = None, output_data: dict = None, duration: int = 0):
        """에이전트 트레이스 수동 전송"""
        try:
            await send_agent_trace(
                job_id, 
                step="TRACE", 
                node_name=node, 
                status=status, 
                input_data=input_data or {}, 
                output_data=output_data or {"message": message}, 
                duration_ms=duration
            )
        except Exception:
            pass

    # [Trace] Job Start
    await _trace("System", "START", "Job Request Received", {"source": source_image_url, "params": {"age": age, "budget": budget, "subject": subject}})

    _log("\u2550" * 70)
    _log(f"\U0001f680 [AI-SERVER] \uc694\uccad \uc2dc\uc791") # jobId 중복 제거
    _log(f"\U0001f4c1 \uc6d0\ubcf8 \uc774\ubbf8\uc9c0 URL: {source_image_url}")
    _log(f"\U0001f4ca \ud30c\ub77c\ubbf8\ud130: subject={subject} | age={age} | budget={budget}")
    s3_label = "ON" if USE_S3 else "OFF"
    _log(f"\u2699\ufe0f  S3 \ubaa8\ub4dc: {s3_label} | bucket={S3_BUCKET or 'N/A'}")
    _log("\u2550" * 70)
    
    # [DECOUPLED] Manual archives removed. SQS Consumer handles auto-flushing.
    # [Restored as comments per user request]
    # async def _async_archive(status: str = "RUNNING"):
    #     try:
    #         await archive_job_logs(job_id, list(job_log_buffer), status=status)
    #     except Exception as e:
    #         print(f"⚠️ [LogArchive] Failed: {e}")

    try:
        # [CHANGE] Global Context 사용
        with JobLogContext(job_log_buffer):
            # 초기 start 로그도 캡쳐됨
            # await _async_archive() # [Restored comment]
            
            with anyio.fail_after(KIDS_TOTAL_TIMEOUT_SEC):

                await _sse("init", "요청을 받았어요. 이미지를 준비하고 있습니다.")

                # 0) S3에서 원본 이미지 다운로드
                step_start = time.time()
                _log("[STEP 0/5] S3에서 원본 이미지 다운로드 중...")
                await _sse("download", "이미지 수신 완료. 구조부터 살펴보겠습니다.")
                img_bytes = await _download_from_s3(source_image_url)
                raw_path = out_req_dir / "raw.png"
                await _write_bytes_async(raw_path, img_bytes)
                _log(f"[STEP 0/5] 다운로드 완료 | {len(img_bytes)/1024:.1f}KB | {time.time()-step_start:.2f}s")
                # await _async_archive() # [Restored comment]
            

                # 1) Gemini 보정
                step_start = time.time()
                _log(f"[STEP 1/5] Gemini 이미지 보정 및 태그 추출 시작... (lang={language})")
                await _sse("gemini", "명암과 형태를 분석합니다. 브릭 색상으로 옮기기 좋은 상태로 보정하고 있어요.")
                corrected_bytes, ai_subject, ai_tags = await render_one_image_async(img_bytes, "image/png", language=language)

                final_subject = subject or ai_subject

                corrected_path = out_req_dir / "corrected.png"
                await _write_bytes_async(corrected_path, corrected_bytes)
                corrected_url = to_generated_url(corrected_path, out_dir=out_req_dir)
                _log(f"[STEP 1/5] Gemini 완료 | Subject: {final_subject} | Tags: {ai_tags} | {time.time()-step_start:.2f}s")
                # await _async_archive() # [Restored comment]
                
                await update_job_suggested_tags(job_id, ai_tags)
 
                # [NEW] 배경 생성 요청 (Screenshot Server로 위임)
                for _bg_attempt in range(1, 4):
                    try:
                        await send_background_request_message(job_id, final_subject or "lego creation")
                        background_requested = True
                        _log(f"배경 생성 요청 전송 완료 | subject={final_subject}")
                        break
                    except Exception as bg_err:
                        _log(f"배경 생성 요청 실패 (시도 {_bg_attempt}/3): {bg_err}")
                        if _bg_attempt < 3:
                            import asyncio as _aio
                            await _aio.sleep(2 ** _bg_attempt)
 
                # 2) Tripo 3D
                step_start = time.time()
                _log(f"[STEP 2/4] Tripo 3D 모델 생성 시작 (image-to-model)... (timeout={TRIPO_WAIT_TIMEOUT_SEC}s)")
                await _sse("tripo", "2D 정보를 바탕으로 3D 형태를 잡아봅니다.")
                await update_job_stage(job_id, "THREE_D_PREVIEW")

                async with TripoClient(api_key=TRIPO_API_KEY) as client:
                    task_id = await client.image_to_model(image=str(corrected_path))
                    _log(f"   Tripo 작업 생성됨 | taskId={task_id}")

                    # [CHANGE] Custom Async Polling (Non-blocking)
                    # wait_for_task might use time.sleep(), blocking the event loop.
                    # We implement our own loop with asyncio.sleep() to allow _auto_flush_logs to run.
                    start_time = time.time()
                    last_progress = 0 # [FIX] Initialize variable
                    while True:
                        if time.time() - start_time > TRIPO_WAIT_TIMEOUT_SEC:
                            raise RuntimeError(f"Tripo task timed out after {TRIPO_WAIT_TIMEOUT_SEC}s")

                        # Check status
                        task = await client.get_task(task_id)
                        if task.status == TaskStatus.SUCCESS:
                            break
                        elif task.status in (TaskStatus.FAILED, TaskStatus.CANCELLED):
                             raise RuntimeError(f"Tripo task failed: status={task.status}")
                        
                        # [CHANGE] Only log if progress changes or every 10s (Heartbeat)
                        # This prevents spamming "Generating..." every 2 seconds.
                        current_progress = task.progress if hasattr(task, 'progress') else '?'
                        elapsed = time.time() - start_time
                        
                        # Log only on progress change or every 10s
                        if current_progress != last_progress or (int(elapsed) % 10 == 0 and int(elapsed) > 0):
                             _log(f"      [Tripo] Generating... ({int(elapsed)}s) | progress={current_progress}")
                             last_progress = current_progress

                        await asyncio.sleep(5.0) # Yield control to _auto_flush_logs

                    _log(f"   Tripo 작업 완료 | status={task.status}")
                    downloaded = await client.download_task_models(task, str(out_tripo_dir))
                    _log(f"   Tripo 파일 다운로드 완료 | files={list(downloaded.keys()) if downloaded else 'None'}")
                    
                    # [Trace] Tripo Success
                    await _trace("Tripo3D", "SUCCESS", "Model Generated", {"task_id": task_id}, {"files": list(downloaded.keys())}, int((time.time() - step_start) * 1000))

                tripo_elapsed = time.time() - step_start
                _log(f"[STEP 2/4] Tripo 완료 | {tripo_elapsed:.2f}s")
                # await _async_archive() # [Restored comment]

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

                _log(f"   GLB 준비완료 | path={glb_path.name} | size={glb_path.stat().st_size/1024:.1f}KB")

                # 4) CoScientist Brickify (Generate → Debate → Evolve)
                step_start = time.time()
                eff_budget = int(budget) if budget is not None else int(AGE_TO_BUDGET.get(age.strip(), 100))
                # PRO 모드(5000개 수준) 시 복셀 제한 상향 (해상도 유지)
                v_limit = 50000 if eff_budget >= 4000 else (20000 if eff_budget >= 1000 else 6000)
                start_target = budget_to_start_target(eff_budget)

                _log(f"[STEP 3/4] CoScientist Brickify 시작... | budget={eff_budget} | target={start_target}")
                await update_job_stage(job_id, "MODEL")
                await _sse("brickify", "브릭 단위로 분해하면서 안정적인 조합을 탐색 중이에요.")

                out_ldr = out_brick_dir / "result.ldr"

                # Brickify 파라미터 (regeneration_loop & fallback 공용)
                brickify_params = dict(
                    target=start_target,
                    budget=eff_budget,
                    min_target=5,
                    shrink=0.6,
                    search_iters=10,
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
                    refine_iters=4,
                    ensure_connected=True,
                    min_embed=2,
                    erosion_iters=1,
                    fast_search=True,
                    step_order="bottomup",
                    extend_catalog=True,
                    max_len=8,
                    avoid_1x1=True,
                )

                # CoScientist regeneration_loop 시도 → 실패 시 단순 brickify fallback
                used_coscientist = False
                try:
                    regen_loop_fn, gemini_cls = load_agent_modules()
                    _log("[CoScientist] LLM 재생성 에이전트 활성화")
                    await _sse("coscientist", "CoScientist가 구조를 검증하며 최적의 브릭 배치를 찾고 있어요.")

                    # SSE 로그 콜백을 params에 주입
                    regen_params = brickify_params.copy()
                    regen_params["log_callback"] = make_agent_log_sender(job_id)
                    regen_params["job_id"] = job_id # [NEW] Trace용

                    # [ASYNC CHANGE] Direct await
                    final_state = await regen_loop_fn(
                        glb_path=str(glb_path),
                        output_ldr_path=str(out_ldr),
                        subject_name=final_subject or "Unknown Object",
                        llm_client=gemini_cls(),
                        max_retries=1,
                        acceptable_failure_ratio=0.1,
                        params=regen_params,
                    )

                    # 결과 추출
                    report = final_state.get('final_report', {})
                    metrics = report.get('final_metrics', {})
                    parts_count = metrics.get('total_bricks', 0)
                    # LDR 파일에서 직접 카운트 (fallback)
                    if parts_count == 0 and out_ldr.exists():
                        ldr_text = out_ldr.read_text(encoding='utf-8')
                        parts_count = sum(1 for line in ldr_text.splitlines() if line.startswith('1 '))
                    result = {"parts": parts_count, "final_target": start_target}
                    used_coscientist = True
                    _log(f"[CoScientist] 완료 | 성공={report.get('success', '?')} | 시도={report.get('total_attempts', '?')}회")

                except Exception as cos_err:
                    _log(f"[CoScientist] 실패, 단순 Brickify로 fallback: {cos_err}")

                    # Fallback: 기존 단순 brickify
                    global _CONVERT_FN
                    if _CONVERT_FN is None:
                        _CONVERT_FN = load_engine_convert()

                    def run_brickify():
                        return _CONVERT_FN(str(glb_path), str(out_ldr), **brickify_params)

                    result = await anyio.to_thread.run_sync(run_brickify)

                brickify_elapsed = time.time() - step_start
                engine_label = "CoScientist" if used_coscientist else "Brickify"
                _log(f"[STEP 3/4] {engine_label} 완료 | parts={result.get('parts')} | target={result.get('final_target')} | {brickify_elapsed:.2f}s")
                # await _async_archive() # [Restored comment]
                

                if not out_ldr.exists() or out_ldr.stat().st_size == 0:
                    raise RuntimeError(f"LDR output missing/empty: {out_ldr}")

                # 5) 결과 URL + BOM
                step_start = time.time()
                s3_mode = "ON" if USE_S3 else "OFF"
                _log(f"[STEP 4/4] 결과 URL 생성 및 BOM 파일 생성 중... (S3={s3_mode})")
                await _sse("bom", "현재 설계를 기준으로 필요한 부품 수를 계산하고 있어요.")
                ldr_url = to_generated_url(out_ldr, out_dir=out_brick_dir)

                _log("   BOM 파일 생성 중...")
                bom_data = await anyio.to_thread.run_sync(generate_bom_from_ldr, out_ldr)
                out_bom = out_brick_dir / "bom.json"
                await _write_bytes_async(out_bom, json.dumps(bom_data, indent=2, ensure_ascii=False).encode("utf-8"))
                bom_url = to_generated_url(out_bom, out_dir=out_brick_dir)
                _log(f"   BOM 파일 생성 완료 | total_parts={bom_data['total_parts']} | unique={len(bom_data['parts'])}")

                # Parts Bundle 생성 (실패해도 파이프라인 계속)
                try:
                    bundle_data = await anyio.to_thread.run_sync(
                        lambda: generate_parts_bundle(out_ldr)
                    )
                    out_bundle = out_brick_dir / "parts-bundle.json"
                    await _write_bytes_async(out_bundle, json.dumps(bundle_data).encode("utf-8"))
                    bundle_url = to_generated_url(out_bundle, out_dir=out_brick_dir)
                    _log(f"   Parts Bundle 생성 완료 | parts={len(bundle_data['parts'])} | url={bundle_url}")
                except Exception as e:
                    _log(f"Parts Bundle 실패 (파이프라인 계속): {e}")

                _log(f"[STEP 4/4] URL 생성 완료 | {time.time()-step_start:.2f}s")

                # 5-2) PDF + Screenshot 생성 요청 (SQS로 위임)
                pdf_url = None
                await _sse("pdf", "조립 순서를 정리해서 설명서로 옮기고 있어요.")
                try:
                    await send_pdf_request_message(
                        job_id=job_id,
                        ldr_url=ldr_url,
                        model_name=final_subject or "Brickers Model",
                    )
                    _log("PDF 생성 요청 전송 (brickers-blueprints-queue)")
                except Exception as pdf_err:
                    _log(f"PDF SQS 전송 실패 (파이프라인 계속): {pdf_err}")

                for _ss_attempt in range(1, 4):
                    try:
                        await send_screenshot_request_message(
                            job_id=job_id,
                            ldr_url=ldr_url,
                            model_name=final_subject or "Brickers Model",
                        )
                        _log("스크린샷 생성 요청 전송 완료")
                        break
                    except Exception as ss_err:
                        _log(f"스크린샷 전송 실패 (시도 {_ss_attempt}/3): {ss_err}")
                        if _ss_attempt < 3:
                            import asyncio as _aio
                            await _aio.sleep(2 ** _ss_attempt)

                await _sse("complete", "설계가 끝났어요. 결과를 한번 살펴볼까요?")

                total_elapsed = time.time() - total_start
                _log("\u2550" * 70)
                _log(f"[AI-SERVER] 요청 완료! | jobId={job_id}")
                _log(f"총 소요시간: {total_elapsed:.2f}s ({total_elapsed/60:.1f}분)")
                _log(f"   - Tripo 3D: {tripo_elapsed:.2f}s")
                _log(f"   - Brickify: {brickify_elapsed:.2f}s")
                _log(f"결과: parts={result.get('parts')} | ldrSize={out_ldr.stat().st_size/1024:.1f}KB")
                _log("\u2550" * 70)
            

            # [Restored comment]
            # try:
            #     await archive_job_logs(job_id, list(job_log_buffer), status="SUCCESS")
            # except:
            #     pass

            # Ensure background URL ready before returning
            # Ensure background requested log
            if background_requested:
                _log("   Background generation requested to Screenshot Server")

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
                "lmmLatency": int(brickify_elapsed * 1000), # [New] ms 단위 변환
                "backgroundUrl": background_url,
            }

    except Exception as e:
        total_elapsed = time.time() - total_start
        tb = traceback.format_exc()
        
        # LogCapture가 이미 Exception 출력도 잡았을 수 있지만,
        # 명시적으로 찍어주는 것이 안전
        _log("\u2550" * 70)
        _log(f"\u274c [AI-SERVER] \uc694\uccad \uc2e4\ud328! | jobId={job_id} | \uc18c\uc694\uc2dc\uac04={total_elapsed:.2f}s")
        _log(f"\u274c \uc5d0\ub7ec: {str(e)}")
        
        # [Trace] Global Failure
        await _trace("System", "FAILURE", str(e), {"error": str(e), "traceback": tb})
        _log("\u2550" * 70)
        log(tb, user_email=user_email) # 이것도 캡쳐됨
        
        _write_error_log(out_req_dir, tb)
        _write_error_log(out_tripo_dir, tb)
        _write_error_log(out_brick_dir, tb)
        

        # [Restored comment]
        # try:
        #     await archive_job_logs(job_id, list(job_log_buffer), status="FAILED")
        # except:
        #     pass

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
            language=request.language or "en",
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
            "backgroundUrl": result.get("backgroundUrl"),
            "subject": result["subject"],
            "tags": result["tags"],
            "parts": result["parts"],
            "finalTarget": result["finalTarget"],
            "lmmLatency": result.get("lmmLatency"), # [New]
        }

    except HTTPException:
        raise
    except Exception as e:
        if DEBUG:
            raise HTTPException(status_code=500, detail={"reqId": req_id, "error": str(e)})
        raise HTTPException(status_code=500, detail="process failed")
