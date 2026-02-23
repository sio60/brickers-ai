"""
SQS Consumer - AI Server에서 Backend로부터 REQUEST 메시지 수신

아키텍처: SQS Poller → asyncio.Queue → 단일 Worker (순차 처리)
- 폴링은 독립적으로 계속 진행 (이벤트루프 non-blocking)
- 작업은 내부 큐에 쌓였다가 1개씩 순차 처리
- 로그 섞임 없음, 리소스 경쟁 없음
"""
from __future__ import annotations

import os
import json
import asyncio
import anyio
from datetime import datetime
from typing import Dict, Any

from route.kids_render import process_kids_request_internal
from route.sqs_producer import send_result_message
from service.backend_client import check_job_canceled
from service.log_context import JobLogContext
from brick_engine.agent.log_analyzer.persistence import archive_job_logs


def log(msg: str, user_email: str = "System") -> None:
    """타임스탬프 및 사용자 정보 포함 로그 출력"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    user_tag = f"[{user_email}]" if user_email else "[System]"
    print(f"[{ts}] {user_tag} {msg}", flush=True)


def _is_truthy(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


# 환경 변수
AWS_REGION = os.environ.get("AWS_REGION", "ap-northeast-2").strip()
SQS_REQUEST_QUEUE_URL = os.environ.get("AWS_SQS_REQUEST_QUEUE_URL", "").strip()
SQS_ENABLED = _is_truthy(os.environ.get("AWS_SQS_ENABLED", "false"))
SQS_POLL_INTERVAL = int(os.environ.get("SQS_POLL_INTERVAL", "5"))
SQS_MAX_MESSAGES = int(os.environ.get("SQS_MAX_MESSAGES", "1"))  # 순차 처리이므로 1개씩
SQS_WAIT_TIME = int(os.environ.get("SQS_WAIT_TIME", "10"))
SQS_VISIBILITY_TIMEOUT = int(os.environ.get("SQS_VISIBILITY_TIMEOUT", "1800"))

# 전역 상태 (모니터링용)
_TOTAL_REQUESTS_RECEIVED = 0
_TOTAL_REQUESTS_COMPLETED = 0
_TOTAL_REQUESTS_FAILED = 0

# 내부 작업 큐 (Lazy Init)
_JOB_QUEUE: asyncio.Queue | None = None


def _get_job_queue() -> asyncio.Queue:
    global _JOB_QUEUE
    if _JOB_QUEUE is None:
        _JOB_QUEUE = asyncio.Queue(maxsize=100)
    return _JOB_QUEUE


# boto3 lazy import
try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None  # type: ignore
    ClientError = Exception  # type: ignore


_SQS_CLIENT = None


def _get_sqs_client():
    """SQS Client 싱글톤"""
    global _SQS_CLIENT
    if _SQS_CLIENT is not None:
        return _SQS_CLIENT

    if not SQS_ENABLED:
        raise RuntimeError("SQS is not enabled (AWS_SQS_ENABLED=false)")

    if boto3 is None:
        raise RuntimeError("boto3 is not installed (pip install boto3)")

    if not SQS_REQUEST_QUEUE_URL:
        raise RuntimeError("AWS_SQS_REQUEST_QUEUE_URL is not set")

    _SQS_CLIENT = boto3.client("sqs", region_name=AWS_REGION)
    return _SQS_CLIENT


# ============================================================================
# SQS Poller — 큐에서 메시지를 가져와 내부 큐에 넣기만 함
# ============================================================================

_POLL_COUNT = 0


async def _sqs_poller():
    """
    SQS 폴링 루프 (독립 코루틴)
    - Long polling으로 메시지 수신
    - 내부 asyncio.Queue에 넣기만 함 (처리는 Worker가 담당)
    - 처리 완료를 기다리지 않으므로 이벤트루프 blocking 없음
    """
    global _POLL_COUNT

    job_queue = _get_job_queue()

    while True:
        try:
            sqs = _get_sqs_client()
            _POLL_COUNT += 1

            if _POLL_COUNT % 10 == 1:
                q_size = job_queue.qsize()
                log(f"🔄 [Poller] 폴링 중... (poll #{_POLL_COUNT}) | 대기 큐: {q_size}개")

            # boto3는 blocking이므로 쓰레드에서 실행
            def _receive():
                return sqs.receive_message(
                    QueueUrl=SQS_REQUEST_QUEUE_URL,
                    MaxNumberOfMessages=SQS_MAX_MESSAGES,
                    WaitTimeSeconds=SQS_WAIT_TIME,
                    VisibilityTimeout=SQS_VISIBILITY_TIMEOUT,
                )

            response = await anyio.to_thread.run_sync(_receive)
            messages = response.get("Messages", [])

            if messages:
                global _TOTAL_REQUESTS_RECEIVED
                for m in messages:
                    _TOTAL_REQUESTS_RECEIVED += 1
                    await job_queue.put((m, _TOTAL_REQUESTS_RECEIVED))
                log(
                    f"📥 [Poller] 메시지 수신 → 내부 큐 전달 | "
                    f"count={len(messages)} | 큐 대기: {job_queue.qsize()}개"
                )
                # 메시지 있으면 즉시 다음 폴링
                await asyncio.sleep(0.1)
            else:
                await asyncio.sleep(SQS_POLL_INTERVAL)

        except Exception as e:
            log(f"❌ [Poller] 폴링 실패 | poll #{_POLL_COUNT} | error={str(e)}")
            await asyncio.sleep(SQS_POLL_INTERVAL)


# ============================================================================
# Job Worker — 내부 큐에서 하나씩 꺼내 순차 처리
# ============================================================================

async def _job_worker():
    """
    단일 워커 코루틴 (순차 처리)
    - 내부 큐에서 메시지를 1개씩 꺼내서 처리
    - 동시 실행 없음 → 로그 섞임 없음, 리소스 경쟁 없음
    - 각 작업 완료 후 다음 작업 시작
    """
    global _TOTAL_REQUESTS_COMPLETED, _TOTAL_REQUESTS_FAILED

    job_queue = _get_job_queue()
    log("[Worker] 🏭 단일 워커 시작 (순차 처리 모드)")

    while True:
        try:
            # 큐에서 다음 작업 대기 (blocking but async-safe)
            message, request_num = await job_queue.get()

            message_id = message.get("MessageId", "unknown")
            receipt_handle = message["ReceiptHandle"]

            log("=" * 60)
            log(f"📨 [Worker] 작업 시작 | #{request_num} | messageId={message_id}")
            log(f"   📋 남은 대기: {job_queue.qsize()}개")

            try:
                body = json.loads(message["Body"])
                user_email = body.get("userEmail", "unknown")
                job_id = body.get("jobId", "unknown")

                # [NEW] Job Log Context Start (Capture logs from here)
                job_log_buffer = []
                with JobLogContext(job_log_buffer):
                    log(f"📨 [Worker] 처리 시작 | jobId={job_id}", user_email=user_email)
                    log(f"   - type: {body.get('type')}", user_email=user_email)

                    # REQUEST 타입만 처리
                    if body.get("type") != "REQUEST":
                        log(f"⚠️ [Worker] RESULT 타입 무시", user_email=user_email)
                        delete_message(receipt_handle)
                        job_queue.task_done()
                        continue

                    source_image_url = body.get("sourceImageUrl")
                    age = body.get("age", "6-7")
                    budget = body.get("budget")
                    language = body.get("language", "en") # [NEW]

                    # 취소 여부 확인 (처리 시작 전)
                    if await check_job_canceled(job_id):
                        log(f"🚫 [Worker] 취소된 작업 스킵 | jobId={job_id}", user_email=user_email)
                        delete_message(receipt_handle)
                        job_queue.task_done()
                        await archive_job_logs(job_id, list(job_log_buffer), status="CANCELED") # [NEW]
                        continue


                # [Legacy: Auto-Flush Logic - Commented out as per user request]
                # async def _auto_flush_logs():
                #     """
                #     주기적으로 로그 버퍼를 DB에 전송 (Real-time UX)
                #     Tripo 생성 등 긴 작업 중에도 사용자가 로그를 볼 수 있게 함.
                #     """
                #     last_sent_count = 0
                #     while True:
                #         await asyncio.sleep(5.0) # 5초마다 체크 (부하 감소)
                #         current_count = len(job_log_buffer)
                #         if current_count > last_sent_count:
                #             # 변경사항이 있을 때만 전송
                #             await archive_job_logs(job_id, list(job_log_buffer), status="RUNNING")
                #             last_sent_count = current_count
                #
                # # Start Auto-Flush Task
                # flusher_task = asyncio.create_task(_auto_flush_logs())

                result = await process_kids_request_internal(
                    job_id=job_id,
                    source_image_url=source_image_url,
                    age=age,
                    budget=budget,
                    user_email=user_email,
                    language=language, # [NEW]
                    external_log_buffer=job_log_buffer, # [NEW]
                )

                if result.get("success", True) is False:
                    # [NEW] Handle Explicit Failure Return (with partial cost)
                    log(f"❌ [Worker] 렌더링 실패 (Graceful Return) | error={result.get('error')}", user_email=user_email)
                    await send_result_message(
                        job_id=job_id,
                        success=False,
                        error_message=result.get("error", "Unknown error"),
                        est_cost=result.get("estCost"),
                        token_count=result.get("tokenCount"),
                    )
                    delete_message(receipt_handle)
                    _TOTAL_REQUESTS_FAILED += 1
                    await archive_job_logs(job_id, list(job_log_buffer), status="FAILED")
                else:
                    # [Existing] Success Logic
                    log(f"✅ AI 렌더링 완료!", user_email=user_email)
                    log(f"   - correctedUrl: {result.get('correctedUrl', '')[:60]}...")
                    log(f"   - modelUrl: {result.get('modelUrl', '')[:60]}...")
                    log(f"   - ldrUrl: {result.get('ldrUrl', '')[:60]}...")
                    log(f"   - parts: {result.get('parts')}, finalTarget: {result.get('finalTarget')}", user_email=user_email)

                    # RESULT 메시지 전송 (성공)
                    log("📤 [Worker] RESULT 메시지 전송 중...")
                    await send_result_message(
                        job_id=job_id,
                        success=True,
                        corrected_url=result.get("correctedUrl", ""),
                        glb_url=result.get("modelUrl", ""),
                        ldr_url=result.get("ldrUrl", ""),
                        bom_url=result.get("bomUrl", ""),
                        pdf_url=result.get("pdfUrl", ""),
                        parts=result.get("parts", 0),
                        final_target=result.get("finalTarget", 0),
                        tags=result.get("tags", []),
                        background_url=result.get("backgroundUrl", ""),
                        est_cost=result.get("estCost"), # [New]
                        token_count=result.get("tokenCount"), # [New]
                        stability_score=result.get("stabilityScore"), # [New]
                    )

                    delete_message(receipt_handle)
                    _TOTAL_REQUESTS_COMPLETED += 1

                    log(f"✅ [Worker] 처리 완료 | jobId={job_id} | "
                        f"완료: {_TOTAL_REQUESTS_COMPLETED} | 실패: {_TOTAL_REQUESTS_FAILED}",
                        user_email=user_email)
                    log("=" * 60, user_email=user_email)
                    
                    # [NEW] Archive Final State (Success)
                    await archive_job_logs(job_id, list(job_log_buffer), status="SUCCESS")

                # finally:
                #     # 작업 종료 시 플러셔 정리
                #     flusher_task.cancel()
                #     try:
                #         await flusher_task
                #     except asyncio.CancelledError:
                #         pass


            except json.JSONDecodeError as e:
                log(f"❌ [Worker] JSON 파싱 실패 | messageId={message_id} | error={str(e)}")
                delete_message(receipt_handle)
                _TOTAL_REQUESTS_FAILED += 1

            except Exception as e:
                u_email = locals().get("user_email", "unknown")
                job_id = locals().get("job_id", "unknown")
                
                # Check if context was active? 
                # If exception happened inside context, logs are in buffer.
                # If before context, buffer might not exist.
                # But here buffer init is very early.
                
                # If job_log_buffer exists in locals():
                j_buf = locals().get("job_log_buffer", [])
                
                # Re-enter context if needed? No, context exited on exception.
                # But we can still use buffer.
                
                # Temporarily re-hook logging to buffer for error logging??
                # Or just append manually?
                # Actually GlobalLogCapture hooks ALL stdout/stderr.
                # But since we exited the context block (due to exception), 
                # future logs won't go to buffer unless we re-enter.
                
                # Let's re-enter context for error logging if buffer exists
                if 'job_log_buffer' in locals():
                    with JobLogContext(j_buf):
                        log(f"❌ [Worker] 처리 실패 | error={str(e)}", user_email=u_email)

                        # RESULT 메시지 전송 (실패)
                        try:
                            # job_id is already set
                            await send_result_message(
                                job_id=job_id,
                                success=False,
                                error_message=str(e),
                            )
                        except Exception as send_error:
                            log(f"❌ [Worker] 결과 전송 실패 | error={str(send_error)}", user_email=u_email)

                        delete_message(receipt_handle)
                        _TOTAL_REQUESTS_FAILED += 1
                        
                        # [NEW] Archive Final State (Failed)
                        await archive_job_logs(job_id, list(j_buf), status="FAILED")
                else:
                    # Fallback system logging
                    log(f"❌ [Worker] 처리 실패 (No Context) | error={str(e)}", user_email=u_email)
                    delete_message(receipt_handle)
                    _TOTAL_REQUESTS_FAILED += 1

            finally:
                job_queue.task_done()

        except Exception as e:
            log(f"❌ [Worker] 워커 예외 | error={str(e)}")
            await asyncio.sleep(1)


# ============================================================================
# Helpers
# ============================================================================

def delete_message(receipt_handle: str):
    """메시지 삭제"""
    try:
        sqs = _get_sqs_client()
        sqs.delete_message(
            QueueUrl=SQS_REQUEST_QUEUE_URL,
            ReceiptHandle=receipt_handle,
        )
    except Exception as e:
        log(f"❌ [SQS] 메시지 삭제 실패 | error={str(e)}")


def get_consumer_stats() -> Dict[str, Any]:
    """모니터링용 통계 (admin 엔드포인트에서 사용 가능)"""
    q = _get_job_queue()
    return {
        "total_received": _TOTAL_REQUESTS_RECEIVED,
        "total_completed": _TOTAL_REQUESTS_COMPLETED,
        "total_failed": _TOTAL_REQUESTS_FAILED,
        "queue_size": q.qsize(),
        "poll_count": _POLL_COUNT,
    }


# ============================================================================
# Entry Point — FastAPI startup에서 호출
# ============================================================================

async def start_consumer():
    """
    SQS Consumer 시작 (2개 코루틴 동시 실행)
    1. _sqs_poller: SQS 폴링 → 내부 큐에 넣기
    2. _job_worker: 내부 큐에서 1개씩 꺼내 순차 처리
    """
    if not SQS_ENABLED:
        log("[SQS Consumer] ⚠️ SQS 비활성화 상태 (AWS_SQS_ENABLED=false)")
        return

    log("═" * 70)
    log("[SQS Consumer] 🚀 시작 (Poller + 단일 Worker 아키텍처)")
    log(f"   - Queue URL: {SQS_REQUEST_QUEUE_URL}")
    log(f"   - Poll Interval: {SQS_POLL_INTERVAL}초")
    log(f"   - Max Messages per poll: {SQS_MAX_MESSAGES}")
    log(f"   - Wait Time: {SQS_WAIT_TIME}초 (Long polling)")
    log(f"   - Visibility Timeout: {SQS_VISIBILITY_TIMEOUT}초")
    log(f"   - Worker: 단일 순차 처리 (동시 실행 없음)")
    log("═" * 70)

    # 두 코루틴을 동시에 실행
    # Poller: SQS → 내부 큐
    # Worker: 내부 큐 → 순차 처리
    asyncio.create_task(_sqs_poller())
    asyncio.create_task(_job_worker())
