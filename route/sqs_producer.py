"""
SQS Producer - AI Server에서 Backend로 RESULT 메시지 전송
"""
from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Optional


def log(msg: str) -> None:
    """타임스탬프 포함 로그 출력"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}")


def _is_truthy(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


# 환경 변수
AWS_REGION = os.environ.get("AWS_REGION", "ap-northeast-2").strip()
SQS_RESULT_QUEUE_URL = os.environ.get("AWS_SQS_RESULT_QUEUE_URL", "").strip()  # AI → Backend (RESULT 전송)
SQS_PDF_QUEUE_URL = os.environ.get("AWS_SQS_PDF_QUEUE_URL", "").strip()  # AI → Blueprint (PDF 생성 요청)
SQS_SCREENSHOT_QUEUE_URL = os.environ.get("AWS_SQS_SCREENSHOT_QUEUE_URL", "").strip()  # AI → Screenshot (스크린샷 요청)
SQS_ENABLED = _is_truthy(os.environ.get("AWS_SQS_ENABLED", "false"))

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

    if not SQS_RESULT_QUEUE_URL:
        raise RuntimeError("AWS_SQS_RESULT_QUEUE_URL is not set")

    # boto3는 아래 env를 자동으로 읽음:
    # AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION
    _SQS_CLIENT = boto3.client("sqs", region_name=AWS_REGION)
    return _SQS_CLIENT


async def send_result_message(
    job_id: str,
    success: bool,
    corrected_url: str = "",
    glb_url: str = "",
    ldr_url: str = "",
    bom_url: str = "",
    pdf_url: str = "",
    parts: int = 0,
    final_target: int = 0,
    tags: list[str] = None,
    background_url: str = "",
    error_message: Optional[str] = None,
) -> None:
    """
    RESULT 메시지를 Backend로 전송

    Args:
        job_id: Job ID
        success: 성공 여부
        corrected_url: 보정된 이미지 URL
        glb_url: GLB 파일 URL
        ldr_url: LDR 파일 URL
        bom_url: BOM 파일 URL
        pdf_url: PDF 파일 URL
        parts: 파츠 수
        final_target: 최종 타겟
        tags: AI가 추출한 태그 목록
        error_message: 실패 시 에러 메시지
    """
    if not SQS_ENABLED:
        log(f"[SQS Producer] ⚠️ SQS 비활성화 상태 (메시지 전송 스킵) | jobId={job_id}")
        return

        log(f"[SQS Producer] RESULT 메시지 생성 시작 | jobId={job_id}")

    try:
        client = _get_sqs_client()

        message = {
            "type": "RESULT",
            "jobId": job_id,
            "timestamp": datetime.now().isoformat(),
            "success": success,
        }

        if success:
            message.update({
                "correctedUrl": corrected_url,
                "glbUrl": glb_url,
                "ldrUrl": ldr_url,
                "bomUrl": bom_url,
                "pdfUrl": pdf_url,
                "parts": parts,
                "finalTarget": final_target,
                "tags": tags or [],
                "backgroundUrl": background_url or "",
            })
            log("   - success=True")
            log(f"   - ldrUrl: {ldr_url[:60]}..." if ldr_url else "   - ldrUrl: (empty)")
            log(f"   - parts: {parts}, finalTarget: {final_target}")
        else:
            message["errorMessage"] = error_message or "Unknown error"
            log("   - success=False")
            log(f"   - errorMessage: {error_message}")

        log(f"   - queueUrl: {SQS_RESULT_QUEUE_URL}")

        response = client.send_message(
            QueueUrl=SQS_RESULT_QUEUE_URL,
            MessageBody=json.dumps(message)
        )

        log(f"✅ [SQS Producer] RESULT 메시지 전송 완료 | jobId={job_id} | messageId={response.get('MessageId', 'N/A')}")

    except Exception as e:
        log(f"❌ [SQS Producer] 메시지 전송 실패 | jobId={job_id} | error={str(e)}")
        raise


async def send_pdf_request_message(
    job_id: str,
    ldr_url: str,
    model_name: str,
) -> None:
    """
    PDF 생성 요청을 brickers-blueprints-queue로 전송

    Args:
        job_id: Job ID
        ldr_url: LDR 파일 S3 URL
        model_name: 모델 이름 (PDF 표지에 표시)
    """
    if not SQS_ENABLED:
        log(f"[SQS Producer] ⚠️ SQS 비활성화 상태 (PDF 요청 전송 스킵) | jobId={job_id}")
        return

    if not SQS_PDF_QUEUE_URL:
        log(f"[SQS Producer] ⚠️ AWS_SQS_PDF_QUEUE_URL 미설정 (PDF 요청 전송 스킵) | jobId={job_id}")
        return

    log(f"📤 [SQS Producer] PDF 요청 메시지 생성 | jobId={job_id}")

    try:
        client = _get_sqs_client()

        message = {
            "type": "PDF_REQUEST",
            "jobId": job_id,
            "ldrUrl": ldr_url,
            "modelName": model_name,
            "timestamp": datetime.now().isoformat(),
        }

        response = client.send_message(
            QueueUrl=SQS_PDF_QUEUE_URL,
            MessageBody=json.dumps(message),
        )

        log(f"✅ [SQS Producer] PDF 요청 전송 완료 | jobId={job_id} | messageId={response.get('MessageId', 'N/A')}")

    except Exception as e:
        log(f"❌ [SQS Producer] PDF 요청 전송 실패 | jobId={job_id} | error={str(e)}")
        # PDF 전송 실패는 파이프라인을 중단시키지 않음 (로그만 남김)
        raise


def _get_celery_producer():
    """Celery producer 모듈 lazy import (없으면 None)"""
    try:
        from route.celery_screenshot_producer import (
            CELERY_ENABLED as _ce,
            send_screenshot_task,
            send_background_task,
        )
        if _ce:
            return send_screenshot_task, send_background_task
    except Exception:
        pass
    return None


async def send_screenshot_request_message(
    job_id: str,
    ldr_url: str,
    model_name: str,
) -> None:
    """
    스크린샷 생성 요청 — Celery 우선, raw SQS 폴백

    Args:
        job_id: Job ID
        ldr_url: LDR 파일 S3 URL
        model_name: 모델 이름
    """
    # 1) Celery 경로 시도
    celery_fns = _get_celery_producer()
    if celery_fns:
        send_ss, _ = celery_fns
        send_ss(job_id, ldr_url, model_name)
        log(f"✅ [Celery] 스크린샷 태스크 전송 완료 | jobId={job_id}")
        return

    # 2) raw SQS 폴백
    if not SQS_ENABLED:
        log(f"[SQS Producer] ⚠️ SQS 비활성화 상태 (스크린샷 요청 전송 스킵) | jobId={job_id}")
        return

    if not SQS_SCREENSHOT_QUEUE_URL:
        log(f"[SQS Producer] ⚠️ AWS_SQS_SCREENSHOT_QUEUE_URL 미설정 (스크린샷 요청 전송 스킵) | jobId={job_id}")
        return

    log(f"📤 [SQS Producer] 스크린샷 요청 메시지 생성 (raw SQS) | jobId={job_id}")

    try:
        client = _get_sqs_client()

        message = {
            "type": "SCREENSHOT_REQUEST",
            "jobId": job_id,
            "ldrUrl": ldr_url,
            "modelName": model_name,
            "timestamp": datetime.now().isoformat(),
        }

        response = client.send_message(
            QueueUrl=SQS_SCREENSHOT_QUEUE_URL,
            MessageBody=json.dumps(message),
        )

        log(f"✅ [SQS Producer] 스크린샷 요청 전송 완료 | jobId={job_id} | messageId={response.get('MessageId', 'N/A')}")

    except Exception as e:
        log(f"❌ [SQS Producer] 스크린샷 요청 전송 실패 | jobId={job_id} | error={str(e)}")
        raise


async def send_background_request_message(
    job_id: str,
    subject: str,
) -> None:
    """
    배경 생성 요청 — Celery 우선, raw SQS 폴백

    Args:
        job_id: Job ID
        subject: 배경 생성 주제 (Gemini 프롬프트용)
    """
    # 1) Celery 경로 시도
    celery_fns = _get_celery_producer()
    if celery_fns:
        _, send_bg = celery_fns
        send_bg(job_id, subject)
        log(f"✅ [Celery] 배경 생성 태스크 전송 완료 | jobId={job_id}")
        return

    # 2) raw SQS 폴백
    if not SQS_ENABLED:
        log(f"[SQS Producer] ⚠️ SQS 비활성화 상태 (배경 요청 전송 스킵) | jobId={job_id}")
        return

    if not SQS_SCREENSHOT_QUEUE_URL:
        log(f"[SQS Producer] ⚠️ AWS_SQS_SCREENSHOT_QUEUE_URL 미설정 (배경 요청 전송 스킵) | jobId={job_id}")
        return

    log(f"[SQS Producer] 배경 생성 요청 메시지 생성 (raw SQS) | jobId={job_id} | subject={subject}")

    try:
        client = _get_sqs_client()

        message = {
            "type": "BACKGROUND_REQUEST",
            "jobId": job_id,
            "subject": subject,
            "timestamp": datetime.now().isoformat(),
        }

        response = client.send_message(
            QueueUrl=SQS_SCREENSHOT_QUEUE_URL,
            MessageBody=json.dumps(message),
        )

        log(f"✅ [SQS Producer] 배경 생성 요청 전송 완료 | jobId={job_id} | messageId={response.get('MessageId', 'N/A')}")

    except Exception as e:
        log(f"❌ [SQS Producer] 배경 생성 요청 전송 실패 | jobId={job_id} | error={str(e)}")
        # 실패하더라도 파이프라인 전체를 중단시키지는 않음 (배경은 부가 기능)
