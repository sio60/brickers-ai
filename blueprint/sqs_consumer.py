# blueprint/sqs_consumer.py
"""
SQS Consumer - brickers-blueprints-queue 폴링
PDF 생성 요청을 수신하여 LDR -> LDView 렌더링 -> PDF 생성 -> S3 업로드 -> Backend 알림
"""
from __future__ import annotations

import os
import re
import json
import uuid
import asyncio
import traceback
from datetime import datetime
from typing import Dict, Any

import anyio
import httpx

from service.render_client import render_ldr_steps, RENDER_ENABLED
from route.instructions_pdf import parse_ldr_step_boms, generate_pdf_with_images_and_bom
from service.s3_client import USE_S3, S3_BUCKET, upload_bytes_to_s3
from service.backend_client import notify_pdf_complete


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{ts}] [Blueprint] {msg}")


def _is_truthy(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


# 환경 변수
AWS_REGION = os.environ.get("AWS_REGION", "ap-northeast-2").strip()
SQS_PDF_QUEUE_URL = os.environ.get("AWS_SQS_PDF_QUEUE_URL", "").strip()
SQS_ENABLED = _is_truthy(os.environ.get("AWS_SQS_ENABLED", "false"))
SQS_POLL_INTERVAL = int(os.environ.get("SQS_POLL_INTERVAL", "5"))
SQS_MAX_MESSAGES = int(os.environ.get("SQS_MAX_MESSAGES", "5"))
SQS_WAIT_TIME = int(os.environ.get("SQS_WAIT_TIME", "10"))
SQS_VISIBILITY_TIMEOUT = int(os.environ.get("SQS_VISIBILITY_TIMEOUT", "600"))

PDF_S3_PREFIX = os.environ.get("S3_PREFIX_PDF", "uploads/pdf").strip().strip("/")

# boto3 lazy import
try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None  # type: ignore
    ClientError = Exception  # type: ignore

_SQS_CLIENT = None


def _get_sqs_client():
    global _SQS_CLIENT
    if _SQS_CLIENT is not None:
        return _SQS_CLIENT

    if not SQS_ENABLED:
        raise RuntimeError("SQS is not enabled (AWS_SQS_ENABLED=false)")
    if boto3 is None:
        raise RuntimeError("boto3 is not installed (pip install boto3)")
    if not SQS_PDF_QUEUE_URL:
        raise RuntimeError("AWS_SQS_PDF_QUEUE_URL is not set")

    _SQS_CLIENT = boto3.client("sqs", region_name=AWS_REGION)
    return _SQS_CLIENT


async def fetch_ldr_text(url: str) -> str:
    """LDR 파일 다운로드"""
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.text


async def process_pdf_message(body: Dict[str, Any]) -> None:
    """
    PDF 생성 메시지 처리
    1. LDR 다운로드
    2. LDView 렌더링
    3. BOM 파싱
    4. PDF 생성
    5. S3 업로드
    6. Backend 알림
    """
    job_id = body["jobId"]
    ldr_url = body["ldrUrl"]
    model_name = body.get("modelName") or body.get("subject") or "Brickers Model"

    _log(f"📋 PDF 생성 시작 | jobId={job_id} | model={model_name}")

    # 1. LDR 다운로드
    _log(f"   [1/6] LDR 다운로드 중... | {ldr_url[:80]}")
    ldr_text = await fetch_ldr_text(ldr_url)
    _log(f"   [1/6] LDR 다운로드 완료 | {len(ldr_text)} chars")

    # 2. LDView 렌더링
    if not RENDER_ENABLED:
        raise RuntimeError("LDView binary not found. PDF generation requires LDView.")

    _log("   [2/6] LDView 렌더링 중...")
    step_images = await render_ldr_steps(ldr_text)
    _log(f"   [2/6] 렌더링 완료 | {len(step_images)} steps")

    # 3. BOM 파싱
    _log("   [3/6] BOM 파싱 중...")
    step_boms = parse_ldr_step_boms(ldr_text)
    _log(f"   [3/6] BOM 파싱 완료 | {len(step_boms)} steps")

    # 4. PDF 생성
    _log("   [4/6] PDF 생성 중...")
    cover_img = None
    if step_images and step_images[-1] and step_images[-1][0]:
        cover_img = step_images[-1][0]

    pdf_bytes = generate_pdf_with_images_and_bom(
        model_name=model_name,
        step_images=step_images,
        step_boms=step_boms,
        cover_image=cover_img,
    )
    _log(f"   [4/6] PDF 생성 완료 | {len(pdf_bytes)} bytes")

    # 5. S3 업로드
    _log("   [5/6] S3 업로드 중...")
    if not (USE_S3 and S3_BUCKET):
        raise RuntimeError("S3 is not configured. Cannot upload PDF.")

    now = datetime.now()
    safe_name = re.sub(r'[\\/:*?"<>|]+', "_", model_name or "instructions")
    s3_key = f"{PDF_S3_PREFIX}/{now.year:04d}/{now.month:02d}/{uuid.uuid4().hex[:8]}_{safe_name}.pdf"
    pdf_url = upload_bytes_to_s3(pdf_bytes, s3_key, "application/pdf")
    _log(f"   [5/6] S3 업로드 완료 | {pdf_url}")

    # 6. Backend 알림
    _log("   [6/6] Backend 알림 전송 중...")
    await notify_pdf_complete(job_id, pdf_url)
    _log("   [6/6] Backend 알림 완료")

    _log(f"✅ PDF 생성 완료 | jobId={job_id} | pdfUrl={pdf_url}")


_POLL_COUNT = 0


async def poll_and_process() -> int:
    """SQS에서 PDF 요청 메시지 폴링 및 처리"""
    global _POLL_COUNT

    if not SQS_ENABLED:
        return 0

    try:
        sqs = _get_sqs_client()
        _POLL_COUNT += 1

        if _POLL_COUNT % 10 == 1:
            _log(f"🔄 폴링 중... (poll #{_POLL_COUNT})")

        def _receive():
            return sqs.receive_message(
                QueueUrl=SQS_PDF_QUEUE_URL,
                MaxNumberOfMessages=SQS_MAX_MESSAGES,
                WaitTimeSeconds=SQS_WAIT_TIME,
                VisibilityTimeout=SQS_VISIBILITY_TIMEOUT,
            )

        response = await anyio.to_thread.run_sync(_receive)
        messages = response.get("Messages", [])

        if messages:
            _log(f"📥 메시지 수신 | count={len(messages)} | poll #{_POLL_COUNT}")

        for m in messages:
            asyncio.create_task(_handle_message(m))

        return len(messages)

    except Exception as e:
        _log(f"❌ 폴링 실패 | poll #{_POLL_COUNT} | error={str(e)}")
        return 0


async def _handle_message(message: Dict[str, Any]) -> None:
    """개별 메시지 처리 (에러 격리)"""
    message_id = message.get("MessageId", "unknown")
    receipt_handle = message["ReceiptHandle"]

    try:
        body = json.loads(message["Body"])

        if body.get("type") != "PDF_REQUEST":
            _log(f"⚠️ 잘못된 메시지 타입 무시 | type={body.get('type')} | messageId={message_id}")
            _delete_message(receipt_handle)
            return

        await process_pdf_message(body)
        _delete_message(receipt_handle)

    except json.JSONDecodeError as e:
        _log(f"❌ JSON 파싱 실패 | messageId={message_id} | error={str(e)}")
        _delete_message(receipt_handle)

    except Exception as e:
        _log(f"❌ PDF 생성 실패 | messageId={message_id} | error={str(e)}")
        _log(traceback.format_exc())
        # 실패해도 메시지 삭제 (재처리 의미 없음 - 결과가 동일할 가능성 높음)
        _delete_message(receipt_handle)


def _delete_message(receipt_handle: str) -> None:
    """SQS 메시지 삭제"""
    try:
        sqs = _get_sqs_client()
        sqs.delete_message(
            QueueUrl=SQS_PDF_QUEUE_URL,
            ReceiptHandle=receipt_handle,
        )
    except Exception as e:
        _log(f"❌ 메시지 삭제 실패 | error={str(e)}")


async def start_pdf_consumer() -> None:
    """
    PDF SQS Consumer 시작 (FastAPI startup에서 호출)
    """
    if not SQS_ENABLED:
        _log("⚠️ SQS 비활성화 상태 (AWS_SQS_ENABLED=false)")
        return

    if not SQS_PDF_QUEUE_URL:
        _log("⚠️ AWS_SQS_PDF_QUEUE_URL 미설정 - PDF Consumer 미시작")
        return

    _log("═" * 60)
    _log("🚀 Blueprint PDF Consumer 시작")
    _log(f"   - Queue URL: {SQS_PDF_QUEUE_URL}")
    _log(f"   - Poll Interval: {SQS_POLL_INTERVAL}초")
    _log(f"   - Max Messages: {SQS_MAX_MESSAGES}")
    _log(f"   - Visibility Timeout: {SQS_VISIBILITY_TIMEOUT}초")
    _log(f"   - LDView: {'✅ OK' if RENDER_ENABLED else '❌ NOT FOUND'}")
    _log("═" * 60)

    while True:
        try:
            msg_count = await poll_and_process()

            if msg_count > 0:
                await asyncio.sleep(0.1)
            else:
                await asyncio.sleep(SQS_POLL_INTERVAL)

        except Exception as e:
            _log(f"❌ 예외 발생 | error={str(e)}")
            await asyncio.sleep(SQS_POLL_INTERVAL)
