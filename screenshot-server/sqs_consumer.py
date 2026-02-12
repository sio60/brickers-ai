# screenshot-server/sqs_consumer.py
"""
SQS Consumer - brickers-screenshots-queue 폴링
스크린샷 생성 요청을 수신하여 LDR -> LDView 6면 렌더링 -> S3 업로드 -> Backend 알림
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

from service.render_client import render_6_views, RENDER_ENABLED
from service.s3_client import USE_S3, S3_BUCKET, upload_bytes_to_s3
from service.backend_client import (
    notify_screenshots_complete, 
    notify_gallery_screenshots_complete,
    notify_background_complete
)
from service.background_composer import generate_background_async


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{ts}] [Screenshot] {msg}")


def _is_truthy(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


# 환경 변수
AWS_REGION = os.environ.get("AWS_REGION", "ap-northeast-2").strip()
SQS_SCREENSHOT_QUEUE_URL = os.environ.get("AWS_SQS_SCREENSHOT_QUEUE_URL", "").strip()
SQS_ENABLED = _is_truthy(os.environ.get("AWS_SQS_ENABLED", "false"))
SQS_POLL_INTERVAL = int(os.environ.get("SQS_POLL_INTERVAL", "5"))
SQS_MAX_MESSAGES = int(os.environ.get("SQS_MAX_MESSAGES", "1"))
SQS_WAIT_TIME = int(os.environ.get("SQS_WAIT_TIME", "10"))
SQS_VISIBILITY_TIMEOUT = int(os.environ.get("SQS_VISIBILITY_TIMEOUT", "300"))

SCREENSHOT_S3_PREFIX = os.environ.get("S3_PREFIX_SCREENSHOT", "uploads/screenshots").strip().strip("/")

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
    if not SQS_SCREENSHOT_QUEUE_URL:
        raise RuntimeError("AWS_SQS_SCREENSHOT_QUEUE_URL is not set")

    _SQS_CLIENT = boto3.client("sqs", region_name=AWS_REGION)
    return _SQS_CLIENT


async def fetch_ldr_text(url: str) -> str:
    """LDR 파일 다운로드"""
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.text


async def process_screenshot_message(body: Dict[str, Any]) -> None:
    """
    스크린샷 생성 메시지 처리
    1. LDR 다운로드
    2. 6면 렌더링 (LDView)
    3. S3 업로드 (6장)
    4. Backend 알림
    source가 "gallery_backfill"이면 갤러리 포스트 직접 업데이트
    """
    source = body.get("source", "job")
    job_id = body.get("jobId", "")
    gallery_post_id = body.get("galleryPostId", "")
    ldr_url = body["ldrUrl"]
    model_name = body.get("modelName", "model")

    identifier = gallery_post_id if source == "gallery_backfill" else job_id
    _log(f"📋 스크린샷 생성 시작 | source={source} | id={identifier} | model={model_name}")

    # S3 key용 ID
    s3_id = gallery_post_id if source == "gallery_backfill" else job_id

    # 1. LDR 다운로드
    _log(f"   [1/4] LDR 다운로드 중... | {ldr_url[:80]}")
    ldr_text = await fetch_ldr_text(ldr_url)
    _log(f"   [1/4] LDR 다운로드 완료 | {len(ldr_text)} chars")

    # 2. 6면 렌더링
    if not RENDER_ENABLED:
        raise RuntimeError("LDView binary not found. Screenshot generation requires LDView.")

    _log("   [2/4] LDView 6면 렌더링 중...")
    screenshots = await render_6_views(ldr_text)
    valid_count = sum(1 for v in screenshots.values() if v)
    _log(f"   [2/4] 렌더링 완료 | {valid_count}/6 views")

    # 3. S3 업로드
    _log("   [3/4] S3 업로드 중...")
    if not (USE_S3 and S3_BUCKET):
        raise RuntimeError("S3 is not configured. Cannot upload screenshots.")

    now = datetime.now()
    screenshot_urls: Dict[str, str] = {}

    for view_name, png_bytes in screenshots.items():
        if not png_bytes:
            _log(f"   [3/4] {view_name}: 빈 이미지 스킵")
            continue

        s3_key = f"{SCREENSHOT_S3_PREFIX}/{now.year:04d}/{now.month:02d}/{s3_id}_{view_name}.png"
        url = upload_bytes_to_s3(png_bytes, s3_key, "image/png")
        screenshot_urls[view_name] = url
        _log(f"   [3/4] {view_name}: 업로드 완료")

    _log(f"   [3/4] S3 업로드 완료 | {len(screenshot_urls)} views")

    # 4. Backend 알림
    _log("   [4/4] Backend 알림 전송 중...")
    if source == "gallery_backfill":
        await notify_gallery_screenshots_complete(gallery_post_id, screenshot_urls)
    else:
        await notify_screenshots_complete(job_id, screenshot_urls)
    _log("   [4/4] Backend 알림 완료")

    _log(f"✅ 스크린샷 생성 완료 | source={source} | id={identifier} | views={list(screenshot_urls.keys())}")


_POLL_COUNT = 0


async def poll_and_process() -> int:
    """SQS에서 스크린샷 요청 메시지 폴링 및 처리"""
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
                QueueUrl=SQS_SCREENSHOT_QUEUE_URL,
                MaxNumberOfMessages=SQS_MAX_MESSAGES,
                WaitTimeSeconds=SQS_WAIT_TIME,
                VisibilityTimeout=SQS_VISIBILITY_TIMEOUT,
            )

        response = await anyio.to_thread.run_sync(_receive)
        messages = response.get("Messages", [])

        if messages:
            _log(f"메시지 수신 | count={len(messages)} | poll #{_POLL_COUNT}")

        for m in messages:
            await _handle_message(m)

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
        message_type = body.get("type")

        if message_type == "SCREENSHOT_REQUEST":
            await process_screenshot_message(body)
        elif message_type == "BACKGROUND_REQUEST":
            await process_background_message(body)
        else:
            _log(f"⚠️ 잘못된 메시지 타입 무시 | type={message_type} | messageId={message_id}")
            _delete_message(receipt_handle)
            return

        _delete_message(receipt_handle)

    except json.JSONDecodeError as e:
        _log(f"❌ JSON 파싱 실패 | messageId={message_id} | error={str(e)}")
        _delete_message(receipt_handle)

    except Exception as e:
        _log(f"❌ 메시지 처리 실패 | messageId={message_id} | error={str(e)}")
        _log(traceback.format_exc())
        _delete_message(receipt_handle)


async def process_background_message(body: Dict[str, Any]) -> None:
    """
    배경 생성 메시지 처리
    1. Gemini 배경 생성
    2. S3 업로드
    3. Backend 알림
    """
    job_id = body.get("jobId", "")
    subject = body.get("subject", "lego creation")

    _log(f"배경 생성 시작 | jobId={job_id} | subject={subject}")

    # 1. Gemini 배경 생성
    _log("   [1/3] Gemini 배경 생성 중...")
    bg_bytes = await generate_background_async(subject)
    _log(f"   [1/3] 배경 생성 완료 | {len(bg_bytes)/1024:.1f}KB")

    # 2. S3 업로드
    _log("   [2/3] S3 업로드 중...")
    if not (USE_S3 and S3_BUCKET):
        raise RuntimeError("S3 is not configured for background upload.")

    now = datetime.now()
    # 경로: uploads/backgrounds/2026/02/jobId_bg.png
    s3_key = f"uploads/backgrounds/{now.year:04d}/{now.month:02d}/{job_id}_bg.png"
    background_url = upload_bytes_to_s3(bg_bytes, s3_key, "image/png")
    _log(f"   [2/3] S3 업로드 완료 | url={background_url[:60]}...")

    # 3. Backend 알림
    _log("   [3/3] Backend 알림 전송 중...")
    await notify_background_complete(job_id, background_url)
    _log("   [3/3] Backend 알림 완료")

    _log(f"배경 생성 완료 | jobId={job_id}")


def _delete_message(receipt_handle: str) -> None:
    """SQS 메시지 삭제"""
    try:
        sqs = _get_sqs_client()
        sqs.delete_message(
            QueueUrl=SQS_SCREENSHOT_QUEUE_URL,
            ReceiptHandle=receipt_handle,
        )
    except Exception as e:
        _log(f"❌ 메시지 삭제 실패 | error={str(e)}")


async def start_screenshot_consumer() -> None:
    """
    Screenshot SQS Consumer 시작 (FastAPI startup에서 호출)
    """
    if not SQS_ENABLED:
        _log("⚠️ SQS 비활성화 상태 (AWS_SQS_ENABLED=false)")
        return

    if not SQS_SCREENSHOT_QUEUE_URL:
        _log("⚠️ AWS_SQS_SCREENSHOT_QUEUE_URL 미설정 - Screenshot Consumer 미시작")
        return

    _log("═" * 60)
    _log("🚀 Screenshot Consumer 시작")
    _log(f"   - Queue URL: {SQS_SCREENSHOT_QUEUE_URL}")
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
