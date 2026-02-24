# blueprint/service/backend_client.py
"""PDF 완료 시 Backend에 pdfUrl 알림 (지수 백오프 재시도 포함)"""
from __future__ import annotations

import asyncio
import logging
import os

import httpx

logger = logging.getLogger(__name__)

BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8080").strip().rstrip("/")
INTERNAL_API_TOKEN = os.environ.get("INTERNAL_API_TOKEN", "").strip()

MAX_RETRIES = 5
INITIAL_DELAY = 5  # 초


def _log(msg: str) -> None:
    logger.info("%s", msg)


async def notify_pdf_complete(job_id: str, pdf_url: str) -> None:
    """Backend에 PDF URL 업데이트 알림 전송 (최대 5회 재시도, 지수 백오프)"""
    url = f"{BACKEND_URL}/api/kids/jobs/{job_id}/pdf"
    headers = {}
    if INTERNAL_API_TOKEN:
        headers["X-Internal-Token"] = INTERNAL_API_TOKEN

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _log(f"📤 Backend 알림 전송 (시도 {attempt}/{MAX_RETRIES}) | jobId={job_id}")

            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.patch(
                    url,
                    json={"pdfUrl": pdf_url},
                    headers=headers,
                )
                r.raise_for_status()

            _log(f"✅ Backend 알림 성공 | jobId={job_id} | status={r.status_code}")
            return

        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
            if attempt < MAX_RETRIES:
                delay = INITIAL_DELAY * (2 ** (attempt - 1))  # 5, 10, 20, 40, 80초
                _log(f"⚠️ 알림 실패 (시도 {attempt}/{MAX_RETRIES}), {delay}초 후 재시도 | error={e}")
                await asyncio.sleep(delay)
            else:
                _log(f"❌ Backend 알림 최종 실패 | jobId={job_id} | error={e}")
                raise
