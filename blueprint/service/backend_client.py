# blueprint/service/backend_client.py
"""PDF 완료 시 Backend에 pdfUrl 알림"""
from __future__ import annotations

import os
from datetime import datetime

import httpx

BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8080").strip().rstrip("/")
INTERNAL_API_TOKEN = os.environ.get("INTERNAL_API_TOKEN", "").strip()


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{ts}] [Blueprint] {msg}")


async def notify_pdf_complete(job_id: str, pdf_url: str) -> None:
    """Backend에 PDF URL 업데이트 알림 전송"""
    url = f"{BACKEND_URL}/api/kids/jobs/{job_id}/pdf"
    headers = {}
    if INTERNAL_API_TOKEN:
        headers["X-Internal-Token"] = INTERNAL_API_TOKEN

    _log(f"📤 Backend 알림 전송 | jobId={job_id} | url={url}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.patch(
            url,
            json={"pdfUrl": pdf_url},
            headers=headers,
        )
        r.raise_for_status()

    _log(f"✅ Backend 알림 성공 | jobId={job_id} | status={r.status_code}")
