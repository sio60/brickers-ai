# screenshot-server/service/backend_client.py
"""Screenshot 완료 시 Backend에 screenshotUrls 알림"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Dict

import httpx

BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8080").strip().rstrip("/")
INTERNAL_API_TOKEN = os.environ.get("INTERNAL_API_TOKEN", "").strip()


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{ts}] [Screenshot] {msg}")


async def notify_screenshots_complete(job_id: str, screenshot_urls: Dict[str, str]) -> None:
    """Backend에 screenshotUrls 업데이트 알림 전송"""
    url = f"{BACKEND_URL}/api/kids/jobs/{job_id}/screenshots"
    headers = {}
    if INTERNAL_API_TOKEN:
        headers["X-Internal-Token"] = INTERNAL_API_TOKEN

    _log(f"📤 Backend 알림 전송 | jobId={job_id} | url={url}")
    _log(f"   - views: {list(screenshot_urls.keys())}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.patch(
            url,
            json={"screenshotUrls": screenshot_urls},
            headers=headers,
        )
        r.raise_for_status()

    _log(f"✅ Backend 알림 성공 | jobId={job_id} | status={r.status_code}")


async def notify_gallery_screenshots_complete(gallery_post_id: str, screenshot_urls: Dict[str, str]) -> None:
    """갤러리 포스트 screenshotUrls 업데이트 알림 전송 (백필용)"""
    url = f"{BACKEND_URL}/api/gallery/{gallery_post_id}/screenshots"
    headers = {}
    if INTERNAL_API_TOKEN:
        headers["X-Internal-Token"] = INTERNAL_API_TOKEN

    _log(f"📤 Gallery 백필 알림 전송 | postId={gallery_post_id} | url={url}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.patch(
            url,
            json={"screenshotUrls": screenshot_urls},
            headers=headers,
        )
        r.raise_for_status()

    _log(f"✅ Gallery \ubc31\ud544 \uc54c\ubbc2 \uc131\uacf5 | postId={gallery_post_id} | status={r.status_code}")
 
 
async def notify_background_complete(job_id: str, background_url: str) -> None:
    """Backend\uc5d0 backgroundUrl \uc5c5\ub370\uc774\ud2b8 \uc54c\ubbc2 \uc804\uc1a1"""
    url = f"{BACKEND_URL}/api/kids/jobs/{job_id}/background"
    headers = {}
    if INTERNAL_API_TOKEN:
        headers["X-Internal-Token"] = INTERNAL_API_TOKEN
 
    _log(f"\ud83d\udce4 Backend \ubc30\uacbd \uc54c\ubbc2 \uc804\uc1a1 | jobId={job_id} | url={url}")
 
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.patch(
            url,
            json={"backgroundUrl": background_url},
            headers=headers,
        )
        r.raise_for_status()
 
    _log(f"\u2705 Backend \ubc30\uacbd \uc54c\ubbc2 \uc131\uacf5 | jobId={job_id} | status={r.status_code}")
