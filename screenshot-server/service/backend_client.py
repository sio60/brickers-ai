# screenshot-server/service/backend_client.py
"""Screenshot 완료 시 Backend에 screenshotUrls 알림 (지수 백오프 재시도 포함)"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Dict

import httpx

BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8080").strip().rstrip("/")
INTERNAL_API_TOKEN = os.environ.get("INTERNAL_API_TOKEN", "").strip()

MAX_RETRIES = 5
INITIAL_DELAY = 5  # 초


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{ts}] [Screenshot] {msg}")


async def _patch_with_retry(url: str, json_body: dict, label: str) -> None:
    """지수 백오프 재시도로 Backend PATCH 호출"""
    headers = {}
    if INTERNAL_API_TOKEN:
        headers["X-Internal-Token"] = INTERNAL_API_TOKEN

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _log(f"📤 {label} (시도 {attempt}/{MAX_RETRIES}) | url={url}")

            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.patch(url, json=json_body, headers=headers)
                r.raise_for_status()

            _log(f"✅ {label} 성공 | status={r.status_code}")
            return

        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
            if attempt < MAX_RETRIES:
                delay = INITIAL_DELAY * (2 ** (attempt - 1))
                _log(f"⚠️ {label} 실패 (시도 {attempt}/{MAX_RETRIES}), {delay}초 후 재시도 | error={e}")
                await asyncio.sleep(delay)
            else:
                _log(f"❌ {label} 최종 실패 | error={e}")
                raise


async def notify_screenshots_complete(job_id: str, screenshot_urls: Dict[str, str]) -> None:
    """Backend에 screenshotUrls 업데이트 알림 전송"""
    url = f"{BACKEND_URL}/api/kids/jobs/{job_id}/screenshots"
    _log(f"   - views: {list(screenshot_urls.keys())}")
    await _patch_with_retry(url, {"screenshotUrls": screenshot_urls}, f"Backend 알림 | jobId={job_id}")


async def notify_gallery_screenshots_complete(gallery_post_id: str, screenshot_urls: Dict[str, str]) -> None:
    """갤러리 포스트 screenshotUrls 업데이트 알림 전송 (백필용)"""
    url = f"{BACKEND_URL}/api/gallery/{gallery_post_id}/screenshots"
    await _patch_with_retry(url, {"screenshotUrls": screenshot_urls}, f"Gallery 백필 알림 | postId={gallery_post_id}")


async def notify_background_complete(job_id: str, background_url: str) -> None:
    """Backend에 backgroundUrl 업데이트 알림 전송"""
    url = f"{BACKEND_URL}/api/kids/jobs/{job_id}/background"
    await _patch_with_retry(url, {"backgroundUrl": background_url}, f"Backend 배경 알림 | jobId={job_id}")
