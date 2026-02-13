# screenshot-server/tasks.py
"""Celery tasks - screenshot & background processing with auto-retry"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Dict, Any

from celery_app import app
from service.render_client import render_6_views, RENDER_ENABLED
from service.s3_client import USE_S3, S3_BUCKET, upload_bytes_to_s3
from service.backend_client import (
    notify_screenshots_complete,
    notify_gallery_screenshots_complete,
    notify_background_complete,
)
from service.background_composer import generate_background_async


SCREENSHOT_S3_PREFIX = "uploads/screenshots"


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{ts}] [CeleryTask] {msg}")


async def _process_screenshot_async(
    job_id: str = "",
    ldr_url: str = "",
    model_name: str = "model",
    source: str = "job",
    gallery_post_id: str = "",
) -> None:
    """6-view screenshot pipeline (reuses sqs_consumer logic)"""
    import httpx

    identifier = gallery_post_id if source == "gallery_backfill" else job_id
    s3_id = gallery_post_id if source == "gallery_backfill" else job_id
    _log(f"screenshot start | source={source} | id={identifier} | model={model_name}")

    # 1. Download LDR
    _log(f"  [1/4] downloading LDR | {ldr_url[:80]}")
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        r = await client.get(ldr_url)
        r.raise_for_status()
        ldr_text = r.text
    _log(f"  [1/4] LDR downloaded | {len(ldr_text)} chars")

    # 2. Render 6 views
    if not RENDER_ENABLED:
        raise RuntimeError("LDView binary not found. Screenshot generation requires LDView.")
    _log("  [2/4] rendering 6 views...")
    screenshots = await render_6_views(ldr_text)
    valid_count = sum(1 for v in screenshots.values() if v)
    _log(f"  [2/4] render done | {valid_count}/6 views")

    # 3. Upload to S3
    _log("  [3/4] uploading to S3...")
    if not (USE_S3 and S3_BUCKET):
        raise RuntimeError("S3 is not configured. Cannot upload screenshots.")

    now = datetime.now()
    screenshot_urls: Dict[str, str] = {}
    for view_name, png_bytes in screenshots.items():
        if not png_bytes:
            continue
        s3_key = f"{SCREENSHOT_S3_PREFIX}/{now.year:04d}/{now.month:02d}/{s3_id}_{view_name}.png"
        url = upload_bytes_to_s3(png_bytes, s3_key, "image/png")
        screenshot_urls[view_name] = url
    _log(f"  [3/4] S3 upload done | {len(screenshot_urls)} views")

    # 4. Notify backend
    _log("  [4/4] notifying backend...")
    if source == "gallery_backfill":
        await notify_gallery_screenshots_complete(gallery_post_id, screenshot_urls)
    else:
        await notify_screenshots_complete(job_id, screenshot_urls)
    _log(f"screenshot done | source={source} | id={identifier} | views={list(screenshot_urls.keys())}")


async def _process_background_async(
    job_id: str = "",
    subject: str = "lego creation",
) -> None:
    """Background generation pipeline (reuses sqs_consumer logic)"""
    _log(f"background start | jobId={job_id} | subject={subject}")

    # 1. Generate background via Gemini
    _log("  [1/3] generating background...")
    bg_bytes = await generate_background_async(subject)
    _log(f"  [1/3] background generated | {len(bg_bytes)/1024:.1f}KB")

    # 2. Upload to S3
    _log("  [2/3] uploading to S3...")
    if not (USE_S3 and S3_BUCKET):
        raise RuntimeError("S3 is not configured for background upload.")
    now = datetime.now()
    s3_key = f"uploads/backgrounds/{now.year:04d}/{now.month:02d}/{job_id}_bg.png"
    background_url = upload_bytes_to_s3(bg_bytes, s3_key, "image/png")
    _log(f"  [2/3] S3 upload done | url={background_url[:60]}...")

    # 3. Notify backend
    _log("  [3/3] notifying backend...")
    await notify_background_complete(job_id, background_url)
    _log(f"background done | jobId={job_id}")


@app.task(
    name="tasks.process_screenshot",
    bind=True,
    acks_late=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
    max_retries=3,
    reject_on_worker_lost=True,
)
def process_screenshot(self, **kwargs):
    _log(f"[retry={self.request.retries}/{self.max_retries}] process_screenshot | jobId={kwargs.get('job_id', '')}")
    asyncio.run(_process_screenshot_async(**kwargs))


@app.task(
    name="tasks.process_background",
    bind=True,
    acks_late=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
    max_retries=3,
    reject_on_worker_lost=True,
)
def process_background(self, **kwargs):
    _log(f"[retry={self.request.retries}/{self.max_retries}] process_background | jobId={kwargs.get('job_id', '')}")
    asyncio.run(_process_background_async(**kwargs))
