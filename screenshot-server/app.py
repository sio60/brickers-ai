# screenshot-server/app.py
"""Brickers Screenshot Server - 6면 스크린샷 생성"""
from __future__ import annotations

import asyncio

from fastapi import FastAPI

from sqs_consumer import start_screenshot_consumer
from service.render_client import RENDER_ENABLED

app = FastAPI(title="Brickers Screenshot Server", version="0.1.0")


@app.on_event("startup")
async def startup():
    asyncio.create_task(start_screenshot_consumer())


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "screenshot-server",
        "ldview": RENDER_ENABLED,
    }
