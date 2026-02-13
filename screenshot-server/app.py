# screenshot-server/app.py
"""Brickers Screenshot Server - health check only (Celery worker handles processing)"""
from __future__ import annotations

from fastapi import FastAPI

from service.render_client import RENDER_ENABLED

app = FastAPI(title="Brickers Screenshot Server", version="0.2.0")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "screenshot-server",
        "ldview": RENDER_ENABLED,
        "worker": "celery",
    }
