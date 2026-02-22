# screenshot-server/app.py
"""Brickers Screenshot Server - health check + render API"""
from __future__ import annotations

from fastapi import FastAPI

from service.render_client import RENDER_ENABLED
from route.render import router as render_router

app = FastAPI(title="Brickers Screenshot Server", version="0.3.0")

app.include_router(render_router)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "screenshot-server",
        "ldview": RENDER_ENABLED,
        "worker": "celery",
    }
