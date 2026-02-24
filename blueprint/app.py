# blueprint/app.py
"""Brickers Blueprint Server - PDF 생성 전용 FastAPI 서버"""
from __future__ import annotations

import asyncio
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from route import instructions_pdf
from sqs_consumer import start_pdf_consumer

# ============================================================================
# Blueprint 서버 logging 설정
# ============================================================================
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="[%(asctime)s] [%(name)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
for _uv_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
    logging.getLogger(_uv_name).setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

logger = logging.getLogger(__name__)

app = FastAPI(title="Brickers Blueprint Server", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 프론트엔드 직접 호출 API 유지
app.include_router(instructions_pdf.router)


@app.on_event("startup")
async def startup():
    logger.info("=" * 60)
    logger.info("🚀 Blueprint PDF Server Startup")
    logger.info("=" * 60)

    # SQS 폴링 백그라운드 태스크 시작
    asyncio.create_task(start_pdf_consumer())
    logger.info("✅ PDF SQS Consumer 백그라운드 태스크 시작")

    # 등록된 라우트 출력
    logger.debug("Registered Routes:")
    for route in app.routes:
        if hasattr(route, "path"):
            methods = getattr(route, "methods", {"?"})
            logger.debug("  - %s %s", methods, route.path)
    logger.info("=" * 60)


@app.get("/health")
def health():
    from service.render_client import RENDER_ENABLED
    return {
        "status": "ok",
        "service": "blueprint",
        "ldview": "available" if RENDER_ENABLED else "not_found",
    }
