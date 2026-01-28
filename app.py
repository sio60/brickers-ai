from __future__ import annotations

import asyncio
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import config
from route import kids_render
from route.sqs_consumer import start_consumer


app = FastAPI(title="Brickers AI API - Kids Mode", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ✅ 배포/로컬 모두 허용 (보안상 필요시 도메인 지정)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 핵심: kids_render의 GENERATED_DIR을 그대로 /api/generated 로 공개
app.mount(
    "/api/generated",
    StaticFiles(directory=str(kids_render.GENERATED_DIR)),
    name="api_generated",
)

# (선택) 호환용
app.mount(
    "/generated",
    StaticFiles(directory=str(kids_render.GENERATED_DIR)),
    name="generated",
)


@app.get("/health")
def health():
    return {"status": "ok", "mode": "kids-only", "env": getattr(config, "ENV", "unknown")}


@app.on_event("startup")
async def startup_event():
    """FastAPI 시작 시 SQS Consumer 백그라운드 태스크 시작"""
    print("=" * 70)
    print("[FastAPI] 🚀 Application Startup")
    print("=" * 70)

    # SQS Consumer 백그라운드 태스크 시작
    asyncio.create_task(start_consumer())
    print("[FastAPI] ✅ SQS Consumer 백그라운드 태스크 시작")


# ✅ Kids Mode router 연결
app.include_router(kids_render.router)
