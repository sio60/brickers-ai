from __future__ import annotations

import asyncio
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import config
from route import kids_render
from route.sqs_consumer import start_consumer
from route import color_variant


import os
import httpx

from chat.router import router as chat_router
from chat.memory import InMemoryConversationStore
from chat.service import ChatService

app = FastAPI()


@app.on_event("startup")
async def startup():
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 환경변수가 없습니다.")

    app.state.openai_http = httpx.AsyncClient(
        base_url="https://api.openai.com",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0),
    )

    store = InMemoryConversationStore(
        max_messages=int(os.getenv("CHAT_MAX_MESSAGES", "20")),
        ttl_seconds=int(os.getenv("CHAT_TTL_SECONDS", "3600")),
    )

    app.state.chat_service = ChatService(http=app.state.openai_http, store=store)


@app.on_event("shutdown")
async def shutdown():
    await app.state.openai_http.aclose()


app.include_router(chat_router)

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
    return {
        "status": "ok",
        "mode": "kids-only",
        "env": getattr(config, "ENV", "unknown"),
    }


@app.on_event("startup")
async def startup_event():
    """FastAPI 시작 시 SQS Consumer 백그라운드 태스크 시작"""
    print("=" * 70)
    print("[FastAPI] 🚀 Application Startup")
    print("=" * 70)

    # SQS Consumer 백그라운드 태스크 시작
    asyncio.create_task(start_consumer())
    print("[FastAPI] ✅ SQS Consumer 백그라운드 태스크 시작")

    # 🔍 라우트 디버깅 (등록된 모든 API 주소 출력)
    print("\n[Debug] Registered Routes:")
    for route in app.routes:
        if hasattr(route, "path"):
            print(f" - {route.methods} {route.path}")
    print("=" * 70)


# ✅ Kids Mode router 연결
app.include_router(kids_render.router)

# ✅ Color Variant router 연결
app.include_router(color_variant.router)

# ✅ Chat router 연결
from chat.router import router as chat_router
app.include_router(chat_router)