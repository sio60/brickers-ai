from __future__ import annotations

import asyncio
import os
import httpx

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import config
from route import kids_render
from route.sqs_consumer import start_consumer
from route import color_variant

from chat.router import router as chat_router
from chat.memory import InMemoryConversationStore
from chat.service import ChatService

# ✅ 단일 FastAPI 인스턴스 (모든 기능 통합)
app = FastAPI(title="Brickers AI API", version="0.2.0")

# ✅ CORS 미들웨어
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포/로컬 모두 허용 (보안상 필요시 도메인 지정)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    """FastAPI 시작 시 초기화"""
    print("=" * 70)
    print("[FastAPI] 🚀 Application Startup")
    print("=" * 70)

    # ✅ 1. OpenAI 클라이언트 초기화 (Chat 기능용)
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        print("[FastAPI] ⚠️ OPENAI_API_KEY 환경변수가 없습니다. Chat 기능이 작동하지 않습니다.")
        app.state.openai_http = None
        app.state.chat_service = None
    else:
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
        print("[FastAPI] ✅ Chat Service 초기화 완료")

    # ✅ 2. SQS Consumer 백그라운드 태스크 시작
    asyncio.create_task(start_consumer())
    print("[FastAPI] ✅ SQS Consumer 백그라운드 태스크 시작")


@app.on_event("shutdown")
async def shutdown():
    """FastAPI 종료 시 정리"""
    if app.state.openai_http:
        await app.state.openai_http.aclose()
    print("[FastAPI] 👋 Application Shutdown")


# ✅ Health Check
@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": "kids-only",
        "env": getattr(config, "ENV", "unknown"),
    }


# ✅ Static Files 마운트
app.mount(
    "/api/generated",
    StaticFiles(directory=str(kids_render.GENERATED_DIR)),
    name="api_generated",
)
app.mount(
    "/generated",
    StaticFiles(directory=str(kids_render.GENERATED_DIR)),
    name="generated",
)

# ✅ 라우터 등록 (Chat, Kids, Color)
app.include_router(chat_router)          # /api/v1/chat
app.include_router(kids_render.router)   # Kids Mode
app.include_router(color_variant.router) # Color Variant
