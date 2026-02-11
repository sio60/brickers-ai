# ============================================================================
# FastAPI 메인 애플리케이션 파일
# Kids Mode AI 서버 + 챗봇 API를 통합 제공
# ============================================================================
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
# instructions_pdf router moved to blueprint server
from route import admin             # ✅ [NEW] Admin/Logs

from chat.router import router as chat_router
from chat.memory import InMemoryConversationStore
from chat.service import ChatService


# ============================================================================
# 앱 인스턴스 생성 (단 한 번만!)
# ============================================================================
app = FastAPI(title="Brickers AI API", version="0.2.0")

# ============================================================================
# CORS 미들웨어
# ============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ✅ 배포/로컬 모두 허용 (보안상 필요시 도메인 지정)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# 정적 파일 마운트
# ============================================================================
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

# ============================================================================
# 라우터 등록 (모든 API 엔드포인트)
# ============================================================================
app.include_router(kids_render.router)      # Kids Mode
app.include_router(color_variant.router)    # Color Variant
# instructions_pdf.router moved to blueprint server (port 8001)
app.include_router(admin.router)            # ✅ [NEW] Admin/Logs
app.include_router(chat_router)             # ✅ 챗봇 (/api/v1/chat)

# --- [Integrate] Brick Judge (Rust Viewer) ---
import brick_judge.server as bj_server  # noqa: E402

# 1. 뷰어 페이지 (HTML)
app.add_api_route("/brick-judge/viewer", bj_server.viewer, methods=["GET"], include_in_schema=False)
# 2. 뷰어용 API (HTML에서 호출하는 절대 경로 /api/verify 대응)
app.add_api_route("/api/verify", bj_server.verify_ldr, methods=["POST"], tags=["viewer"])
# 3. LLM용 Judge API
app.add_api_route("/api/judge", bj_server.judge_ldr, methods=["POST"], tags=["judge"])
# 4. 정보 API
app.add_api_route("/api/info", bj_server.info, methods=["GET"], tags=["info"])


# ============================================================================
# Startup 이벤트
# ============================================================================
@app.on_event("startup")
async def startup():
    """서버 시작 시 초기화"""
    print("=" * 70, flush=True)
    print("[FastAPI] 🚀 Application Startup", flush=True)
    print("=" * 70, flush=True)

    # --- OpenAI/Gemini HTTP 클라이언트 초기화 ---
    openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    gemini_key = (os.getenv("GEMINI_API_KEY") or "").strip()

    api_key = openai_key or gemini_key
    base_url = "https://api.openai.com/v1/" if openai_key else "https://generativelanguage.googleapis.com/v1beta/openai/"

    if not api_key:
        print("⚠️ [Warn] OPENAI_API_KEY/GEMINI_API_KEY 둘 다 없음. 챗봇 기능 비활성화.", flush=True)
        app.state.openai_http = None
        app.state.chat_service = None
    else:
        print(f"[Startup] Using API at {base_url}", flush=True)
        app.state.openai_http = httpx.AsyncClient(
            base_url=base_url,
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

        # --- [NEW] Analytics Agent Service 초기화 ---
        from service.analytics_agent_service import AnalyticsAgentService
        app.state.analytics_agent = AnalyticsAgentService(http_client=app.state.openai_http)
        print("[FastAPI] ✅ Analytics Agent Service Enabled", flush=True)

    # --- [NEW] Global log capture ---
    try:
        from service.log_context import GlobalLogCapture
        glc = GlobalLogCapture() # 싱글톤이라 한번만 실행됨
        glc.start_flusher() # Start Start Smart Batching Flusher
        print("[FastAPI] ✅ Global Logging Enabled", flush=True)
    except ImportError as e:
        print(f"⚠️ [Warn] GlobalLogCapture failed: {e}", flush=True)

    # --- SQS Consumer 백그라운드 태스크 시작 ---
    asyncio.create_task(start_consumer())
    print("[FastAPI] ✅ SQS Consumer 백그라운드 태스크 시작", flush=True)

    # --- 라우트 디버깅 (등록된 모든 API 주소 출력) ---
    print("\n[Debug] Registered Routes:", flush=True)
    for route in app.routes:
        if hasattr(route, "path"):
            methods = getattr(route, "methods", {"?"})
            print(f"  - {methods} {route.path}", flush=True)
    print("=" * 70, flush=True)


@app.on_event("shutdown")
async def shutdown():
    """서버 종료 시 정리"""
    if app.state.openai_http:
        await app.state.openai_http.aclose()


# ============================================================================
# Health Check
# ============================================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": "kids-only",
        "env": getattr(config, "ENV", "unknown"),
    }
