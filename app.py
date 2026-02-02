from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from db import get_db, get_parts_collection
from vectordb.seed import seed_dummy_parts
from vectordb.search import parts_vector_search
from ldr.import_to_mongo import import_ldr_bom_with_steps, import_car_ldr

import asyncio
import os
import httpx

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter

import config
from route import kids_render
from route.sqs_consumer import start_consumer
from route import color_variant

# ✅ instructions routers
from route.instructions_pdf import router as instructions_router
from route.instructions_upload import router as instructions_upload_router

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

# ✅ 정적 서빙 (로컬 CDN)
# 업로드가 uploads/instructions/... 에 저장한다고 했으니
# 프론트/서버에서 http://localhost:8000/static/instructions/<file> 로 접근 가능해야 함.

# [선택 A] 지금 방식 유지: uploads 전체를 /static 으로 노출
# app.mount("/static", StaticFiles(directory="uploads"), name="static")

# ✅ [추천 B] instructions만 노출 (더 안전/명확)
app.mount(
    "/static/instructions",
    StaticFiles(directory="uploads/instructions"),
    name="instructions_static",
)

api = APIRouter(prefix="/api/v1")
# ✅ 핵심: kids_render의 GENERATED_DIR을 그대로 /api/generated 로 공개

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


class VectorSearchRequest(BaseModel):
    query_vector: List[float] = Field(...)
    limit: int = 10
    num_candidates: int = 200
    category: Optional[List[str]] = None


class LdrImportRequest(BaseModel):
    job_id: str
    ldr_path: Optional[str] = None


@api.get("/health", tags=["system"])
def health():
    return {"status": "ok", "mode": "kids-only", "env": getattr(config, "ENV", "unknown")}


@api.get("/mongo/ping", tags=["system"])
def mongo_ping():
    db = get_db()
    return {
        "db": getattr(config, "MONGODB_DB", "unknown"),
        "collections": db.list_collection_names()
    }


@api.post("/ldr/import", tags=["ldr"])
def api_ldr_import(req: LdrImportRequest):
    if req.ldr_path:
        result = import_ldr_bom_with_steps(job_id=req.job_id, ldr_path=req.ldr_path)
    else:
        result = import_car_ldr(job_id=req.job_id)
    return {"ok": True, **result}


@api.post("/vectordb/seed", tags=["vectordb"])
def api_seed():
    n = seed_dummy_parts(overwrite=True)
    return {"inserted": n}


@api.post("/vectordb/parts/search", tags=["vectordb"])
def api_search(req: VectorSearchRequest):
    dims = getattr(config, "EMBEDDING_DIMS", None)
    if dims is not None and len(req.query_vector) != dims:
        raise HTTPException(
            status_code=400,
            detail=f"query_vector must be length {dims}, got {len(req.query_vector)}",
        )

    col = get_parts_collection()
    filters = {"category": req.category} if req.category else None

    hits = parts_vector_search(
        col=col,
        query_vector=req.query_vector,
        limit=req.limit,
        num_candidates=req.num_candidates,
        filters=filters,
    )
    return {"count": len(hits), "items": hits}


# ✅ 메인 api 라우터 등록
app.include_router(api)

# ✅ kids 라우터
app.include_router(kids_render.router, prefix="/api/v1/kids", tags=["kids"])

# ✅ instructions 라우터들
# (라우터 파일 내부 prefix="/api/instructions" 이므로 프론트는 /api/instructions/...로 호출)
app.include_router(instructions_router)
app.include_router(instructions_upload_router)
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
# app.include_router(kids_render.router)
# ✅ 라우터 등록 (Chat, Kids, Color)
app.include_router(chat_router)          # /api/v1/chat
app.include_router(kids_render.router)   # Kids Mode
app.include_router(color_variant.router) # Color Variant
