from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional

import config
from db import get_db, get_parts_collection
from vectordb.seed import seed_dummy_parts
from vectordb.search import parts_vector_search
from ldr.import_to_mongo import import_ldr_bom_with_steps, import_car_ldr

# ✅ kids router
from route import kids_render

# ✅ instructions routers
from route.instructions_pdf import router as instructions_router
from route.instructions_upload import router as instructions_upload_router


app = FastAPI(title="Brickers AI API", version="0.1.0")

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=getattr(config, "CORS_ORIGINS", ["*"]),
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


class VectorSearchRequest(BaseModel):
    query_vector: List[float] = Field(...)
    limit: int = 10
    num_candidates: int = 200
    category: Optional[List[str]] = None


class LdrImportRequest(BaseModel):
    job_id: str
    ldr_path: Optional[str] = None  # 없으면 car.ldr 기본 사용


@api.get("/health", tags=["system"])
def health():
    return {"status": "ok", "env": getattr(config, "ENV", "unknown")}


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
