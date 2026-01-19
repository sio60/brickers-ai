from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

# ✅ ai. 접두사 제거 (같은 repo 루트 기준 import)
import config
from db import get_db, get_parts_collection
from vectordb.seed import seed_dummy_parts
from vectordb.search import parts_vector_search
from ldr.import_to_mongo import import_ldr_bom, import_car_ldr

# ✅ kids router
from route import kids_render

app = FastAPI(title="Brickers AI API", version="0.1.0")


class VectorSearchRequest(BaseModel):
    query_vector: List[float] = Field(...)
    limit: int = 10
    num_candidates: int = 200
    category: Optional[List[str]] = None


class LdrImportRequest(BaseModel):
    job_id: str
    ldr_path: Optional[str] = None  # 없으면 car.ldr 기본 사용


@app.get("/health")
def health():
    return {"status": "ok", "env": getattr(config, "ENV", "unknown")}


@app.post("/ldr/import")
def api_ldr_import(req: LdrImportRequest):
    if req.ldr_path:
        result = import_ldr_bom(job_id=req.job_id, ldr_path=req.ldr_path)
    else:
        result = import_car_ldr(job_id=req.job_id)
    return {"ok": True, **result}


@app.get("/mongo/ping")
def mongo_ping():
    db = get_db()
    return {"db": getattr(config, "MONGODB_DB", "unknown"), "collections": db.list_collection_names()}


@app.post("/vectordb/seed")
def api_seed():
    n = seed_dummy_parts(overwrite=True)
    return {"inserted": n}


@app.post("/vectordb/parts/search")
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


# ✅ kids router 연결
app.include_router(kids_render.router)
