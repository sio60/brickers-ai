# vectordb/resolver.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pymongo.collection import Collection

import config
from db import get_db
from vectordb.utils import ResolvedPart, ensure_dat_ext

def resolve_part(part_file: str, parts_col: Collection, alias_col: Optional[Collection] = None) -> Optional[ResolvedPart]:
    """파일명이나 별칭을 통해 실제 DB 문서와 매칭되는 부품을 찾습니다."""
    """파일명, ID, 별칭을 통해 파츠를 식별합니다."""
    pf = (part_file or "").lower().replace("\\", "/").split("/")[-1].strip()
    if not pf: return None

    # 1) Direct match
    doc = parts_col.find_one({"$or": [{"canonicalFile": pf}, {"partFile": pf}, {"name": pf}]})
    if doc:
        return ResolvedPart(part_file=pf, part_id=str(doc.get("partId", "")), canonical_file=str(doc.get("canonicalFile") or doc.get("partFile") or pf), name=str(doc.get("name") or ""), doc=doc)

    # 2) Alias match
    if alias_col is not None:
        a = alias_col.find_one({"$or": [{"alias": pf}, {"from": pf}, {"fromFile": pf}]})
        if a:
            canonical = (a.get("toFile") or a.get("to") or a.get("canonicalFile") or "").lower().strip()
            if canonical: return resolve_part(canonical, parts_col)
    return None

def parts_vector_search(col: Collection, query_vector: List[float], limit: int = 10, num_candidates: int = 200, filters: Optional[Dict[str, Any]] = None) -> List[Dict]:
    """입력된 벡터와 유사한 부품들을 Atlas Vector Search로 검색합니다."""
    """아틀라스 벡터 시퀀스를 사용하여 유사 부품을 검색합니다."""
    if len(query_vector) != config.EMBEDDING_DIMS: raise ValueError(f"Vector length must be {config.EMBEDDING_DIMS}")
    
    stage = {"$vectorSearch": {"index": config.ATLAS_VECTOR_INDEX_PARTS, "path": config.VECTOR_FIELD, "queryVector": query_vector, "numCandidates": int(num_candidates), "limit": int(limit)}}
    if filters:
        if "category" in filters: stage["$vectorSearch"]["filter"] = {"category": {"$in": filters["category"]}}
        
    pipeline = [stage, {"$project": {"_id": 0, "partId": 1, "category": 1, "bbox": 1, "score": {"$meta": "vectorSearchScore"}}}]
    return list(col.aggregate(pipeline))

def get_bbox_doc(part_file: str) -> Optional[Dict]:
    """특정 부품 파일의 BBox 정보를 빠르게 조회합니다."""
    """특정 파일의 BBox 정보를 조회합니다 (공용 유틸)."""
    db = get_db()
    res = resolve_part(part_file, db[config.PARTS_COLLECTION], db["ldraw_aliases"])
    return res.doc.get("bbox") if res and res.doc else None
