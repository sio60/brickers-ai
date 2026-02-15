# ============================================================================
# Vector Search: Atlas Vector Search 기반 RAG 검색
# ============================================================================

import logging
from typing import Dict, List, Any

from .embeddings import get_embedding

logger = logging.getLogger("CoScientistMemory")


def format_context_for_embedding(
    observation: str,
    verification: Dict[str, Any] = None,
    subject_name: str = None,
) -> str:
    """임베딩용 문맥 포맷팅 (의미론적 사물 이름 및 물리적 결함 정보 주입)"""
    context_parts = []

    # 0. 사물 이름
    metrics = verification.get("metrics_after", verification) if verification else {}
    subject = subject_name or metrics.get("subject_name", "Unknown Object")
    context_parts.append(f"Subject: {subject}")

    if verification:
        metrics = verification.get("metrics_after", verification)

        if not verification.get("stable", True):
            context_parts.append("Status: Unstable (Structural Collapse)")

        f_count = metrics.get("floating_count", 0)
        if f_count > 0:
            f_ids = metrics.get("floating_ids", [])
            id_str = f" IDs:{f_ids[:5]}" if f_ids else ""
            context_parts.append(f"Status: Floating Bricks ({f_count} bricks{id_str})")

        fallen_count = metrics.get("fallen_count", 0)
        if fallen_count > 0:
            context_parts.append(f"Status: Fallen Bricks ({fallen_count} bricks)")

        if metrics.get("budget_exceeded"):
            context_parts.append(f"Status: Budget Exceeded (Max:{metrics.get('target_budget')})")

        vol = metrics.get("total_volume", 0)
        if vol > 0:
            context_parts.append(f"Volume: {vol:.1f}")

        dims = metrics.get("dimensions", {})
        if dims:
            context_parts.append(f"Size: {dims.get('width', 0):.0f}x{dims.get('height', 0):.0f}x{dims.get('depth', 0):.0f}")

        ratio = metrics.get("failure_ratio", 0)
        context_parts.append(f"FailureRatio: {ratio:.2f}")

        s_ratio = metrics.get("small_brick_ratio", 0)
        context_parts.append(f"SmallBrickRatio: {s_ratio:.2f}")

    context_parts.append(f"Observation: {observation}")

    return " | ".join(context_parts)


def verify_vector_index(collection_exps, vector_index_name: str, _cache: dict = {"verified": False}) -> bool:
    """Vector Search 인덱스 존재 여부 확인 (캐시됨)"""
    if _cache["verified"]:
        return True

    if collection_exps is None:
        return False

    try:
        test_pipeline = [
            {"$vectorSearch": {
                "index": vector_index_name,
                "path": "embedding",
                "queryVector": [0.1] * 384,
                "numCandidates": 1,
                "limit": 1
            }}
        ]
        list(collection_exps.aggregate(test_pipeline))
        _cache["verified"] = True
        logger.info(f"Vector index '{vector_index_name}' verified")
        return True
    except Exception as e:
        if "index not found" in str(e).lower() or "Atlas" in str(e):
            logger.warning(f"Vector index '{vector_index_name}' not found. RAG disabled.")
        else:
            logger.warning(f"Vector index check failed: {e}")
        return False


def search_similar_cases(
    collection_exps,
    vector_index_name: str,
    use_vector: bool,
    observation: str,
    limit: int = 10,
    min_score: float = 0.5,
    verification_metrics: Dict[str, Any] = None,
    subject_name: str = None,
) -> List[Dict]:
    """RAG: 유사 후보군 검색 (Re-ranking을 위해 넉넉히 검색)"""
    if not use_vector or collection_exps is None:
        return []

    if not verify_vector_index(collection_exps, vector_index_name):
        return []

    query_text = format_context_for_embedding(observation, verification_metrics, subject_name=subject_name)
    query_vector = get_embedding(query_text)
    if not query_vector:
        return []

    pipeline = [
        {
            "$vectorSearch": {
                "index": vector_index_name,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": limit * 10,
                "limit": limit * 2
            }
        },
        {
            "$addFields": {
                "similarity_score": {"$meta": "vectorSearchScore"}
            }
        },
        {
            "$match": {
                "similarity_score": {"$gte": min_score}
            }
        },
        {
            "$limit": limit
        },
        {
            "$project": {
                "_id": 0,
                "session_id": 0,
            }
        }
    ]

    try:
        results = list(collection_exps.aggregate(pipeline))

        if not results:
            logger.info(f"No matches with score >= {min_score}. Trying fallback...")
            fallback_pipeline = [
                {
                    "$vectorSearch": {
                        "index": vector_index_name,
                        "path": "embedding",
                        "queryVector": query_vector,
                        "numCandidates": 10,
                        "limit": 1
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "session_id": 0,
                    }
                }
            ]
            results = list(collection_exps.aggregate(fallback_pipeline))
            if results:
                logger.info("Fallback successful: Found 1 similar case.")
        else:
            logger.info(f"Found {len(results)} similar cases (score >= {min_score})")

        for res in results:
            score = res.get("similarity_score", 0)
            if score >= min_score:
                res["reliability"] = "high"
            elif score >= (min_score - 0.1):
                res["reliability"] = "medium"
            else:
                res["reliability"] = "low"

        return results
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []


def search_success_and_failure(
    collection_exps,
    vector_index_name: str,
    use_vector: bool,
    observation: str,
    limit: int = 5,
    min_score: float = 0.5,
    verification_metrics: Dict[str, Any] = None,
    shape_metrics: Dict[str, Any] = None,
    subject_name: str = None,
) -> Dict[str, List[Dict]]:
    """성공/실패 사례를 구분하여 검색"""
    results = {"success": [], "failure": []}

    candidates = search_similar_cases(
        collection_exps, vector_index_name, use_vector,
        observation, limit=limit*3, min_score=min_score,
        verification_metrics=verification_metrics,
        subject_name=subject_name
    )

    for case in candidates:
        if case.get("result_success", False):
            results["success"].append(case)
        else:
            results["failure"].append(case)

    results["success"] = results["success"][:limit]
    results["failure"] = results["failure"][:limit]

    return results
