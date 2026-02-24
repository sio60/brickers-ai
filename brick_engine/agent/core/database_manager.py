# ============================================================================
# 데이터베이스 관리 및 MongoDB 연결 모듈
# 핵심 데이터 조회를 위한 MongoDB 연결을 설정하고 관리합니다.
# ============================================================================

import logging
from pymongo import MongoClient
import config

logger = logging.getLogger(__name__)

_client = None

def get_client() -> MongoClient:
    """MongoDB 클라이언트를 생성하거나 반환합니다."""
    global _client
    if _client is None:
        if not config.MONGODB_URI:
            logger.warning("⚠️ [Warn] MONGODB_URI is missing. DB features will be disabled.")
            return None
        # 연결 타임아웃을 짧게 설정하여 빠른 실패 유도
        _client = MongoClient(config.MONGODB_URI, serverSelectionTimeoutMS=2000)
    return _client

def get_parts_collection():
    """파트 컬렉션을 반환합니다."""
    client = get_client()
    return client[config.MONGODB_DB][config.PARTS_COLLECTION] if client else None

def get_db():
    """데이터베이스 인스턴스를 반환합니다."""
    client = get_client()
    return client[config.MONGODB_DB] if client else None
