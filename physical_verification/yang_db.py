# ============================================================================
# MongoDB 데이터베이스 연결 및 조회 모듈
# 이 파일은 물리 검증 시스템이 필요한 레고 파트 데이터를 조회할 수 있도록
# MongoDB 데이터베이스와의 연결을 설정하고 관리합니다.
# MongoDB 클라이언트, 파트 컬렉션 및 데이터베이스 인스턴스에 접근하는
# 유틸리티 함수들을 제공합니다.
# ============================================================================

from pymongo import MongoClient
import config

_client = None

def get_client() -> MongoClient:
    global _client
    if _client is None:
        if not config.MONGODB_URI:
            raise RuntimeError("MONGODB_URI가 비어 있습니다. .env 파일을 확인하세요.")
        # DB가 다운되었거나 연결할 수 없는 경우 빠르게 실패하도록 짧은 타임아웃 설정
        _client = MongoClient(config.MONGODB_URI, serverSelectionTimeoutMS=2000)
    return _client

def get_parts_collection():
    client = get_client()
    return client[config.MONGODB_DB][config.PARTS_COLLECTION]

def get_db():
    client = get_client()
    return client[config.MONGODB_DB]
