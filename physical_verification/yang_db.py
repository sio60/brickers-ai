# 이 파일은 db.py와 유사하게 MongoDB 연결을 처리하지만 사용자별 설정을 분리하기 위한 대체 DB 파일입니다.
# 검증할때 mongodb에서 부품 불러오는 파일

from pymongo import MongoClient
import config

_client = None

def get_client() -> MongoClient:
    global _client
    if _client is None:
        if not config.MONGODB_URI:
            raise RuntimeError("MONGODB_URI is empty. Check .env")
        # Set short timeout to fail fast if DB is down/unreachable
        _client = MongoClient(config.MONGODB_URI, serverSelectionTimeoutMS=2000)
    return _client

def get_parts_collection():
    client = get_client()
    return client[config.MONGODB_DB][config.PARTS_COLLECTION]

def get_db():
    client = get_client()
    return client[config.MONGODB_DB]
