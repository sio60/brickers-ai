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
