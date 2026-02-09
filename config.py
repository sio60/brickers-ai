import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# ✅ 실행 위치와 상관없이 ai/.env 로드
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ✅ 전역 Import 경로 등록 (memory_utils 등을 어디서든 import 가능)
AGENT_DIR = BASE_DIR / "brick_engine" / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

ENV = os.getenv("ENV", "local")

################################
# MongoDB Atlas (옵션)
################################
MONGODB_URI = os.getenv("MONGODB_URI", "")
MONGODB_DB = os.getenv("MONGODB_DB", "brickers")
PARTS_COLLECTION = os.getenv("PARTS_COLLECTION", "ldraw_parts")

################################
# Atlas Vector Search (옵션)
################################
ATLAS_VECTOR_INDEX_PARTS = os.getenv("ATLAS_VECTOR_INDEX_PARTS", "")
ATLAS_VECTOR_INDEX_MEMORY = os.getenv("ATLAS_VECTOR_INDEX_MEMORY", "co_scientist_memory_index") # Default index name for memory
VECTOR_FIELD = os.getenv("VECTOR_FIELD", "embedding")
EMBEDDING_DIMS = int(os.getenv("EMBEDDING_DIMS", "512"))
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "intfloat/multilingual-e5-small")  # 384차원 (경량화)

################################
# Kids Render (옵션이지만 거의 필수)
################################
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
NANO_BANANA_MODEL = os.getenv("NANO_BANANA_MODEL", "gemini-2.5-flash-image")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

################################
# Local Base URL (for static files)
################################
API_PUBLIC_BASE_URL = os.getenv("API_PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/")




def require_env(name: str) -> str:
    """
    ✅ 특정 기능에서만 필요한 env를 그때 검증하기 위한 함수.
    - 서버 부팅 시점엔 터지지 않음
    - 실제로 Mongo/Vector 기능 호출할 때만 강제 가능
    """
    v = os.getenv(name)
    if v is None or not str(v).strip():
        raise RuntimeError(f"{name} is empty. Check ai/.env")
    return v


def mongo_ready() -> bool:
    """Mongo 기능 사용 가능 여부"""
    return bool(MONGODB_URI.strip())


def vector_ready() -> bool:
    """Vector Search 기능 사용 가능 여부"""
    return bool(MONGODB_URI.strip()) and bool(ATLAS_VECTOR_INDEX_PARTS.strip())


def kids_render_ready() -> bool:
    """Kids 렌더 기능 사용 가능 여부"""
    return bool(OPENAI_API_KEY.strip())

################################
# LEGO Units (LDraw LDU)
################################
STUD_PITCH_LDU = 20.0
PLATE_HEIGHT_LDU = 8.0
BRICK_HEIGHT_LDU = 24.0
RENDER_FRAME_PAD_LDU = 10.0

LDRAW_BASE_DIR = Path(r"C:\complete\ldraw")