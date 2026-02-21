# ============================================================================
# 임베딩: HuggingFace 로컬 + Gemini API Fallback
# ============================================================================

import logging
import threading
from typing import List

logger = logging.getLogger("CoScientistMemory")

# Thread-safe 모델 로딩
_embed_lock = threading.Lock()
_hf_model = None
_hf_tokenizer = None


def _load_hf_model():
    """HuggingFace 임베딩 모델 로드 (Lazy)"""
    global _hf_model, _hf_tokenizer

    if _hf_model is not None:
        return

    with _embed_lock:
        if _hf_model is not None:
            return

        try:
            from transformers import AutoTokenizer, AutoModel
            try:
                import config as _cfg
            except ImportError:
                _cfg = None
            
            # config에서 모델 이름을 안전하게 추출
            raw_model_name = getattr(_cfg, "HF_EMBED_MODEL", None) if _cfg else None
            
            # config.HF_EMBED_MODEL이 문자열이 아니거나 callable이면 기본값 사용
            if not isinstance(raw_model_name, str) or callable(raw_model_name) or not raw_model_name.strip():
                model_name = "intfloat/multilingual-e5-small"
            else:
                model_name = raw_model_name.strip()
                
            logger.info(f"Loading HF embedding model: {model_name}")
            _hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
            _hf_model = AutoModel.from_pretrained(model_name)
            logger.info("HF embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load HF model: {type(e).__name__}: {e}")
            # 로딩 실패 시 sentinel 값 설정하여 매번 재시도 방지
            _hf_model = False


def get_embedding(text: str, max_retries: int = 2) -> List[float]:
    """텍스트를 벡터로 변환 (HuggingFace 우선, Gemini Fallback)"""
    if not text:
        return []

    # 1차: HuggingFace 로컬 모델
    try:
        _load_hf_model()
        if _hf_model and _hf_tokenizer:
            import torch
            inputs = _hf_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = _hf_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
            return embedding
    except Exception as e:
        logger.warning(f"HF embedding failed: {e}")

    # Fallback: Gemini API (새 SDK: google-genai)
    for attempt in range(max_retries):
        try:
            try:
                import config
            except ImportError:
                break
            if getattr(config, "GEMINI_API_KEY", ""):
                from google import genai
                from google.genai import types
                client = genai.Client(
                    api_key=config.GEMINI_API_KEY,
                    http_options=types.HttpOptions(api_version='v1beta')
                )
                result = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=text,
                    config={"output_dimensionality": 384},  # MongoDB 인덱스 384차원에 맞춤
                )
                return result.embeddings[0].values
        except Exception as e:
            logger.warning(f"Gemini embedding attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(0.5 * (attempt + 1))

    return []
