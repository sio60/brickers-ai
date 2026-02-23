# brickers-ai/service/cost_calculator.py
"""운영 비용(LLM 토큰 및 외부 API) 계산 및 추적 모듈"""
from typing import Dict, Any, Optional
from .kids_config import MODEL_PRICING, TRIPO_GEN_COST

class CostTracker:
    def __init__(self, initial_cost: float = 0.0, initial_tokens: int = 0):
        self.total_cost = initial_cost
        self.total_tokens = initial_tokens

    def add_llm_cost(self, model_name: str, usage: Dict[str, int]):
        """LLM 토큰 사용량을 기반으로 비용을 추가합니다."""
        if not usage:
            return
            
        m_lower = model_name.lower()
        pricing = None
        
        # 1. OpenAI Matching
        if "gpt-4o-mini" in m_lower:
            pricing = MODEL_PRICING["gpt-4o-mini"]
        elif "gpt-4o" in m_lower:
            pricing = MODEL_PRICING["gpt-4o"]
        # 2. Anthropic Matching
        elif "claude-3-5-sonnet" in m_lower or "claude-3.5-sonnet" in m_lower:
            pricing = MODEL_PRICING["claude-3-5-sonnet-20241022"]
        # 3. Gemini Matching
        elif "flash" in m_lower:
            if "2.0" in m_lower or "2.5" in m_lower:
                pricing = MODEL_PRICING["gemini-2.0-flash"]
            else:
                pricing = MODEL_PRICING["gemini-1.5-flash"]
        elif "pro" in m_lower:
            pricing = MODEL_PRICING["gemini-1.5-pro"]
            
        # Fallback
        if not pricing:
            pricing = MODEL_PRICING["gemini-1.5-flash"]
            
        in_tokens = usage.get("input_tokens", 0)
        out_tokens = usage.get("output_tokens", 0)
        
        cost = (in_tokens * (pricing["input"] / 1_000_000)) + (out_tokens * (pricing["output"] / 1_000_000))
        
        self.total_cost += cost
        self.total_tokens += (in_tokens + out_tokens)

    def add_fixed_cost(self, cost: float):
        """고정 비용(예: Tripo $0.30)을 추가합니다."""
        self.total_cost += cost

    def add_tripo_cost(self):
        """Tripo 생성 고정 비용을 추가합니다."""
        self.total_cost += TRIPO_GEN_COST

    def get_result(self) -> Dict[str, Any]:
        """최종 합산된 비용 및 토큰 정보를 반환합니다."""
        return {
            "est_cost": round(self.total_cost, 6),
            "token_count": self.total_tokens
        }
