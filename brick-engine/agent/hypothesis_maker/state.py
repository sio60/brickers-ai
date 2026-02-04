from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage

class HypothesisState(TypedDict):
    """
    Hypothesis Generation Subgraph State
    """
    # [Input]
    observation: str                     # 현재 상황 관찰 텍스트
    verification_result: Dict[str, Any]  # 물리 검증 결과 (메트릭 포함)
    similar_cases: List[Dict[str, Any]]  # RAG 검색 결과 (없을 수도 있음)
    
    # [Internal/Output]
    messages: List[BaseMessage]          # LLM 대화 기록
    final_hypothesis: Dict[str, Any]     # 최종 생성된 가설 (JSON)
