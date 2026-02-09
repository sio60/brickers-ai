from typing import TypedDict, List, Dict, Any, Optional, Annotated
try:
    from langchain_core.messages import BaseMessage
    from langgraph.graph.message import add_messages
except ImportError:
    BaseMessage = Any
    def add_messages(x, y): return x

class HypothesisState(TypedDict):
    """
    가설 수립 서브그래프 상태 정의
    최소 2회의 티키타카 토론을 지원합니다.
    """
    # [입력 및 공통]
    subject_name: str              # 사물 태그 (RAG 검색용)
    observation: str
    verification_raw_result: Dict[str, Any]
    hypothesis_maker: Any
    messages: Annotated[List[BaseMessage], add_messages]
    
    # [티키타카 진행 상태]
    round_count: int               # 현재 라운드 (최소 2회 보장)
    internal_score: int             # 가설 완성도 점수 (0~100)
    debate_history: List[str]       # 토론 로그 기록
    
    # [RAG 사례 풀]
    success_cases: Optional[List[Dict]]  # Gemini용 (Round당 1개 소모)
    failure_cases: Optional[List[Dict]]  # GPT용 (Round당 3개 소모)
    
    # [중간 결과 및 비평]
    draft_hypothesis: Optional[Dict]
    algo_evaluation: Optional[str]
    critique_feedback: Optional[str]
    
    # [최종 출력]
    current_hypothesis: Optional[Dict[str, Any]]
    next_action: str
