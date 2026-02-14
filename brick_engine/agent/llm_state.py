from typing import Dict, Any, Optional, List, Literal, TypedDict, Annotated
from dataclasses import dataclass, field
try:
    from langchain_core.messages import BaseMessage
    from langgraph.graph.message import add_messages
except ImportError:
    # Fallback for environments without langgraph/langchain
    BaseMessage = Any
    def add_messages(x, y): return x

@dataclass
class VerificationFeedback:
    """PyBullet/brick_judge 검증 결과를 LLM에게 전달하기 위한 구조화된 피드백"""
    stable: bool = True
    total_bricks: int = 0
    fallen_bricks: int = 0
    floating_bricks: int = 0
    floating_brick_ids: List[str] = field(default_factory=list)
    fallen_brick_ids: List[str] = field(default_factory=list)
    failure_ratio: float = 0.0
    first_failure_brick: Optional[str] = None
    max_drift: float = 0.0
    collision_count: int = 0
    stability_grade: str = "STABLE"
    stability_score: int = 100
    small_brick_count: int = 0
    small_brick_ratio: float = 0.0

class AgentState(TypedDict):
    # 입력 및 설정
    glb_path: str
    ldr_path: str
    subject_name: str
    params: Dict[str, Any]
    max_retries: int
    acceptable_failure_ratio: float
    verification_duration: float
    gui: bool
    
    # 실행 상태
    attempts: int
    session_id: str
    merged: bool  # 1x1 브릭 병합 완료 여부
    messages: Annotated[List[BaseMessage], add_messages]
    
    # 검증 결과 캐시
    verification_raw_result: Any 
    floating_bricks_ids: List[str]
    verification_errors: int

    # 도구 사용 추적
    tool_usage_count: Dict[str, int]
    last_tool_used: Optional[str]
    consecutive_same_tool: int
    
    # 도구 효과 측정용 상태 저장
    previous_metrics: Dict[str, Any]
    current_metrics: Dict[str, Any]
    
    # 최종 결과 리포트
    final_report: Dict[str, Any]
    
    # Co-Scientist Memory
    memory: Dict[str, Any]

    # [v2] Co-Scientist 아키텍처 추가 필드
    current_hypothesis: Optional[Dict[str, Any]]
    strategy_plan: Optional[Dict[str, Any]]
    llm_config: Optional[Dict[str, str]]

    # Hypothesizer 전용 상태 필드
    success_cases: Optional[List[Dict]]
    failure_cases: Optional[List[Dict]]
    draft_hypothesis: Optional[Dict]
    algo_evaluation: Optional[str]
    critique_feedback: Optional[str]
    hypothesis_maker: Optional[Any]
    
    # [티키타카 추가 필드]
    round_count: int                              # 현재 토론 라운드
    internal_score: int                            # 가설 완성도 점수
    debate_history: List[str]                      # 토론 로그 기록

    # 다음 노드 제어
    next_action: Literal["generate", "verify", "model", "tool", "reflect", "hypothesize", "strategy", "merge", "end", "search", "draft", "critic", "refine"]
