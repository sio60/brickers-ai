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
    
    # [Best of 3] 최적 결과 추적용
    modification_attempts: int          # 실제 도구 실행(LDR 수정) 횟수
    best_score: int                     # 최고 물리 점수
    best_ldr_content: Optional[str]     # 최고 점수 시점의 LDR 내용
    hallucination_count: int            # 존재하지 않는 도구 호출 횟수
    
    # [Hypothesis] 관찰 결과 (Verifier -> Reflect -> Hypothesize)
    observation: str

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

    # 다음 노드 제어 (graph.py의 라우터가 참조)
    next_action: Literal[
        "generate",     # GLB→LDR 변환 실행
        "verify",       # PyBullet/brick_judge 검증
        "verifier",     # 검증 노드 재진입 (에러 재시도)
        "model",        # LLM 호출 (CoScientist)
        "tool",         # 도구 실행 (TuneParameters/RemoveBricks/MergeBricks)
        "reflect",      # 검증 결과 분석
        "hypothesize",  # 가설 생성
        "strategy",     # 전략 수립
        "end",          # 파이프라인 종료
    ]

    # [시스템 컨텍스트]
    job_id: str
    initial_ldr_path: Optional[str] # [NEW] 초기 생성 LDR 백업 경로
