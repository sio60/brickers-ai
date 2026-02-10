from typing import TypedDict, Annotated, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage

class LogAnalysisState(TypedDict):
    """
    로그 분석 에이전트의 상태 정의
    """
    messages: Annotated[List[BaseMessage], "에이전트와 LLM 간의 대화 기록"]
    logs: str # 분석 대상 로그
    container_name: str # 대상 컨테이너 명
    analysis_result: Optional[str] # 최종 분석 결과 (JSON)
    error_count: int
    iteration: int # 루프 카운터
    job_id: Optional[str] # 특정 Job 추적용 추가
