from typing import TypedDict, Annotated, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage
import operator


class LogAnalysisState(TypedDict):
    """
    로그 분석 에이전트의 상태 정의 (LangGraph State)
    
    - messages: LLM과의 대화 이력 (Tool Calling에 필수, 누적)
    - investigation_notes: 각 조사 라운드의 결과 기록 (누적)
    
    사용 위치:
        - nodes.py: 각 노드에서 읽기/쓰기
        - graph.py: 조건부 엣지에서 읽기
        - admin.py: 초기 상태 생성
    """
    # ── LLM 대화 이력 (add 방식으로 누적) ──
    messages: Annotated[List[BaseMessage], operator.add]
    
    # ── 기본 정보 ──
    logs: str                           # 분석 대상 로그 원문
    container_name: str                 # 대상 컨테이너 명 (Docker SDK 폴백용)
    job_id: Optional[str]               # 특정 Job 추적용 (없으면 자동 탐색)
    
    # ── 루프 제어 ──
    iteration: int                      # 조사 라운드 카운터 (fetch에서 0 초기화)
    
    # ── 에러 파싱 결과 ──
    error_context: Optional[Dict[str, Any]]  # parse_error에서 구조화 (type, message, call_stack 등)
    error_category: Optional[str]            # parse_error에서 분류 (code_bug / infra_issue)
    
    # ── 조사 결과 (누적) ──
    investigation_notes: Annotated[List[str], operator.add]  # 각 라운드별 조사 기록
    
    # ── 리포트 검증 ──
    report_retry_count: int                  # validate_report에서 재시도 카운트 (max 2)
    
    # ── 최종 출력 ──
    analysis_result: Optional[str]           # 최종 분석 리포트 (JSON)

