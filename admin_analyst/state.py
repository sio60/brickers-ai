"""
Admin AI Analyst — State 정의
랭그래프 에이전트가 노드 간 전달하는 전역 상태(메모리) 구조.
"""
from typing import TypedDict, List, Dict, Any, Optional


class AdminAnalystState(TypedDict):
    # ─── 📥 입력 (Miner) ───
    raw_metrics: Dict[str, Any]       # GA4 + 백엔드 지표 원본
    temporal_context: Dict[str, Any]  # 시간/요일/시즌 맥락

    # ─── 🔍 분석 (Evaluator) ───
    anomalies: List[Dict[str, Any]]   # 감지된 이상 징후 리스트
    risk_score: float                 # 종합 위험 점수 0.0~1.0

    # ─── 🧠 추론 (Diagnoser) ───
    diagnosis: Optional[Dict[str, Any]]
    # {root_cause, confidence, evidence[], affected_segment}

    # ─── 🎯 전략 (Strategist) ───
    proposed_actions: List[Dict[str, Any]]
    # [{action, target, expected_impact, risk, priority}]

    # ─── 🔄 제어 흐름 ───
    iteration: int          # 현재 루프 횟수
    max_iterations: int     # 최대 허용 반복 (기본 3)
    next_action: str        # 다음 노드 결정 키
    final_report: Optional[str]  # 최종 보고서 (마크다운)
