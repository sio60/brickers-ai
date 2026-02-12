"""
Admin AI Analyst — Graph 빌드
StateGraph를 조립하고 컴파일하여 실행 가능한 에이전트를 생성.
"""
from __future__ import annotations

import logging
from langgraph.graph import StateGraph, END

from .state import AdminAnalystState
from .nodes import (
    miner_node,
    evaluator_node,
    diagnoser_node,
    strategist_node,
    deep_investigator_node,
    reporter_green_node,
    finalizer_node,
)

log = logging.getLogger("admin_analyst.graph")


# ─── Conditional Edge 라우터 ───
def route_after_eval(state: AdminAnalystState) -> str:
    """Evaluator 후: 이상 있으면 diagnose, 없으면 report_green"""
    return state.get("next_action", "report_green")


def route_after_strategy(state: AdminAnalystState) -> str:
    """Strategist 후: 확신도 낮으면 deep_investigate, 높으면 finalize"""
    return state.get("next_action", "finalize")


# ─── Graph 빌드 ───
def build_analyst_graph():
    """LangGraph StateGraph를 조립하고 컴파일."""
    builder = StateGraph(AdminAnalystState)

    # 노드 등록
    builder.add_node("mine", miner_node)
    builder.add_node("evaluate", evaluator_node)
    builder.add_node("diagnose", diagnoser_node)
    builder.add_node("strategize", strategist_node)
    builder.add_node("deep_investigate", deep_investigator_node)
    builder.add_node("report_green", reporter_green_node)
    builder.add_node("finalize", finalizer_node)

    # 엣지
    builder.set_entry_point("mine")
    builder.add_edge("mine", "evaluate")

    # Conditional 1: 이상 여부 분기
    builder.add_conditional_edges("evaluate", route_after_eval, {
        "diagnose": "diagnose",
        "report_green": "report_green",
    })

    builder.add_edge("diagnose", "strategize")

    # Conditional 2: 확신도 분기 (루프백 가능)
    builder.add_conditional_edges("strategize", route_after_strategy, {
        "deep_investigate": "deep_investigate",
        "finalize": "finalize",
    })

    # 루프백: 심층 조사 → 다시 진단
    builder.add_edge("deep_investigate", "diagnose")

    # 종료
    builder.add_edge("report_green", END)
    builder.add_edge("finalize", END)

    log.info("[Graph] Admin Analyst Graph 빌드 완료")
    return builder.compile()


# 전역 그래프 인스턴스
analyst_graph = build_analyst_graph()
