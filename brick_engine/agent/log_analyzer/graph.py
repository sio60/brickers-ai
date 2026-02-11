"""
Log Analyzer — 그래프 구성
=========================
LangGraph StateGraph의 노드 연결, 조건부 엣지, 루프 구성 (분기 5개 확장 버전).

Graph Structure:
    fetch → [route_after_fetch]
              ├─ "has_logs" → parse_error → [route_by_category]
              │                  ├─ "code_bug" → investigate ↔ [should_continue] ─┐
              │                  ├─ "infra_issue" → invest_infra ↔ [should_continue] ─┤
              │                  └─ "no_error" → simple_summary ──→ END           │
              └─ "no_logs" → no_logs_report ──────────────────────→ END           │
                                                                                 ↓
                                                                         generate_report
                                                                                 ↓
                                                                         validate_report
                                                                                 ↓
                                                                      [route_after_validation]
                                                                         ├─ "retry" ─→ generate_report
                                                                         ├─ "critical" → alert_admin → END
                                                                         └─ "normal" ──→ END
"""

import logging
import json
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END

from .state import LogAnalysisState
from .config import MAX_INVESTIGATION_ROUNDS, MAX_REPORT_RETRIES
from .nodes import (
    fetch_logs_node,
    no_logs_report_node,
    parse_error_node,
    agent_investigate_node,
    investigate_infra_node,
    simple_summary_node,
    generate_report_node,
    validate_report_node,
    alert_admin_node,
)

logger = logging.getLogger("agent.log_analyzer.graph")


# ============================================================
# CONDITIONAL EDGES
# ============================================================

def route_after_fetch(state: LogAnalysisState) -> str:
    """로그 확보 여부에 따른 분기 (분기 1)"""
    logs = state.get("logs", "")
    if not logs or logs == "로그 수집 실패":
        logger.info("🔀 [Router: fetch] 로그 수집 실패 → no_logs_report 경로")
        return "no_logs"
    return "has_logs"


def route_by_category(state: LogAnalysisState) -> str:
    """에러 카테고리에 따른 조사 경로 분기 (분기 2)"""
    error_ctx = state.get("error_context", {})
    has_error = bool(error_ctx.get("error_type") or error_ctx.get("error_message"))
    category = state.get("error_category", "code_bug")

    if not has_error:
        logger.info("🔀 [Router: category] 에러 미감지 → simple_summary")
        return "no_error"
    
    if category == "infra_issue":
        logger.info("🔀 [Router: category] 인프라 장애 감지 → investigate_infra")
        return "infra_issue"
    
    logger.info("🔀 [Router: category] 코드 버그 감지 → investigate")
    return "code_bug"


def should_continue(state: LogAnalysisState) -> str:
    """ReAct 루프 제어 (분기 3)"""
    iteration = state.get("iteration", 0)
    messages = state.get("messages", [])

    # 마지막 AI 메시지의 도구 호출 확인
    last_ai_msg = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    has_tool_calls = bool(last_ai_msg and getattr(last_ai_msg, "tool_calls", None))

    if iteration >= MAX_INVESTIGATION_ROUNDS or not has_tool_calls:
        logger.info(f"🔀 [Router: loop] 조사 종료 ({iteration} rounds) → generate_report")
        return "done"

    logger.info(f"🔀 [Router: loop] 추가 조사 진행 (Round {iteration + 1})")
    return "continue"


def route_after_validation(state: LogAnalysisState) -> str:
    """리포트 검증 및 심각도 분기 (분기 4-5)"""
    retry_count = state.get("report_retry_count", 0)
    result_str = state.get("analysis_result", "{}")
    
    # 1. 재시도 여부 (분기 4)
    if retry_count > 0 and retry_count <= MAX_REPORT_RETRIES:
        logger.warning(f"🔀 [Router: validate] 검증 실패 → 재시도 ({retry_count}/{MAX_REPORT_RETRIES})")
        return "retry"

    # 2. 심각도 분기 (분기 5)
    try:
        data = json.loads(result_str)
        severity = data.get("error_identification", {}).get("severity", "normal")
        if severity == "critical":
            logger.info("🔀 [Router: severity] Critical 에러 → alert_admin")
            return "critical"
    except:
        pass

    logger.info("🔀 [Router: validate/severity] 분석 성공 → 종료")
    return "normal"


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================
workflow = StateGraph(LogAnalysisState)

# ── Nodes 등록 ──
workflow.add_node("fetch", fetch_logs_node)
workflow.add_node("no_logs_report", no_logs_report_node)
workflow.add_node("parse_error", parse_error_node)
workflow.add_node("investigate", agent_investigate_node)
workflow.add_node("investigate_infra", investigate_infra_node)
workflow.add_node("simple_summary", simple_summary_node)
workflow.add_node("generate_report", generate_report_node)
workflow.add_node("validate_report", validate_report_node)
workflow.add_node("alert_admin", alert_admin_node)

# ── Entry Point ──
workflow.set_entry_point("fetch")

# ── 분기 1: 로그 확보 여부 ──
workflow.add_conditional_edges(
    "fetch",
    route_after_fetch,
    {
        "has_logs": "parse_error",
        "no_logs": "no_logs_report"
    }
)

# ── 분기 2: 에러 카테고리별 조사 ──
workflow.add_conditional_edges(
    "parse_error",
    route_by_category,
    {
        "code_bug": "investigate",
        "infra_issue": "investigate_infra",
        "no_error": "simple_summary"
    }
)

# ── 분기 3: ReAct 루프 제어 (공통) ──
workflow.add_conditional_edges(
    "investigate",
    should_continue,
    {"continue": "investigate", "done": "generate_report"}
)
workflow.add_conditional_edges(
    "investigate_infra",
    should_continue,
    {"continue": "investigate_infra", "done": "generate_report"}
)

# ── 리포트 생성 및 검증 ──
workflow.add_edge("generate_report", "validate_report")

# ── 분기 4-5: 검증 재시도 및 심각도 알림 ──
workflow.add_conditional_edges(
    "validate_report",
    route_after_validation,
    {
        "retry": "generate_report",
        "critical": "alert_admin",
        "normal": END
    }
)

# ── 단방향 엣지 ──
workflow.add_edge("no_logs_report", END)
workflow.add_edge("simple_summary", END)
workflow.add_edge("alert_admin", END)

# ── Compile ──
app = workflow.compile()
