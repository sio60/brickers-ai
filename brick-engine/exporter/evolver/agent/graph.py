"""LangGraph Configuration
Enhanced with:
1. Checkpointer (Time Travel)
2. Send() API ready (parallel proposals)
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from .state import AgentState
from .constants import FIXED_ITERATIONS
from .nodes import (
    node_observe,
    node_supervisor,
    node_generate,
    node_debate,
    node_evolve,
    node_reflect,
    node_finish
)

def route_after_supervisor(state: AgentState) -> str:
    """Route after supervisor decision"""
    if state.get("should_finish"):
        return "finish"
    return "generate"

def route_after_reflect(state: AgentState) -> str:
    """Route after reflection - 고정 3회 반복"""
    iteration = state.get("iteration", 0)

    # 3회 다 돌았으면 종료
    if iteration >= FIXED_ITERATIONS:
        print(f"  [COMPLETE] {FIXED_ITERATIONS}회 반복 완료")
        return "finish"

    # 아직 3회 안 됐으면 계속
    print(f"  [CONTINUE] {iteration}/{FIXED_ITERATIONS}회 완료, 계속 진행")
    return "observe"

def build_graph() -> StateGraph:
    """Build the LangGraph state machine"""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("observe", node_observe)
    graph.add_node("supervisor", node_supervisor)
    graph.add_node("generate", node_generate)
    graph.add_node("debate", node_debate)
    graph.add_node("evolve", node_evolve)
    graph.add_node("reflect", node_reflect)
    graph.add_node("finish", node_finish)

    # Set entry point
    graph.set_entry_point("observe")

    # Add edges
    graph.add_edge("observe", "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"generate": "generate", "finish": "finish"}
    )

    graph.add_edge("generate", "debate")
    graph.add_edge("debate", "evolve")
    graph.add_edge("evolve", "reflect")

    graph.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {"observe": "observe", "finish": "finish"}
    )

    graph.add_edge("finish", END)

    # Compile with checkpointer (Time Travel 지원)
    # 버그 수정: 전역 싱글톤 대신 함수 내 생성
    checkpointer = InMemorySaver()
    return graph.compile(checkpointer=checkpointer)
