from langgraph.graph import StateGraph, END
from .state import HypothesisState
from .nodes import (
    node_search_cases,
    node_draft_creator,
    node_refiner
)

def build_hypothesis_graph():
    """
    가설 수립 서브그래프 (1회 실행, 티키타카 없음)
    [Search] -> [Draft (Gemini 1회)] -> [Finalize]
    """
    workflow = StateGraph(HypothesisState)
    
    # 노드 등록 (Critic 제거)
    workflow.add_node("search", node_search_cases)
    workflow.add_node("draft_refine", node_draft_creator)
    workflow.add_node("finalize", node_refiner)
    
    # 엣지 연결 (직선 흐름, 루프 없음)
    workflow.set_entry_point("search")
    workflow.add_edge("search", "draft_refine")
    workflow.add_edge("draft_refine", "finalize")
    workflow.add_edge("finalize", END)
    
    return workflow.compile()
