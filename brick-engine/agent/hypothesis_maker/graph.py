from langgraph.graph import StateGraph, END
from .state import HypothesisState
from .nodes import (
    node_search_cases,
    node_draft_creator,
    node_critic,
    node_refiner
)

def build_hypothesis_graph():
    """
    가설 수립 서브그래프 생성 (최소 2회 티키타카)
    [Search] -> [Draft/Refine] -> [Critic] -> (Loop?) -> [Final Refine]
    """
    workflow = StateGraph(HypothesisState)
    
    # 노드 등록
    workflow.add_node("search", node_search_cases)
    workflow.add_node("draft_refine", node_draft_creator) # Draft와 Refine 통합 노드
    workflow.add_node("critic", node_critic)
    workflow.add_node("finalize", node_refiner)
    
    # 루프 조건 함수
    def should_continue(state: HypothesisState):
        round_count = state.get("round_count", 0)
        score = state.get("internal_score", 0)
        
        # 1. 최소 2라운드 보장
        if round_count < 2:
            return "continue"
        
        # 2. 점수 기준 (95점) 및 최대 5회 제한
        if score < 95 and round_count < 5:
            return "continue"
            
        return "end"

    # 엣지 연결
    workflow.set_entry_point("search")
    workflow.add_edge("search", "draft_refine")
    workflow.add_edge("draft_refine", "critic")
    
    # Critic 이후 루프 여부 결정
    workflow.add_conditional_edges(
        "critic",
        should_continue,
        {
            "continue": "draft_refine",
            "end": "finalize"
        }
    )
    
    workflow.add_edge("finalize", END)
    
    return workflow.compile()
