from langgraph.graph import StateGraph, END
from .state import HypothesisState
from .nodes import HypothesisNodes

def create_hypothesis_graph():
    """
    가설 생성 서브그래프 생성
    [Start] -> [Generate] -> [End]
    """
    nodes = HypothesisNodes()
    workflow = StateGraph(HypothesisState)
    
    # 노드 등록
    workflow.add_node("generate", nodes.generate_hypothesis)
    
    # 엣지 연결
    workflow.set_entry_point("generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()
