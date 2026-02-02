"""FINISH Node - Complete and save"""
from ..state import AgentState

def node_finish(state: AgentState) -> AgentState:
    """Finish the agent run"""
    print(f"\n[FINISH] {state['finish_reason']}")
    return state
