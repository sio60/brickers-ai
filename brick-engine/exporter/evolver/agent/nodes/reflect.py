"""REFLECT Node - Learn from results"""
from ..state import AgentState
from ..tools import get_model_state

def node_reflect(state: AgentState) -> AgentState:
    """Analyze results and update memory"""
    print(f"\n[REFLECT] Analyzing...")

    # Get new state
    new_state = get_model_state(state["model"], state["parts_db"])
    before = state["floating_count"]
    after = new_state["floating_count"]

    improvement = before - after
    last_action = state["action_history"][-1] if state["action_history"] else None
    action_type = last_action["type"] if last_action else "unknown"

    memory = state["memory"].copy()

    if improvement > 0:
        lesson = f"SUCCESS: {action_type} reduced floating by {improvement}"
        memory["successful_patterns"].append(action_type)
        memory["consecutive_failures"] = 0
        print(f"  {lesson}")
    elif improvement < 0:
        lesson = f"FAILED: {action_type} increased floating by {-improvement}"
        memory["failed_approaches"].append(action_type)
        memory["consecutive_failures"] += 1
        print(f"  {lesson}")

        if memory["consecutive_failures"] >= 3:
            print(f"  WARNING: 3 consecutive failures!")
    else:
        lesson = f"NEUTRAL: {action_type} had no effect"
        print(f"  {lesson}")

    memory["lessons"].append(lesson)

    return {
        **state,
        "floating_count": after,
        "memory": memory,
        "iteration": state["iteration"] + 1
    }
