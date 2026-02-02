"""OBSERVE Node - Analyze current model state using 승준's PhysicalVerifier"""
from ..state import AgentState
from ..tools import get_model_state

MAX_REMOVAL_RATIO = 0.10
MAX_ITERATIONS = 10

def node_observe(state: AgentState) -> AgentState:
    """Observe current model state with 승준's verification"""
    print(f"\n[OBSERVE] Iteration {state['iteration']}")

    model_state = get_model_state(state["model"], state["parts_db"])

    print(f"  Bricks: {model_state['total_bricks']}")
    print(f"  Score: {model_state.get('score', 'N/A')}")
    print(f"  Valid: {model_state.get('is_valid', 'N/A')}")
    print(f"  Floating: {model_state['floating_count']}")
    print(f"  Collisions: {model_state['collision_count']}")
    print(f"  Removed: {state['total_removed']}/{int(state['original_brick_count'] * MAX_REMOVAL_RATIO)}")

    # Show evidence from 승준's verifier
    evidence = model_state.get('evidence', [])
    if evidence:
        print(f"  Evidence ({len(evidence)} issues):")
        for ev in evidence[:5]:  # Show first 5
            print(f"    - [{ev.severity}] {ev.type}: {ev.message[:80]}...")

    return {
        **state,
        "floating_count": model_state["floating_count"],
        "collision_count": model_state["collision_count"],
        "floating_bricks": model_state["floating_bricks"],
        "verification_result": model_state.get("verification_result"),
        "verification_score": model_state.get("score", 100.0),
        "verification_evidence": evidence
    }
