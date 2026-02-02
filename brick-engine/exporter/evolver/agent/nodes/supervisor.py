"""SUPERVISOR Node - Decide strategy direction"""
import os
import json
from openai import OpenAI
from ..state import AgentState
from ..prompts import SUPERVISOR_SYSTEM

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MAX_REMOVAL_RATIO = 0.10
MAX_ITERATIONS = 10

def node_supervisor(state: AgentState) -> AgentState:
    """Decide repair strategy"""
    print(f"\n[SUPERVISOR] Deciding strategy...")

    # Check finish conditions
    if state["floating_count"] == 0:
        return {**state, "should_finish": True, "finish_reason": "All fixed!"}

    limit = int(state["original_brick_count"] * MAX_REMOVAL_RATIO)
    if state["total_removed"] >= limit:
        return {**state, "should_finish": True, "finish_reason": f"Removal limit ({limit})"}

    if state["iteration"] >= MAX_ITERATIONS:
        return {**state, "should_finish": True, "finish_reason": "Max iterations"}

    # Build context
    glb_info = ""
    if state.get("glb_reference") and state["glb_reference"].get("available"):
        ref = state["glb_reference"]
        glb_info = f"""
[GLB Reference]
Model: {ref.get('name')} ({ref.get('model_type')})
Legs: {ref.get('legs')}
Features: {ref.get('key_features')}
Notes: {ref.get('structure_notes')}
"""

    context = f"""
{glb_info}
Floating: {state['floating_count']}
Removed: {state['total_removed']}/{limit}
Iteration: {state['iteration']}/{MAX_ITERATIONS}
Failed: {state['memory']['failed_approaches'][-3:]}
Success: {state['memory']['successful_patterns'][-3:]}
Consecutive failures: {state['memory']['consecutive_failures']}

Sample floating bricks:
{json.dumps(state['floating_bricks'][:5], indent=2)}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SUPERVISOR_SYSTEM},
            {"role": "user", "content": context}
        ],
        temperature=0.3
    )

    result = response.choices[0].message.content.strip()

    # 전략 파싱 (여러 형태 대응)
    valid = ["AUTO_EVOLVE", "ADD_SUPPORT", "SELECTIVE_REMOVE", "BRIDGE", "ROLLBACK"]
    strategy = None

    # 응답에서 유효 전략 찾기
    result_upper = result.upper()
    for v in valid:
        if v in result_upper:
            strategy = v
            break

    # 못 찾으면 기본값
    if not strategy:
        # 연속 실패 3회 이상이면 SELECTIVE_REMOVE
        if state["memory"]["consecutive_failures"] >= 3:
            strategy = "SELECTIVE_REMOVE"
        else:
            strategy = "AUTO_EVOLVE"

    print(f"  Strategy: {result}")

    # Select target brick for single-brick strategies
    target = None
    if strategy in ["ADD_SUPPORT", "SELECTIVE_REMOVE", "BRIDGE"] and state["floating_bricks"]:
        target = state["floating_bricks"][0]

    return {**state, "strategy": strategy, "target_brick": target}
