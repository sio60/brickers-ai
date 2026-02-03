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

    # Unified Logging (표준화된 헬퍼 함수 사용)
    try:
        import config  # This registers AGENT_DIR in sys.path
        from memory_utils import memory_manager, build_hypothesis, build_experiment, build_verification, build_improvement
        
        if memory_manager:
            # 모델 이름 추출 (가능하면)
            model_name = state.get("model", {}).get("name", "unknown_glb") if isinstance(state.get("model"), dict) else "unknown_glb"
            
            # action_history에서 파라미터 추출
            action_params = last_action.get("params", {}) if last_action else {}
            
            memory_manager.log_experiment(
                session_id=state.get('session_id', 'evolver_session'),
                model_id=model_name,
                agent_type="evolver",
                iteration=state["iteration"],
                hypothesis=build_hypothesis(
                    observation=f"floating={before}, action_history={len(state.get('action_history', []))}",
                    hypothesis=f"{action_type} 적용 시 floating 감소 예상",
                    reasoning=f"Based on memory: consecutive_failures={memory.get('consecutive_failures', 0)}",
                    prediction=f"floating: {before}→{after} 예상"
                ),
                experiment=build_experiment(
                    tool=action_type,
                    parameters=action_params,
                    model_name="gpt-4o"
                ),
                verification=build_verification(
                    passed=improvement > 0,
                    metrics_before={"floating": before, "total_actions": len(state.get("action_history", []))},
                    metrics_after={"floating": after, "improvement": improvement},
                    numerical_analysis=f"floating {before}→{after} ({improvement:+d})"
                ),
                improvement=build_improvement(
                    lesson_learned=lesson,
                    next_hypothesis="Continue" if improvement > 0 else "Try different approach"
                )
            )
    except Exception as e:
        print(f"⚠️ [Reflect] Log failed: {e}")

    return {
        **state,
        "floating_count": after,
        "memory": memory,
        "iteration": state["iteration"] + 1
    }
