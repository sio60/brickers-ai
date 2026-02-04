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

    # LLM에게 로그 내용 생성 요청
    try:
        from openai import OpenAI
        import os
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        prompt = f"""
        실험 결과를 분석해줘.
        - 행동: {action_type}
        - 결과: 부유 브릭 수가 {before}에서 {after}로 변함 (변화량: {-improvement if improvement < 0 else "+" + str(improvement)})
        - 연속 실패 횟수: {memory.get("consecutive_failures", 0) + (1 if improvement <= 0 else 0)}
        
        간결한 "교훈(Lesson Learned)" 로그를 한글로 작성해줘 (최대 2문장).
        "성공:", "실패:", "중립:" 중 하나로 시작해.
        물리적/논리적 관점에서 왜 성공했거나 실패했는지 원인을 분석해서 포함해줘.
        특히, 지지대를 추가했는데도 부유 브릭이 줄지 않았다면 지지대 자체가 허공에 설치된 것은 아닌지 의심해봐.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        lesson = response.choices[0].message.content.strip()
    except Exception as e:
        lesson = f"Error generating log: {e}"
        # Fallback
        if improvement > 0:
            lesson = f"SUCCESS: {action_type} reduced floating by {improvement}"
        elif improvement < 0:
            lesson = f"FAILED: {action_type} increased floating by {-improvement}"
        else:
            lesson = f"NEUTRAL: {action_type} had no effect"

    # 메모리 업데이트
    if improvement > 0:
        memory["successful_patterns"].append(action_type)
        memory["consecutive_failures"] = 0
    else:
        # 실패/중립인 경우
        memory["failed_approaches"].append(action_type)
        memory["consecutive_failures"] += 1
        if memory["consecutive_failures"] >= 3:
             print(f"  WARNING: 3 consecutive failures!")

    print(f"  [Reflect LLM]: {lesson}")

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
                    metrics_before={
                        "floating": before,
                        "total_actions": len(state.get("action_history", [])),
                        "score": state.get("verification_score", 0.0)
                    },
                    metrics_after={
                        "floating": after,
                        "improvement": improvement,
                        "score": new_state.get("score", 0.0)
                    },
                    numerical_analysis=f"floating {before}→{after} ({improvement:+d})"
                ),
                improvement=build_improvement(
                    lesson_learned=f"{lesson}. Rationale: The strategy {action_type} was chosen because of {memory.get('consecutive_failures', 0)} consecutive failures. Result was {improvement:+d} improvement.",
                    next_hypothesis="Continue strategy" if improvement > 0 else "Switch strategy or parameters"
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
