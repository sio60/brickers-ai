# ============================================================================
# Reflect 노드: 회고 + 학습 데이터 기록
# ============================================================================

from pathlib import Path
from typing import Dict, Any


def node_reflect(graph, state) -> Dict[str, Any]:
    """
    회고 노드: 검증 결과를 분석하고 성공/실패를 Memory에 기록합니다.
    Co-Scientist의 핵심 학습 메커니즘입니다.
    """
    from ...memory_utils import (
        memory_manager, build_hypothesis, build_experiment,
        build_verification, build_improvement,
    )

    print("\n[Reflect] 실제 결과 분석 중...")
    graph._log("REFLECT", "이전 시도와 비교해서 개선된 점을 정리하고 있어요.")

    # Memory 초기화
    memory = state.get('memory', {
        "failed_approaches": [],
        "successful_patterns": [],
        "lessons": [],
        "consecutive_failures": 0
    })

    previous_metrics = state.get('previous_metrics', {})
    current_metrics = state.get('current_metrics', {})
    last_tool = state.get('last_tool_used', 'unknown')

    # 이전 메트릭이 없으면 첫 실행 (비교 대상 없음)
    if not previous_metrics:
        print("  (첫 검증 - 기준점 설정)")
        return {
            "memory": memory,
            "previous_metrics": current_metrics,
            "next_action": "hypothesize"
        }

    # 메트릭 비교
    prev_failure = previous_metrics.get('failure_ratio', 0)
    curr_failure = current_metrics.get('failure_ratio', 0)
    prev_floating = previous_metrics.get('floating_count', 0)
    curr_floating = current_metrics.get('floating_count', 0)
    prev_small_ratio = previous_metrics.get('small_brick_ratio', 0)
    curr_small_ratio = current_metrics.get('small_brick_ratio', 0)

    failure_improved = curr_failure < prev_failure
    floating_improved = curr_floating < prev_floating
    overall_improved = failure_improved or floating_improved

    # 결과 분석 및 학습 데이터 저장
    current_hypothesis = state.get('current_hypothesis', {})
    hyp_text = current_hypothesis.get('hypothesis', 'No hypothesis')

    if overall_improved:
        lesson = f"✅ {last_tool} 성공: {hyp_text} (Gained Improvement)"
        memory["successful_patterns"].append(f"{last_tool}: 효과 있음")
        memory["consecutive_failures"] = 0
        print(f"  {lesson}")
    else:
        lesson = f"❌ {last_tool} 실패: {hyp_text} (No Improvement)"
        memory["failed_approaches"].append(f"{last_tool}: 효과 미미")
        memory["consecutive_failures"] += 1
        print(f"  {lesson}")

    memory["lessons"].append(lesson)

    # 리스트 크기 관리
    memory["lessons"] = memory["lessons"][-10:]
    memory["failed_approaches"] = memory["failed_approaches"][-5:]
    memory["successful_patterns"] = memory["successful_patterns"][-5:]

    # 통합 로그 저장
    if memory_manager:
        try:
            observation = f"ratio={prev_small_ratio:.2f}, floating={prev_floating}, failure={prev_failure:.2f}"

            memory_manager.log_experiment(
                session_id=state.get('session_id', 'unknown_session'),
                model_id=Path(state['glb_path'] or state['ldr_path']).name,
                agent_type="main_agent",
                iteration=state['attempts'],
                hypothesis=build_hypothesis(
                    observation=observation,
                    hypothesis=current_hypothesis.get('hypothesis', hyp_text),
                    reasoning=current_hypothesis.get('reasoning', f"Based on memory lessons: {memory.get('lessons', [])[-1] if memory.get('lessons') else 'None'}"),
                    prediction=current_hypothesis.get('prediction', f"floating: {prev_floating}→{curr_floating}, ratio: {prev_small_ratio:.2f}→?")
                ) if build_hypothesis else {"observation": observation, "reasoning": current_hypothesis.get('reasoning')},
                experiment=build_experiment(
                    tool=last_tool,
                    parameters=state.get('params', {}),
                    model_name="gemini-2.5-flash"
                ) if build_experiment else {"tool": last_tool},
                verification=build_verification(
                    passed=overall_improved,
                    metrics_before=previous_metrics,
                    metrics_after=current_metrics,
                    numerical_analysis=f"floating {prev_floating}→{curr_floating}, ratio {prev_small_ratio:.2f}→{curr_small_ratio:.2f}, failure {prev_failure:.2f}→{curr_failure:.2f}"
                ) if build_verification else {"passed": overall_improved},
                improvement=build_improvement(
                    lesson_learned=lesson,
                    next_hypothesis="Maintain strategy" if overall_improved else "Change strategy"
                ) if build_improvement else {"lesson_learned": lesson}
            )
        except Exception as e:
            print(f"⚠️ [Memory] 통합 로그 저장 실패: {e}")

    print("\n" + "🎓" * 20)
    print(" [Deep Debate] 비평가와 설계자의 심층 토론 단계로 진입합니다.")
    print("🎓" * 20)

    return {
        "memory": memory,
        "observation": f"실패율={curr_failure:.2f}, 공중부양={curr_floating}개, 작은브릭={curr_small_ratio:.2f}",
        "previous_metrics": current_metrics,
        "next_action": "hypothesize"
    }
