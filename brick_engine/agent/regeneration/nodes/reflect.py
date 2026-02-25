# ============================================================================
# Reflect 노드: 회고 + 학습 데이터 기록
# ============================================================================

import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger("agent.regeneration.reflect")


def node_reflect(graph, state) -> Dict[str, Any]:
    """
    회고 노드: 검증 결과를 분석하고 성공/실패를 Memory에 기록합니다.
    Co-Scientist의 핵심 학습 메커니즘입니다.
    """
    from ...core.memory_utils import (
        memory_manager, build_hypothesis, build_experiment,
        build_verification, build_improvement,
    )

    logger.info("[Reflect] 실제 결과 분석 중...")
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
    session_id = state.get('session_id', 'unknown_session') # session_id를 미리 정의

    # [FIX] 첫 번째 시도(Baseline)도 무조건 기록하여 전/후 비교 및 통계 지원
    if not previous_metrics:
        logger.info("[Reflect] 첫 번째 시도 기록 (Baseline)")
        if memory_manager:
            try:
                memory_manager.log_experiment(
                    session_id=session_id,
                    model_id=Path(state.get('glb_path') or state['ldr_path']).name,
                    agent_type="main_agent",
                    iteration=state.get("round_count", 0) + 1,
                    hypothesis=build_hypothesis(
                        observation="Initial baseline",
                        hypothesis="First attempt baseline for comparison",
                        reasoning="No previous metrics available",
                        prediction="Establishing baseline metrics"
                    ) if build_hypothesis else {"observation": "Initial baseline"},
                    experiment=build_experiment(
                        tool="baseline",
                        parameters={},
                        model_name="gemini-2.5-flash"
                    ) if build_experiment else {"tool": "baseline"},
                    verification=build_verification(
                        passed=True,
                        metrics_before={},
                        metrics_after=current_metrics,
                        numerical_analysis="Baseline measurement"
                    ) if build_verification else {"passed": True, "metrics_after": current_metrics},
                    improvement=build_improvement(
                        lesson_learned="Initial baseline for comparison",
                        next_hypothesis="Start optimization"
                    ) if build_improvement else {"lesson_learned": "Initial baseline"}
                )
            except Exception as e:
                logger.error(f"⚠️ [Memory] 통합 로그 저장 실패: {e}")
        # Baseline 기록 후 다음 단계로
        # 첫 라운드만 가설 수립, 이후는 바로 도구 선택으로
        round_count = state.get("round_count", 0) + 1
        if round_count >= 1:
            next_step = "model"
            logger.info(f"[Reflect] 라운드 {round_count}: 가설 이미 수립됨 → 바로 도구 선택으로")
        else:
            next_step = state.get("next_action_override") or "hypothesize"
        return {
            "memory": memory,
            "previous_metrics": current_metrics,
            "next_action": next_step,
            "round_count": round_count,
            "verification_raw_result": state.get("verification_raw_result"), # [FIX] Preserve data flow
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
        logger.info(lesson)
    else:
        lesson = f"❌ {last_tool} 실패: {hyp_text} (No Improvement)"
        memory["failed_approaches"].append(f"{last_tool}: 효과 미미")
        memory["consecutive_failures"] += 1
        logger.info(lesson)

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
            logger.warning("⚠️ [Memory] 통합 로그 저장 실패: %s", e)

    logger.info("🎓 학습 데이터 기록 완료")

    # force_end 플래그가 있으면 (max_retries 초과) 저장만 하고 종료
    if state.get("force_end", False):
        logger.info("[Reflect] 마지막 결과 기록 완료. 종료합니다.")
        return {
            "memory": memory,
            "observation": f"실패율={curr_failure:.2f}, 공중부양={curr_floating}개, 작은브릭={curr_small_ratio:.2f}",
            "previous_metrics": current_metrics,
            "next_action": "end"
        }

    # 1라운드 이상이면 가설 수립 건너뛰고 바로 도구 선택으로
    round_count = state.get('round_count', 0)
    if round_count >= 1:
        logger.info(f"[Reflect] 라운드 {round_count}: 가설 이미 수립됨 → 바로 도구 선택으로")
        next_step = "model"
    else:
        logger.info("[Deep Debate] 비평가와 설계자의 심층 토론 단계로 진입합니다.")
        next_step = "hypothesize"

    return {
        "memory": memory,
        "observation": f"실패율={curr_failure:.2f}, 공중부양={curr_floating}개, 작은브릭={curr_small_ratio:.2f}",
        "previous_metrics": current_metrics,
        "round_count": round_count + 1,
        "verification_raw_result": state.get("verification_raw_result"), # [FIX] Preserve data flow
        "next_action": next_step
    }
