# ============================================================================
# Strategy 노드: 난이도에 따른 LLM 모델 선택
# ============================================================================

from typing import Dict, Any


def node_strategy(graph, state) -> Dict[str, Any]:
    """전략 결정 노드: 난이도에 따른 LLM 모델 선택"""
    graph._log("STRATEGY", "현재 조건에서 가장 합리적인 설계 전략을 세우고 있어요.")
    hypothesis = state.get("current_hypothesis", {})
    difficulty = hypothesis.get("difficulty", "Medium")

    # [Rollback] GPT 사용 안 함 -> 무조건 Gemini 선택
    model_selection = "gemini-2.5-flash"

    if difficulty == "Hard":
        reason = "난이도 높음 (Hard) - Gemini 집중 모드 권장"
    elif difficulty == "Easy":
        reason = "난이도 낮음 (Easy)"
    else:
        reason = "일반 난이도"

    print(f"\n[Strategy] 전략 결정: {model_selection} ({reason})")

    return {
        "llm_config": {"model": model_selection},
        "strategy_plan": {"selected_model": model_selection, "reason": reason},
        "next_action": "model"
    }
