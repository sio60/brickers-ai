# ============================================================================
# Helper Functions: 표준화된 데이터 빌더
# ============================================================================

from typing import Dict, Any


def calculate_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """metrics_before와 metrics_after의 차이 자동 계산"""
    delta = {}
    for key in before:
        if key in after:
            try:
                if isinstance(before[key], (int, float)) and isinstance(after[key], (int, float)):
                    change = after[key] - before[key]
                    delta[key] = round(change, 4) if isinstance(change, float) else change
            except:
                pass
    return delta


def ensure_not_empty(value: Any, default: Any) -> Any:
    """빈 값이면 기본값 반환"""
    if value is None or value == "" or value == {} or value == []:
        return default
    return value


def build_hypothesis(observation: str, hypothesis: str = None, reasoning: str = None, prediction: str = None) -> Dict[str, str]:
    """표준화된 hypothesis 객체 생성 (빈값 방지)"""
    return {
        "observation": ensure_not_empty(observation, "No observation"),
        "hypothesis": ensure_not_empty(hypothesis, "No explicit hypothesis"),
        "reasoning": ensure_not_empty(reasoning, "Automatic tool selection"),
        "prediction": ensure_not_empty(prediction, "Improvement expected")
    }


def build_experiment(tool: str, parameters: Dict = None, model_name: str = None, duration_sec: float = None) -> Dict[str, Any]:
    """표준화된 experiment 객체 생성 (빈값 방지)"""
    return {
        "tool": ensure_not_empty(tool, "unknown"),
        "parameters": ensure_not_empty(parameters, {}),
        "model_name": ensure_not_empty(model_name, "gemini-2.5-flash"),
        "duration_sec": ensure_not_empty(duration_sec, 0.0)
    }


def build_verification(passed: bool, metrics_before: Dict, metrics_after: Dict, numerical_analysis: str = None) -> Dict[str, Any]:
    """표준화된 verification 객체 생성 (delta 자동 계산)"""
    delta = calculate_delta(metrics_before, metrics_after)
    return {
        "passed": passed,
        "metrics_before": ensure_not_empty(metrics_before, {}),
        "metrics_after": ensure_not_empty(metrics_after, {}),
        "delta": delta,
        "numerical_analysis": ensure_not_empty(numerical_analysis, f"Delta: {delta}")
    }


def build_improvement(lesson_learned: str, next_hypothesis: str = None) -> Dict[str, str]:
    """표준화된 improvement 객체 생성 (빈값 방지)"""
    return {
        "lesson_learned": ensure_not_empty(lesson_learned, "No lesson recorded"),
        "next_hypothesis": ensure_not_empty(next_hypothesis, "Continue current strategy")
    }
