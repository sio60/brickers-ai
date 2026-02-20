"""SUPERVISOR Node - 전략 결정 (단일 전략)

핵심 원칙:
1. 문제 유형에 따라 동적으로 전략 풀 구성
2. 모델 타입에 따라 대칭 분석 자동 스킵 (animal, plant 등)
3. ADD_SUPPORT는 연속 실패 시에만 등장
4. 한 번에 하나의 전략만 실행

Enhanced: Structured Output (Pydantic) + Dynamic Strategy Pooling
"""
import json
import sys
from typing import List, Set, Dict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from ..state import AgentState
from ..prompts import SUPERVISOR_SYSTEM
from ..constants import (
    MAX_REMOVAL_RATIO, FIXED_ITERATIONS, SKIP_SYMMETRY_TYPES, LLM_MODEL, LLM_TIMEOUT
)

llm = ChatOpenAI(model=LLM_MODEL, temperature=0.3, timeout=LLM_TIMEOUT)


# ===== Structured Output Models =====
class StrategyDecision(BaseModel):
    """전략 결정 구조화 출력 - 단일 전략"""
    strategy: str = Field(
        description="적용할 전략 (하나만 선택). 예: 'SYMMETRY_FIX'"
    )
    confidence: int = Field(
        ge=0, le=100,
        description="전략 선택 확신도 (0-100)"
    )
    reasoning: str = Field(
        description="전략 선택 이유. 반드시 한국어로 작성."
    )


# Structured output 적용된 LLM
strategy_llm = llm.with_structured_output(StrategyDecision)


# ===== 전략 매핑 =====
STRATEGY_MAP = {
    "vision": ["RELOCATE", "ROTATE"],  # SYMMETRY_FIX 제거 - 후순위로 이동
    "symmetry": ["SYMMETRY_FIX"],  # 대칭 문제는 별도로, floating 해결 후 처리
    "floating": ["SELECTIVE_REMOVE", "BRIDGE"],
    "fallback": ["ADD_SUPPORT", "REBUILD", "SYMMETRY_FIX", "ROLLBACK"],  # SYMMETRY_FIX를 fallback으로 이동
}


def _get_valid_strategies(state: AgentState) -> List[str]:
    """문제 유형에 따라 동적으로 전략 풀 구성"""
    strategies: Set[str] = set()
    model_type = state.get("model_type", "unknown")

    # Vision 문제 → 형태 전략
    if state.get("vision_problems"):
        strategies.update(STRATEGY_MAP["vision"])
        strategies.add("REBUILD")  # 풀에는 포함, 프롬프트에서 후순위 처리

    # 대칭 문제 → 모델 타입에 따라 동적 판단
    if state.get("symmetry_issues"):
        if model_type not in SKIP_SYMMETRY_TYPES:
            strategies.update(STRATEGY_MAP["symmetry"])

    # Floating → 물리 전략
    if state["floating_count"] > 0:
        strategies.update(STRATEGY_MAP["floating"])
        # ADD_SUPPORT는 연속 실패 시에만
        if state["memory"]["consecutive_failures"] >= 2:
            strategies.add("ADD_SUPPORT")

    # Collision(충돌) → 겹침 해소를 위한 전략 추가
    if state.get("collision_count", 0) > 0:
        strategies.update(["SELECTIVE_REMOVE", "RELOCATE"])

    # 아무 전략도 없으면 종료 신호
    if not strategies:
        return []

    return list(strategies)


def _build_context(state: AgentState, valid_strategies: List[str]) -> str:
    """문제 상황에 따라 동적으로 컨텍스트 생성"""
    parts = []

    # GLB 참조 정보
    if state.get("glb_reference") and state["glb_reference"].get("available"):
        ref = state["glb_reference"]
        parts.append(f"""[모델 참조]
이름: {ref.get('name')} (타입: {ref.get('model_type')})
다리 수: {ref.get('legs')}
특징: {ref.get('key_features')}""")

    # 모델 타입
    model_type = state.get("model_type", "unknown")
    parts.append(f"[모델 타입] {model_type}")

    # 현재 상태 요약
    vision_problems = state.get("vision_problems", [])
    symmetry_issues = state.get("symmetry_issues", [])
    floating_count = state["floating_count"]

    status_lines = []
    if vision_problems:
        status_lines.append(f"Vision 문제: {len(vision_problems)}개")
    if symmetry_issues and model_type not in SKIP_SYMMETRY_TYPES:
        status_lines.append(f"대칭 문제: {len(symmetry_issues)}개")
    if floating_count > 0:
        status_lines.append(f"부유 브릭: {floating_count}개")

    parts.append(f"[현재 상태]\n" + "\n".join(status_lines) if status_lines else "[현재 상태] 문제 없음")

    # 문제 상세
    if vision_problems:
        parts.append(f"[Vision 문제 상세]\n{json.dumps(vision_problems[:3], indent=2, ensure_ascii=False)}")

    if symmetry_issues and model_type not in SKIP_SYMMETRY_TYPES:
        # PlacedBrick 객체를 직렬화 가능한 dict로 변환
        sym_sample = []
        for issue in symmetry_issues[:3]:
            if isinstance(issue, dict):
                # mirror_brick 필드에 PlacedBrick 있을 수 있음
                clean_issue = {}
                for k, v in issue.items():
                    if hasattr(v, 'position'):
                        # PlacedBrick → dict
                        clean_issue[k] = {
                            "id": getattr(v, 'id', 'unknown'),
                            "x": v.position.x, "y": v.position.y, "z": v.position.z
                        }
                    elif isinstance(v, tuple):
                        clean_issue[k] = list(v)
                    else:
                        clean_issue[k] = v
                sym_sample.append(clean_issue)
            else:
                sym_sample.append(str(issue))
        parts.append(f"[대칭 문제 상세]\n{json.dumps(sym_sample, indent=2, ensure_ascii=False)}")

    if floating_count > 0:
        parts.append(f"[부유 브릭 샘플]\n{json.dumps(state['floating_bricks'][:3], indent=2)}")

    # 제약 조건
    limit = int(state["original_brick_count"] * MAX_REMOVAL_RATIO)
    parts.append(f"""[제약]
삭제됨: {state['total_removed']}/{limit}
반복: {state['iteration']}/{FIXED_ITERATIONS}
연속 실패: {state['memory']['consecutive_failures']}""")

    # 사용 가능한 전략
    strategy_desc = {
        "RELOCATE": "브릭을 올바른 위치로 이동 (형태 개선)",
        "ROTATE": "브릭을 올바른 방향으로 회전 (형태 개선)",
        "REBUILD": "잘못된 부분 삭제 후 재구축",
        "SYMMETRY_FIX": "빠진 대칭 브릭 추가",
        "SELECTIVE_REMOVE": "불필요한 부유 브릭 삭제",
        "BRIDGE": "부유 클러스터를 안정 부분에 연결",
        "ROLLBACK": "원본 복원",
    }

    strategy_lines = [f"- {s}: {strategy_desc.get(s, s)}" for s in valid_strategies]
    parts.append(f"[사용 가능한 전략]\n" + "\n".join(strategy_lines))

    # 이력 (참고용 - 전략 회피가 아닌 실행 개선 목적)
    if state['memory'].get('lessons'):
        parts.append(f"[이전 교훈 (같은 전략을 더 잘 실행하기 위한 참고)]\n" + "\n".join(state['memory']['lessons'][-5:]))
    if state['memory']['successful_patterns']:
        parts.append(f"[성공 패턴] {state['memory']['successful_patterns'][-3:]}")

    return "\n\n".join(parts)


def _get_fallback_strategy(state: AgentState, valid_strategies: List[str]) -> str:
    """LLM 실패 시 fallback 전략 선택"""
    # 우선순위: 형태 > 물리 > 대칭 (SYMMETRY_FIX는 최후순위)
    priority = ["RELOCATE", "ROTATE", "REBUILD",
                "SELECTIVE_REMOVE", "BRIDGE", "ADD_SUPPORT",
                "SYMMETRY_FIX", "ROLLBACK"]

    for strategy in priority:
        if strategy in valid_strategies:
            return strategy

    return "ROLLBACK"


def _match_target_to_strategy(strategy: str, vision_problems: List[Dict], reasoning: str) -> Dict:
    """전략에 맞는 Vision 문제 매칭

    ROTATE: pointing, direction, angle, upward, downward, sideways, wrong way
    RELOCATE: detached, position, place, moved, shifted, disconnected
    REBUILD: missing, broken, incomplete
    """
    if not vision_problems:
        return None

    # 전략별 키워드
    strategy_keywords = {
        "ROTATE": ["pointing", "direction", "angle", "upward", "downward",
                   "sideways", "wrong way", "rotated", "facing", "orientation"],
        "RELOCATE": ["detached", "position", "place", "moved", "shifted",
                     "disconnected", "separated", "wrong location", "misplaced"],
        "REBUILD": ["missing", "broken", "incomplete", "absent", "gone",
                    "not visible", "없", "빠진", "누락"]
    }

    keywords = strategy_keywords.get(strategy, [])

    # 1. reasoning에서 언급된 위치 찾기
    reasoning_lower = reasoning.lower()
    for vp in vision_problems:
        loc = vp.get("location", "").lower()
        if loc in reasoning_lower:
            return vp

    # 2. issue 키워드 매칭
    for vp in vision_problems:
        issue = vp.get("issue", "").lower()
        for kw in keywords:
            if kw in issue:
                return vp

    # 3. 매칭 없으면 첫 번째
    return vision_problems[0]


def _flush_print(msg: str):
    """실시간 출력"""
    print(msg)
    sys.stdout.flush()


def _safe_truncate(text: str, max_len: int = 80) -> str:
    """한글 안전 truncation (멀티바이트 문자 고려)"""
    if not text or len(text) <= max_len:
        return text
    # 단어 경계에서 자르기 시도
    truncated = text[:max_len]
    last_space = truncated.rfind(' ')
    if last_space > max_len // 2:
        return truncated[:last_space] + "..."
    return truncated + "..."

def node_supervisor(state: AgentState) -> AgentState:
    """전략 결정 - 동적 전략 풀링"""
    print(f"\n[SUPERVISOR] 전략 결정 중... (반복 {state['iteration']}/{FIXED_ITERATIONS})")

    # 삭제 한도 체크
    limit = int(state["original_brick_count"] * MAX_REMOVAL_RATIO)
    if state["total_removed"] >= limit:
        return {**state, "should_finish": True, "finish_reason": f"삭제 한도 도달 ({limit})"}

    # 동적 전략 풀 구성
    valid = _get_valid_strategies(state)

    if not valid:
        return {**state, "should_finish": True, "finish_reason": "모든 문제 해결됨"}

    # 상태 출력
    model_type = state.get("model_type", "unknown")
    vision_count = len(state.get("vision_problems", []))
    symmetry_count = len(state.get("symmetry_issues", []))
    floating_count = state["floating_count"]

    print(f"  [모델] {model_type}")
    print(f"  [상태] vision={vision_count}, symmetry={symmetry_count}, floating={floating_count}")
    print(f"  [전략 풀] {valid}")

    # 동적 컨텍스트 생성
    context = _build_context(state, valid)

    # LLM 호출
    strategy = ""
    confidence = 0
    reasoning = ""

    try:
        messages = [
            SystemMessage(content=SUPERVISOR_SYSTEM),
            HumanMessage(content=context)
        ]
        decision: StrategyDecision = strategy_llm.invoke(
            messages,
            config={"run_name": "전략결정", "tags": ["supervisor"]}
        )
        # 유효한 전략인지 확인
        if decision.strategy.upper() in valid:
            strategy = decision.strategy.upper()
        confidence = decision.confidence
        reasoning = decision.reasoning

        if not strategy:
            print(f"  [WARNING] 유효한 전략 없음, fallback 사용")
    except Exception as e:
        print(f"  [WARNING] LLM 에러: {e}")

    # Fallback - 첫 번째 유효 전략 사용
    if not strategy:
        strategy = valid[0]
        confidence = 50
        reasoning = f"fallback: {strategy}"

    print(f"  전략: {strategy} (확신도: {confidence}%)")
    print(f"  이유: {_safe_truncate(reasoning, 200)}")

    # 대상 선택
    target = None
    vision_problems = state.get("vision_problems", [])
    symmetry_issues = state.get("symmetry_issues", [])

    if strategy in ["RELOCATE", "ROTATE", "REBUILD"] and vision_problems:
        target = _match_target_to_strategy(strategy, vision_problems, reasoning)
    elif strategy in ["ADD_SUPPORT", "SELECTIVE_REMOVE", "BRIDGE"] and state["floating_bricks"]:
        target = state["floating_bricks"][0]
    elif strategy == "SYMMETRY_FIX" and symmetry_issues:
        target = symmetry_issues[0]

    # 대화 기록 추가 (LangSmith 트레이싱용)
    new_messages = [
        HumanMessage(content=f"[SUPERVISOR] {context[:500]}..."),
        AIMessage(content=f"전략: {strategy} (확신도: {confidence}%)\n이유: {reasoning}")
    ]

    return {**state, "strategy": strategy, "target_brick": target, "messages": new_messages}
