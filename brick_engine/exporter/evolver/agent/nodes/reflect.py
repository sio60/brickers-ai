"""REFLECT Node - Learn from results + 원본 비교 롤백

Enhanced:
- Memory Summarize (lessons 10개 초과 시 요약)
- LLM 기반 lesson 생성 (REFLECT_PROMPT 템플릿 사용)
"""
import sys
import copy
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from ..state import AgentState
from ..tools import get_model_state
from ..constants import LESSONS_THRESHOLD, SUMMARIZE_TIMEOUT, LLM_MODEL, LLM_TIMEOUT
from ..prompts import (
    MEMORY_SUMMARIZE_SYSTEM, MEMORY_SUMMARIZE_PROMPT,
    REFLECT_SYSTEM, REFLECT_PROMPT
)
from ..config import get_config


# ===== Structured Output for Reflect =====
class ReflectOutput(BaseModel):
    """회고 결과 구조화 출력"""
    improved: bool = Field(description="모델이 개선되었는지 여부")
    stability_maintained: bool = Field(description="물리적 안정성이 유지되었는지 여부")
    what_changed: str = Field(description="실제로 변경된 내용 요약 (한국어)")
    lesson: str = Field(description="이번 진화에서 배운 핵심 교훈 (한국어, 2-3문장)")
    lesson_tag: Literal["SUCCESS", "FAILED", "INSIGHT"] = Field(description="교훈 태그")


# Reflect용 LLM
reflect_llm = ChatOpenAI(model=LLM_MODEL, temperature=0.3, timeout=LLM_TIMEOUT)
reflect_structured_llm = reflect_llm.with_structured_output(ReflectOutput)

# Memory Summarize용 LLM (constants에서 timeout 가져옴)
summarize_llm = ChatOpenAI(model=LLM_MODEL, temperature=0.3, timeout=SUMMARIZE_TIMEOUT)


def _generate_lesson_llm(
    strategy: str,
    before_state: dict,
    after_state: dict,
    vision_improved: bool,
    collision_worsened: bool
) -> tuple[str, str]:
    """LLM으로 lesson 생성 (REFLECT_PROMPT 템플릿 사용)

    Returns:
        tuple: (lesson 텍스트, lesson_tag)
    """
    # 간략화된 정보
    before_brick_list = f"부유: {before_state.get('floating', 0)}개, 충돌: {before_state.get('collision', 0)}개"
    after_brick_list = f"부유: {after_state.get('floating', 0)}개, 충돌: {after_state.get('collision', 0)}개"

    before_vision = f"품질: {before_state.get('vision_quality', 'N/A')}, 문제: {before_state.get('vision_problems', 0)}개"
    after_vision = f"품질: {after_state.get('vision_quality', 'N/A')}, 문제: {after_state.get('vision_problems', 0)}개"

    context = REFLECT_PROMPT.format(
        strategy=strategy,
        before_brick_list=before_brick_list,
        after_brick_list=after_brick_list,
        before_vision=before_vision,
        after_vision=after_vision,
        debate_result=f"Vision 개선: {vision_improved}, 충돌 악화: {collision_worsened}"
    )

    try:
        messages = [
            SystemMessage(content=REFLECT_SYSTEM),
            HumanMessage(content=context)
        ]
        result: ReflectOutput = reflect_structured_llm.invoke(
            messages,
            config={"run_name": "회고", "tags": ["reflect", "lesson"]}
        )
        return result.lesson, result.lesson_tag
    except Exception as e:
        print(f"  [WARNING] LLM 회고 실패: {e}")

    # fallback: 규칙 기반 lesson
    if vision_improved and not collision_worsened:
        return f"SUCCESS: {strategy} 전략이 효과적이었음", "SUCCESS"
    elif collision_worsened:
        return f"FAILED: {strategy} 전략이 충돌을 악화시킴", "FAILED"
    else:
        return f"INSIGHT: {strategy} 전략의 효과가 불분명함", "INSIGHT"


def _summarize_lessons(lessons: list) -> list:
    """Lessons가 너무 많으면 LLM으로 요약 (Memory Management)"""
    if len(lessons) <= LESSONS_THRESHOLD:
        return lessons

    print(f"  [MEMORY] Lessons {len(lessons)}개 → 요약 중...")

    # prompts.py의 템플릿 사용
    lessons_text = '\n'.join(f'- {l}' for l in lessons)
    prompt = MEMORY_SUMMARIZE_PROMPT.format(lessons=lessons_text)

    try:
        messages = [
            SystemMessage(content=MEMORY_SUMMARIZE_SYSTEM),
            HumanMessage(content=prompt)
        ]
        response = summarize_llm.invoke(
            messages,
            config={"run_name": "메모리요약", "tags": ["reflect", "memory"]}
        )

        # 응답을 줄 단위로 파싱
        summarized = []
        for line in response.content.strip().split('\n'):
            line = line.strip()
            if line and (line.startswith('SUCCESS:') or line.startswith('FAILED:') or line.startswith('INSIGHT:')):
                summarized.append(line)

        if summarized:
            print(f"  [MEMORY] 요약 완료: {len(lessons)}개 → {len(summarized)}개")
            return summarized
    except Exception as e:
        print(f"  [MEMORY] 요약 실패: {e}")

    # 실패 시 최근 5개만 유지
    return lessons[-5:]

# Vision 분석용 5방향 (TOP, LEFT 제외)
VISION_ANGLES = ["FRONT", "BACK", "RIGHT", "BOTTOM", "FRONT_RIGHT"]

def node_reflect(state: AgentState) -> AgentState:
    """Analyze results and update memory (물리 + Vision 통합 검증 + 롤백)"""
    print(f"\n[REFLECT] Analyzing...")

    # Get new state
    new_state = get_model_state(state["model"], get_config().parts_db)
    before_floating = state["floating_count"]
    after_floating = new_state["floating_count"]
    before_collision = state.get("collision_count", 0)
    after_collision = new_state.get("collision_count", 0)

    # 물리적 개선 여부
    floating_improvement = before_floating - after_floating
    collision_worsened = after_collision > before_collision

    last_action = state["action_history"][-1] if state["action_history"] else None
    action_type = last_action["type"] if last_action else "unknown"

    memory = state["memory"].copy()
    lesson = ""
    should_rollback = False

    # Vision 전략인지 확인 (형태 변경 전략들)
    is_vision_strategy = action_type in ["relocate", "rebuild", "rotate"]
    vision_check_completed = False  # Critical 버그 수정: dir() 대신 플래그 사용
    after_quality = 0  # 변수 미리 초기화

    # Vision 전략이면 Vision 검증 필요
    if is_vision_strategy:
        print(f"  [Vision Strategy] Re-rendering for quality check...")

        # Vision 검증 (re-render + re-analyze)
        try:
            # 절대 경로 import
            evolver_dir = Path(__file__).parent.parent.parent
            if str(evolver_dir) not in sys.path:
                sys.path.insert(0, str(evolver_dir))

            from vision_analyzer import find_problems
            from ldr_renderer import render_model_multi_angle

            # 렌더링 (5방향)
            images = render_model_multi_angle(state["model"], get_config().parts_db, angles=VISION_ANGLES)

            # Vision 분석
            target_name = state["model"].name if hasattr(state["model"], "name") else "unknown"
            vision_result = find_problems(images, target_name)

            after_problems = len(vision_result.get("problems", []))
            after_quality = vision_result.get("overall_quality", 0)

            # 이전 값 (List에서 len으로 계산)
            before_problems = len(state.get("vision_problems", []))
            before_quality = state.get("vision_quality_score") or 0

            print(f"  Vision: problems {before_problems} → {after_problems}, quality {before_quality} → {after_quality}")

            # Vision 기준 성공 판정
            vision_improved = (after_problems < before_problems) or (after_quality > before_quality)

            # LLM으로 lesson 생성 (REFLECT_PROMPT 템플릿 사용)
            before_state_info = {
                "floating": before_floating,
                "collision": before_collision,
                "vision_quality": before_quality,
                "vision_problems": before_problems
            }
            after_state_info = {
                "floating": after_floating,
                "collision": after_collision,
                "vision_quality": after_quality,
                "vision_problems": after_problems
            }
            lesson, lesson_tag = _generate_lesson_llm(
                action_type, before_state_info, after_state_info,
                vision_improved, collision_worsened
            )

            if lesson_tag == "SUCCESS":
                memory["successful_patterns"].append(action_type)
                memory["consecutive_failures"] = 0
                vision_check_completed = True
                print(f"  {lesson}")
            else:
                memory["failed_approaches"].append(action_type)
                memory["consecutive_failures"] += 1
                should_rollback = True
                print(f"  {lesson}")
                print(f"  [ROLLBACK] 원본으로 복원합니다.")

        except Exception as e:
            print(f"  WARNING: Vision check failed - {e}")
            # Vision 실패 시 물리 검증으로 폴백
            lesson = f"NEUTRAL: {action_type} (vision check failed)"
            after_problems = before_problems
            after_quality = before_quality

    else:
        # 물리 전략 - LLM으로 lesson 생성
        physics_improved = floating_improvement > 0 and not collision_worsened

        before_state_info = {
            "floating": before_floating,
            "collision": before_collision,
            "vision_quality": state.get("vision_quality_score", "N/A"),
            "vision_problems": len(state.get("vision_problems", []))
        }
        after_state_info = {
            "floating": after_floating,
            "collision": after_collision,
            "vision_quality": state.get("vision_quality_score", "N/A"),
            "vision_problems": len(state.get("vision_problems", []))
        }
        lesson, lesson_tag = _generate_lesson_llm(
            action_type, before_state_info, after_state_info,
            physics_improved, collision_worsened
        )

        if lesson_tag == "SUCCESS":
            memory["successful_patterns"].append(action_type)
            memory["consecutive_failures"] = 0
            print(f"  {lesson}")
        elif lesson_tag == "FAILED":
            memory["failed_approaches"].append(action_type)
            memory["consecutive_failures"] += 1
            should_rollback = True
            print(f"  {lesson}")
            print(f"  [ROLLBACK] 물리 상태 악화로 원본 복원")
        else:  # INSIGHT
            print(f"  {lesson}")

    if memory["consecutive_failures"] >= 3:
        print(f"  WARNING: 3 consecutive failures!")

    memory["lessons"].append(lesson)

    # Memory Summarize: lessons가 너무 많으면 요약
    memory["lessons"] = _summarize_lessons(memory["lessons"])

    # 롤백 필요 시 원본 복원
    if should_rollback and state.get("model_backup"):
        print(f"  [ROLLBACK] 모델을 원본으로 복원")
        restored_model = copy.deepcopy(state["model_backup"])
        restored_state = get_model_state(restored_model, get_config().parts_db)

        result_state = {
            **state,
            "model": restored_model,
            "floating_count": restored_state["floating_count"],
            "collision_count": restored_state.get("collision_count", 0),
            "memory": memory,
            "iteration": state["iteration"] + 1
        }
    else:
        result_state = {
            **state,
            "floating_count": after_floating,
            "collision_count": after_collision,
            "memory": memory,
            "iteration": state["iteration"] + 1
        }

        # Vision 전략인 경우에만 Vision 상태 업데이트
        if is_vision_strategy and vision_check_completed:
            result_state["vision_quality_score"] = after_quality

    return result_state
