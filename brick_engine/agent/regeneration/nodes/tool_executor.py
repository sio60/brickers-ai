# ============================================================================
# Tool Executor 노드: 선택된 도구 실행
# 리팩토링: 도구별 로직 분리, state 직접 변경 제거, print→logging
# ============================================================================

import logging
from typing import Dict, Any, List, Tuple

from langchain_core.messages import AIMessage, ToolMessage

from ..constants import MAX_TOOL_USES, MAX_REMOVE_FALLBACK
from ... import merger as ldr_modifier  # 네이티브 패키지 임포트 (merger로 리팩토링됨)

logger = logging.getLogger("agent.regeneration.tool_executor")


# ---------------------------------------------------------------------------
# 도구별 실행 함수
# ---------------------------------------------------------------------------
def _execute_tune_parameters(
    args: Dict, params: Dict[str, Any]
) -> Tuple[str, str, Dict[str, Any], bool]:
    """TuneParameters 도구 실행. (result_content, next_step, new_params, reset_merged) 반환."""
    new_params = params.copy()
    new_params.update(args)
    new_params['shrink'] = 0.7
    result_content = f"파라미터가 업데이트되었습니다. ({args})"
    # 재생성 시 병합 상태 리셋 (다시 병합 단계를 거치도록)
    return result_content, "generator", new_params, True  # reset_merged=True


def _execute_remove_bricks(
    args: Dict, ldr_path: str
) -> Tuple[str, str]:
    """RemoveBricks 도구 실행. (result_content, next_step) 반환."""
    brick_ids = args.get('brick_ids', [])
    if not brick_ids:
        return "삭제할 브릭 ID가 제공되지 않았습니다.", "model"

    decisions = [{"brick_id": bid, "action": "delete"} for bid in brick_ids]
    stats = ldr_modifier.apply_llm_decisions(ldr_path, decisions)

    if stats['deleted'] > 0:
        return f"브릭 {stats['deleted']}개를 성공적으로 삭제했습니다.", "verifier"
    else:
        return "브릭 삭제에 실패했습니다. (ID를 찾을 수 없거나 이미 삭제됨)", "model"


def _execute_merge_bricks(
    state: Dict[str, Any]
) -> Tuple[str, str, bool]:
    """MergeBricks 도구 실행. (result_content, next_step, merged_flag) 반환."""
    ldr_path = state['ldr_path']
    raw_result = state.get('verification_raw_result') or {}
    issues = raw_result.get('issues', [])

    # [사용자 추가 반영] 불안정 브릭 ID 추출 (floating, isolated, unstable_base, top_only)
    unstable_ids = []
    for issue in issues:
        itype = issue.get('type', '').lower()
        bid = issue.get('brick_id')
        # top_only(아래 지지 없음) 등 모든 불안정 유형 포함
        if any(word in itype for word in ['floating', 'isolated', 'unstable', 'top_only']):
            if bid is not None:
                unstable_ids.append(bid)
    
    unstable_ids = list(set(unstable_ids))

    # 불안정 브릭이 없으면 파편화 분석으로 대상 선정
    if not unstable_ids:
        small_ratio = raw_result.get('small_brick_ratio', 0)
        if small_ratio > 0.05:
            logger.info(
                f"[Merge] 구조적 문제는 없으나 파편화율({small_ratio:.1%})이 높아 공격적 병합(Aggressive)을 수행합니다."
            )
            unstable_ids = _find_fragmented_bricks(ldr_path)

    if not unstable_ids:
        # 정말로 병합할 게 아무것도 없는 경우 (Fallback)
        logger.info("[Merge] 불안정 브릭 없음 -> 단순 병합(simple) Fallback")
        merge_stats = ldr_modifier.merge_small_bricks(ldr_path, min_merge_count=2)
        if merge_stats.get('merged', 0) > 0:
            return (
                f"구조적 문제는 없으나, 1x1 브릭 {merge_stats['merged']}개 그룹을 같은 색상 단위로 병합하여 정리했습니다.",
                "verifier",
                True,
            )
        else:
            return (
                "현재 구조상 더 이상 인접한 같은 색상의 1x1 브릭을 병합할 수 없습니다. "
                "파라미터 튜닝(TuneParameters) 등 다른 전략을 고려하세요.",
                "model",
                False,
            )

    # 구조적 병합 수행
    try:
        # [Refinement] 먼저 같은 색상의 1x1 브릭들을 최대한 병합하여 토대를 만듭니다.
        ldr_modifier.merge_small_bricks(ldr_path, min_merge_count=2, group_by_color=True)
        
        logger.info(f"[Merge] 구조적 병합 시작 (Target: {len(unstable_ids)} unstable bricks)")
        struct_stats = ldr_modifier.structural_merge(ldr_path, unstable_ids)

        merged_cnt = struct_stats.get('merged', 0)
        split_cnt = struct_stats.get('split', 0)

        if merged_cnt > 0 or split_cnt > 0:
            return (
                f"구조적 병합 완료: 불안정 부위 {split_cnt}곳을 분해하고 "
                f"{merged_cnt}개 그룹으로 재조립하여 보강했습니다.",
                "verifier",
                True,
            )
        else:
            return "구조적 병합을 시도했으나 변경된 부분이 없습니다.", "model", False
    except Exception as e:
        logger.error(f"구조적 병합 중 오류 발생: {e}", exc_info=True)
        return f"구조적 병합 중 오류 발생: {e}", "model", False


def _find_fragmented_bricks(ldr_path: str) -> List[int]:
    """파편화된 1x1/1x2 브릭의 인덱스를 반환."""
    fragmented_ids = []
    try:
        with open(ldr_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        target_parts = {"3005.dat", "3024.dat", "3004.dat", "3023.dat"}
        brick_idx = 0
        for line in lines:
            p = ldr_modifier.parse_ldr_line(line)
            if p:
                if p['part'] in target_parts:
                    fragmented_ids.append(brick_idx)
                brick_idx += 1

        if fragmented_ids:
            logger.info(f"[Merge] 파편화 브릭 {len(fragmented_ids)}개를 재결합 대상으로 선정했습니다.")
    except Exception as e:
        logger.error(f"⚠️ 파편화 분석 중 오류: {e}")

    return fragmented_ids


# ---------------------------------------------------------------------------
# 메인 노드 함수
# ---------------------------------------------------------------------------
def node_tool_executor(graph, state) -> Dict[str, Any]:
    """선택된 도구를 실행하는 노드"""
    last_message = state['messages'][-1]

    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {"next_action": "model"}

    tool_results = []
    next_step = "model"

    tool_usage_count = dict(state.get('tool_usage_count', {}))
    last_tool_used = state.get('last_tool_used', None)
    consecutive_same_tool = state.get('consecutive_same_tool', 0)
    
    # [사용자 추가 추적 변수]
    hallucination_count = state.get('hallucination_count', 0)
    mod_attempts = state.get('modification_attempts', 0)

    # 반환할 상태 업데이트
    updated_params = dict(state['params'])
    updated_merged = state.get('merged', False)

    for tool_call in last_message.tool_calls:
        tool_name = tool_call['name']
        args = tool_call['args']
        tool_call_id = tool_call['id']

        # 도구별 사용 횟수 체크
        current_count = tool_usage_count.get(tool_name, 0)
        max_allowed = MAX_REMOVE_FALLBACK if tool_name == "RemoveBricks" else MAX_TOOL_USES

        if tool_name not in ["TuneParameters", "RemoveBricks", "MergeBricks"]:
            # [사용자 추가] 존재하지 않는 도구 (Hallucination)
            hallucination_count += 1
            logger.warning(f"⚠️ [Hallucination] 존재하지 않는 도구 호출: {tool_name} (총 {hallucination_count}회)")
            result_content = (
                f"❌ 알 수 없는 도구: '{tool_name}'. 사용 가능한 도구는 [TuneParameters, RemoveBricks, MergeBricks] 뿐입니다. "
                f"존재하지 않는 도구를 생성하지 마세요. (현재 {hallucination_count}회 경고)"
            )
            tool_results.append(ToolMessage(content=result_content, tool_call_id=tool_call_id))
            continue

        if current_count >= max_allowed:
            logger.warning(f"⛔ {tool_name} 사용 한도 초과! ({current_count}/{max_allowed})")
            warning_msg = (
                f"'{tool_name}'은(는) 최대 {max_allowed}회까지만 사용 가능하며 이미 도달했습니다. "
                f"다른 전략을 고려해 주세요."
            )
            tool_results.append(ToolMessage(content=warning_msg, tool_call_id=tool_call_id))
            continue

        # 연속 사용 추적
        if tool_name == last_tool_used:
            consecutive_same_tool += 1
        else:
            consecutive_same_tool = 1

        tool_usage_count[tool_name] = current_count + 1
        logger.info(f"\n[Tool Execution] {tool_name} 실행... (총 {tool_usage_count[tool_name]}/{max_allowed}회)")
        graph._log("TOOL", f"도구를 활용해 브릭 구조를 조정하고 있어요. ({tool_name})")

        result_content = ""

        if tool_name == "TuneParameters":
            result_content, next_step, updated_params, _reset = _execute_tune_parameters(
                args, updated_params
            )
            if _reset:
                updated_merged = False

        elif tool_name == "RemoveBricks":
            result_content, next_step = _execute_remove_bricks(args, state['ldr_path'])
            mod_attempts += 1 # 수정 시도 카운트 증가

        elif tool_name == "MergeBricks":
            result_content, next_step, merged_flag = _execute_merge_bricks(state)
            if merged_flag:
                updated_merged = True
            mod_attempts += 1 # 수정 시도 카운트 증가

        logger.info(f"  결과: {result_content}")
        tool_results.append(ToolMessage(content=result_content, tool_call_id=tool_call_id))

    return {
        "messages": tool_results,
        "next_action": next_step,
        "params": updated_params,
        "merged": updated_merged,
        "tool_usage_count": tool_usage_count,
        "last_tool_used": tool_name if 'tool_name' in locals() else None,
        "consecutive_same_tool": consecutive_same_tool,
        "modification_attempts": mod_attempts,
        "hallucination_count": hallucination_count
    }
