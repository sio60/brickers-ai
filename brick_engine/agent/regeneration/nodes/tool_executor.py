# ============================================================================
# Tool Executor 노드: 선택된 도구 실행
# ============================================================================

from typing import Dict, Any

from langchain_core.messages import AIMessage, ToolMessage


def node_tool_executor(graph, state) -> Dict[str, Any]:
    """선택된 도구를 실행하는 노드"""
    from ... import ldr_modifier

    last_message = state['messages'][-1]

    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {"next_action": "model"}

    tool_results = []
    next_step = "model"

    tool_usage_count = state.get('tool_usage_count', {})
    last_tool_used = state.get('last_tool_used', None)
    consecutive_same_tool = state.get('consecutive_same_tool', 0)

    # 도구별 최대 사용 횟수 제한
    MAX_TOOL_USES = 5
    MAX_REMOVE_FALLBACK = 1

    for tool_call in last_message.tool_calls:
        tool_name = tool_call['name']
        args = tool_call['args']
        tool_call_id = tool_call['id']

        graph._log("TOOL", f"도구를 활용해 브릭 구조를 조정하고 있어요. ({tool_name})")

        # 도구별 사용 횟수 체크
        current_count = tool_usage_count.get(tool_name, 0)
        max_allowed = MAX_REMOVE_FALLBACK if tool_name == "RemoveBricks" else MAX_TOOL_USES

        if current_count >= max_allowed:
            print(f"  ⛔ {tool_name} 사용 한도 초과! ({current_count}/{max_allowed})")
            warning_msg = (
                f"'{tool_name}'은(는) 최대 {max_allowed}회까지만 사용 가능하며, "
                f"이미 {current_count}회 사용하여 한도에 도달했습니다. "
                f"다른 도구를 사용하거나 현재 상태를 수락해 주세요."
            )
            tool_results.append(ToolMessage(content=warning_msg, tool_call_id=tool_call_id))
            return {
                "messages": tool_results,
                "next_action": "model",
                "tool_usage_count": tool_usage_count,
                "last_tool_used": tool_name,
                "consecutive_same_tool": consecutive_same_tool,
            }

        # 연속 사용 추적 (로깅용)
        if tool_name == last_tool_used:
            consecutive_same_tool += 1
        else:
            consecutive_same_tool = 1

        tool_usage_count[tool_name] = current_count + 1
        print(f"\n[Tool Execution] {tool_name} 실행... (총 {tool_usage_count[tool_name]}/{max_allowed}회)")

        result_content = ""

        if tool_name == "TuneParameters":
            new_params = state['params'].copy()
            new_params.update(args)
            new_params['shrink'] = 0.7
            result_content = f"파라미터가 업데이트되었습니다. ({args})"
            next_step = "generator"
            state['params'] = new_params
            # 재생성 시 병합 상태 리셋 (다시 병합 단계를 거치도록)
            state['merged'] = False

        elif tool_name == "RemoveBricks":
            brick_ids = args.get('brick_ids', [])
            if not brick_ids:
                result_content = "삭제할 브릭 ID가 제공되지 않았습니다."
            else:
                decisions = [{"brick_id": bid, "action": "delete"} for bid in brick_ids]
                stats = ldr_modifier.apply_llm_decisions(state['ldr_path'], decisions)

                if stats['deleted'] > 0:
                    result_content = f"브릭 {stats['deleted']}개를 성공적으로 삭제했습니다."
                    next_step = "verifier"
                else:
                    result_content = "브릭 삭제에 실패했습니다. (ID를 찾을 수 없거나 이미 삭제됨)"
                    
        elif tool_name == "MergeBricks":
            # [전략 통합] 사용자 요청에 따라 무조건 'structural_merge' (구조적 병합) 수행
            # 불안정 브릭(Floating, Isolated)을 식별하여 그 주변을 분해/재조립함
            
            raw_result = state.get('verification_raw_result') or {}
            issues = raw_result.get('issues', [])
            
            # 불안정 브릭 ID 추출 (top_only 포함: 아래 지지 없음)
            unstable_ids = []
            for issue in issues:
                # [BUG FIX] issue_type 케이스 불일치 방지 및 brick_id 0 누락 해결
                itype = issue.get('type', '').lower()
                bid = issue.get('brick_id')
                
                if itype in ['floating', 'isolated', 'unstable_base']:
                    if bid is not None:
                        unstable_ids.append(bid)
            
            # 중복 제거
            unstable_ids = list(set(unstable_ids))
            
            if not unstable_ids:
                # [STRATEGY] 불안정 브릭은 없으나, 모델이 너무 파편화(많은 1x1/1x2)된 경우 강제 병합 수행
                # 파편화 지표: small_brick_ratio (verifier에서 제공)
                small_ratio = raw_result.get('small_brick_ratio', 0)
                
                if small_ratio > 0.05: # 5% 이상이면 파편화로 간주
                    print(f"  [Merge] 구조적 문제는 없으나 파편화율({small_ratio:.1%})이 높아 공격적 병합(Aggressive)을 수행합니다.")
                    
                    # 모든 1x1 및 1x2 브릭을 분해/재병합 대상으로 선정
                    import brick_engine.agent.ldr_modifier as mod
                    fragmented_ids = []
                    try:
                        with open(state['ldr_path'], 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                        
                        target_parts = {"3005.dat", "3024.dat", "3004.dat", "3023.dat"}
                        brick_idx = 0
                        for line in lines:
                            p = mod.parse_ldr_line(line)
                            if p:
                                if p['part'] in target_parts:
                                    fragmented_ids.append(brick_idx)
                                brick_idx += 1
                                
                        if fragmented_ids:
                            unstable_ids = fragmented_ids
                            print(f"  [Merge] 파편화 브릭 {len(unstable_ids)}개를 재결합 대상으로 선정했습니다.")
                    except Exception as e:
                        print(f"  ⚠️ 파편화 분석 중 오류: {e}")

            if not unstable_ids:
                # 정말로 병합할 게 아무것도 없는 경우 (Fallback)
                print("  [Merge] 불안정 브릭 없음 -> 단순 병합(simple) Fallback")
                merge_stats = ldr_modifier.merge_small_bricks(state['ldr_path'], min_merge_count=2) # group_by_color=True (기본값)
                if merge_stats.get('merged', 0) > 0:
                    result_content = f"구조적 문제는 없으나, 1x1 브릭 {merge_stats['merged']}개 그룹을 같은 색상 단위로 병합하여 정리했습니다."
                    next_step = "verifier"
                    state['merged'] = True
                else:
                    result_content = "현재 구조상 더 이상 인접한 같은 색상의 1x1 브릭을 병합할 수 없습니다. 파라미터 튜닝(TuneParameters) 등 다른 전략을 고려하세요."
            else:
                try:
                    print(f"  [Merge] 구조적 병합 시작 (Target: {len(unstable_ids)} unstable bricks)")
                    struct_stats = ldr_modifier.structural_merge(state['ldr_path'], unstable_ids)
                    
                    merged_cnt = struct_stats.get('merged', 0)
                    split_cnt = struct_stats.get('split', 0)
                    
                    if merged_cnt > 0 or split_cnt > 0:
                        result_content = f"구조적 병합 완료: 불안정 부위 {split_cnt}곳을 분해하고 {merged_cnt}개 그룹으로 재조립하여 보강했습니다."
                        next_step = "verifier"
                        state['merged'] = True # 병합 완료 플래그
                    else:
                        result_content = "구조적 병합을 시도했으나 변경된 부분이 없습니다."
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    result_content = f"구조적 병합 중 오류 발생: {e}"
        else:
            result_content = f"알 수 없는 도구: {tool_name}"

        print(f"  결과: {result_content}")
        tool_results.append(ToolMessage(content=result_content, tool_call_id=tool_call_id))

    return {
        "messages": tool_results,
        "next_action": next_step,
        "params": state['params'],
        "tool_usage_count": tool_usage_count,
        "last_tool_used": tool_name,
        "consecutive_same_tool": consecutive_same_tool,
    }
