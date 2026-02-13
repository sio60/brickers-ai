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

    for tool_call in last_message.tool_calls:
        tool_name = tool_call['name']
        args = tool_call['args']
        tool_call_id = tool_call['id']

        # 무한 루프 방지
        if tool_name == last_tool_used:
            consecutive_same_tool += 1
        else:
            consecutive_same_tool = 1

        if consecutive_same_tool >= 3:
            print(f"  ⚠️ 경고: {tool_name}을(를) {consecutive_same_tool}회 연속 사용 중!")
            warning_msg = f"'{tool_name}'을(를) 3회 연속 사용했습니다. 다른 전략을 고려해주세요."
            tool_results.append(ToolMessage(content=warning_msg, tool_call_id=tool_call_id))
            return {
                "messages": tool_results,
                "next_action": "model",
                "tool_usage_count": tool_usage_count,
                "last_tool_used": tool_name,
                "consecutive_same_tool": consecutive_same_tool,
            }

        tool_usage_count[tool_name] = tool_usage_count.get(tool_name, 0) + 1
        print(f"\n[Tool Execution] {tool_name} 실행... (총 {tool_usage_count[tool_name]}회)")

        result_content = ""

        if tool_name == "TuneParameters":
            new_params = state['params'].copy()
            new_params.update(args)
            new_params['shrink'] = 0.7
            result_content = f"파라미터가 업데이트되었습니다. ({args})"
            next_step = "generator"
            state['params'] = new_params

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
