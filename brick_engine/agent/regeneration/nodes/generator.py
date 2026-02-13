# ============================================================================
# Generator 노드: GLB → LDR 변환
# ============================================================================

from typing import Dict, Any

from langchain_core.messages import HumanMessage


def node_generator(graph, state) -> Dict[str, Any]:
    """GLB -> LDR 변환 노드"""
    from glb_to_ldr_embedded import convert_glb_to_ldr

    print(f"\n[Generator] 변환 시도 {state['attempts'] + 1}/{state['max_retries']}")
    graph._log("GENERATE", f"설계안을 하나씩 구현해 보는 중이에요. ({state['attempts'] + 1}/{state['max_retries']})")
    print(f"  Params: target={state['params'].get('target')}, budget={state['params'].get('budget')}")

    try:
        result = convert_glb_to_ldr(
            state['glb_path'],
            state['ldr_path'],
            **state['params']
        )

        brick_count = result.get('parts', 0)
        final_target = result.get('final_target', 0)

        print(f"  [OK] 변환 완료: {brick_count}개 브릭 (Final Target: {final_target})")
        return {"attempts": state['attempts'] + 1, "next_action": "verify"}

    except Exception as e:
        print(f"  [Error] 변환 실패: {e}")
        error_msg = f"변환 중 치명적 오류 발생: {e}. 파라미터를 크게 변경해야 합니다."
        return {
            "attempts": state['attempts'] + 1,
            "messages": [HumanMessage(content=error_msg)],
            "next_action": "model"
        }
