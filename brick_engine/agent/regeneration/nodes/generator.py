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
        
        # [FIX] 초기 모델 백업 (첫 번째 시도에서만)
        initial_ldr_path = state.get("initial_ldr_path")
        if state['attempts'] == 0:
            import shutil
            from pathlib import Path
            
            p = Path(state['ldr_path'])
            initial_path = p.parent / f"{p.stem}_initial{p.suffix}"
            try:
                shutil.copy2(p, initial_path)
                initial_ldr_path = str(initial_path)
                print(f"  [Backup] 초기 모델 백업 완료: {initial_path.name}")
            except Exception as e:
                print(f"  [Warning] 초기 모델 백업 실패: {e}")

        result = {
            "attempts": state['attempts'] + 1, 
            "next_action": "verify",
        }
        if initial_ldr_path:
            result["initial_ldr_path"] = initial_ldr_path
        return result

    except Exception as e:
        print(f"  [Error] 변환 실패: {e}")
        error_msg = f"변환 중 치명적 오류 발생: {e}. 파라미터를 크게 변경해야 합니다."
        return {
            "attempts": state['attempts'] + 1,
            "messages": [HumanMessage(content=error_msg)],
            "next_action": "model"
        }
