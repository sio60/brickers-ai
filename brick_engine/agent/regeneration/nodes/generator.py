# ============================================================================
# Generator 노드: GLB → LDR 변환
# ============================================================================

import logging
from typing import Dict, Any

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


def node_generator(graph, state) -> Dict[str, Any]:
    """GLB -> LDR 변환 노드"""
    from ....exporter.ldr_converter.glb_to_ldr_embedded import convert_glb_to_ldr

    logger.info("[Generator] 변환 시도 %s/%s", state['attempts'] + 1, state['max_retries'])
    graph._log("GENERATE", f"설계안을 하나씩 구현해 보는 중이에요. ({state['attempts'] + 1}/{state['max_retries']})")
    logger.info("  Params: target=%s, budget=%s", state['params'].get('target'), state['params'].get('budget'))

    try:
        result = convert_glb_to_ldr(
            state['glb_path'],
            state['ldr_path'],
            **state['params']
        )

        brick_count = result.get('parts', 0)
        final_target = result.get('final_target', 0)

        logger.info("  [OK] 변환 완료: %s개 브릭 (Final Target: %s)", brick_count, final_target)
        
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
                logger.info("  [Backup] 초기 모델 백업 완료: %s", initial_path.name)
            except Exception as e:
                logger.warning("  [Warning] 초기 모델 백업 실패: %s", e)

        result = {
            "attempts": state['attempts'] + 1, 
            "next_action": "verify",
        }
        if initial_ldr_path:
            result["initial_ldr_path"] = initial_ldr_path
        return result

    except Exception as e:
        logger.error("  [Error] 변환 실패: %s", e)
        error_msg = f"변환 중 치명적 오류 발생: {e}. 파라미터를 크게 변경해야 합니다."
        return {
            "attempts": state['attempts'] + 1,
            "messages": [HumanMessage(content=error_msg)],
            "next_action": "model"
        }
