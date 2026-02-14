# ============================================================================
# LLM 재생성 에이전트 (호환성 유지 wrapper)
#
# 실제 구현은 regeneration/ 패키지로 분리됨.
# 기존 import 경로 (route/kids_render.py 등) 호환을 위해 re-export.
# ============================================================================

from .regeneration.pipeline import regeneration_loop
from .regeneration.graph import RegenerationGraph
from .regeneration.constants import DEFAULT_PARAMS
from .regeneration.feedback import extract_verification_feedback, format_feedback
from .regeneration.prompts import SYSTEM_PROMPT

__all__ = [
    "regeneration_loop",
    "RegenerationGraph",
    "DEFAULT_PARAMS",
    "extract_verification_feedback",
    "format_feedback",
    "SYSTEM_PROMPT",
]


if __name__ == "__main__":
    import argparse
    from .llm_clients import GeminiClient

    parser = argparse.ArgumentParser()
    parser.add_argument("glb", help="입력 GLB 파일")
    parser.add_argument("--out", default="output.ldr", help="출력 LDR")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--api-key", help="API Key")

    args = parser.parse_args()

    client = GeminiClient(api_key=args.api_key)

    import asyncio
    asyncio.run(regeneration_loop(
        args.glb,
        args.out,
        llm_client=client,
        max_retries=args.max_retries,
        gui=args.gui
    ))
