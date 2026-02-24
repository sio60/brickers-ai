# ============================================================================
# LLM 재생성 에이전트 (Shim)
#
# 위치: agent/regeneration/shim.py
# 기존 agent/llm_regeneration_agent.py의 기능을 담당하며 패키지 구조에 통합됨.
# ============================================================================

from .pipeline import regeneration_loop
from .graph import RegenerationGraph
from .constants import DEFAULT_PARAMS
from .feedback import extract_verification_feedback, format_feedback
from .prompts import SYSTEM_PROMPT

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
    import asyncio
    import sys
    from pathlib import Path

    # 프로젝트 루트 추가
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from agent.core.llm_clients import GeminiClient

    parser = argparse.ArgumentParser()
    parser.add_argument("glb", help="입력 GLB 파일")
    parser.add_argument("--out", default="output.ldr", help="출력 LDR")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--api-key", help="API Key")

    args = parser.parse_args()

    client = GeminiClient(api_key=args.api_key)

    asyncio.run(regeneration_loop(
        args.glb,
        args.out,
        llm_client=client,
        max_retries=args.max_retries,
        gui=args.gui
    ))
