# ============================================================================
# LLM 재생성 에이전트 실행부 (__main__)
#
# 위치: agent/regeneration/__main__.py
# python -m agent.regeneration 명령으로 직접 실행 시 호출됨.
# ============================================================================
import argparse
import asyncio
import sys
from pathlib import Path

# 프로젝트 루트 설정 (상위 3단계 위)
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.core.llm_clients import GeminiClient
from agent.regeneration.pipeline import regeneration_loop

async def main():
    parser = argparse.ArgumentParser(description="LLM 재생성 에이전트 CLI")
    parser.add_argument("glb", help="입력 GLB 파일 경로")
    parser.add_argument("--out", default="output.ldr", help="출력 LDR 파일 경로")
    parser.add_argument("--max-retries", type=int, default=5, help="최대 재시도 횟수")
    parser.add_argument("--gui", action="store_true", help="물리 엔진 GUI 활성화")
    parser.add_argument("--api-key", help="Gemini API Key (환경변수 미설정 시)")

    args = parser.parse_args()

    client = GeminiClient(api_key=args.api_key)

    await regeneration_loop(
        args.glb,
        args.out,
        llm_client=client,
        max_retries=args.max_retries,
        gui=args.gui
    )

if __name__ == "__main__":
    asyncio.run(main())
