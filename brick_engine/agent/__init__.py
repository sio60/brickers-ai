# ============================================================================
# brick_engine.agent 통합 패키지
# 모든 하위 에이전트 및 핵심 로직의 통합 인터페이스를 제공합니다.
# ============================================================================

# 1. 핵심 인프라 및 클라이언트
from .core.llm_clients import GroqClient, GeminiClient, BaseLLMClient

# 2. 개별 에이전트 패키지 노출
from .regeneration import regeneration_loop, RegenerationGraph

# 3. 브릭 조작 및 분석 도구
from .merger import structural_merge

# Note: evolver.run_agent는 sys.path 조작 + evolver_agent 의존성이 있어
# 직접 import하면 ModuleNotFoundError 발생. evolver_runner.py에서 lazy import로 사용.

__all__ = [
    "GroqClient",
    "GeminiClient",
    "BaseLLMClient",
    "regeneration_loop",
    "RegenerationGraph",
    "structural_merge",
]

