# ============================================================================
# 통합 메모리 관리 모듈 (호환성 유지 wrapper)
#
# 실제 구현은 memory/ 패키지로 분리됨.
# 기존 import 경로 호환을 위해 re-export.
# ============================================================================

try:
    from .memory.manager import MemoryUtils, memory_manager
    from .memory.builders import (
        calculate_delta,
        ensure_not_empty,
        build_hypothesis,
        build_experiment,
        build_verification,
        build_improvement,
    )
except ImportError:
    # 패키지 구조가 복잡할 경우를 위한 상대 경로 지원
    from ..memory.manager import MemoryUtils, memory_manager
    from ..memory.builders import (
        calculate_delta,
        ensure_not_empty,
        build_hypothesis,
        build_experiment,
        build_verification,
        build_improvement,
    )

__all__ = [
    "MemoryUtils",
    "memory_manager",
    "calculate_delta",
    "ensure_not_empty",
    "build_hypothesis",
    "build_experiment",
    "build_verification",
    "build_improvement",
]
