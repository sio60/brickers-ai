# ============================================================================
# brick_engine 통합 패키지
# 이 패키지는 브릭 생성, 수정, 분석 및 내보내기를 위한 핵심 엔진 기능을 제공합니다.
# ============================================================================

# 1. 에이전트 핵심 인터페이스 노출 (agent 패키지로부터)
from .agent import (
    regeneration_loop,
    RegenerationGraph,
    run_color_variant,
    structural_merge,
    GroqClient,
    GeminiClient,
    BaseLLMClient,
)

# 2. 데이터베이스 유틸리티 노출
from .agent.core.database_manager import get_db, get_parts_collection

# 3. LDR 변환 및 모델 핵심 노출 (exporter 패키지로부터)
try:
    from .exporter.ldr_converter import (
        BrickModel,
        PlacedBrick,
        parse_ldr_file,
        save_ldr_file,
        validate_physics,
    )
except ImportError:
    # exporter 모듈이 아직 준비되지 않았거나 의존성 문제가 있는 경우 대비
    pass

__all__ = [
    "regeneration_loop",
    "RegenerationGraph",
    "run_color_variant",
    "structural_merge",
    "GroqClient",
    "GeminiClient",
    "BaseLLMClient",
    "get_db",
    "get_parts_collection",
    "BrickModel",
    "PlacedBrick",
    "parse_ldr_file",
    "save_ldr_file",
    "validate_physics",
]
