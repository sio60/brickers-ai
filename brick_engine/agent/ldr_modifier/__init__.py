# ============================================================================
# LDR 파일 수정 패키지 (리팩토링 버전)
# 기존 ldr_modifier.py를 기능별로 분리한 패키지
# 외부 호환성을 위해 기존 API를 그대로 re-export
# ============================================================================

# 파서 유틸리티
from .parser import parse_ldr_line, build_ldr_line

# LLM 결정 적용
from .decisions import apply_llm_decisions, modify_brick_position, remove_brick

# 병합 기능
from .merge_linear import merge_small_bricks

# 구조적 병합
from .structural_merge import structural_merge

# 상수 (필요 시 접근 가능)
from .constants import (
    STUD_SPACING, BRICK_HEIGHT, PLATE_HEIGHT,
    SMALL_BRICK_PARTS, MERGE_TARGET_BRICKS, PLATE_MERGE_TARGETS,
    BRICK_STUD_COUNT, BRICK_DIMENSIONS, BRICK_2X2_MAPPING,
    is_plate
)

__all__ = [
    # 파서
    "parse_ldr_line", "build_ldr_line",
    # LLM 결정
    "apply_llm_decisions", "modify_brick_position", "remove_brick",
    # 병합
    "merge_small_bricks",
    # 구조적 병합
    "structural_merge",
    # 상수
    "STUD_SPACING", "BRICK_HEIGHT", "PLATE_HEIGHT",
    "SMALL_BRICK_PARTS", "MERGE_TARGET_BRICKS", "PLATE_MERGE_TARGETS",
    "BRICK_STUD_COUNT", "BRICK_DIMENSIONS", "BRICK_2X2_MAPPING",
    "is_plate",
]
