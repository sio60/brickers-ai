# ============================================================================
# Merger 패키지 (브릭 병합 전담)
# ============================================================================

from .parser import parse_ldr_line, build_ldr_line
from .decisions import apply_llm_decisions, modify_brick_position, remove_brick
from .merge_linear import merge_small_bricks
from .structural_merge import structural_merge

from .constants import (
    STUD_SPACING, BRICK_HEIGHT, PLATE_HEIGHT,
    SMALL_BRICK_PARTS, MERGE_TARGET_BRICKS, PLATE_MERGE_TARGETS,
    is_plate
)

__all__ = [
    "parse_ldr_line", "build_ldr_line",
    "apply_llm_decisions", "modify_brick_position", "remove_brick",
    "merge_small_bricks",
    "structural_merge",
    "STUD_SPACING", "BRICK_HEIGHT", "PLATE_HEIGHT",
    "SMALL_BRICK_PARTS", "MERGE_TARGET_BRICKS", "PLATE_MERGE_TARGETS",
    "is_plate",
]
