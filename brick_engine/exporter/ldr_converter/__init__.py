"""
LDR Converter 패키지

기존 `from ldr_converter import X` 호환을 위한 re-export.
"""

# models
from .models import Vector3, PlacedBrick, BrickModel, BBox

# rotation / bbox
from .rotation import (
    ROTATION_MATRICES,
    get_rotation_matrix,
    get_rotated_size,
    get_brick_bbox,
    SLOPE_PARTS,
    SLOPE_TOLERANCE,
    PART_SIZE_TABLE,
)

# validation
from .validation import (
    ValidationError,
    VALID_COLOR_RANGE,
    VALID_ROTATIONS,
    validate_brick,
    validate_model,
    L2ValidationResult,
    check_collisions,
    check_floating,
    validate_physics,
)

# ldr writer
from .ldr_writer import (
    STEP_MODE_NONE,
    STEP_MODE_LAYER,
    STEP_MODE_BRICK,
    VALID_STEP_MODES,
    brick_to_ldr_line,
    model_to_ldr,
    model_to_ldr_unsafe,
    save_ldr_file,
)

# ldr parser
from .ldr_parser import (
    matrix_to_rotation,
    parse_ldr_line,
    parse_ldr_file,
    ldr_to_brick_model,
    change_colors,
)

# parts db
from .parts_db import load_parts_db, parse_brick_model

__all__ = [
    # models
    'Vector3', 'PlacedBrick', 'BrickModel', 'BBox',
    # rotation
    'ROTATION_MATRICES', 'get_rotation_matrix', 'get_rotated_size',
    'get_brick_bbox', 'SLOPE_PARTS', 'SLOPE_TOLERANCE', 'PART_SIZE_TABLE',
    # validation
    'ValidationError', 'VALID_COLOR_RANGE', 'VALID_ROTATIONS',
    'validate_brick', 'validate_model', 'L2ValidationResult',
    'check_collisions', 'check_floating', 'validate_physics',
    # writer
    'STEP_MODE_NONE', 'STEP_MODE_LAYER', 'STEP_MODE_BRICK', 'VALID_STEP_MODES',
    'brick_to_ldr_line', 'model_to_ldr', 'model_to_ldr_unsafe', 'save_ldr_file',
    # parser
    'matrix_to_rotation', 'parse_ldr_line', 'parse_ldr_file',
    'ldr_to_brick_model', 'change_colors',
    # parts_db
    'load_parts_db', 'parse_brick_model',
]
