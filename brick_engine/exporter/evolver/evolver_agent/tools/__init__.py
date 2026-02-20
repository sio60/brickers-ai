"""Agent Tools - Helper functions for LangGraph Evolver Agent

CoScientist 원칙:
- LLM이 "무엇을" 할지 결정 (strategy)
- 알고리즘이 "어떻게" 실행 (tools)

이 패키지의 함수들은 LLM이 결정한 전략을 정확하게 실행하는 도구들입니다.
"""

# --- 상수 ---
from .constants import (
    BRICK_HEIGHT,
    PLATE_HEIGHT,
    LDU_PER_STUD,
    CELL_SIZE,
    GROUND_TOLERANCE,
    MAX_SUPPORT_PER_BRICK,
    DEFAULT_MAX_REMOVE,
    KIDS_MIN_PART_SIZE,
    ADULT_DEFAULT_PART,
    PLATE_PART_IDS,
    SYMMETRY_TOLERANCE,
    SYMMETRY_CENTER_MARGIN,
    SPARSE_LAYER_THRESHOLD,
    DEFAULT_SUPPORT_RATIO,
    OPTIMAL_BOND_MIN,
    OPTIMAL_BOND_MAX,
    LDrawColor,
    SupportParts,
)

# --- 모델 조작 ---
from .model_ops import (
    get_brick_direction_label,
    label_brick_list,
    analyze_glb,
    get_model_state,
    remove_brick,
    add_brick,
    rollback_model,
)

# --- 충돌 / 점유 맵 ---
from .collision import (
    _build_occupancy_set,
    _mark_occupied,
    _check_collision_simple,
)

# --- 배치 점수 ---
from .scoring import (
    calc_support_ratio,
    calc_bond_score,
    can_place_brick,
    find_best_placement,
)

# --- 지지대 ---
from .support import (
    find_nearby_stable_bricks,
    generate_support_candidates,
    try_add_brick_with_validation,
)

# --- 대칭 / 고립 ---
from .symmetry import (
    analyze_symmetry,
    find_isolated_ground_bricks,
    remove_isolated_bricks,
    fix_symmetry,
)
