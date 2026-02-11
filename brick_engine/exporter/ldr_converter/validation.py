"""
LDR Converter - 검증 (L1 스키마 + L2 물리)

충돌 검사, 부유 검사, 모델 검증
"""

from dataclasses import dataclass
from typing import List, Dict

from .models import PlacedBrick, BrickModel
from .rotation import get_brick_bbox, SLOPE_PARTS, SLOPE_TOLERANCE


class ValidationError(Exception):
    """검증 실패 시 발생하는 예외"""
    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"Validation failed: {errors}")


# LDraw 색상 코드 범위 (0-511, 일반적으로 사용되는 범위)
VALID_COLOR_RANGE = (0, 511)
VALID_ROTATIONS = [0, 90, 180, 270]


def validate_brick(brick: PlacedBrick, parts_db: Dict) -> List[str]:
    """
    개별 브릭의 L1 검증 (스키마/타입)

    Returns:
        에러 메시지 리스트 (빈 리스트면 검증 통과)
    """
    errors = []

    # 1. part_id 존재 여부
    if brick.part_id not in parts_db:
        errors.append(f"[{brick.id}] Unknown part_id: '{brick.part_id}'")

    # 2. rotation 범위 (0, 90, 180, 270만 허용)
    if brick.rotation not in VALID_ROTATIONS:
        errors.append(
            f"[{brick.id}] Invalid rotation: {brick.rotation}, "
            f"must be one of {VALID_ROTATIONS}"
        )

    # 3. color_code 범위 (LDraw 표준: 0-511)
    if not (VALID_COLOR_RANGE[0] <= brick.color_code <= VALID_COLOR_RANGE[1]):
        errors.append(
            f"[{brick.id}] Invalid color_code: {brick.color_code}, "
            f"must be {VALID_COLOR_RANGE[0]}-{VALID_COLOR_RANGE[1]}"
        )

    # 4. layer 음수 체크
    if brick.layer < 0:
        errors.append(f"[{brick.id}] Invalid layer: {brick.layer}, must be >= 0")

    # 5. id 빈 문자열 체크
    if not brick.id or not brick.id.strip():
        errors.append("Brick has empty id")

    return errors


def validate_model(model: BrickModel, parts_db: Dict) -> List[str]:
    """
    BrickModel 전체의 L1 검증

    Returns:
        에러 메시지 리스트 (빈 리스트면 검증 통과)
    """
    errors = []

    # 1. 모델 기본 필드 검증
    if not model.model_id or not model.model_id.strip():
        errors.append("Model has empty model_id")

    if not model.name or not model.name.strip():
        errors.append("Model has empty name")

    if model.mode not in ['pro', 'kids']:
        errors.append(f"Invalid mode: '{model.mode}', must be 'pro' or 'kids'")

    # 2. kids 모드일 때 target_age 검증
    if model.mode == 'kids':
        valid_ages = ['4-6', '7-9', '10-12']
        if model.target_age not in valid_ages:
            errors.append(
                f"Kids mode requires target_age: {valid_ages}, "
                f"got '{model.target_age}'"
            )

    # 3. 브릭 리스트 검증
    if not model.bricks:
        errors.append("Model has no bricks")

    # 4. 개별 브릭 검증
    brick_ids = set()
    for brick in model.bricks:
        # 중복 ID 체크
        if brick.id in brick_ids:
            errors.append(f"Duplicate brick id: '{brick.id}'")
        brick_ids.add(brick.id)

        # 개별 브릭 검증
        brick_errors = validate_brick(brick, parts_db)
        errors.extend(brick_errors)

    return errors


# ============================================
# L2 검증 (물리: 충돌, 부유)
# ============================================

@dataclass
class L2ValidationResult:
    """L2 검증 결과"""
    collisions: List[tuple]      # [(brick1_id, brick2_id), ...]
    floating_bricks: List[str]   # [brick_id, ...]
    warnings: List[str]          # 경고 메시지


def check_collisions(bricks: List[PlacedBrick], parts_db: Dict) -> List[tuple]:
    """
    브릭 간 충돌 검사

    Returns:
        충돌하는 브릭 쌍 리스트 [(id1, id2), ...]
    """
    collisions = []
    bboxes = []

    # 모든 브릭의 바운딩 박스 계산 (part_id도 함께 저장)
    for brick in bricks:
        bbox = get_brick_bbox(brick, parts_db)
        if bbox:
            bboxes.append((brick.id, brick.part_id, bbox))

    # 모든 쌍에 대해 충돌 검사
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            id1, part1, bbox1 = bboxes[i]
            id2, part2, bbox2 = bboxes[j]

            # 슬로프끼리는 tolerance를 더 크게 적용 (A자 지붕 등 허용)
            is_both_slope = part1 in SLOPE_PARTS and part2 in SLOPE_PARTS
            tolerance = SLOPE_TOLERANCE if is_both_slope else 10.0

            if bbox1.intersects(bbox2, tolerance):
                collisions.append((id1, id2))

    return collisions


def _are_bricks_connected(bbox1, bbox2, tolerance: float = 2.0) -> bool:
    """
    두 브릭이 연결되어 있는지 확인 (수직 연결만)

    레고 브릭은 스터드를 통해 수직으로만 연결됨.
    수평으로 맞닿는 것은 물리적 연결이 아님.

    연결 조건:
    - Y축으로 맞닿고 X/Z가 겹침 (위아래로 쌓임)
    """
    # X, Z 겹침 여부
    x_overlap = bbox1.min_x < bbox2.max_x and bbox1.max_x > bbox2.min_x
    z_overlap = bbox1.min_z < bbox2.max_z and bbox1.max_z > bbox2.min_z

    # 수직 연결: Y축으로 맞닿음 (위아래)
    y_touch_top = abs(bbox1.min_y - bbox2.max_y) < tolerance  # bbox1이 bbox2 위
    y_touch_bottom = abs(bbox1.max_y - bbox2.min_y) < tolerance  # bbox1이 bbox2 아래

    if (y_touch_top or y_touch_bottom) and x_overlap and z_overlap:
        return True

    return False


def check_floating(bricks: List[PlacedBrick], parts_db: Dict, ground_y: float = None) -> List[str]:
    """
    부유 브릭 검사 - 연결성 기반 (v2)

    바닥에서 시작해서 연결된 브릭 네트워크를 BFS로 탐색.
    네트워크에 속하지 않은 브릭만 부유로 판정.

    Args:
        bricks: 브릭 리스트
        parts_db: 파츠 DB
        ground_y: 바닥 Y좌표 (None이면 자동 계산 - 가장 아래 브릭 기준)

    Returns:
        부유 중인 브릭 ID 리스트
    """
    from collections import deque

    if not bricks:
        return []

    bboxes = {}
    brick_ids = []

    # 1. 모든 브릭의 바운딩 박스 계산
    for brick in bricks:
        bbox = get_brick_bbox(brick, parts_db)
        if bbox:
            bboxes[brick.id] = bbox
            brick_ids.append(brick.id)

    if not bboxes:
        return []

    # 2. 바닥 Y 자동 계산 (가장 아래 브릭의 max_y)
    if ground_y is None:
        ground_y = max(bbox.max_y for bbox in bboxes.values())

    # 3. 바닥에 닿은 브릭들 찾기 (시작점)
    grounded = set()
    for brick_id, bbox in bboxes.items():
        if bbox.max_y >= ground_y - 2:  # 바닥에 닿음
            grounded.add(brick_id)

    # 바닥에 닿은 브릭이 없으면 전부 부유
    if not grounded:
        return brick_ids

    # 3. 연결 그래프 생성 (인접 리스트)
    adjacency = {bid: [] for bid in brick_ids}

    # O(n^2) 연결 체크 - 브릭 수가 많으면 느릴 수 있음
    id_list = list(bboxes.keys())
    for i in range(len(id_list)):
        for j in range(i + 1, len(id_list)):
            id1, id2 = id_list[i], id_list[j]
            if _are_bricks_connected(bboxes[id1], bboxes[id2]):
                adjacency[id1].append(id2)
                adjacency[id2].append(id1)

    # 4. BFS로 바닥에서 도달 가능한 브릭 찾기
    supported = set(grounded)
    queue = deque(grounded)

    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor not in supported:
                supported.add(neighbor)
                queue.append(neighbor)

    # 5. 도달 불가능한 브릭 = 부유
    floating = [bid for bid in brick_ids if bid not in supported]

    return floating


def validate_physics(
    model: BrickModel,
    parts_db: Dict,
    check_collision: bool = True,
    check_float: bool = True
) -> L2ValidationResult:
    """
    L2 물리 검증 (충돌 + 부유)

    Args:
        model: 검증할 모델
        parts_db: 파츠 DB (bbox 정보 포함)
        check_collision: 충돌 검사 여부
        check_float: 부유 검사 여부

    Returns:
        L2ValidationResult
    """
    collisions = []
    floating_bricks = []
    warnings = []

    # 충돌 검사
    if check_collision:
        collisions = check_collisions(model.bricks, parts_db)
        if collisions:
            warnings.append(f"[L2] {len(collisions)}개 충돌 발견")

    # 부유 검사
    if check_float:
        floating_bricks = check_floating(model.bricks, parts_db)
        if floating_bricks:
            warnings.append(f"[L2] {len(floating_bricks)}개 부유 브릭 발견")

    return L2ValidationResult(
        collisions=collisions,
        floating_bricks=floating_bricks,
        warnings=warnings
    )
