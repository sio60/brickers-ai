"""
LDR Converter - JSON BrickModel을 LDraw 포맷(.ldr)으로 변환

작성자: 성빈
작성일: 2026-01-14

LDraw 스펙 참고: docs/LDraw_Reference.md
"""

import json
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime


# ============================================
# 데이터 타입 정의
# ============================================

@dataclass
class Vector3:
    x: float
    y: float
    z: float


@dataclass
class PlacedBrick:
    id: str
    part_id: str
    position: Vector3
    rotation: int  # 0, 90, 180, 270
    color_code: int
    layer: int


@dataclass
class BrickModel:
    model_id: str
    name: str
    mode: str  # 'pro' or 'kids'
    bricks: List[PlacedBrick]
    target_age: Optional[str] = None
    created_at: Optional[str] = None


# ============================================
# 회전 행렬 (Y축 기준)
# ============================================

# LDraw는 -Y가 위쪽인 오른손 좌표계
# 회전 행렬: 3x3 = [a b c / d e f / g h i]
ROTATION_MATRICES = {
    0:   [1, 0, 0, 0, 1, 0, 0, 0, 1],      # 회전 없음
    90:  [0, 0, -1, 0, 1, 0, 1, 0, 0],     # Y축 90도
    180: [-1, 0, 0, 0, 1, 0, 0, 0, -1],    # Y축 180도
    270: [0, 0, 1, 0, 1, 0, -1, 0, 0],     # Y축 270도
}


def get_rotation_matrix(rotation: int) -> List[int]:
    """회전 각도(0, 90, 180, 270)에 해당하는 3x3 행렬 반환"""
    return ROTATION_MATRICES.get(rotation, ROTATION_MATRICES[0])


# ============================================
# 슬로프 파츠 (bbox 충돌 검사 시 특별 처리)
# ============================================

# 슬로프는 경사면이라 bbox가 실제보다 크게 계산됨
# A자 지붕처럼 마주보는 배치 시 bbox 겹침 → false positive 발생
# 슬로프끼리는 tolerance를 더 크게 적용
#
# NOTE: 하드코딩 이유 - MongoDB에 실제 파츠명 없음 (파일명만 저장)
# 추후 MongoDB 데이터 업데이트 시 동적 판단으로 전환 가능
SLOPE_PARTS = {
    '3037', '3038', '3039', '3040',  # Slope 45 2xN
    '3044', '3048', '3049',          # Slope 45 1xN
    '3665', '3660', '3676', '3678',  # Slope Inverted
    '3747', '4286', '4287',          # Slope Triple
    '4460', '4461',                  # Slope 75
    '3298', '3299', '3300',          # Slope 33
    '85984', '60481',                # Slope 30
    '54200', '30363',                # Slope 30 1x1
    '15571', '92946',                # Slope Curved
    '4861', '4856', '4857', '4858',  # Slope 45 (roof wedge)
}

SLOPE_TOLERANCE = 45.0  # 슬로프끼리 충돌 검사 시 tolerance (A자 지붕 허용)


# ============================================
# 바운딩 박스 계산
# ============================================

@dataclass
class BBox:
    """3D 바운딩 박스"""
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float

    def intersects(self, other: 'BBox', tolerance: float = 10.0) -> bool:
        """
        다른 바운딩 박스와 겹치는지 확인

        tolerance: 허용 오차 (LDU 단위)
        - 레고 브릭은 스터드 그리드(20 LDU)에 맞춰 나란히 배치됨
        - 슬로프, 바퀴 등 특수 파츠는 bbox가 실제보다 크게 계산됨
        - 기본값 10.0 (0.5 스터드)으로 정상 배치 허용
        """
        return (
            self.min_x < other.max_x - tolerance and
            self.max_x > other.min_x + tolerance and
            self.min_y < other.max_y - tolerance and
            self.max_y > other.min_y + tolerance and
            self.min_z < other.max_z - tolerance and
            self.max_z > other.min_z + tolerance
        )

    def is_supported_by(self, other: 'BBox', tolerance: float = 2.0) -> bool:
        """다른 박스 위에 올라가 있는지 확인 (Y축 기준)"""
        # 이 브릭의 바닥(max_y)이 다른 브릭의 상단(min_y)과 닿아있고
        # X, Z가 겹치는 영역이 있어야 함
        y_contact = abs(self.max_y - other.min_y) < tolerance
        x_overlap = self.min_x < other.max_x and self.max_x > other.min_x
        z_overlap = self.min_z < other.max_z and self.max_z > other.min_z
        return y_contact and x_overlap and z_overlap


def get_rotated_size(size: List[float], rotation: int) -> List[float]:
    """회전 적용 후 크기 반환 (Y축 회전)"""
    sx, sy, sz = size
    if rotation == 0 or rotation == 180:
        return [sx, sy, sz]
    elif rotation == 90 or rotation == 270:
        return [sz, sy, sx]  # X와 Z 교환
    return [sx, sy, sz]


# ============================================
# 파츠 ID 기반 크기 테이블 (MongoDB 정보 부족 대비)
# ============================================
# 형식: part_id -> (studs_x, studs_z, height)
PART_SIZE_TABLE = {
    # === 브릭 (높이 24) ===
    # 형식: (studs_x, studs_z, height_ldu)
    "3001": (4, 2, 24),   # Brick 2x4
    "3002": (3, 2, 24),   # Brick 2x3
    "3003": (2, 2, 24),   # Brick 2x2
    "3004": (2, 1, 24),   # Brick 1x2 (주의: 1x4 아님!)
    "3005": (1, 1, 24),   # Brick 1x1
    "3006": (10, 2, 24),  # Brick 2x10
    "3007": (8, 2, 24),   # Brick 2x8
    "3008": (8, 1, 24),   # Brick 1x8
    "3009": (6, 1, 24),   # Brick 1x6
    "3010": (4, 1, 24),   # Brick 1x4
    "3622": (3, 1, 24),   # Brick 1x3
    "3245": (12, 2, 24),  # Brick 2x12
    "2456": (6, 2, 24),   # Brick 2x6
    "3062b": (1, 1, 24),  # Brick Round 1x1

    # === 플레이트 (높이 8) ===
    "3020": (4, 2, 8),    # Plate 2x4
    "3021": (3, 2, 8),    # Plate 2x3
    "3022": (2, 2, 8),    # Plate 2x2
    "3023": (2, 1, 8),    # Plate 1x2
    "3024": (1, 1, 8),    # Plate 1x1
    "3026": (6, 2, 8),    # Plate 2x6
    "3028": (12, 6, 8),   # Plate 6x12
    "3029": (12, 4, 8),   # Plate 4x12
    "3030": (10, 4, 8),   # Plate 4x10
    "3031": (6, 4, 8),    # Plate 4x6
    "3032": (4, 4, 8),    # Plate 4x4
    "3033": (10, 6, 8),   # Plate 6x10
    "3034": (8, 2, 8),    # Plate 2x8
    "3035": (8, 4, 8),    # Plate 4x8
    "3036": (8, 6, 8),    # Plate 6x8
    "3037": (10, 4, 8),   # Plate 4x10
    "3460": (8, 1, 8),    # Plate 1x8
    "3666": (6, 1, 8),    # Plate 1x6
    "3710": (4, 1, 8),    # Plate 1x4
    "3795": (2, 2, 8),    # Plate 2x2 corner

    # === 슬로프 (높이 24, 다양) ===
    "3039": (2, 2, 24),   # Slope 45 2x2
    "3040": (1, 2, 24),   # Slope 45 2x1 (Z방향으로 긺)
    "3040a": (1, 2, 24),
    "3040b": (1, 2, 24),
    "3044": (3, 1, 24),   # Slope 45 1x3
    "3037": (4, 2, 24),   # Slope 45 2x4
    "3038": (3, 2, 24),   # Slope 45 2x3
    "3298": (3, 2, 24),   # Slope 33 2x3
    "3300": (4, 2, 24),   # Slope 33 2x4
    "3678b": (4, 2, 24),  # Slope 65 2x4
    "4286": (3, 1, 24),   # Slope 33 1x3
    "4287": (3, 1, 8),    # Slope 33 1x3 inverted (낮음)
    "3665": (2, 1, 8),    # Slope 45 1x2 inverted

    # === 타일 (높이 8) ===
    "3068b": (2, 2, 8),   # Tile 2x2
    "3069b": (2, 1, 8),   # Tile 1x2
    "3070b": (1, 1, 8),   # Tile 1x1
    "6636": (6, 1, 8),    # Tile 1x6
    "2431": (4, 1, 8),    # Tile 1x4
    "63864": (3, 1, 8),   # Tile 1x3

    # === 바퀴/타이어 (특수 - 원형이라 bbox 계산 다름) ===
    "3641": (2, 2, 14),    # Wheel 11mm D. x 8mm (원형, Z가 두께)
    "3137c01": (4, 2, 32), # Wheel Holder 2x4 with Wheels
    "4624": (2, 2, 12),    # Wheel 8mm D. x 6mm
    "6014": (2, 2, 20),    # Wheel 11mm D. x 12mm
    "6015": (2, 2, 24),    # Wheel 18mm D. x 8mm

    # === 기타 특수 파츠 ===
    "3942": (2, 2, 24),   # Cone 2x2x2
    "3062": (1, 1, 24),   # Round Brick 1x1
    "6143": (2, 2, 8),    # Round Brick 2x2 Dome
}


def get_brick_bbox(brick: PlacedBrick, parts_db: Dict) -> Optional[BBox]:
    """
    브릭의 실제 바운딩 박스 계산

    LDraw 표준 단위:
    - 1 스터드 = 20 LDU
    - 브릭 높이 = 24 LDU (스터드 제외)
    - 플레이트 높이 = 8 LDU (스터드 제외)
    - 스터드 높이 = 4 LDU

    우선순위:
    1. PART_SIZE_TABLE (특수 파츠 - 바퀴 등)
    2. MongoDB bbox.size (일반 파츠)
    3. 기본값 2x2 브릭
    """
    part_id = brick.part_id.lower()
    part_info = parts_db.get(part_id)
    bbox_size = None

    # 1순위: PART_SIZE_TABLE (특수 파츠는 MongoDB bbox가 부정확)
    if part_id in PART_SIZE_TABLE:
        studs_x, studs_z, height = PART_SIZE_TABLE[part_id]
        width_x = studs_x * 20
        width_z = studs_z * 20
        bbox_size = [width_x, height, width_z]

    # 2순위: MongoDB bbox.size
    elif part_info and 'bbox' in part_info:
        bbox_data = part_info['bbox']
        if isinstance(bbox_data, dict) and 'size' in bbox_data:
            size = bbox_data['size']
            if isinstance(size, list) and len(size) >= 3:
                # MongoDB bbox는 스터드 포함 높이 (+4)
                width_x = size[0]
                height = size[1] - 4  # 스터드 높이 제외
                width_z = size[2]
                bbox_size = [width_x, height, width_z]

    # 3순위: 기본값 (2x2 브릭)
    if bbox_size is None:
        bbox_size = [40, 24, 40]

    # 회전 적용
    size = get_rotated_size(bbox_size, brick.rotation)
    half_x, half_z = size[0] / 2, size[2] / 2
    height = size[1]

    # LDraw 좌표계: -Y가 위쪽
    # position은 브릭의 기준점 (보통 상단 중앙)
    x, y, z = brick.position.x, brick.position.y, brick.position.z

    return BBox(
        min_x=x - half_x,
        max_x=x + half_x,
        min_y=y,              # 상단 (LDraw에서 작은 Y가 위)
        max_y=y + height,     # 하단
        min_z=z - half_z,
        max_z=z + half_z
    )


# ============================================
# L1 검증 (스키마/타입)
# ============================================

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


def _are_bricks_connected(bbox1: BBox, bbox2: BBox, tolerance: float = 2.0) -> bool:
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


# ============================================
# LDR 변환 함수
# ============================================

def brick_to_ldr_line(brick: PlacedBrick, parts_db: Dict) -> str:
    """
    PlacedBrick을 LDR Line Type 1 형식으로 변환

    형식: 1 <색상> <x> <y> <z> <a> <b> <c> <d> <e> <f> <g> <h> <i> <파일명>
    """
    # 파츠 정보 조회
    part_info = parts_db.get(brick.part_id)
    if not part_info:
        raise ValueError(f"Unknown part ID: {brick.part_id}")

    ldraw_file = part_info['ldrawFile']

    # 회전 행렬
    matrix = get_rotation_matrix(brick.rotation)

    # LDR 라인 생성
    # 좌표는 이미 LDU 단위로 들어온다고 가정
    line = f"1 {brick.color_code} {brick.position.x:.0f} {brick.position.y:.0f} {brick.position.z:.0f} "
    line += " ".join(str(m) for m in matrix)
    line += f" {ldraw_file}"

    return line


# STEP 모드 옵션
STEP_MODE_NONE = 'none'    # STEP 없음
STEP_MODE_LAYER = 'layer'  # 레이어마다 STEP
STEP_MODE_BRICK = 'brick'  # 브릭마다 STEP
VALID_STEP_MODES = [STEP_MODE_NONE, STEP_MODE_LAYER, STEP_MODE_BRICK]


def model_to_ldr(
    model: BrickModel,
    parts_db: Dict,
    skip_validation: bool = False,
    skip_physics: bool = False,
    step_mode: str = STEP_MODE_NONE
) -> str:
    """
    BrickModel 전체를 LDR 파일 내용으로 변환

    Args:
        model: 변환할 브릭 모델
        parts_db: 파츠 데이터베이스 (bbox 정보 포함 권장)
        skip_validation: True면 L1 검증 스킵 (디버깅용, 기본값 False)
        skip_physics: True면 L2 물리 검증 스킵 (기본값 False)
        step_mode: STEP 출력 모드
            - 'none': STEP 없음 (기본값)
            - 'layer': 레이어마다 0 STEP 추가
            - 'brick': 브릭마다 0 STEP 추가 (조립 설명서용)

    Raises:
        ValidationError: 검증 실패 시 (skip_validation=False일 때)
        ValueError: 잘못된 step_mode
    """
    # step_mode 검증
    if step_mode not in VALID_STEP_MODES:
        raise ValueError(f"Invalid step_mode: '{step_mode}', must be one of {VALID_STEP_MODES}")

    # L1 검증 (스킵 옵션 없으면 항상 실행)
    if not skip_validation:
        errors = validate_model(model, parts_db)
        if errors:
            raise ValidationError(errors)

    # L2 물리 검증 (충돌, 부유) - 경고만 출력, 에러 아님
    l2_warnings = []
    if not skip_physics:
        physics_result = validate_physics(model, parts_db)
        if physics_result.collisions:
            l2_warnings.append(f"[L2 Warning] {len(physics_result.collisions)}개 충돌 발견")
            for c in physics_result.collisions[:3]:
                l2_warnings.append(f"  - {c[0]} <-> {c[1]}")
            if len(physics_result.collisions) > 3:
                l2_warnings.append(f"  ... 외 {len(physics_result.collisions) - 3}개")
        if physics_result.floating_bricks:
            l2_warnings.append(f"[L2 Warning] {len(physics_result.floating_bricks)}개 부유 브릭 발견")
            for f in physics_result.floating_bricks[:3]:
                l2_warnings.append(f"  - {f}")
            if len(physics_result.floating_bricks) > 3:
                l2_warnings.append(f"  ... 외 {len(physics_result.floating_bricks) - 3}개")
        # 경고 출력 (에러 아님)
        for w in l2_warnings:
            print(w)

    lines = []

    # 헤더 주석
    lines.append(f"0 {model.name}")
    lines.append(f"0 Name: {model.model_id}.ldr")
    lines.append(f"0 Author: Brick CoScientist")
    lines.append(f"0 Mode: {model.mode}")
    if model.target_age:
        lines.append(f"0 TargetAge: {model.target_age}")
    lines.append(f"0 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")  # 빈 줄

    # 레이어별로 정렬 (아래부터 위로)
    sorted_bricks = sorted(model.bricks, key=lambda b: b.layer)

    current_layer = -1
    for i, brick in enumerate(sorted_bricks):
        # 레이어 변경 시
        if brick.layer != current_layer:
            # 이전 레이어 끝에 STEP 추가 (layer 모드, 첫 레이어 제외)
            if step_mode == STEP_MODE_LAYER and current_layer != -1:
                lines.append("")
                lines.append("0 STEP")
                lines.append("")

            current_layer = brick.layer
            lines.append(f"0 // Layer {current_layer}")

        # 브릭 라인 추가
        try:
            ldr_line = brick_to_ldr_line(brick, parts_db)
            lines.append(ldr_line)

            # brick 모드: 매 브릭마다 STEP (마지막 브릭 제외)
            if step_mode == STEP_MODE_BRICK and i < len(sorted_bricks) - 1:
                lines.append("")
                lines.append("0 STEP")
                lines.append("")
        except ValueError as e:
            l2_warnings.append(f"[WARNING] Skipped brick: {e}")
            continue

    # 마지막 STEP 추가 (layer 모드)
    if step_mode == STEP_MODE_LAYER and len(sorted_bricks) > 0:
        lines.append("")
        lines.append("0 STEP")

    return "\n".join(lines)


def model_to_ldr_unsafe(model: BrickModel, parts_db: Dict) -> str:
    """
    검증 없이 LDR 변환 (디버깅/테스트용)

    주의: 프로덕션에서 사용 금지. 잘못된 데이터도 그대로 변환됨.
    """
    return model_to_ldr(model, parts_db, skip_validation=True)


def save_ldr_file(content: str, filepath: str):
    """LDR 파일 저장"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"LDR 파일 저장 완료: {filepath}")


# ============================================
# JSON 파싱 헬퍼
# ============================================

def load_parts_db(filepath: str) -> Dict:
    """BrickParts_Database.json 로드 후 partId로 인덱싱"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # partId를 키로 하는 딕셔너리 생성
    parts_dict = {}
    for part in data['parts']:
        parts_dict[part['partId']] = part

    return parts_dict


def parse_brick_model(json_data: dict) -> BrickModel:
    """JSON 딕셔너리를 BrickModel로 변환"""
    bricks = []
    for b in json_data.get('bricks', []):
        pos = b['position']
        brick = PlacedBrick(
            id=b['id'],
            part_id=b['partId'],
            position=Vector3(x=pos['x'], y=pos['y'], z=pos['z']),
            rotation=b['rotation'],
            color_code=b['colorCode'],
            layer=b['layer']
        )
        bricks.append(brick)

    return BrickModel(
        model_id=json_data['modelId'],
        name=json_data['name'],
        mode=json_data['mode'],
        bricks=bricks,
        target_age=json_data.get('targetAge'),
        created_at=json_data.get('createdAt')
    )


# ============================================
# LDR 파서 (LDR → BrickModel)
# ============================================

def matrix_to_rotation(matrix: List[float]) -> int:
    """
    회전 행렬을 각도(0, 90, 180, 270)로 변환

    가장 가까운 표준 회전 각도로 근사
    """
    # 표준 회전 행렬들
    standard_matrices = {
        0:   [1, 0, 0, 0, 1, 0, 0, 0, 1],
        90:  [0, 0, -1, 0, 1, 0, 1, 0, 0],
        180: [-1, 0, 0, 0, 1, 0, 0, 0, -1],
        270: [0, 0, 1, 0, 1, 0, -1, 0, 0],
    }

    # 가장 가까운 행렬 찾기
    best_match = 0
    best_diff = float('inf')

    for angle, std_matrix in standard_matrices.items():
        diff = sum(abs(a - b) for a, b in zip(matrix, std_matrix))
        if diff < best_diff:
            best_diff = diff
            best_match = angle

    return best_match


def parse_ldr_line(line: str) -> Optional[dict]:
    """
    LDR Line Type 1 파싱

    형식: 1 <색상> <x> <y> <z> <a> <b> <c> <d> <e> <f> <g> <h> <i> <파일명>

    Returns:
        파싱된 정보 딕셔너리 또는 None
    """
    line = line.strip()
    if not line or line.startswith('0'):
        return None

    parts = line.split()
    if len(parts) < 15 or parts[0] != '1':
        return None

    try:
        color = int(parts[1])
        x = float(parts[2])
        y = float(parts[3])
        z = float(parts[4])

        # 회전 행렬 (9개 값)
        matrix = [float(parts[i]) for i in range(5, 14)]

        # 파츠 파일명
        part_file = parts[14]

        # part_id 추출 (확장자 제거)
        part_id = part_file.replace('.dat', '').replace('.DAT', '')

        return {
            'color': color,
            'x': x,
            'y': y,
            'z': z,
            'matrix': matrix,
            'rotation': matrix_to_rotation(matrix),
            'part_file': part_file,
            'part_id': part_id,
        }
    except (ValueError, IndexError):
        return None


def parse_ldr_file(filepath: str) -> dict:
    """
    LDR 파일을 파싱하여 딕셔너리로 반환

    Returns:
        {
            'name': 모델 이름,
            'bricks': [파싱된 브릭 정보 리스트],
            'comments': [주석 리스트],
        }
    """
    result = {
        'name': '',
        'bricks': [],
        'comments': [],
    }

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()

            # 주석 (Line Type 0)
            if line.startswith('0 '):
                comment = line[2:]
                result['comments'].append(comment)
                # 첫 번째 주석을 이름으로
                if not result['name'] and not comment.startswith('!'):
                    result['name'] = comment

            # 파츠 (Line Type 1)
            elif line.startswith('1 '):
                parsed = parse_ldr_line(line)
                if parsed:
                    result['bricks'].append(parsed)

    return result


def ldr_to_brick_model(
    filepath: str,
    model_id: Optional[str] = None,
    mode: str = 'pro'
) -> BrickModel:
    """
    LDR 파일을 BrickModel로 변환

    Args:
        filepath: LDR 파일 경로
        model_id: 모델 ID (없으면 파일명 사용)
        mode: 'pro' 또는 'kids'

    Returns:
        BrickModel 객체
    """
    from pathlib import Path

    parsed = parse_ldr_file(filepath)

    if not model_id:
        model_id = Path(filepath).stem

    bricks = []
    for i, b in enumerate(parsed['bricks']):
        brick = PlacedBrick(
            id=f"b{i+1:03d}",
            part_id=b['part_id'].lower(),
            position=Vector3(x=b['x'], y=b['y'], z=b['z']),
            rotation=b['rotation'],
            color_code=b['color'],
            layer=0  # LDR에서는 레이어 정보 없음, 나중에 Y좌표로 계산 가능
        )
        bricks.append(brick)

    # Y좌표로 레이어 자동 계산
    if bricks:
        # Y좌표 기준 정렬 (LDraw: 작은 Y가 위)
        y_values = sorted(set(b.position.y for b in bricks))
        y_to_layer = {y: i for i, y in enumerate(y_values)}
        for brick in bricks:
            brick.layer = y_to_layer[brick.position.y]

    return BrickModel(
        model_id=model_id,
        name=parsed['name'] or model_id,
        mode=mode,
        bricks=bricks
    )


def change_colors(model: BrickModel, color_map: Dict[int, int]) -> BrickModel:
    """
    BrickModel의 색상 변경

    Args:
        model: 원본 모델
        color_map: {원본색상: 새색상} 매핑

    Returns:
        색상이 변경된 새 BrickModel
    """
    new_bricks = []
    for brick in model.bricks:
        new_color = color_map.get(brick.color_code, brick.color_code)
        new_brick = PlacedBrick(
            id=brick.id,
            part_id=brick.part_id,
            position=brick.position,
            rotation=brick.rotation,
            color_code=new_color,
            layer=brick.layer
        )
        new_bricks.append(new_brick)

    return BrickModel(
        model_id=model.model_id,
        name=model.name,
        mode=model.mode,
        bricks=new_bricks,
        target_age=model.target_age,
        created_at=model.created_at
    )


# ============================================
# 메인 실행
# ============================================

if __name__ == "__main__":
    print("LDR Converter 모듈 로드 완료")
    print("사용법:")
    print("  from ldr_converter import model_to_ldr, load_parts_db")
    print("  from ldr_converter import ldr_to_brick_model, change_colors")
