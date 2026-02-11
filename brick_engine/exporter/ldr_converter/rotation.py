"""
LDR Converter - 회전/좌표/바운딩박스 계산

회전 행렬, 파츠 크기 테이블, bbox 계산
"""

from typing import List, Dict, Optional

from .models import PlacedBrick, BBox


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
