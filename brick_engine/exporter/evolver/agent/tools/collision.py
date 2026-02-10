"""충돌 검사 / 점유 맵 헬퍼 함수"""
from typing import Dict, Set, Tuple, TYPE_CHECKING

from .constants import (
    BRICK_HEIGHT, PLATE_HEIGHT, LDU_PER_STUD, CELL_SIZE, PLATE_PART_IDS,
)

if TYPE_CHECKING:
    from ldr_converter import BrickModel


def _build_occupancy_set(model: "BrickModel", parts_db: Dict) -> Set[Tuple[int, int, int]]:
    """기존 브릭들의 점유 셀 계산"""
    from ldr_converter import get_brick_bbox
    occupied = set()

    for brick in model.bricks:
        bbox = get_brick_bbox(brick, parts_db)
        if not bbox:
            continue

        # 셀 범위 계산
        min_cx = int(bbox.min_x // CELL_SIZE)
        max_cx = int(bbox.max_x // CELL_SIZE)
        min_cy = int(bbox.min_y // CELL_SIZE)
        max_cy = int(bbox.max_y // CELL_SIZE)
        min_cz = int(bbox.min_z // CELL_SIZE)
        max_cz = int(bbox.max_z // CELL_SIZE)

        for cx in range(min_cx, max_cx + 1):
            for cy in range(min_cy, max_cy + 1):
                for cz in range(min_cz, max_cz + 1):
                    occupied.add((cx, cy, cz))

    return occupied


def _mark_occupied(occupied: Set, x: float, y: float, z: float,
                   height: int, width_studs: int = 1, depth_studs: int = 1):
    """
    셀을 점유 상태로 마킹 (bbox 기반 - _build_occupancy_set과 동일한 방식)

    Args:
        occupied: 점유 셀 Set
        x, y, z: 브릭 위치 (LDU)
        height: 브릭 높이 (LDU)
        width_studs: 브릭 너비 (스터드 단위, 기본 1)
        depth_studs: 브릭 깊이 (스터드 단위, 기본 1)
    """
    # bbox 계산 (LDraw 좌표계: 브릭 중심이 position)
    half_width = (width_studs * LDU_PER_STUD) / 2
    half_depth = (depth_studs * LDU_PER_STUD) / 2

    min_x = x - half_width
    max_x = x + half_width
    min_y = y
    max_y = y + height
    min_z = z - half_depth
    max_z = z + half_depth

    # 셀 범위 계산
    min_cx = int(min_x // CELL_SIZE)
    max_cx = int(max_x // CELL_SIZE)
    min_cy = int(min_y // CELL_SIZE)
    max_cy = int(max_y // CELL_SIZE)
    min_cz = int(min_z // CELL_SIZE)
    max_cz = int(max_z // CELL_SIZE)

    for cx in range(min_cx, max_cx + 1):
        for cy in range(min_cy, max_cy + 1):
            for cz in range(min_cz, max_cz + 1):
                occupied.add((cx, cy, cz))


def _check_collision_simple(model: "BrickModel", x: float, y: float, z: float,
                            part_id: str, parts_db: Dict,
                            occupied: Set = None) -> bool:
    """
    충돌 체크 (bbox 기반 - _build_occupancy_set과 동일한 방식)

    Args:
        model: BrickModel (occupied 없을 때 사용)
        x, y, z: 브릭 위치 (LDU)
        part_id: 파츠 ID
        parts_db: 파츠 DB
        occupied: 점유 셀 Set (None이면 새로 계산)

    Returns:
        True if collision exists
    """
    if occupied is None:
        occupied = _build_occupancy_set(model, parts_db)

    # 파츠 크기 (LDU 단위로 변환)
    part_info = parts_db.get(part_id.lower().replace('.dat', ''), {})
    width_studs = part_info.get('width', 1)
    depth_studs = part_info.get('depth', 1)
    height = PLATE_HEIGHT if part_id.lower() in PLATE_PART_IDS else BRICK_HEIGHT

    # bbox 계산 (LDraw 좌표계: 브릭 중심이 position)
    # width는 X 방향, depth는 Z 방향
    half_width = (width_studs * LDU_PER_STUD) / 2
    half_depth = (depth_studs * LDU_PER_STUD) / 2

    min_x = x - half_width
    max_x = x + half_width
    min_y = y  # LDraw에서 Y는 아래가 양수, position.y가 브릭 상단
    max_y = y + height
    min_z = z - half_depth
    max_z = z + half_depth

    # 셀 범위 계산 (_build_occupancy_set과 동일하게)
    min_cx = int(min_x // CELL_SIZE)
    max_cx = int(max_x // CELL_SIZE)
    min_cy = int(min_y // CELL_SIZE)
    max_cy = int(max_y // CELL_SIZE)
    min_cz = int(min_z // CELL_SIZE)
    max_cz = int(max_z // CELL_SIZE)

    # 모든 셀 체크
    for cx in range(min_cx, max_cx + 1):
        for cy in range(min_cy, max_cy + 1):
            for cz in range(min_cz, max_cz + 1):
                if (cx, cy, cz) in occupied:
                    return True

    return False
