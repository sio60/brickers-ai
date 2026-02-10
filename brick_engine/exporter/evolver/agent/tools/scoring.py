"""배치 점수 계산 및 최적 위치 선택"""
from typing import Dict, List, Set, Tuple, Optional

from .constants import (
    LDU_PER_STUD, DEFAULT_SUPPORT_RATIO, OPTIMAL_BOND_MIN, OPTIMAL_BOND_MAX,
)
from .collision import _check_collision_simple


def calc_support_ratio(x: float, y: float, z: float, part_id: str,
                       parts_db: Dict, occupied: Set) -> float:
    """
    브릭 아래층의 지지 비율 계산 (glb_to_ldr_embedded.py 참고)

    Args:
        x, y, z: 브릭 위치 (LDU)
        part_id: 파츠 ID
        parts_db: 파츠 DB
        occupied: 점유 셀 Set

    Returns:
        0.0 ~ 1.0 사이의 지지 비율
    """
    cell_size = LDU_PER_STUD  # 20

    # 파츠 크기 가져오기
    part_info = parts_db.get(part_id.lower().replace('.dat', ''), {})
    width = part_info.get('width', 1)  # 스터드 단위
    depth = part_info.get('depth', 1)

    cx = int(round(x / cell_size))
    cy = int(round(y / cell_size))
    cz = int(round(z / cell_size))

    # 바닥(y=0)이면 100% 지지
    if cy >= 0:
        return 1.0

    # 아래층 셀 체크
    below_y = cy + 1  # LDraw에서 Y가 클수록 아래
    total_cells = width * depth
    supported_cells = 0

    for dx in range(width):
        for dz in range(depth):
            if (cx + dx, below_y, cz + dz) in occupied:
                supported_cells += 1

    return supported_cells / total_cells if total_cells > 0 else 0.0


def calc_bond_score(x: float, y: float, z: float, part_id: str,
                    parts_db: Dict, occupied: Set) -> float:
    """
    브릭 결합 점수 계산 (glb_to_ldr_embedded.py 참고)

    30-70% 겹침이 최적 (인터락 품질)

    Args:
        x, y, z: 브릭 위치 (LDU)
        part_id: 파츠 ID
        parts_db: 파츠 DB
        occupied: 점유 셀 Set

    Returns:
        0.5 (지지 없음), 1.0 (지지 있음), 1.5 (최적 겹침)
    """
    cell_size = LDU_PER_STUD
    cy = int(round(y / cell_size))

    # 바닥이면 1.0
    if cy >= 0:
        return 1.0

    support_ratio = calc_support_ratio(x, y, z, part_id, parts_db, occupied)

    # 30-70% 겹침이 최적
    if OPTIMAL_BOND_MIN <= support_ratio <= OPTIMAL_BOND_MAX:
        return 1.5
    elif support_ratio > 0:
        return 1.0
    return 0.5


def can_place_brick(x: float, y: float, z: float, part_id: str,
                    parts_db: Dict, occupied: Set,
                    support_ratio: float = DEFAULT_SUPPORT_RATIO) -> Tuple[bool, str]:
    """
    브릭 배치 가능 여부 검사 (glb_to_ldr_embedded.py 참고)

    충돌 체크 + 지지율 체크

    Args:
        x, y, z: 브릭 위치 (LDU)
        part_id: 파츠 ID
        parts_db: 파츠 DB
        occupied: 점유 셀 Set
        support_ratio: 최소 지지율 (기본 0.3)

    Returns:
        (can_place: bool, reason: str)
    """
    # 1. 충돌 체크
    if _check_collision_simple(None, x, y, z, part_id, parts_db, occupied):
        return False, "collision"

    # 2. 지지율 체크
    actual_ratio = calc_support_ratio(x, y, z, part_id, parts_db, occupied)
    if actual_ratio < support_ratio:
        return False, f"insufficient_support ({actual_ratio:.1%} < {support_ratio:.0%})"

    return True, "ok"


def find_best_placement(candidates: List[Dict], parts_db: Dict,
                        occupied: Set) -> Optional[Dict]:
    """
    후보 위치 중 최적 배치 위치 선택 (bond_score 기준)

    Args:
        candidates: [{"x": ..., "y": ..., "z": ..., "part_id": ...}, ...]
        parts_db: 파츠 DB
        occupied: 점유 셀 Set

    Returns:
        최적 후보 또는 None
    """
    valid_candidates = []

    for cand in candidates:
        can_place, reason = can_place_brick(
            cand["x"], cand["y"], cand["z"],
            cand.get("part_id", "3005"),
            parts_db, occupied
        )

        if can_place:
            score = calc_bond_score(
                cand["x"], cand["y"], cand["z"],
                cand.get("part_id", "3005"),
                parts_db, occupied
            )
            valid_candidates.append({**cand, "bond_score": score})

    if not valid_candidates:
        return None

    # bond_score 높은 순 정렬
    valid_candidates.sort(key=lambda c: c["bond_score"], reverse=True)
    return valid_candidates[0]
