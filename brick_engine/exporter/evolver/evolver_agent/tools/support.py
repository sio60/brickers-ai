"""지지대 탐색 및 브릭 추가 검증 함수"""
from collections import deque
from typing import Dict, Any, List, Set, TYPE_CHECKING

from .constants import (
    BRICK_HEIGHT, PLATE_HEIGHT, LDU_PER_STUD, PLATE_PART_IDS,
    GROUND_TOLERANCE, KIDS_MIN_PART_SIZE, ADULT_DEFAULT_PART,
)
from .collision import _build_occupancy_set, _mark_occupied
from .scoring import calc_bond_score, can_place_brick
from .model_ops import add_brick, get_model_state

if TYPE_CHECKING:
    from ldr_converter import BrickModel


def find_nearby_stable_bricks(model, floating_brick, parts_db, search_radius=60):
    """
    부유 브릭 주변의 안정된 브릭 찾기
    Returns: 안정 브릭 리스트 (거리순 정렬)
    """
    from ldr_converter import get_brick_bbox

    # 1. 모든 브릭의 bbox 계산
    bboxes = {}
    for b in model.bricks:
        bbox = get_brick_bbox(b, parts_db)
        if bbox:
            bboxes[b.id] = {"brick": b, "bbox": bbox}

    if not bboxes:
        return []

    # 2. 바닥에 연결된 브릭 찾기 (BFS)
    ground_y = max(bb["bbox"].max_y for bb in bboxes.values())
    grounded = set()
    for bid, bb in bboxes.items():
        if bb["bbox"].max_y >= ground_y - GROUND_TOLERANCE:
            grounded.add(bid)

    # 인접 관계
    adjacency = {bid: [] for bid in bboxes}
    ids = list(bboxes.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id1, id2 = ids[i], ids[j]
            bb1, bb2 = bboxes[id1]["bbox"], bboxes[id2]["bbox"]
            x_overlap = bb1.min_x < bb2.max_x and bb1.max_x > bb2.min_x
            z_overlap = bb1.min_z < bb2.max_z and bb1.max_z > bb2.min_z
            y_touch = abs(bb1.min_y - bb2.max_y) < 2 or abs(bb1.max_y - bb2.min_y) < 2
            if y_touch and x_overlap and z_overlap:
                adjacency[id1].append(id2)
                adjacency[id2].append(id1)

    # BFS로 안정 브릭 찾기
    stable = set(grounded)
    queue = deque(grounded)
    while queue:
        current = queue.popleft()
        for neighbor in adjacency.get(current, []):
            if neighbor not in stable:
                stable.add(neighbor)
                queue.append(neighbor)

    # 3. 부유 브릭 주변의 안정 브릭 찾기
    if floating_brick.id not in bboxes:
        return []

    f_bbox = bboxes[floating_brick.id]["bbox"]
    f_center_x = (f_bbox.min_x + f_bbox.max_x) / 2
    f_center_z = (f_bbox.min_z + f_bbox.max_z) / 2

    nearby = []
    for bid in stable:
        if bid == floating_brick.id:
            continue
        bb = bboxes[bid]["bbox"]
        s_center_x = (bb.min_x + bb.max_x) / 2
        s_center_z = (bb.min_z + bb.max_z) / 2

        dist = ((f_center_x - s_center_x) ** 2 + (f_center_z - s_center_z) ** 2) ** 0.5
        if dist <= search_radius:
            nearby.append({
                "brick": bboxes[bid]["brick"],
                "bbox": bb,
                "distance": dist
            })

    # 거리순 정렬
    nearby.sort(key=lambda x: x["distance"])
    return nearby


def generate_support_candidates(floating_brick, nearby_stable, parts_db,
                                 kids_mode: bool = False):
    """
    지지대 후보 위치 생성

    Args:
        floating_brick: 부유 브릭
        nearby_stable: 주변 안정 브릭 리스트
        parts_db: 파츠 DB
        kids_mode: Kids Mode일 때 2x2 이상만 사용

    Returns: 후보 위치 리스트 [{x, y, z, part_id, color}, ...]
    """
    from ldr_converter import get_brick_bbox

    candidates = []
    f_bbox = get_brick_bbox(floating_brick, parts_db)
    if not f_bbox:
        return []

    # Kids Mode: 2x2 이상만 사용 (안전 기준)
    default_part = KIDS_MIN_PART_SIZE if kids_mode else ADULT_DEFAULT_PART
    bridge_part = "3003" if kids_mode else "3004"  # 2x2 vs 1x2

    # 부유 브릭 아래에 지지대 위치 계산
    # LDraw에서 Y가 클수록 아래
    support_y = f_bbox.max_y  # 부유 브릭 바닥 바로 아래

    # 스터드 정렬 (20 LDU 단위)
    def snap_to_stud(val):
        return round(val / LDU_PER_STUD) * LDU_PER_STUD

    # 후보 1: 부유 브릭 바로 아래
    candidates.append({
        "x": snap_to_stud(floating_brick.position.x),
        "y": support_y,
        "z": snap_to_stud(floating_brick.position.z),
        "part_id": default_part,
        "color": floating_brick.color_code,
        "description": "directly below floating brick"
    })

    # 후보 2-4: 주변 안정 브릭 방향으로
    for i, stable_info in enumerate(nearby_stable[:3]):
        s_brick = stable_info["brick"]
        s_bbox = stable_info["bbox"]

        # 안정 브릭 상단에서 부유 브릭 방향으로
        mid_x = snap_to_stud((floating_brick.position.x + s_brick.position.x) / 2)
        mid_z = snap_to_stud((floating_brick.position.z + s_brick.position.z) / 2)

        candidates.append({
            "x": mid_x,
            "y": s_bbox.min_y,  # 안정 브릭 위
            "z": mid_z,
            "part_id": bridge_part,
            "color": s_brick.color_code,
            "description": f"bridge toward stable brick {s_brick.id}"
        })

    return candidates


def try_add_brick_with_validation(model: "BrickModel", candidate: Dict,
                                   parts_db: Dict,
                                   occupied: Set = None) -> Dict[str, Any]:
    """
    브릭 추가 시도 + 검증 (can_place_brick 사용)

    Args:
        model: BrickModel
        candidate: {"x", "y", "z", "part_id", "color", ...}
        parts_db: 파츠 DB
        occupied: 점유 셀 Set (None이면 새로 계산)

    Returns:
        {"success": bool, "brick_id": str, "floating_count": int, ...}
    """
    # 점유 맵 생성 (없으면)
    if occupied is None:
        occupied = _build_occupancy_set(model, parts_db)

    part_id = candidate.get("part_id", "3005")
    x, y, z = candidate["x"], candidate["y"], candidate["z"]

    # 1. 배치 가능 여부 검사 (충돌 + 지지율)
    can_place, reason = can_place_brick(x, y, z, part_id, parts_db, occupied)
    if not can_place:
        return {"success": False, "reason": reason}

    # 2. bond_score 계산
    bond = calc_bond_score(x, y, z, part_id, parts_db, occupied)

    # 3. 브릭 추가
    result = add_brick(
        model,
        part_id,
        x, y, z,
        candidate.get("color", 15),
        candidate.get("rotation", 0)
    )

    if not result["success"]:
        return {"success": False, "reason": "add_brick failed"}

    brick_id = result["brick_id"]

    # 4. 점유 맵 업데이트
    part_info = parts_db.get(part_id.lower().replace('.dat', ''), {})
    width = part_info.get('width', 1)
    depth = part_info.get('depth', 1)
    height = PLATE_HEIGHT if part_id.lower() in PLATE_PART_IDS else BRICK_HEIGHT
    _mark_occupied(occupied, x, y, z, height, width, depth)

    # 5. 검증
    state = get_model_state(model, parts_db)

    return {
        "success": True,
        "brick_id": brick_id,
        "bond_score": bond,
        "floating_count": state["floating_count"],
        "collision_count": state["collision_count"],
        "occupied": occupied
    }
