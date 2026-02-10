"""대칭 분석 및 고립 브릭 탐지/삭제"""
from typing import Dict, Any, List, TYPE_CHECKING

from .constants import (
    BRICK_HEIGHT, PLATE_HEIGHT, LDU_PER_STUD, PLATE_PART_IDS,
    SYMMETRY_TOLERANCE, SYMMETRY_CENTER_MARGIN, SPARSE_LAYER_THRESHOLD,
)
from .collision import _build_occupancy_set, _mark_occupied, _check_collision_simple

if TYPE_CHECKING:
    from ldr_converter import BrickModel


def analyze_symmetry(model: "BrickModel", parts_db: Dict) -> List[Dict]:
    """
    좌우 대칭 분석 - 한쪽에만 있는 브릭 찾기

    Returns:
        [{'missing_side': 'left'|'right', 'mirror_brick': brick,
          'suggested_pos': (x,y,z), 'part_id': str, 'color': int}, ...]
    """
    if not model.bricks:
        return []

    margin = SYMMETRY_CENTER_MARGIN
    tolerance = SYMMETRY_TOLERANCE

    # Y 레이어별 브릭 수 카운트 + 최상단 레이어 판단
    y_layer_count = {}
    for b in model.bricks:
        y_key = round(b.position.y / tolerance) * tolerance
        y_layer_count[y_key] = y_layer_count.get(y_key, 0) + 1

    # 최상단 Y 찾기 (LDraw에서 Y가 음수일수록 위)
    min_y = min(b.position.y for b in model.bricks)

    def is_accent(brick):
        """악센트/장식인지 판단: 최상단 + 희소일 때만"""
        y_key = round(brick.position.y / tolerance) * tolerance
        is_sparse = y_layer_count.get(y_key, 0) <= SPARSE_LAYER_THRESHOLD
        is_top = abs(brick.position.y - min_y) < tolerance
        return is_sparse and is_top

    # X=0 기준 좌우 분류 (악센트 브릭만 제외)
    left_bricks = [b for b in model.bricks
                   if b.position.x < -margin and not is_accent(b)]
    right_bricks = [b for b in model.bricks
                    if b.position.x > margin and not is_accent(b)]

    missing = []

    # 오른쪽 브릭에 대해 왼쪽 대칭 브릭이 있는지 확인
    for rb in right_bricks:
        mirror_x = -rb.position.x
        found = False
        for lb in left_bricks:
            if (abs(lb.position.x - mirror_x) < tolerance and
                abs(lb.position.y - rb.position.y) < tolerance and
                abs(lb.position.z - rb.position.z) < tolerance and
                lb.part_id.lower() == rb.part_id.lower()):
                found = True
                break

        if not found:
            missing.append({
                'missing_side': 'left',
                'mirror_brick': rb,
                'suggested_pos': (mirror_x, rb.position.y, rb.position.z),
                'part_id': rb.part_id,
                'color': rb.color_code
            })

    # 왼쪽 브릭에 대해 오른쪽 대칭 브릭이 있는지 확인
    for lb in left_bricks:
        mirror_x = -lb.position.x
        found = False
        for rb in right_bricks:
            if (abs(rb.position.x - mirror_x) < tolerance and
                abs(rb.position.y - lb.position.y) < tolerance and
                abs(rb.position.z - lb.position.z) < tolerance and
                rb.part_id.lower() == lb.part_id.lower()):
                found = True
                break

        if not found:
            missing.append({
                'missing_side': 'right',
                'mirror_brick': lb,
                'suggested_pos': (mirror_x, lb.position.y, lb.position.z),
                'part_id': lb.part_id,
                'color': lb.color_code
            })

    return missing


# =============================================================================
# ISOLATED BRICK DETECTION (고립 브릭 탐지)
# =============================================================================

def find_isolated_ground_bricks(model: "BrickModel", parts_db: Dict = None) -> List[Dict]:
    """
    바닥에 배치된 고립 브릭 탐지 (실제 스터드 연결 체크)

    고립 브릭 = 바닥 근처에 있으면서 다른 브릭과 스터드 연결이 없는 브릭
    - 스터드 연결: bbox가 실제로 겹쳐야 함 (위치만 가까운 건 연결 아님)

    Returns:
        [{"brick": PlacedBrick, "id": str, "part_id": str, "position": (x,y,z)}, ...]
    """
    from ldr_converter import get_brick_bbox as ldr_get_bbox  # 회전 처리된 bbox 함수

    if not model.bricks or not parts_db:
        return []

    tolerance = 2.0  # 작은 오차 허용
    isolated = []

    # 바닥 y 좌표 찾기 (가장 높은 y = 가장 아래 in LDraw)
    ground_y = max(b.position.y for b in model.bricks)

    def get_brick_bbox(brick):
        """브릭의 실제 bbox 계산 (회전 포함)"""
        bbox = ldr_get_bbox(brick, parts_db)
        if bbox:
            return {
                'min_x': bbox.min_x,
                'max_x': bbox.max_x,
                'min_y': bbox.min_y,
                'max_y': bbox.max_y,
                'min_z': bbox.min_z,
                'max_z': bbox.max_z
            }
        # fallback: 회전 무시
        part_id = brick.part_id.lower().replace('.dat', '')
        part_info = parts_db.get(part_id, {})
        width = part_info.get('width_studs', 1) * LDU_PER_STUD
        depth = part_info.get('depth_studs', 1) * LDU_PER_STUD
        height = PLATE_HEIGHT if part_id in PLATE_PART_IDS else BRICK_HEIGHT
        half_w = width / 2
        half_d = depth / 2
        return {
            'min_x': brick.position.x - half_w,
            'max_x': brick.position.x + half_w,
            'min_y': brick.position.y,
            'max_y': brick.position.y + height,
            'min_z': brick.position.z - half_d,
            'max_z': brick.position.z + half_d
        }

    def bbox_overlap_xz(bb1, bb2):
        """X-Z 평면에서 bbox가 겹치는지 (스터드 연결 가능)"""
        x_overlap = bb1['min_x'] < bb2['max_x'] and bb1['max_x'] > bb2['min_x']
        z_overlap = bb1['min_z'] < bb2['max_z'] and bb1['max_z'] > bb2['min_z']
        return x_overlap and z_overlap

    def is_stud_connected(brick, other):
        """
        두 브릭이 스터드로 연결되어 있는지 체크

        중요: 레고 브릭은 수직 연결(위/아래)만 진짜 스터드 연결임!
        옆에 나란히 붙어있는 건 연결이 아님 (수평 연결 제거)
        """
        bb1 = get_brick_bbox(brick)
        bb2 = get_brick_bbox(other)

        # 수직 연결만 체크: Y가 딱 맞닿고 X-Z 겹침
        y_touch_above = abs(bb1['min_y'] - bb2['max_y']) < tolerance  # other가 위에
        y_touch_below = abs(bb1['max_y'] - bb2['min_y']) < tolerance  # other가 아래에

        if (y_touch_above or y_touch_below) and bbox_overlap_xz(bb1, bb2):
            return True

        return False

    ground_bricks = []
    for brick in model.bricks:
        # 바닥 근처 브릭만 대상
        if abs(brick.position.y - ground_y) > BRICK_HEIGHT:
            continue
        ground_bricks.append(brick)


    for brick in ground_bricks:
        # 스터드 연결된 브릭이 있는지 확인
        has_connection = False
        for other in model.bricks:
            if other.id == brick.id:
                continue

            if is_stud_connected(brick, other):
                has_connection = True
                break

        if not has_connection:
            isolated.append({
                "brick": brick,
                "id": brick.id,
                "part_id": brick.part_id,
                "position": (brick.position.x, brick.position.y, brick.position.z),
                "color": brick.color_code
            })

    return isolated


def remove_isolated_bricks(model: "BrickModel", parts_db: Dict = None,
                            isolated_bricks: List[Dict] = None) -> Dict[str, Any]:
    """
    고립된 바닥 브릭 삭제

    Args:
        model: BrickModel
        parts_db: 파츠 DB (파츠 이름 조회용)
        isolated_bricks: find_isolated_ground_bricks() 결과 (없으면 자동 탐지)

    Returns:
        {"deleted": int, "changes": [str, ...], "deleted_bricks": [Dict, ...]}
    """
    if isolated_bricks is None:
        isolated_bricks = find_isolated_ground_bricks(model, parts_db)

    if not isolated_bricks:
        return {"deleted": 0, "changes": [], "deleted_bricks": []}

    deleted = 0
    changes = []
    deleted_bricks = []

    for iso in isolated_bricks:
        brick = iso["brick"]
        if brick in model.bricks:
            # 파츠 이름 조회
            part_name = brick.part_id
            if parts_db:
                part_info = parts_db.get(brick.part_id.lower().replace('.dat', ''), {})
                part_name = part_info.get("name", brick.part_id)

            pos = f"({brick.position.x:.0f}, {brick.position.y:.0f}, {brick.position.z:.0f})"

            model.bricks.remove(brick)
            deleted += 1
            changes.append(f"고립 브릭 삭제: [{part_name}] {pos}")
            deleted_bricks.append({
                "id": brick.id,
                "part_id": brick.part_id,
                "position": (brick.position.x, brick.position.y, brick.position.z),
                "color": brick.color_code
            })

    return {"deleted": deleted, "changes": changes, "deleted_bricks": deleted_bricks}


def fix_symmetry(model: "BrickModel", parts_db: Dict,
                 symmetry_issues: List[Dict] = None,
                 delete_extras: bool = True) -> Dict[str, Any]:
    """
    대칭성 분석 결과로 빠진 브릭 추가 + 여분 브릭 삭제

    Args:
        model: BrickModel
        parts_db: 파츠 DB
        symmetry_issues: analyze_symmetry() 결과 (없으면 자동 분석)
        delete_extras: True면 여분 브릭(대칭 없는 브릭) 먼저 삭제

    Returns:
        {"added": int, "deleted": int, "changes": [str, ...]}
    """
    from ldr_converter import PlacedBrick, Vector3

    if symmetry_issues is None:
        symmetry_issues = analyze_symmetry(model, parts_db)

    if not symmetry_issues:
        return {"added": 0, "deleted": 0, "changes": []}

    added = 0
    deleted = 0
    changes = []
    brick_counter = 3000

    # === 1단계: 고립된 바닥 브릭 삭제 (다른 브릭과 연결 안 된 브릭) ===
    if delete_extras:
        iso_result = remove_isolated_bricks(model, parts_db)
        deleted += iso_result["deleted"]
        changes.extend(iso_result["changes"])

    # === 2단계: 점유 맵 재생성 (삭제 후) ===
    occupied = _build_occupancy_set(model, parts_db)

    # === 3단계: 빠진 브릭 추가 ===
    skipped = 0
    for issue in symmetry_issues:
        x, y, z = issue['suggested_pos']
        part_id = issue['part_id']

        # 충돌 체크
        if _check_collision_simple(model, x, y, z, part_id, parts_db, occupied):
            skipped += 1
            continue

        # 파츠 정보 가져오기
        part_info = parts_db.get(part_id.lower().replace('.dat', ''), {})
        part_height = PLATE_HEIGHT if part_id.lower() in PLATE_PART_IDS else BRICK_HEIGHT
        width_studs = part_info.get('width_studs', 1)
        depth_studs = part_info.get('depth_studs', 1)

        brick_counter += 1
        new_brick = PlacedBrick(
            id=f"sym_{brick_counter}",
            part_id=part_id,
            position=Vector3(x=x, y=y, z=z),
            rotation=issue['mirror_brick'].rotation,
            color_code=issue['color'],
            layer=issue['mirror_brick'].layer
        )
        model.bricks.append(new_brick)

        # 점유 맵 업데이트 (파츠 크기 포함)
        _mark_occupied(occupied, x, y, z, part_height, width_studs, depth_studs)

        added += 1
        changes.append(
            f"대칭 보완: {part_id} at ({x:.0f}, {y:.0f}, {z:.0f}) - {issue['missing_side']}쪽"
        )

    if skipped > 0:
        changes.append(f"충돌로 스킵: {skipped}개")

    return {"added": added, "deleted": deleted, "changes": changes}
