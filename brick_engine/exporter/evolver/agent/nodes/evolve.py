"""EVOLVE Node - Execute selected proposal (Dispatch 패턴)

리팩토링:
1. sys.path 중복 제거 → 모듈 상단 초기화
2. Dispatch 패턴으로 핸들러 분리
3. 예외 처리 개선
"""
import copy
import sys
import json
from pathlib import Path
from typing import Dict, Any, Callable, List

from ..state import AgentState
from ..tools import (
    remove_brick, add_brick, rollback_model, get_model_state,
    try_add_brick_with_validation, fix_symmetry,
    _check_collision_simple, _build_occupancy_set, _mark_occupied,
    calc_support_ratio, can_place_brick,
    find_isolated_ground_bricks, remove_isolated_bricks,
    LDU_PER_STUD, BRICK_HEIGHT, PLATE_HEIGHT, PLATE_PART_IDS,
    DEFAULT_SUPPORT_RATIO
)
from ..constants import MAX_REMOVAL_RATIO
from ..config import get_config

# Vision 분석용 5방향 (TOP, LEFT 제외)
VISION_ANGLES = ["FRONT", "BACK", "RIGHT", "BOTTOM", "FRONT_RIGHT"]

# 모듈 레벨에서 한 번만 경로 설정
_evolver_dir = Path(__file__).parent.parent.parent
if str(_evolver_dir) not in sys.path:
    sys.path.insert(0, str(_evolver_dir))

# 지연 import를 위한 캐시
_vision_modules = {}

def _get_vision_modules():
    """Vision 관련 모듈 지연 로드 (최초 1회만)"""
    if not _vision_modules:
        try:
            from vision_analyzer import map_to_coordinates, find_reference_part, plan_rebuild
            from ldr_renderer import render_model_multi_angle
            _vision_modules["map_to_coordinates"] = map_to_coordinates
            _vision_modules["find_reference_part"] = find_reference_part
            _vision_modules["plan_rebuild"] = plan_rebuild
            _vision_modules["render_model_multi_angle"] = render_model_multi_angle
        except ImportError as e:
            print(f"  [WARNING] Vision 모듈 로드 실패: {e}")
    return _vision_modules


def _get_brick_list(model) -> list:
    """모델에서 브릭 리스트 추출 (공통 유틸리티)"""
    return [
        {"id": b.id, "x": b.position.x, "y": b.position.y, "z": b.position.z,
         "part_id": b.part_id, "color": b.color_code}
        for b in model.bricks
    ]


def _snap_to_stud(x: float, z: float) -> tuple:
    """좌표를 스터드 단위로 스냅"""
    snap_x = round(x / LDU_PER_STUD) * LDU_PER_STUD
    snap_z = round(z / LDU_PER_STUD) * LDU_PER_STUD
    return snap_x, snap_z


# ===== 핸들러 함수들 =====

def handle_remove(state: AgentState, proposal: Dict, action: Dict, before_floating: int) -> bool:
    """부유 브릭 삭제 핸들러"""
    brick_id = proposal["brick_id"]
    brick = next((b for b in state["model"].bricks if b.id == brick_id), None)
    result = remove_brick(state["model"], brick_id)
    if result["success"]:
        state["total_removed"] += 1
        action["backup"] = result["backup"]
        # 파츠 이름 출력
        if brick:
            parts_db = get_config().parts_db
            part_info = parts_db.get(brick.part_id.lower(), {})
            part_name = part_info.get("name", brick.part_id)
            pos = f"({int(brick.position.x)}, {int(brick.position.y)}, {int(brick.position.z)})"
            print(f"  삭제됨: [{part_name}] {pos}")
        else:
            print(f"  삭제됨: {brick_id}")
        return True
    print(f"  실패: {result.get('error', 'unknown')}")
    return False


def handle_add_support_candidates(state: AgentState, proposal: Dict, action: Dict, before_floating: int) -> bool:
    """지지대 추가 핸들러 (후보 목록)"""
    candidates = proposal["candidates"]
    print(f"  {len(candidates)}개 후보 위치 시도...")

    for i, candidate in enumerate(candidates):
        model_backup = copy.deepcopy(state["model"])
        result = try_add_brick_with_validation(state["model"], candidate, get_config().parts_db)

        if result["success"]:
            new_floating = result["floating_count"]
            print(f"    후보 {i+1}: floating {before_floating} -> {new_floating}")

            if new_floating < before_floating:
                print(f"    [OK] 개선됨!")
                action["brick_id"] = result["brick_id"]
                action["position"] = candidate
                return True
            else:
                print(f"    [X] 개선 없음, 롤백")
                state["model"] = model_backup
        else:
            print(f"    후보 {i+1}: 추가 실패")

    print(f"  모든 후보 실패")
    return False


def handle_add_support_single(state: AgentState, proposal: Dict, action: Dict, before_floating: int) -> bool:
    """지지대 추가 핸들러 (단일 위치)"""
    pos = proposal["position"]
    color = 15
    if state["floating_bricks"]:
        color = state["floating_bricks"][0].get("color", 15)

    result = add_brick(
        state["model"],
        proposal.get("part_id", "3023"),
        pos["x"], pos["y"], pos["z"],
        color
    )
    if result["success"]:
        action["brick_id"] = result["brick_id"]
        print(f"  추가됨: {result['brick_id']} at ({pos['x']}, {pos['y']}, {pos['z']})")
        return True
    return False


def handle_bridge(state: AgentState, proposal: Dict, action: Dict, before_floating: int) -> bool:
    """연결 브릭 핸들러"""
    candidates = proposal.get("candidates", [])
    print(f"  연결 브릭 시도: {len(candidates)}개 후보...")

    for i, candidate in enumerate(candidates):
        model_backup = copy.deepcopy(state["model"])
        result = try_add_brick_with_validation(state["model"], candidate, get_config().parts_db)

        if result["success"]:
            new_floating = result["floating_count"]
            print(f"    후보 {i+1}: floating {before_floating} -> {new_floating}")

            if new_floating < before_floating:
                print(f"    [OK] 연결 성공!")
                action["brick_id"] = result["brick_id"]
                action["position"] = candidate
                return True
            else:
                print(f"    [X] 개선 없음, 롤백")
                state["model"] = model_backup
        else:
            print(f"    후보 {i+1}: 추가 실패")

    print(f"  모든 후보 실패")
    return False


def handle_rollback(state: AgentState, proposal: Dict, action: Dict, before_floating: int) -> bool:
    """원본 복원 핸들러"""
    state["model"] = rollback_model(state["model_backup"])
    state["total_removed"] = 0
    state["action_history"] = []
    print(f"  원본으로 복원: {state['original_brick_count']}개 브릭")
    return True


def handle_relocate(state: AgentState, proposal: Dict, action: Dict, before_floating: int) -> bool:
    """위치 이동 핸들러"""
    problem = proposal.get("problem", {})
    print(f"  위치 이동: {problem.get('location')} - {problem.get('issue')}")

    modules = _get_vision_modules()
    if not modules:
        action["error"] = "Vision 모듈 로드 실패"
        return False

    map_to_coordinates = modules["map_to_coordinates"]
    render_model_multi_angle = modules["render_model_multi_angle"]

    images = render_model_multi_angle(state["model"], get_config().parts_db, angles=VISION_ANGLES)
    if not images:
        print(f"    이미지 렌더링 실패")
        return False

    brick_list = _get_brick_list(state["model"])
    mapping = map_to_coordinates(images, {"problem_location": problem.get("location")}, brick_list)

    if not mapping.get("target_brick_ids") or mapping.get("confidence", 0) <= 50:
        print(f"    신뢰도 낮음 ({mapping.get('confidence', 0)}), 스킵")
        return False

    parts_db = get_config().parts_db
    occupied = _build_occupancy_set(state["model"], parts_db)
    moved_count = 0

    NUDGE_OFFSETS = [
        (0, 0),
        (LDU_PER_STUD, 0), (-LDU_PER_STUD, 0),
        (0, LDU_PER_STUD), (0, -LDU_PER_STUD),
        (LDU_PER_STUD, LDU_PER_STUD), (-LDU_PER_STUD, -LDU_PER_STUD),
        (LDU_PER_STUD, -LDU_PER_STUD), (-LDU_PER_STUD, LDU_PER_STUD),
    ]

    for brick_id in mapping["target_brick_ids"]:
        transform = mapping.get("transform", {})
        if not transform.get("translation"):
            continue

        t = transform["translation"]
        brick = next((b for b in state["model"].bricks if b.id == brick_id), None)
        if not brick:
            continue

        old_x, old_y, old_z = brick.position.x, brick.position.y, brick.position.z
        base_x = old_x + t.get("x", 0)
        new_y = old_y + t.get("y", 0)
        base_z = old_z + t.get("z", 0)

        base_x, base_z = _snap_to_stud(base_x, base_z)

        # nudge: 충돌 시 주변 ±1 stud 탐색
        placed = False
        for dx, dz in NUDGE_OFFSETS:
            try_x = base_x + dx
            try_z = base_z + dz

            can_place, reason = can_place_brick(
                try_x, new_y, try_z, brick.part_id, parts_db, occupied,
                support_ratio=DEFAULT_SUPPORT_RATIO
            )

            if can_place:
                brick.position.x = try_x
                brick.position.y = new_y
                brick.position.z = try_z
                moved_count += 1
                part_info = parts_db.get(brick.part_id.lower(), {})
                part_name = part_info.get("name", brick.part_id)
                if dx != 0 or dz != 0:
                    print(f"    이동됨 [{part_name}] ({int(old_x)}, {int(old_y)}, {int(old_z)}) -> ({int(try_x)}, {int(new_y)}, {int(try_z)}) (nudge)")
                else:
                    print(f"    이동됨 [{part_name}] ({int(old_x)}, {int(old_y)}, {int(old_z)}) -> ({int(try_x)}, {int(new_y)}, {int(try_z)})")
                placed = True
                break

        if not placed:
            part_info = parts_db.get(brick.part_id.lower(), {})
            part_name = part_info.get("name", brick.part_id)
            print(f"    스킵 [{part_name}]: 주변에도 배치 불가")

    if moved_count > 0:
        print(f"    {moved_count}개 브릭 이동 완료")
        return True
    return False


def _apply_y_rotation(brick, degrees: int):
    """
    브릭에 Y축 회전 적용 (LDraw 회전 행렬 변환)

    LDraw 회전 행렬 (3x3):
    90도: [0, 0, -1, 0, 1, 0, 1, 0, 0]
    180도: [-1, 0, 0, 0, 1, 0, 0, 0, -1]
    270도: [0, 0, 1, 0, 1, 0, -1, 0, 0]
    """
    import math

    # 현재 회전 값에 추가 (0, 90, 180, 270 중 하나)
    current_rotation = getattr(brick, 'rotation', 0) or 0
    new_rotation = (current_rotation + degrees) % 360

    # rotation 속성 업데이트
    brick.rotation = new_rotation

    # rotation_matrix가 있으면 업데이트 (선택적)
    if hasattr(brick, 'rotation_matrix'):
        rad = math.radians(new_rotation)
        cos_r = round(math.cos(rad))
        sin_r = round(math.sin(rad))
        # Y축 회전 행렬: [cos, 0, -sin, 0, 1, 0, sin, 0, cos]
        brick.rotation_matrix = [cos_r, 0, -sin_r, 0, 1, 0, sin_r, 0, cos_r]

    return new_rotation


def handle_rotate(state: AgentState, proposal: Dict, action: Dict, before_floating: int) -> bool:
    """방향 회전 핸들러 (실제 회전 구현)"""
    problem = proposal.get("problem", {})
    print(f"  방향 회전: {problem.get('location')} - {problem.get('issue')}")

    modules = _get_vision_modules()
    if not modules:
        action["error"] = "Vision 모듈 로드 실패"
        return False

    map_to_coordinates = modules["map_to_coordinates"]
    render_model_multi_angle = modules["render_model_multi_angle"]

    images = render_model_multi_angle(state["model"], get_config().parts_db, angles=VISION_ANGLES)
    if not images:
        print(f"    이미지 렌더링 실패")
        return False

    brick_list = _get_brick_list(state["model"])
    mapping = map_to_coordinates(images, {"problem_location": problem.get("location"), "action": "rotate"}, brick_list)

    if not mapping.get("target_brick_ids") or mapping.get("confidence", 0) <= 50:
        print(f"    신뢰도 낮음 ({mapping.get('confidence', 0)}), 스킵")
        return False

    parts_db = get_config().parts_db
    rotated_count = 0
    rotations_applied = []

    for brick_id in mapping["target_brick_ids"]:
        transform = mapping.get("transform", {})
        rotation = transform.get("rotation", {})
        brick = next((b for b in state["model"].bricks if b.id == brick_id), None)

        if brick and rotation.get("y", 0) != 0:
            y_rot = int(rotation["y"])
            # 90도 단위로 정규화
            y_rot = round(y_rot / 90) * 90
            if y_rot == 0:
                continue

            old_rotation = getattr(brick, 'rotation', 0) or 0
            new_rotation = _apply_y_rotation(brick, y_rot)
            rotated_count += 1
            rotations_applied.append({
                "brick_id": brick_id,
                "old": old_rotation,
                "new": new_rotation,
                "delta": y_rot
            })
            # 파츠 이름 + 위치 출력
            part_info = parts_db.get(brick.part_id.lower(), {})
            part_name = part_info.get("name", brick.part_id)
            pos = f"({int(brick.position.x)}, {int(brick.position.y)}, {int(brick.position.z)})"
            print(f"    회전 [{part_name}] {pos}: {old_rotation}° → {new_rotation}°")

    if rotated_count > 0:
        action["rotations"] = rotations_applied
        print(f"    {rotated_count}개 브릭 회전 완료")
        return True

    print(f"    회전할 브릭 없음")
    return False


def _calculate_mirror_positions(reference_bricks: List[Dict], relationship: str, occupied: set) -> List[Dict]:
    """
    참조 브릭들의 좌표를 변환하여 새 위치 계산 (알고리즘)
    충돌 시 주변 ±1~2 스터드 오프셋 탐색

    Args:
        reference_bricks: 참조 브릭 정보 [{"x": 20, "y": -48, "z": -80, "part_id": "3005", "color": 71}, ...]
        relationship: "mirror_x", "mirror_z", "same_pattern"
        occupied: 이미 점유된 좌표 set

    Returns:
        새 위치 리스트 (충돌 없는 것만)
    """
    new_positions = []
    # 주변 탐색 오프셋: 원래 위치 → ±1 스터드 → ±2 스터드
    OFFSETS = [
        (0, 0),
        (LDU_PER_STUD, 0), (-LDU_PER_STUD, 0),
        (0, LDU_PER_STUD), (0, -LDU_PER_STUD),
        (LDU_PER_STUD, LDU_PER_STUD), (-LDU_PER_STUD, -LDU_PER_STUD),
        (LDU_PER_STUD, -LDU_PER_STUD), (-LDU_PER_STUD, LDU_PER_STUD),
        (2*LDU_PER_STUD, 0), (-2*LDU_PER_STUD, 0),
        (0, 2*LDU_PER_STUD), (0, -2*LDU_PER_STUD),
    ]

    for ref in reference_bricks:
        x, y, z = ref["x"], ref["y"], ref["z"]

        # 변환 적용
        if relationship == "mirror_x":
            new_x = -x  # X 반전
            new_z = z
        elif relationship == "mirror_z":
            new_x = x
            new_z = -z  # Z 반전
        else:  # same_pattern
            new_x = x
            new_z = z

        # 스터드 단위로 스냅
        snap_x, snap_z = _snap_to_stud(new_x, new_z)
        snap_y = int(round(y))

        # 충돌 시 주변 위치 탐색
        placed = False
        for dx, dz in OFFSETS:
            try_x = snap_x + dx
            try_z = snap_z + dz
            pos_key = (try_x, snap_y, try_z)
            if pos_key not in occupied:
                if dx != 0 or dz != 0:
                    print(f"      ({snap_x}, {snap_y}, {snap_z}) 충돌 → ({try_x}, {snap_y}, {try_z})로 이동")
                new_positions.append({
                    "part_id": ref.get("part_id", "3005"),
                    "x": try_x,
                    "y": snap_y,
                    "z": try_z,
                    "color": ref.get("color", 15)
                })
                placed = True
                break

        if not placed:
            print(f"      스킵 ({snap_x}, {snap_y}, {snap_z}): 주변에도 빈 자리 없음")

    return new_positions


def handle_rebuild(state: AgentState, proposal: Dict, action: Dict, before_floating: int) -> bool:
    """재배치 핸들러 - LLM은 브릭 ID만 찾고, 좌표 계산은 알고리즘이 함"""
    problem = proposal.get("problem", {})
    print(f"  재배치: {problem.get('location')} - {problem.get('issue')}")

    modules = _get_vision_modules()
    if not modules:
        action["error"] = "Vision 모듈 로드 실패"
        return False

    find_reference_part = modules["find_reference_part"]
    plan_rebuild = modules["plan_rebuild"]
    render_model_multi_angle = modules["render_model_multi_angle"]

    images = render_model_multi_angle(state["model"], get_config().parts_db, angles=VISION_ANGLES)
    if not images:
        print(f"    이미지 렌더링 실패")
        return False

    model_name = getattr(state["model"], "name", "unknown")

    # 참조 부분 찾기 (LLM)
    ref = find_reference_part(images, problem.get("location"), model_name)
    if ref.get("confidence", 0) <= 50:
        print(f"    적합한 참조 부분 없음")
        return False

    # 브릭 ID만 찾기 (LLM) - 좌표 계산 X
    brick_list = _get_brick_list(state["model"])
    plan = plan_rebuild(images, problem.get("location"), ref.get("reference_part"),
                       ref.get("relationship", "mirror_x"), brick_list)

    if plan.get("error"):
        print(f"    계획 생성 실패: {plan.get('error')}")
        return False

    # 삭제할 브릭 처리
    deleted_backups = []
    parts_db = get_config().parts_db
    for brick_id in plan.get("delete_brick_ids", []):
        brick = next((b for b in state["model"].bricks if b.id == brick_id), None)
        result = remove_brick(state["model"], brick_id)
        if result["success"]:
            deleted_backups.append(result["backup"])
            state["total_removed"] += 1
            if brick:
                part_info = parts_db.get(brick.part_id.lower(), {})
                part_name = part_info.get("name", brick.part_id)
                pos = f"({int(brick.position.x)}, {int(brick.position.y)}, {int(brick.position.z)})"
                print(f"    삭제: [{part_name}] {pos}")
            else:
                print(f"    삭제: {brick_id}")

    # 참조 브릭 정보 가져오기 (알고리즘)
    reference_brick_ids = plan.get("reference_brick_ids", [])
    brick_dict = {b["id"]: b for b in brick_list}
    reference_bricks = []
    for ref_id in reference_brick_ids:
        if ref_id in brick_dict:
            reference_bricks.append(brick_dict[ref_id])

    if not reference_bricks:
        print(f"    참조 브릭을 찾을 수 없음")
        return len(deleted_backups) > 0  # 삭제만 했어도 변경은 됨

    print(f"    참조 브릭 {len(reference_bricks)}개 발견")

    # 점유 맵 생성 (bbox 기반 - 회전/크기 반영)
    occupied = _build_occupancy_set(state["model"], parts_db)

    # 좌표 계산 (알고리즘) - LLM이 아닌 알고리즘이 계산
    relationship = plan.get("relationship", "mirror_x")
    new_positions = _calculate_mirror_positions(reference_bricks, relationship, occupied)

    print(f"    새 위치 {len(new_positions)}개 계산됨 (relationship: {relationship})")

    # 브릭 추가 (알고리즘)
    added_ids = []
    parts_db = get_config().parts_db

    NUDGE_OFFSETS = [
        (0, 0), (LDU_PER_STUD, 0), (-LDU_PER_STUD, 0),
        (0, LDU_PER_STUD), (0, -LDU_PER_STUD),
    ]
    for pos in new_positions:
        placed = False
        for dx, dz in NUDGE_OFFSETS:
            try_x = pos["x"] + dx
            try_z = pos["z"] + dz
            if _check_collision_simple(state["model"], try_x, pos["y"], try_z,
                                       pos["part_id"], parts_db, occupied):
                continue

            result = add_brick(state["model"], pos["part_id"], try_x, pos["y"], try_z, pos["color"])
            if result["success"]:
                added_ids.append(result["brick_id"])
                occupied.add((try_x, pos["y"], try_z))
                part_data = parts_db.get(pos["part_id"].lower(), {})
                p_height = part_data.get("height", BRICK_HEIGHT)
                p_width = part_data.get("width", 1)
                p_depth = part_data.get("depth", 1)
                _mark_occupied(occupied, try_x, pos["y"], try_z, p_height, p_width, p_depth)
                part_info = parts_db.get(pos["part_id"].lower(), {})
                part_name = part_info.get("name", pos["part_id"])
                if dx != 0 or dz != 0:
                    print(f"    추가: [{part_name}] ({pos['x']}, {pos['y']}, {pos['z']}) → nudge ({try_x}, {pos['y']}, {try_z})")
                else:
                    print(f"    추가: [{part_name}] ({try_x}, {pos['y']}, {try_z})")
                placed = True
                break

        if not placed:
            print(f"      최종 스킵 ({pos['x']}, {pos['y']}, {pos['z']}): 주변에도 배치 불가")

    action["deleted_backups"] = deleted_backups
    action["added_ids"] = added_ids
    print(f"    재배치 완료: 삭제 {len(deleted_backups)}개, 추가 {len(added_ids)}개")
    return len(deleted_backups) > 0 or len(added_ids) > 0


def _filter_symmetry_by_vision(symmetry_issues: List[Dict], vision_problems: List[Dict]) -> List[Dict]:
    """Vision 문제 위치 기반으로 대칭 이슈 필터링

    Vision: "front left leg", "back right leg" 등
    → front: z < 0, back: z > 0, left: x < 0, right: x > 0
    """
    if not vision_problems:
        return symmetry_issues[:5]  # Vision 없으면 최대 5개만

    # Vision 문제에서 위치 키워드 추출
    locations = set()
    for vp in vision_problems:
        loc = vp.get("location", "").lower()
        if "front" in loc:
            locations.add("front")
        if "back" in loc:
            locations.add("back")
        if "left" in loc:
            locations.add("left")
        if "right" in loc:
            locations.add("right")
        if "leg" in loc:
            locations.add("leg")

    print(f"    Vision 위치 키워드: {locations}")

    if not locations:
        return symmetry_issues[:5]

    # 대칭 이슈 필터링
    filtered = []
    for issue in symmetry_issues:
        pos = issue.get("suggested_pos", (0, 0, 0))
        if isinstance(pos, (list, tuple)) and len(pos) >= 3:
            x, y, z = pos[0], pos[1], pos[2]
        else:
            continue

        match = False
        # 다리 영역: 바닥 근처 (y > -50) + 모델 가장자리
        is_leg_area = y > -50 and (abs(x) > 20 or abs(z) > 30)

        if "leg" in locations and is_leg_area:
            # front/back + left/right 조합 체크
            if "front" in locations and z < 0:
                match = True
            elif "back" in locations and z > 0:
                match = True
            elif "left" in locations and x < 0:
                match = True
            elif "right" in locations and x > 0:
                match = True
            elif len(locations) == 1:  # "leg"만 있으면 모든 다리 영역
                match = True

        if match:
            filtered.append(issue)

    return filtered if filtered else symmetry_issues[:3]  # 필터 결과 없으면 3개만


def handle_symmetry_fix(state: AgentState, proposal: Dict, action: Dict, before_floating: int) -> bool:
    """대칭 브릭 추가 핸들러 - Vision 문제 위치만 수정"""
    symmetry_issues = state.get("symmetry_issues", [])
    vision_problems = state.get("vision_problems", [])

    if not symmetry_issues:
        print(f"  대칭 문제 없음")
        return False

    # Vision 문제 기반 필터링
    filtered_issues = _filter_symmetry_by_vision(symmetry_issues, vision_problems)

    print(f"  대칭 수정: {len(filtered_issues)}/{len(symmetry_issues)}개 처리")
    result = fix_symmetry(state["model"], get_config().parts_db, filtered_issues, delete_extras=True)

    deleted = result.get("deleted", 0)
    added = result.get("added", 0)

    if deleted > 0 or added > 0:
        action["deleted"] = deleted
        action["added"] = added
        print(f"  결과: 삭제 {deleted}개, 추가 {added}개")
        return True

    print(f"  대칭: 변경 없음")
    return False


# ===== Dispatch 테이블 =====

HANDLERS: Dict[str, Callable] = {
    "remove": handle_remove,
    "add_support": None,  # 특수 처리 (candidates vs position)
    "bridge": handle_bridge,
    "rollback": handle_rollback,
    "relocate": handle_relocate,
    "rotate": handle_rotate,
    "rebuild": handle_rebuild,
    "symmetry_fix": handle_symmetry_fix,
}


def node_evolve(state: AgentState) -> AgentState:
    """단일 전략 실행"""
    strategy = state.get("strategy", "")
    proposals = state.get("proposals", [])

    # 바닥 고립 브릭 자동 제거 - 비활성화 (오탐으로 정상 브릭 삭제됨)
    # TODO: 스터드 연결 판정 정확도 개선 후 재활성화

    if not strategy and not proposals:
        print("\n[EVOLVE] 실행할 전략/제안 없음")
        return state

    print(f"\n[EVOLVE] 전략: {strategy}")

    before_floating = state["floating_count"]
    strategy_lower = strategy.lower()

    # 해당 전략의 proposal 찾기 (없으면 기본 proposal 생성)
    proposal = next(
        (p for p in proposals if p.get("type", "").lower() == strategy_lower),
        {"type": strategy_lower, "id": f"auto_{strategy_lower}"}
    )

    action = {"type": strategy_lower, "proposal": proposal, "success": False}

    # add_support 특수 처리
    if strategy_lower == "add_support":
        if proposal.get("candidates"):
            action["success"] = handle_add_support_candidates(state, proposal, action, before_floating)
        elif proposal.get("position"):
            action["success"] = handle_add_support_single(state, proposal, action, before_floating)
        else:
            print(f"    add_support: candidates/position 없음")
    else:
        # Dispatch 테이블에서 핸들러 찾기
        handler = HANDLERS.get(strategy_lower)
        if handler:
            try:
                action["success"] = handler(state, proposal, action, before_floating)
            except Exception as e:
                print(f"    [ERROR] {strategy} 핸들러 예외: {e}")
                action["error"] = str(e)
                action["success"] = False
        else:
            print(f"    [WARNING] 알 수 없는 전략: {strategy}")

    state["action_history"].append(action)

    # selected_proposal 실행 (debate에서 선택된 경우)
    if state.get("selected_proposal"):
        selected = state["selected_proposal"]
        if selected["type"].lower() != strategy_lower:
            print(f"\n  --- 선택된 제안 실행: {selected['id']} ---")
            sel_action = {"type": selected["type"], "proposal": selected, "success": False}
            handler = HANDLERS.get(selected["type"].lower())
            if handler:
                try:
                    sel_action["success"] = handler(state, selected, sel_action, before_floating)
                except Exception as e:
                    print(f"    [ERROR] 핸들러 예외: {e}")
            state["action_history"].append(sel_action)

    return state
