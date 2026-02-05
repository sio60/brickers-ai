"""Agent Tools - Helper functions for LangGraph Evolver Agent

CoScientist 원칙:
- LLM이 "무엇을" 할지 결정 (strategy)
- 알고리즘이 "어떻게" 실행 (tools)

이 파일의 함수들은 LLM이 결정한 전략을 정확하게 실행하는 도구들입니다.
"""
import os
import sys
import json
import copy
import base64
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set, Optional, TYPE_CHECKING

# Configuration dataclass 사용 (전역 변수 대체)
from .config import get_config, init_config, AgentConfig

# ============================================================================
# 모듈 레벨 경로 설정 (함수 내부 sys.path 조작 제거)
# ============================================================================
_PROJECT_ROOT = Path(__file__).resolve().parents[4]  # brickers-ai
_PHYS_PATH = _PROJECT_ROOT / "physical_verification"
_AGENT_PATH = _PROJECT_ROOT / "brick-engine" / "agent"
_EXPORTER_PATH = Path(__file__).resolve().parents[1]  # evolver/../exporter

for _p in [_PROJECT_ROOT, _PHYS_PATH, _AGENT_PATH, _EXPORTER_PATH]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# 타입 힌팅용 (런타임에는 import 안 함)
if TYPE_CHECKING:
    from ldr_converter import BrickModel, PlacedBrick


# =============================================================================
# 상수 정의
# =============================================================================

BRICK_HEIGHT = 24  # LDU
PLATE_HEIGHT = 8   # LDU
LDU_PER_STUD = 20  # 1 스터드 = 20 LDU
CELL_SIZE = LDU_PER_STUD  # 점유 맵 셀 크기

# 검증 관련 상수
GROUND_TOLERANCE = 2  # 바닥 연결 판정 허용 오차 (LDU)
MAX_SUPPORT_PER_BRICK = 20  # 브릭당 최대 지지대 수
DEFAULT_MAX_REMOVE = 5  # 기본 최대 삭제 개수

# Kids Mode 안전 기준
KIDS_MIN_PART_SIZE = "3003"  # 2x2 이상만 허용
ADULT_DEFAULT_PART = "3005"  # 1x1 브릭


class LDrawColor:
    """LDraw 색상 코드"""
    BLACK = 0
    BLUE = 1
    GREEN = 2
    RED = 4
    BROWN = 6
    LIGHT_GRAY = 7
    DARK_GRAY = 8
    YELLOW = 14
    WHITE = 15
    LIGHT_BLUISH_GRAY = 71
    DARK_BLUISH_GRAY = 72


class SupportParts:
    """지지대용 파츠 ID"""
    BRICK_1X1 = '3005'
    BRICK_1X2 = '3004'
    BRICK_1X4 = '3010'
    BRICK_2X2 = '3003'
    BRICK_2X4 = '3001'
    PLATE_1X1 = '3024'
    PLATE_1X2 = '3023'
    PLATE_2X2 = '3022'
    PLATE_2X4 = '3020'


# 플레이트 파츠 ID (높이 8 LDU)
PLATE_PART_IDS = frozenset([
    "3024", "3023", "3022", "3020", "3021", "3710", "3666"
])


# 대칭 분석 상수
SYMMETRY_TOLERANCE = 5  # 위치 매칭 허용 오차 (LDU)
SYMMETRY_CENTER_MARGIN = 5  # 중앙 영역 판정 마진
SPARSE_LAYER_THRESHOLD = 2  # 이 개수 이하면 희소 레이어


def _get_parts_db() -> Dict[str, Any]:
    """파츠 DB 가져오기 (Configuration 사용)"""
    config = get_config()
    return config.parts_db if config.is_initialized else {}


def _get_exporter_dir() -> Optional[Path]:
    """exporter 디렉토리 경로 가져오기 (Configuration 사용)"""
    config = get_config()
    return config.exporter_dir


def get_brick_direction_label(x: float, z: float) -> str:
    """
    브릭 좌표를 앞/뒤/좌/우 라벨로 변환

    LDraw 좌표계:
    - Z 음수 = 앞 (FRONT)
    - Z 양수 = 뒤 (BACK)
    - X 음수 = 왼쪽 (LEFT)
    - X 양수 = 오른쪽 (RIGHT)

    Returns:
        예: "FRONT-LEFT", "BACK-RIGHT", "CENTER"
    """
    parts = []

    # 앞/뒤 판단
    if z < -10:
        parts.append("FRONT")
    elif z > 10:
        parts.append("BACK")

    # 좌/우 판단
    if x < -10:
        parts.append("LEFT")
    elif x > 10:
        parts.append("RIGHT")

    if not parts:
        return "CENTER"

    return "-".join(parts)


def label_brick_list(bricks: List[Dict]) -> List[Dict]:
    """
    브릭 목록에 방향 라벨 추가

    Args:
        bricks: [{"id": ..., "x": ..., "y": ..., "z": ..., ...}, ...]

    Returns:
        [{"id": ..., "x": ..., "direction": "FRONT-LEFT", ...}, ...]
    """
    labeled = []
    for b in bricks:
        brick_copy = dict(b)
        brick_copy["direction"] = get_brick_direction_label(
            b.get("x", 0), b.get("z", 0)
        )
        labeled.append(brick_copy)
    return labeled

def analyze_glb(glb_path: str, vision_model: str = None) -> Dict[str, Any]:
    """
    GLB 파일을 렌더링하고 Vision LLM으로 분석

    Args:
        glb_path: GLB 파일 경로
        vision_model: Vision 모델명 (None이면 config에서 가져옴)
    """
    try:
        import trimesh
        from openai import OpenAI

        if not Path(glb_path).exists():
            return {"available": False, "error": "GLB not found"}

        # Vision 모델 설정
        config = get_config()
        model_name = vision_model or getattr(config, 'vision_model', None) or "gpt-4o-mini"

        # 1. GLB 로드 및 렌더링
        try:
            mesh = trimesh.load(glb_path)
            png_data = mesh.save_image(resolution=[512, 512])
            image_b64 = base64.b64encode(png_data).decode("utf-8")
        except Exception as e:
            return {"available": False, "error": f"Render failed: {e}"}

        # 2. Vision LLM으로 분석
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": """Analyze this 3D model image.
Return JSON with:
1. model_type: what is it (animal, vehicle, building, etc)
2. name: specific name (horse, car, house, etc)
3. legs: number of legs (0 if none)
4. key_features: list of main features
5. structure_notes: notes for building with bricks
Return only JSON."""},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                    ]
                }],
                max_tokens=500
            )
        except Exception as e:
            return {"available": False, "error": f"OpenAI API failed: {e}"}

        # 3. JSON 파싱
        try:
            content = response.choices[0].message.content
            start = content.find('{')
            end = content.rfind('}') + 1
            if start == -1 or end == 0:
                return {"available": False, "error": "No JSON in response"}
            result = json.loads(content[start:end])
            result["available"] = True
            return result
        except json.JSONDecodeError as e:
            return {"available": False, "error": f"JSON parse failed: {e}"}

    except Exception as e:
        return {"available": False, "error": f"Unknown error: {e}"}

def get_model_state(model: "BrickModel", parts_db: Dict) -> Dict:
    """Get current model state using PhysicalVerifier"""
    from verifier import PhysicalVerifier
    from models import Brick, BrickPlan

    # Convert BrickModel to BrickPlan (Verifier format)
    from ldr_converter import get_brick_bbox

    bricks = []
    for b in model.bricks:
        bbox = get_brick_bbox(b, parts_db)
        if bbox:
            # 좌표 변환: LDraw -> Verifier
            # LDraw: X=가로, Y=위아래(음수가 위, 양수가 아래), Z=앞뒤
            # Verifier: X=가로, Y=앞뒤, Z=위아래(양수가 위)
            #
            # Verifier Brick은 최소 좌표(왼쪽 아래 모서리) 기준
            # LDraw bbox를 직접 사용
            brick = Brick(
                id=b.id,
                x=bbox.min_x,                    # X는 그대로
                y=bbox.min_z,                    # LDraw Z -> Verifier Y
                z=-bbox.max_y,                   # LDraw -max_Y -> Verifier Z (바닥)
                width=bbox.max_x - bbox.min_x,
                depth=bbox.max_z - bbox.min_z,
                height=bbox.max_y - bbox.min_y,
                mass=1.0
            )
            bricks.append(brick)

    if not bricks:
        return {
            "total_bricks": 0,
            "floating_count": 0,
            "collision_count": 0,
            "floating_bricks": [],
            "verification_result": None
        }

    plan = BrickPlan(bricks)
    verifier = PhysicalVerifier(plan)
    result = verifier.run_all_checks()

    # Extract floating bricks from evidence
    floating_ids = []
    collision_count = 0
    for ev in result.evidence:
        if ev.type == "FLOATING":
            floating_ids.extend(ev.brick_ids)
        elif ev.type == "COLLISION":
            collision_count += 1

    floating_list = []
    for bid in floating_ids[:15]:
        b = next((x for x in model.bricks if x.id == bid), None)
        if b:
            floating_list.append({
                "id": b.id,
                "part_id": b.part_id,
                "position": {"x": b.position.x, "y": b.position.y, "z": b.position.z},
                "color": b.color_code
            })

    return {
        "total_bricks": len(model.bricks),
        "floating_count": len(floating_ids),
        "collision_count": collision_count,
        "floating_bricks": floating_list,
        "verification_result": result,
        "score": result.score,
        "is_valid": result.is_valid,
        "evidence": result.evidence
    }

def remove_brick(model: "BrickModel", brick_id: str) -> Dict[str, Any]:
    """Remove a brick from model"""
    brick = next((b for b in model.bricks if b.id == brick_id), None)
    if not brick:
        return {"success": False, "error": f"Brick {brick_id} not found"}

    backup = {
        "id": brick.id,
        "part_id": brick.part_id,
        "x": brick.position.x,
        "y": brick.position.y,
        "z": brick.position.z,
        "color": brick.color_code
    }
    model.bricks.remove(brick)
    return {"success": True, "backup": backup}

def add_brick(model: "BrickModel", part_id: str, x: float, y: float, z: float,
              color: int, rotation: int = 0) -> Dict[str, Any]:
    """Add a new brick to model"""
    from ldr_converter import PlacedBrick, Vector3
    import uuid

    new_id = f"support_{uuid.uuid4().hex[:8]}"

    # 타입 변환 (LLM이 이상한 값 줄 수 있음)
    part_id_str = str(part_id) if part_id else "3023"
    x_int = int(round(float(x)))
    y_int = int(round(float(y)))
    z_int = int(round(float(z)))

    brick = PlacedBrick(
        id=new_id,
        part_id=part_id_str,
        position=Vector3(x=x_int, y=y_int, z=z_int),
        rotation=rotation,
        color_code=color,
        layer=0
    )
    model.bricks.append(brick)
    return {"success": True, "brick_id": new_id}

def rollback_model(model_backup: "BrickModel") -> "BrickModel":
    """
    모델을 원본으로 복원

    Args:
        model_backup: 백업된 원본 모델

    Returns:
        복원된 모델 (deep copy)

    Note:
        호출하는 쪽에서 state["model"], state["total_removed"],
        state["action_history"]를 직접 업데이트해야 함
    """
    return copy.deepcopy(model_backup)


# ========================================
# 좌표 계산 알고리즘 (LLM이 아닌 알고리즘이 계산)
# ========================================

def find_nearby_stable_bricks(model, floating_brick, parts_db, search_radius=60):
    """
    부유 브릭 주변의 안정된 브릭 찾기
    Returns: 안정 브릭 리스트 (거리순 정렬)
    """
    from collections import deque
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


# =============================================================================
# SYMMETRY_FIX 알고리즘 (evolver_cos.py에서 추출)
# =============================================================================

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
            # print는 node_evolve에서 하므로 여기선 생략

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


# =============================================================================
# 충돌 검사 / 점유 맵 헬퍼 함수
# =============================================================================

# 배치 검증 상수 (glb_to_ldr_embedded.py에서 추출)
DEFAULT_SUPPORT_RATIO = 0.3  # 최소 30% 지지 필요
OPTIMAL_BOND_MIN = 0.3  # 최적 겹침 범위
OPTIMAL_BOND_MAX = 0.7


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


