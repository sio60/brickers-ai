"""Agent Tools - Helper functions"""
import os
import json
import copy
import base64
from pathlib import Path
from typing import Dict, Any

# Will be set by run_agent.py
_parts_db = None
_exporter_dir = None

def init_tools(parts_db: Dict, exporter_dir: Path):
    global _parts_db, _exporter_dir
    _parts_db = parts_db
    _exporter_dir = exporter_dir


def analyze_glb(glb_path: str) -> Dict[str, Any]:
    """GLB 파일을 렌더링하고 GPT-4o-mini로 분석"""
    import trimesh
    from openai import OpenAI

    if not Path(glb_path).exists():
        return {"available": False, "error": "GLB not found"}

    # 1. GLB 로드 및 렌더링
    mesh = trimesh.load(glb_path)
    png_data = mesh.save_image(resolution=[512, 512])
    image_b64 = base64.b64encode(png_data).decode("utf-8")

    # 2. GPT-4o-mini로 분석
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
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

    # 3. JSON 파싱
    content = response.choices[0].message.content
    start = content.find('{')
    end = content.rfind('}') + 1
    result = json.loads(content[start:end])
    result["available"] = True
    return result

def load_parts_db() -> Dict:
    if _parts_db:
        return _parts_db
    cache = _exporter_dir / "parts_cache.json" if _exporter_dir else Path("parts_cache.json")
    if cache.exists():
        with open(cache, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def get_model_state(model, parts_db) -> Dict:
    """Get current model state using 승준's PhysicalVerifier"""
    import sys
    from pathlib import Path

    # Add project root and physical_verification to path
    project_root = Path(__file__).parent.parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    phys_path = project_root / "physical_verification"
    if str(phys_path) not in sys.path:
        sys.path.insert(0, str(phys_path))

    from verifier import PhysicalVerifier
    from models import Brick, BrickPlan

    # Convert BrickModel to BrickPlan (승준's format)
    from ldr_converter import get_brick_bbox

    bricks = []
    for b in model.bricks:
        bbox = get_brick_bbox(b, parts_db)
        if bbox:
            # 좌표 변환: LDraw -> 승준
            # LDraw: X=가로, Y=위아래(음수가 위, 양수가 아래), Z=앞뒤
            # 승준: X=가로, Y=앞뒤, Z=위아래(양수가 위)
            #
            # 승준 Brick은 최소 좌표(왼쪽 아래 모서리) 기준
            # LDraw bbox를 직접 사용
            brick = Brick(
                id=b.id,
                x=bbox.min_x,                    # X는 그대로
                y=bbox.min_z,                    # LDraw Z -> 승준 Y
                z=-bbox.max_y,                   # LDraw -max_Y -> 승준 Z (바닥)
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

def remove_brick(model, brick_id: str) -> Dict:
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

def restore_brick(model, backup: Dict):
    """Restore a removed brick"""
    from ldr_converter import PlacedBrick, Vector3
    brick = PlacedBrick(
        id=backup["id"],
        part_id=backup["part_id"],
        position=Vector3(x=backup["x"], y=backup["y"], z=backup["z"]),
        rotation=0,
        color_code=backup["color"],
        layer=0
    )
    model.bricks.append(brick)
    return {"success": True, "restored": backup["id"]}

def add_brick(model, part_id, x, y, z, color: int) -> Dict:
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
        rotation=0,
        color_code=color,
        layer=0
    )
    model.bricks.append(brick)
    return {"success": True, "brick_id": new_id}

def rollback_model(state) -> Dict:
    """Restore model to original"""
    state["model"] = copy.deepcopy(state["model_backup"])
    state["total_removed"] = 0
    state["action_history"] = []
    return {"success": True, "bricks": state["original_brick_count"]}


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
        if bb["bbox"].max_y >= ground_y - 2:
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


def generate_support_candidates(floating_brick, nearby_stable, parts_db):
    """
    지지대 후보 위치 생성
    Returns: 후보 위치 리스트 [{x, y, z, part_id, color}, ...]
    """
    from ldr_converter import get_brick_bbox

    candidates = []
    f_bbox = get_brick_bbox(floating_brick, parts_db)
    if not f_bbox:
        return []

    # 부유 브릭 아래에 지지대 위치 계산
    # LDraw에서 Y가 클수록 아래
    support_y = f_bbox.max_y  # 부유 브릭 바닥 바로 아래

    # 스터드 정렬 (20 LDU 단위)
    def snap_to_stud(val):
        return round(val / 20) * 20

    # 후보 1: 부유 브릭 바로 아래
    candidates.append({
        "x": snap_to_stud(floating_brick.position.x),
        "y": support_y,
        "z": snap_to_stud(floating_brick.position.z),
        "part_id": "3005",  # 1x1 brick
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
            "part_id": "3004",  # 1x2 brick
            "color": s_brick.color_code,
            "description": f"bridge toward stable brick {s_brick.id}"
        })

    return candidates


def try_add_brick_with_validation(model, candidate, parts_db):
    """
    브릭 추가 시도 + 검증
    Returns: {"success": bool, "brick_id": str, "new_floating": int}
    """
    # 1. 브릭 추가
    result = add_brick(
        model,
        candidate["part_id"],
        candidate["x"],
        candidate["y"],
        candidate["z"],
        candidate["color"]
    )

    if not result["success"]:
        return {"success": False, "reason": "add_brick failed"}

    brick_id = result["brick_id"]

    # 2. 검증
    state = get_model_state(model, parts_db)

    # 3. 결과 반환
    return {
        "success": True,
        "brick_id": brick_id,
        "floating_count": state["floating_count"],
        "collision_count": state["collision_count"]
    }
