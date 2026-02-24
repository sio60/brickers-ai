"""모델 상태 조회 및 브릭 추가/삭제/롤백 함수"""
import os
import json
import copy
import base64
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, TYPE_CHECKING

from ..config import get_config
from ..constants import (
    BRICK_HEIGHT, PLATE_HEIGHT, LDU_PER_STUD, PLATE_PART_IDS,
    GROUND_TOLERANCE, _get_parts_db,
)

if TYPE_CHECKING:
    from ldr_converter import BrickModel, PlacedBrick


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
        model_name = vision_model or getattr(config, 'vision_model', None) or "gpt-4o"

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
    """Get current model state using brick_judge (Rust)"""
    import sys
    from ldr_converter import model_to_ldr

    # brick_judge 프로젝트 루트 경로 추가
    project_root = str(Path(__file__).resolve().parents[5])  # brickers-ai
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from brick_judge.parser import parse_ldr_string
    from brick_judge.physics import full_judge, calc_score_from_issues

    if not model.bricks:
        return {
            "total_bricks": 0,
            "floating_count": 0,
            "collision_count": 0,
            "floating_bricks": [],
            "verification_result": None
        }

    # BrickModel → LDR 문자열 → ParsedModel → Rust 물리 검증
    ldr_text = model_to_ldr(model, parts_db, skip_validation=True, skip_physics=True)
    parsed = parse_ldr_string(ldr_text, model.name if hasattr(model, "name") else "model")
    issues = full_judge(parsed)
    score = calc_score_from_issues(issues, len(parsed.bricks))

    # floating/isolated 카운트
    floating_ids = set()
    isolated_ids = set()
    for issue in issues:
        if issue.issue_type.value == "floating":
            if issue.brick_id is not None:
                floating_ids.add(issue.brick_id)
        elif issue.issue_type.value == "isolated":
            if issue.brick_id is not None:
                isolated_ids.add(issue.brick_id)

    # floating_bricks 리스트 (brick_judge ID = 순서 인덱스 → BrickModel 매핑)
    floating_list = []
    for bid in list(floating_ids)[:15]:
        if 0 <= bid < len(model.bricks):
            b = model.bricks[bid]
            floating_list.append({
                "id": b.id,
                "part_id": b.part_id,
                "position": {"x": b.position.x, "y": b.position.y, "z": b.position.z},
                "color": b.color_code
            })

    return {
        "total_bricks": len(model.bricks),
        "floating_count": len(floating_ids),
        "collision_count": len(isolated_ids),
        "floating_bricks": floating_list,
        "verification_result": None,
        "score": score,
        "is_valid": score >= 70,
        "evidence": issues
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
