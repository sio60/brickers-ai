"""비전 LLM 기반 브릭 모델 분석 (7방향 멀티앵글)

리팩토링:
- _extract_json(): 공통 JSON 파싱 헬퍼
- _call_vision_api(): 공통 API 호출 헬퍼
"""
import os
import json
from typing import Dict, Any, List, Optional
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# API 타임아웃 (초)
API_TIMEOUT = 120.0


def _extract_json(text: str, is_array: bool = False) -> Optional[Dict | List]:
    """텍스트에서 JSON 추출 (공통 헬퍼)

    Args:
        text: LLM 응답 텍스트
        is_array: True면 배열 [], False면 객체 {} 찾기

    Returns:
        파싱된 JSON 또는 None
    """
    try:
        if is_array:
            start = text.find('[')
            end = text.rfind(']') + 1
        else:
            start = text.find('{')
            end = text.rfind('}') + 1

        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except json.JSONDecodeError as e:
        print(f"  [WARNING] JSON 파싱 실패: {e}")
    return None


def _call_vision_api(content: List[Dict], max_tokens: int = 1000) -> str:
    """Vision API 호출 (공통 헬퍼)

    Args:
        content: 메시지 content (텍스트 + 이미지들)
        max_tokens: 최대 응답 토큰

    Returns:
        응답 텍스트 또는 빈 문자열
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=0.0,
            timeout=API_TIMEOUT
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  [WARNING] Vision API 호출 실패: {e}")
        return ""


def _add_images_to_content(content: List[Dict], images: Dict[str, str]) -> None:
    """이미지들을 content에 추가 (공통 헬퍼)"""
    for angle_name, b64 in images.items():
        if b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })


def analyze_multi_angle(images: Dict[str, str], model_name: str = "unknown") -> Dict[str, Any]:
    """
    7방향 이미지를 분석하여 모델 정보 + 보정 필요 영역 추출

    Args:
        images: {angle_name: base64_image, ...} 7방향 이미지
        model_name: 모델 이름 (힌트)

    Returns:
        {
            "model_type": "animal",
            "name": "cat",
            "description": "4족 동물, 검정색 몸통, 회색 다리",
            "legs": 4,
            "support_needed": [
                {"location": "front left leg bottom", "reason": "다리 끝에 지지대 필요"},
                {"location": "front right leg bottom", "reason": "다리 끝에 지지대 필요"},
                ...
            ],
            "do_not_touch": [
                {"location": "belly area", "reason": "배 아래 빈 공간은 자연스러운 형태"},
                {"location": "neck area", "reason": "목 아래 공간 유지"}
            ],
            "overall_assessment": "4족 동물로 다리 4개 끝에 지지대 필요, 배 아래는 비워둬야 함"
        }
    """
    # 이미지들을 content로 구성
    content = [
        {"type": "text", "text": f"""You are a LEGO brick model expert. Analyze these 5 images of the same model from different angles.

Model name hint: {model_name} (ignore if it conflicts with what you see)

These are 5 camera angles of the SAME model:
- FRONT: 정면 (Z- 방향)
- BACK: 뒷면 (Z+ 방향)
- RIGHT: 오른쪽 측면 (X+ 방향)
- BOTTOM: 아래에서 올려다본 뷰 (다리 바닥 등 보임)
- FRONT_RIGHT: 대각선 뷰 (전체 형태 파악)

CLASSIFICATION RULES:
1. Only classify if you are 80%+ confident
2. If unsure, use "unknown" - do NOT guess

Categories:
- animal: living creature with body parts (legs, head, tail)
- vehicle: transportation (wheels, wings, hull)
- building: static structure (walls, roof)
- robot: mechanical humanoid
- figure: human character
- plant: tree, flower, vegetation
- unknown: cannot determine with confidence

Analyze ALL images and determine:
1. What is this model? (type, name, features)
2. Where does it need support to stand properly?
3. Where should NOT be touched? (intentional empty spaces)

IMPORTANT for 4-legged animals:
- Legs need support at the BOTTOM (feet/hooves)
- Belly area should stay EMPTY (this is natural, not a problem)
- Look at BOTTOM view carefully to see the leg positions

Return JSON:
{{
    "model_type": "one of the categories above",
    "confidence": 0-100,
    "name": "specific name (cat, horse, car, etc)",
    "description": "brief visual description",
    "legs": number or 0,
    "support_needed": [
        {{"location": "specific area", "reason": "why support needed"}}
    ],
    "do_not_touch": [
        {{"location": "specific area", "reason": "why should not touch"}}
    ],
    "overall_assessment": "summary of what needs to be done"
}}

Return only valid JSON."""}
    ]

    # 5방향 이미지 추가
    for angle_name, b64 in images.items():
        if b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": content}],
        max_tokens=1500,
        temperature=0.0
    )

    result_text = response.choices[0].message.content
    start = result_text.find('{')
    end = result_text.rfind('}') + 1
    if start >= 0 and end > start:
        return json.loads(result_text[start:end])
    return {"error": "Failed to parse", "raw": result_text}


def score_similarity_multi(images: Dict[str, str], target: str) -> Dict[str, Any]:
    """
    7방향 이미지로 유사도 점수 매기기 (멘토 제안)

    Args:
        images: 7방향 이미지
        target: 목표 설명 ("고양이", "말" 등)

    Returns:
        {"score": 0-100, "reasoning": "...", "looks_like": "...", ...}
    """
    content = [
        {"type": "text", "text": f"""Rate how well this LEGO brick model looks like: "{target}"

You are seeing 7 different angles of the same model.

Score from 0 to 100:
- 90-100: Clearly recognizable as {target}
- 70-89: Mostly looks like {target}
- 50-69: Somewhat resembles {target}
- 30-49: Hard to recognize as {target}
- 0-29: Does not look like {target}

Return JSON:
{{
    "score": 0-100,
    "reasoning": "why this score",
    "looks_like": "what it actually looks like",
    "missing": ["missing features"],
    "extra": ["unnecessary elements"]
}}

Return only valid JSON."""}
    ]

    for angle_name, b64 in images.items():
        if b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": content}],
        max_tokens=500,
        temperature=0.0
    )

    result_text = response.choices[0].message.content
    start = result_text.find('{')
    end = result_text.rfind('}') + 1
    if start >= 0 and end > start:
        return json.loads(result_text[start:end])
    return {"error": "Failed to parse", "raw": result_text}


def verify_correction_multi(
    before_images: Dict[str, str],
    after_images: Dict[str, str],
    target: str
) -> Dict[str, Any]:
    """
    보정 전후 7방향 비교

    Args:
        before_images: 보정 전 7방향 이미지
        after_images: 보정 후 7방향 이미지
        target: 목표 설명

    Returns:
        {
            "improved": true/false,
            "before_score": 0-100,
            "after_score": 0-100,
            "shape_preserved": true/false,
            "recommendation": "keep" or "rollback",
            "reasoning": "..."
        }
    """
    content = [
        {"type": "text", "text": f"""Compare BEFORE and AFTER correction of this LEGO model.
Target: {target}

First 7 images = BEFORE correction
Next 7 images = AFTER correction

Evaluate:
1. Does AFTER still look like {target}? (shape_preserved)
2. Is AFTER better than BEFORE? (improved)
3. Score both (0-100)

Return JSON:
{{
    "improved": true/false,
    "before_score": 0-100,
    "after_score": 0-100,
    "shape_preserved": true/false,
    "recommendation": "keep" or "rollback",
    "reasoning": "brief explanation"
}}

IMPORTANT: If shape is destroyed (e.g., animal became a block), recommend "rollback".

Return only valid JSON."""}
    ]

    # Before 이미지들
    for angle_name, b64 in before_images.items():
        if b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })

    # After 이미지들
    for angle_name, b64 in after_images.items():
        if b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": content}],
        max_tokens=500,
        temperature=0.0
    )

    result_text = response.choices[0].message.content
    start = result_text.find('{')
    end = result_text.rfind('}') + 1
    if start >= 0 and end > start:
        return json.loads(result_text[start:end])
    return {"error": "Failed to parse", "raw": result_text}


def find_problems(images: Dict[str, str], target: str) -> Dict[str, Any]:
    """
    5방향 이미지에서 잘못 배치된 브릭 찾기

    Args:
        images: 5방향 이미지 ({"FRONT": ..., "BACK": ..., "RIGHT": ..., "BOTTOM": ..., "FRONT_RIGHT": ...})
        target: 목표 모델 (예: "cat", "horse")

    Returns:
        {
            "problems": [
                {
                    "location": "back right leg",
                    "issue": "leg is pointing outward instead of downward",
                    "severity": "high",
                    "visible_in_angles": ["FRONT", "RIGHT"]
                }
            ],
            "overall_quality": 0-100,
            "major_issues_count": int
        }
    """
    content = [
        {"type": "text", "text": f"""You are a LEGO brick model expert. Analyze these 5 images of a model that should look like: "{target}"

CAMERA ANGLES (each image has a label in the top-left corner):
- FRONT: Front view (model facing you, Z- direction)
- BACK: Back view (behind the model, Z+ direction)
- RIGHT: Right side view (model's right side, X+ direction)
- BOTTOM: Bottom view (looking up, see legs/underside)
- FRONT_RIGHT: Front-right diagonal view (overall shape)

COORDINATE SYSTEM (LDraw):
- Z negative = FRONT (앞)
- Z positive = BACK (뒤)
- X negative = LEFT (왼쪽)
- X positive = RIGHT (오른쪽)

DIRECTION RULES:
- "front left leg" = leg at Z- and X- position
- "back right leg" = leg at Z+ and X+ position
- In FRONT view: LEFT side of model appears on RIGHT side of image
- In BACK view: LEFT side of model appears on LEFT side of image

Your job is to find INCORRECTLY POSITIONED bricks, NOT to suggest adding support bricks.

Look for:
1. Parts in WRONG POSITION (e.g., leg pointing wrong direction)
2. Parts MISSING (e.g., only 3 legs when there should be 4)
3. Parts ROTATED incorrectly
4. Parts DETACHED or floating
5. Parts that break overall SHAPE

For each problem:
- Location: use "front/back" + "left/right" + part name (e.g., "front right leg")
- Issue: be specific (e.g., "pointing outward instead of downward")
- Angles: which views show this problem

Return JSON:
{{
    "problems": [
        {{
            "location": "front right leg",
            "issue": "pointing outward instead of downward",
            "severity": "high/medium/low",
            "visible_in_angles": ["FRONT", "RIGHT"]
        }}
    ],
    "overall_quality": 0-100,
    "major_issues_count": number of high severity issues
}}

Quality scoring guide:
- 90-100: No issues, model looks perfect
- 70-89: Minor issues, overall shape is good
- 50-69: Some problems, shape is recognizable
- 30-49: Major issues, hard to recognize
- 0-29: Severe problems, shape is broken

If the model looks correct, return empty problems array with high quality score (90+).

Return only valid JSON."""}
    ]

    for angle_name, b64 in images.items():
        if b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": content}],
        max_tokens=1000,
        temperature=0.0
    )

    result_text = response.choices[0].message.content
    start = result_text.find('{')
    end = result_text.rfind('}') + 1
    if start >= 0 and end > start:
        return json.loads(result_text[start:end])
    return {"error": "Failed to parse", "raw": result_text}


def suggest_corrections(
    images: Dict[str, str],
    problems: List[Dict],
    target: str
) -> Dict[str, Any]:
    """
    발견된 문제에 대한 수정 방법 제안

    Args:
        images: 7방향 이미지
        problems: find_problems()에서 반환된 문제 목록
        target: 목표 모델

    Returns:
        {
            "corrections": [
                {
                    "problem_location": "back right leg",
                    "action": "rotate",  # rotate, move, delete, add
                    "direction": "rotate 90 degrees clockwise when viewed from bottom",
                    "priority": 1,
                    "expected_improvement": "leg will point downward like the other legs"
                }
            ],
            "correction_order": ["back right leg", "tail"],
            "estimated_improvement": 0-100
        }
    """
    problems_text = json.dumps(problems, ensure_ascii=False, indent=2)

    content = [
        {"type": "text", "text": f"""You are a LEGO brick model expert. These 7 images show a model that should look like: "{target}"

The following problems were identified:
{problems_text}

For each problem, suggest HOW to fix it. Be SPECIFIC about:
1. What ACTION to take: rotate, move, delete, or add bricks
2. What DIRECTION to move/rotate (use angle references like "when viewed from angle_6")
3. The PRIORITY (fix most important issues first)

IMPORTANT CONSTRAINTS:
- Suggest the MINIMUM changes needed
- Don't suggest adding support bricks unless absolutely necessary
- Focus on repositioning existing bricks correctly
- Consider how fixing one part might affect others

Return JSON:
{{
    "corrections": [
        {{
            "problem_location": "location from problems list",
            "action": "rotate/move/delete/add",
            "direction": "specific direction description",
            "priority": 1-5 (1 is highest),
            "expected_improvement": "what this fix will achieve"
        }}
    ],
    "correction_order": ["location1", "location2"],
    "estimated_improvement": 0-100 (how much better the model will look after all fixes)
}}

Return only valid JSON."""}
    ]

    for angle_name, b64 in images.items():
        if b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": content}],
        max_tokens=1000,
        temperature=0.0
    )

    result_text = response.choices[0].message.content
    start = result_text.find('{')
    end = result_text.rfind('}') + 1
    if start >= 0 and end > start:
        return json.loads(result_text[start:end])
    return {"error": "Failed to parse", "raw": result_text}


def map_to_coordinates(
    images: Dict[str, str],
    correction: Dict,
    brick_list: List[Dict]
) -> Dict[str, Any]:
    """
    LLM의 텍스트 설명을 실제 브릭 좌표로 매핑

    Args:
        images: 7방향 이미지
        correction: suggest_corrections()에서 반환된 단일 수정 사항
        brick_list: 모델의 브릭 목록 [{id, x, y, z, part_id, color}, ...]

    Returns:
        {
            "target_brick_ids": ["brick_123", "brick_124"],
            "action": "rotate",
            "transform": {
                "rotation": {"x": 0, "y": 90, "z": 0},  # or
                "translation": {"x": 20, "y": 0, "z": 0}  # or
                "delete": true
            },
            "confidence": 0-100
        }
    """
    # 브릭 목록에 방향 라벨 추가
    def get_direction(x, z):
        parts = []
        if z < -10: parts.append("FRONT")
        elif z > 10: parts.append("BACK")
        if x < -10: parts.append("LEFT")
        elif x > 10: parts.append("RIGHT")
        return "-".join(parts) if parts else "CENTER"

    brick_summary = []
    for b in brick_list[:50]:
        brick_summary.append({
            "id": b.get("id", "unknown"),
            "x": b.get("x", 0),
            "y": b.get("y", 0),
            "z": b.get("z", 0),
            "part": b.get("part_id", "unknown"),
            "direction": get_direction(b.get("x", 0), b.get("z", 0))
        })

    content = [
        {"type": "text", "text": f"""You are a LEGO brick coordinate expert.

Given these 7 images of a model and the following correction needed:
- Location: {correction.get('problem_location', 'unknown')}
- Action: {correction.get('action', 'unknown')}
- Direction: {correction.get('direction', 'unknown')}

Brick list with direction labels:
{json.dumps(brick_summary, indent=2)}

COORDINATE SYSTEM (LDraw):
- Z negative = FRONT (앞)
- Z positive = BACK (뒤)
- X negative = LEFT (왼쪽)
- X positive = RIGHT (오른쪽)
- Y negative = UP (위)
- Y positive = DOWN (아래)

Each brick has a "direction" field showing its position (e.g., "FRONT-LEFT", "BACK-RIGHT").
Use this to match the problem location to the correct brick.

Example: "front right leg" → look for bricks with direction "FRONT-RIGHT"

TRANSLATION RULES:
- X and Z: multiples of 20
- Y: multiples of 24 (brick) or 8 (plate)

Return JSON:
{{
    "target_brick_ids": ["id1", "id2"],
    "action": "rotate/move/delete",
    "transform": {{
        "rotation": {{"x": 0, "y": 90, "z": 0}},
        "translation": {{"x": 20, "y": 0, "z": -20}}
    }},
    "confidence": 0-100,
    "reasoning": "why these bricks"
}}

Return only valid JSON."""}
    ]

    for angle_name, b64 in images.items():
        if b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": content}],
        max_tokens=800,
        temperature=0.0
    )

    result_text = response.choices[0].message.content
    start = result_text.find('{')
    end = result_text.rfind('}') + 1
    if start >= 0 and end > start:
        return json.loads(result_text[start:end])
    return {"error": "Failed to parse", "raw": result_text}


def find_reference_part(
    images: Dict[str, str],
    bad_part: str,
    target: str
) -> Dict[str, Any]:
    """
    잘못된 부분의 참조가 될 정상 부분 찾기
    예: "back left leg"이 잘못됐으면 → "back right leg"을 참조로 찾기

    Args:
        images: 7방향 이미지
        bad_part: 잘못된 부분 (예: "back left leg")
        target: 목표 모델 (예: "cat")

    Returns:
        {
            "reference_part": "back right leg",
            "relationship": "mirror_x",  # X축 대칭
            "visible_in_angles": ["angle_1", "angle_3"],
            "confidence": 0-100
        }
    """
    content = [
        {"type": "text", "text": f"""You are a LEGO brick model expert analyzing a model that should look like: "{target}"

The part "{bad_part}" is incorrectly positioned.

Find a REFERENCE part that can be used to fix it. For example:
- If "back left leg" is wrong, find "back right leg" as reference
- If "left ear" is wrong, find "right ear" as reference
- If "tail" is wrong, describe where it should connect

Return JSON:
{{
    "reference_part": "name of the correct part to use as reference",
    "relationship": "mirror_x" or "mirror_z" or "same_pattern" or "connect_to",
    "visible_in_angles": ["angle_X", "angle_Y"],
    "description": "how the bad part should look based on the reference",
    "confidence": 0-100
}}

If no good reference exists, return confidence: 0.

Return only valid JSON."""}
    ]

    for angle_name, b64 in images.items():
        if b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": content}],
        max_tokens=500,
        temperature=0.0
    )

    result_text = response.choices[0].message.content
    start = result_text.find('{')
    end = result_text.rfind('}') + 1
    if start >= 0 and end > start:
        return json.loads(result_text[start:end])
    return {"error": "Failed to parse", "raw": result_text}


def plan_rebuild(
    images: Dict[str, str],
    bad_part: str,
    reference_part: str,
    relationship: str,
    brick_list: List[Dict]
) -> Dict[str, Any]:
    """
    잘못된 부분과 참조 부분의 브릭 ID만 찾기 (좌표 계산은 알고리즘이 함)

    Args:
        images: 7방향 이미지
        bad_part: 잘못된 부분 이름
        reference_part: 참조할 정상 부분 이름
        relationship: "mirror_x", "mirror_z", "same_pattern"
        brick_list: 모델의 브릭 목록

    Returns:
        {
            "delete_brick_ids": ["b001", "b002"],
            "reference_brick_ids": ["b010", "b011"],
            "relationship": "mirror_x",
            "confidence": 0-100
        }
    """
    # 브릭 목록에 방향 라벨 추가
    def get_direction(x, z):
        parts = []
        if z < -10: parts.append("FRONT")
        elif z > 10: parts.append("BACK")
        if x < -10: parts.append("LEFT")
        elif x > 10: parts.append("RIGHT")
        return "-".join(parts) if parts else "CENTER"

    brick_summary = []
    for b in brick_list[:80]:
        brick_summary.append({
            "id": b.get("id"),
            "x": b.get("x"),
            "y": b.get("y"),
            "z": b.get("z"),
            "part": b.get("part_id"),
            "color": b.get("color"),
            "direction": get_direction(b.get("x", 0), b.get("z", 0))
        })

    # LLM은 브릭 ID만 찾음 (좌표 계산 X)
    content = [
        {"type": "text", "text": f"""You are a LEGO brick identification expert.

Given these images and brick list, identify which bricks belong to "{bad_part}" and "{reference_part}".

Brick list with direction labels:
{json.dumps(brick_summary, indent=2)}

COORDINATE SYSTEM (LDraw):
- Z negative = FRONT (앞)
- Z positive = BACK (뒤)
- X negative = LEFT (왼쪽)
- X positive = RIGHT (오른쪽)

Each brick has a "direction" field showing its position (e.g., "FRONT-LEFT", "BACK-RIGHT").

Example mappings:
- "front right leg" → bricks with direction containing "FRONT" and "RIGHT"
- "back left leg" → bricks with direction containing "BACK" and "LEFT"
- "front left leg" → bricks with direction containing "FRONT" and "LEFT"

Tasks (ONLY identify brick IDs, do NOT calculate coordinates):
1. Find bricks belonging to "{bad_part}" (these will be deleted)
2. Find bricks belonging to "{reference_part}" (these will be used as reference)

IMPORTANT: Only return brick IDs that exist in the brick list above.

Return JSON:
{{
    "delete_brick_ids": ["id1", "id2"],
    "reference_brick_ids": ["id3", "id4"],
    "confidence": 0-100,
    "reasoning": "why these bricks were selected"
}}

Return only valid JSON."""}
    ]

    for angle_name, b64 in images.items():
        if b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": content}],
        max_tokens=800,
        temperature=0.0
    )

    result_text = response.choices[0].message.content
    start = result_text.find('{')
    end = result_text.rfind('}') + 1
    if start >= 0 and end > start:
        result = json.loads(result_text[start:end])
        result["relationship"] = relationship  # 알고리즘에서 사용할 변환 방식
        return result
    return {"error": "Failed to parse", "raw": result_text}


if __name__ == "__main__":
    from ldr_renderer import render_ldr_multi_angle

    test_ldr = r"C:\Users\301\Desktop\Brickers 관련 문서\테스트 LDR\냥이.ldr"

    # 5방향 (TOP, LEFT 제외)
    VISION_ANGLES = ["FRONT", "BACK", "RIGHT", "BOTTOM", "FRONT_RIGHT"]

    print("=== 5방향 렌더링 ===")
    images = render_ldr_multi_angle(test_ldr, angles=VISION_ANGLES)
    print(f"렌더링 완료: {len(images)}개 각도")

    print("\n=== 5방향 분석 ===")
    analysis = analyze_multi_angle(images, "cat")
    print(json.dumps(analysis, indent=2, ensure_ascii=False))

    print("\n=== 유사도 점수 ===")
    score = score_similarity_multi(images, "cat")
    print(json.dumps(score, indent=2, ensure_ascii=False))

    print("\n=== 문제점 찾기 ===")
    problems = find_problems(images, "cat")
    print(json.dumps(problems, indent=2, ensure_ascii=False))

    if problems.get("problems"):
        print("\n=== 수정 방법 제안 ===")
        corrections = suggest_corrections(images, problems["problems"], "cat")
        print(json.dumps(corrections, indent=2, ensure_ascii=False))
