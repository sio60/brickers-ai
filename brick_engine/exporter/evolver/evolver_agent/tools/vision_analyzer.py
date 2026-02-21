"""비전 LLM 기반 브릭 모델 분석 (멀티앵글)

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
    멀티앵글 이미지를 분석하여 모델 정보 + 보정 필요 영역 추출

    Args:
        images: {angle_name: base64_image, ...} 멀티앵글 이미지
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
        {"type": "text", "text": f"""당신은 레고 브릭 모델 전문가입니다. 같은 모델을 다른 각도에서 촬영한 5장의 이미지를 분석하세요.

모델 이름 힌트: {model_name} (보이는 것과 다르면 무시)

동일 모델의 5개 카메라 앵글:
- FRONT: 정면 (Z- 방향)
- BACK: 뒷면 (Z+ 방향)
- RIGHT: 오른쪽 측면 (X+ 방향)
- BOTTOM: 아래에서 올려다본 뷰 (다리 바닥 등 보임)
- FRONT_RIGHT: 대각선 뷰 (전체 형태 파악)

분류 규칙:
1. 80% 이상 확신할 때만 분류
2. 확실하지 않으면 "unknown" 사용 — 추측 금지

카테고리:
- animal: 신체 부위가 있는 생물 (다리, 머리, 꼬리)
- vehicle: 이동수단 (바퀴, 날개, 선체)
- building: 정적 구조물 (벽, 지붕)
- robot: 기계식 인간형
- figure: 사람 캐릭터
- plant: 나무, 꽃, 식물
- unknown: 확신할 수 없음

모든 이미지를 보고 판단:
1. 이 모델은 무엇인가? (유형, 이름, 특징)
2. 어디에 지지대가 필요한가?
3. 어디를 건드리면 안 되는가? (의도적 빈 공간)

4족 동물 주의사항:
- 다리 바닥(발/발굽)에 지지대 필요
- 배 아래 빈 공간은 자연스러운 형태 — 문제 아님
- BOTTOM 뷰를 주의 깊게 보고 다리 위치 확인

JSON으로 반환:
{{
    "model_type": "위 카테고리 중 하나",
    "confidence": 0-100,
    "name": "구체적 이름 (cat, horse, car 등)",
    "description": "간단한 시각적 설명",
    "legs": 다리 수 또는 0,
    "support_needed": [
        {{"location": "구체적 위치", "reason": "지지대가 필요한 이유"}}
    ],
    "do_not_touch": [
        {{"location": "구체적 위치", "reason": "건드리면 안 되는 이유"}}
    ],
    "overall_assessment": "필요한 작업 요약"
}}

유효한 JSON만 반환."""}
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
    멀티앵글 이미지로 유사도 점수 매기기 (멘토 제안)

    Args:
        images: 멀티앵글 이미지
        target: 목표 설명 ("고양이", "말" 등)

    Returns:
        {"score": 0-100, "reasoning": "...", "looks_like": "...", ...}
    """
    content = [
        {"type": "text", "text": f"""이 레고 브릭 모델이 "{target}"과(와) 얼마나 닮았는지 평가하세요.

같은 모델을 여러 각도에서 촬영한 이미지입니다.

0~100점 채점 기준:
- 90-100: {target}임이 명확히 인식됨
- 70-89: 대체로 {target}처럼 보임
- 50-69: 어느 정도 {target}과 닮음
- 30-49: {target}으로 인식하기 어려움
- 0-29: {target}과 전혀 다름

JSON으로 반환:
{{
    "score": 0-100,
    "reasoning": "이 점수를 준 이유",
    "looks_like": "실제로 무엇처럼 보이는지",
    "missing": ["빠진 특징들"],
    "extra": ["불필요한 요소들"]
}}

유효한 JSON만 반환."""}
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
    보정 전후 멀티앵글 비교

    Args:
        before_images: 보정 전 멀티앵글 이미지
        after_images: 보정 후 멀티앵글 이미지
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
        {"type": "text", "text": f"""이 레고 모델의 보정 전후를 비교하세요.
목표 모델: {target}

앞쪽 이미지들 = 보정 전
뒤쪽 이미지들 = 보정 후

평가 항목:
1. 보정 후에도 {target}처럼 보이는가? (shape_preserved)
2. 보정 후가 보정 전보다 나은가? (improved)
3. 각각 점수 (0-100)

JSON으로 반환:
{{
    "improved": true/false,
    "before_score": 0-100,
    "after_score": 0-100,
    "shape_preserved": true/false,
    "recommendation": "keep" 또는 "rollback",
    "reasoning": "간단한 설명"
}}

중요: 형태가 망가졌으면 (예: 동물이 블록이 됨) "rollback" 권장.

유효한 JSON만 반환."""}
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
        {"type": "text", "text": f"""당신은 레고 브릭 모델 전문가입니다. "{target}"처럼 보여야 하는 모델의 5방향 이미지를 분석하세요.

카메라 앵글 (각 이미지 좌상단에 라벨 표시):
- FRONT: 정면 (Z- 방향)
- BACK: 뒷면 (Z+ 방향)
- RIGHT: 오른쪽 측면 (X+ 방향)
- BOTTOM: 아래에서 올려다본 뷰 (다리/밑면 보임)
- FRONT_RIGHT: 앞-오른쪽 대각선 뷰 (전체 형태)

좌표계 (LDraw):
- Z 음수 = FRONT (앞)
- Z 양수 = BACK (뒤)
- X 음수 = LEFT (왼쪽)
- X 양수 = RIGHT (오른쪽)

방향 규칙:
- "front left leg" = Z-이고 X-인 위치의 다리
- "back right leg" = Z+이고 X+인 위치의 다리
- FRONT 뷰에서: 모델의 왼쪽이 이미지의 오른쪽에 보임
- BACK 뷰에서: 모델의 왼쪽이 이미지의 왼쪽에 보임

당신의 역할: 잘못 배치된 브릭 찾기. 지지대 추가 제안이 아닙니다.

찾아야 할 것:
1. 잘못된 위치의 부품 (예: 다리가 엉뚱한 방향)
2. 빠진 부품 (예: 4개여야 할 다리가 3개)
3. 잘못 회전된 부품
4. 분리되거나 떠 있는 부품
5. 전체 형태를 깨뜨리는 부품

각 문제에 대해:
- location: "front/back" + "left/right" + 부위명 (예: "front right leg")
- issue: 구체적으로 (예: "아래로 향해야 하는데 바깥으로 향함")
- angles: 어떤 뷰에서 보이는지

JSON으로 반환:
{{
    "problems": [
        {{
            "location": "front right leg",
            "issue": "아래로 향해야 하는데 바깥으로 향함",
            "severity": "high/medium/low",
            "visible_in_angles": ["FRONT", "RIGHT"]
        }}
    ],
    "overall_quality": 0-100,
    "major_issues_count": high severity 문제 수
}}

품질 채점 기준:
- 90-100: 문제 없음, 완벽
- 70-89: 사소한 문제, 전체 형태 양호
- 50-69: 일부 문제, 형태 인식 가능
- 30-49: 심각한 문제, 인식 어려움
- 0-29: 매우 심각, 형태 붕괴

모델이 정상이면 problems를 빈 배열로, quality를 90+ 으로 반환.

유효한 JSON만 반환."""}
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
        images: 멀티앵글 이미지
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
        {"type": "text", "text": f"""당신은 레고 브릭 모델 전문가입니다. "{target}"처럼 보여야 하는 모델의 이미지입니다.

발견된 문제점:
{problems_text}

각 문제에 대해 수정 방법을 구체적으로 제안하세요:
1. 어떤 행동: rotate(회전), move(이동), delete(삭제), add(추가)
2. 어떤 방향으로: 구체적 방향 설명
3. 우선순위: 가장 중요한 문제부터

제약 조건:
- 최소한의 변경만 제안
- 지지대 추가는 꼭 필요한 경우만
- 기존 브릭 재배치에 집중
- 한 부분 수정이 다른 부분에 미치는 영향 고려

JSON으로 반환:
{{
    "corrections": [
        {{
            "problem_location": "문제 목록의 location",
            "action": "rotate/move/delete/add",
            "direction": "구체적 방향 설명",
            "priority": 1-5 (1이 최우선),
            "expected_improvement": "이 수정으로 기대되는 개선"
        }}
    ],
    "correction_order": ["location1", "location2"],
    "estimated_improvement": 0-100 (전체 수정 후 예상 개선도)
}}

유효한 JSON만 반환."""}
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
        images: 멀티앵글 이미지
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
        {"type": "text", "text": f"""당신은 레고 브릭 좌표 전문가입니다.

모델 이미지와 필요한 수정 사항:
- 위치: {correction.get('problem_location', 'unknown')}
- 행동: {correction.get('action', 'unknown')}
- 방향: {correction.get('direction', 'unknown')}

방향 라벨이 포함된 브릭 목록:
{json.dumps(brick_summary, indent=2)}

좌표계 (LDraw):
- Z 음수 = FRONT (앞)
- Z 양수 = BACK (뒤)
- X 음수 = LEFT (왼쪽)
- X 양수 = RIGHT (오른쪽)
- Y 음수 = UP (위)
- Y 양수 = DOWN (아래)

각 브릭의 "direction" 필드로 위치 확인 (예: "FRONT-LEFT", "BACK-RIGHT").
문제 위치와 올바른 브릭을 매칭하세요.

예시: "front right leg" → direction이 "FRONT-RIGHT"인 브릭 찾기

이동 규칙:
- X, Z: 20의 배수
- Y: 24의 배수 (브릭) 또는 8의 배수 (플레이트)

JSON으로 반환:
{{
    "target_brick_ids": ["id1", "id2"],
    "action": "rotate/move/delete",
    "transform": {{
        "rotation": {{"x": 0, "y": 90, "z": 0}},
        "translation": {{"x": 20, "y": 0, "z": -20}}
    }},
    "confidence": 0-100,
    "reasoning": "이 브릭을 선택한 이유"
}}

유효한 JSON만 반환."""}
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
        images: 멀티앵글 이미지
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
        {"type": "text", "text": f"""당신은 레고 브릭 모델 전문가입니다. "{target}"처럼 보여야 하는 모델을 분석 중입니다.

"{bad_part}" 부분이 잘못 배치되어 있습니다.

수정의 참조가 될 정상 부분을 찾으세요. 예시:
- "back left leg"이 잘못됐으면 → "back right leg"을 참조로
- "left ear"가 잘못됐으면 → "right ear"를 참조로
- "tail"이 잘못됐으면 → 어디에 연결되어야 하는지 설명

JSON으로 반환:
{{
    "reference_part": "참조로 사용할 정상 부분의 이름",
    "relationship": "mirror_x" 또는 "mirror_z" 또는 "same_pattern" 또는 "connect_to",
    "visible_in_angles": ["angle_X", "angle_Y"],
    "description": "참조를 기반으로 잘못된 부분이 어떻게 보여야 하는지",
    "confidence": 0-100
}}

적절한 참조가 없으면 confidence: 0 반환.

유효한 JSON만 반환."""}
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
        images: 멀티앵글 이미지
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
        {"type": "text", "text": f"""당신은 레고 브릭 식별 전문가입니다.

이미지와 브릭 목록을 보고, "{bad_part}"과 "{reference_part}"에 해당하는 브릭을 찾으세요.

방향 라벨이 포함된 브릭 목록:
{json.dumps(brick_summary, indent=2)}

좌표계 (LDraw):
- Z 음수 = FRONT (앞)
- Z 양수 = BACK (뒤)
- X 음수 = LEFT (왼쪽)
- X 양수 = RIGHT (오른쪽)

각 브릭의 "direction" 필드로 위치 확인 (예: "FRONT-LEFT", "BACK-RIGHT").

매핑 예시:
- "front right leg" → direction에 "FRONT"과 "RIGHT"가 포함된 브릭
- "back left leg" → direction에 "BACK"과 "LEFT"가 포함된 브릭
- "front left leg" → direction에 "FRONT"과 "LEFT"가 포함된 브릭

할 일 (브릭 ID만 찾기, 좌표 계산 금지):
1. "{bad_part}"에 해당하는 브릭 찾기 (삭제 대상)
2. "{reference_part}"에 해당하는 브릭 찾기 (참조 대상)

중요: 위 브릭 목록에 존재하는 ID만 반환하세요.

JSON으로 반환:
{{
    "delete_brick_ids": ["id1", "id2"],
    "reference_brick_ids": ["id3", "id4"],
    "confidence": 0-100,
    "reasoning": "이 브릭을 선택한 이유"
}}

유효한 JSON만 반환."""}
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
