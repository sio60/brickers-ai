"""
Color Variant Agent - 색상 변형 에이전트

기존 모델의 형태는 유지하고 색상 팔레트만 변경
LLM이 색상 테마를 제안하거나 사용자가 지정

사용법:
    python color_variant_agent.py <ldr_path> [theme]

예시:
    python color_variant_agent.py model.ldr "sunset"
    python color_variant_agent.py model.ldr "ocean"
    python color_variant_agent.py model.ldr  # LLM이 자동 제안
"""

import os
import sys
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# 경로 설정
EXPORTER_DIR = Path(__file__).parent
sys.path.insert(0, str(EXPORTER_DIR))

# LangChain + OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 내부 모듈
from ldr_converter import ldr_to_brick_model, model_to_ldr

# ========================================
# LDraw 색상 팔레트
# ========================================

LDRAW_COLORS = {
    # 기본 색상
    0: {"name": "Black", "rgb": (0, 0, 0)},
    1: {"name": "Blue", "rgb": (0, 85, 191)},
    2: {"name": "Green", "rgb": (0, 123, 40)},
    3: {"name": "Teal", "rgb": (0, 133, 125)},
    4: {"name": "Red", "rgb": (179, 0, 6)},
    5: {"name": "Dark Pink", "rgb": (171, 26, 91)},
    6: {"name": "Brown", "rgb": (92, 32, 0)},
    7: {"name": "Light Gray", "rgb": (156, 156, 156)},
    8: {"name": "Dark Gray", "rgb": (99, 99, 99)},
    9: {"name": "Light Blue", "rgb": (107, 171, 220)},
    10: {"name": "Bright Green", "rgb": (0, 186, 66)},
    11: {"name": "Cyan", "rgb": (0, 170, 164)},
    12: {"name": "Salmon", "rgb": (255, 133, 122)},
    13: {"name": "Pink", "rgb": (252, 151, 172)},
    14: {"name": "Yellow", "rgb": (255, 214, 0)},
    15: {"name": "White", "rgb": (255, 255, 255)},

    # 추가 색상
    17: {"name": "Light Green", "rgb": (173, 221, 80)},
    18: {"name": "Light Yellow", "rgb": (251, 230, 150)},
    19: {"name": "Tan", "rgb": (232, 207, 161)},
    20: {"name": "Light Violet", "rgb": (215, 196, 230)},
    22: {"name": "Purple", "rgb": (129, 0, 123)},
    25: {"name": "Orange", "rgb": (255, 126, 20)},
    26: {"name": "Magenta", "rgb": (144, 31, 118)},
    27: {"name": "Lime", "rgb": (187, 233, 11)},
    28: {"name": "Dark Tan", "rgb": (149, 125, 98)},

    # 메탈릭/특수
    71: {"name": "Light Bluish Gray", "rgb": (163, 162, 165)},
    72: {"name": "Dark Bluish Gray", "rgb": (99, 95, 98)},
    85: {"name": "Dark Purple", "rgb": (63, 26, 100)},
    320: {"name": "Dark Red", "rgb": (114, 0, 18)},
    484: {"name": "Dark Orange", "rgb": (145, 80, 28)},
}

# 색상 테마 프리셋
COLOR_THEMES = {
    "sunset": {
        "description": "노을 테마 - 따뜻한 오렌지, 빨강, 노랑 계열",
        "primary": [25, 4, 14],      # Orange, Red, Yellow
        "secondary": [484, 320, 18],  # Dark Orange, Dark Red, Light Yellow
        "accent": [15, 0],            # White, Black
    },
    "ocean": {
        "description": "바다 테마 - 시원한 파랑, 청록 계열",
        "primary": [1, 9, 11],        # Blue, Light Blue, Cyan
        "secondary": [3, 72, 71],     # Teal, Dark Bluish Gray, Light Bluish Gray
        "accent": [15, 14],           # White, Yellow
    },
    "forest": {
        "description": "숲 테마 - 자연의 초록, 갈색 계열",
        "primary": [2, 10, 17],       # Green, Bright Green, Light Green
        "secondary": [6, 28, 19],     # Brown, Dark Tan, Tan
        "accent": [14, 15],           # Yellow, White
    },
    "night": {
        "description": "밤 테마 - 어두운 보라, 파랑, 검정 계열",
        "primary": [0, 72, 85],       # Black, Dark Bluish Gray, Dark Purple
        "secondary": [22, 1, 8],      # Purple, Blue, Dark Gray
        "accent": [15, 14],           # White, Yellow (별빛)
    },
    "candy": {
        "description": "캔디 테마 - 밝은 핑크, 민트 계열",
        "primary": [13, 5, 20],       # Pink, Dark Pink, Light Violet
        "secondary": [11, 9, 17],     # Cyan, Light Blue, Light Green
        "accent": [15, 14],           # White, Yellow
    },
    "monochrome": {
        "description": "흑백 테마 - 그레이스케일",
        "primary": [0, 8, 7],         # Black, Dark Gray, Light Gray
        "secondary": [72, 71, 15],    # Dark Bluish Gray, Light Bluish Gray, White
        "accent": [15, 0],            # White, Black
    },
    "fire": {
        "description": "불꽃 테마 - 빨강, 주황, 노랑의 그라데이션",
        "primary": [4, 25, 14],       # Red, Orange, Yellow
        "secondary": [320, 484, 18],  # Dark Red, Dark Orange, Light Yellow
        "accent": [0, 8],             # Black, Dark Gray
    },
    "ice": {
        "description": "얼음 테마 - 차가운 하늘색, 흰색 계열",
        "primary": [15, 9, 11],       # White, Light Blue, Cyan
        "secondary": [71, 1, 3],      # Light Bluish Gray, Blue, Teal
        "accent": [72, 0],            # Dark Bluish Gray, Black
    },
}

# ========================================
# 설정
# ========================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,  # 창의성을 위해 약간 높임
    api_key=OPENAI_API_KEY
)

# ========================================
# 상태 관리
# ========================================

_state = {
    "model": None,
    "parts_db": None,
    "ldr_path": None,
    "original_colors": {}  # 원본 색상 저장
}


def load_parts_db():
    """파츠 DB 로드"""
    if _state["parts_db"] is not None:
        return _state["parts_db"]

    cache_file = EXPORTER_DIR / "parts_cache.json"
    if cache_file.exists():
        with open(cache_file, 'r', encoding='utf-8') as f:
            _state["parts_db"] = json.load(f)
        print(f"[OK] Parts DB: {len(_state['parts_db'])} parts")
    else:
        print("[WARN] parts_cache.json not found")
        _state["parts_db"] = {}

    return _state["parts_db"]


# ========================================
# 색상 분석 함수
# ========================================

def analyze_model_colors(model) -> Dict:
    """모델의 색상 분포 분석"""
    color_count = {}

    for brick in model.bricks:
        color = brick.color_code
        if color not in color_count:
            color_count[color] = 0
        color_count[color] += 1

    # 정렬 (많은 순)
    sorted_colors = sorted(color_count.items(), key=lambda x: -x[1])

    # 색상 정보 추가
    color_info = []
    for color_code, count in sorted_colors:
        info = LDRAW_COLORS.get(color_code, {"name": f"Unknown({color_code})", "rgb": (128, 128, 128)})
        color_info.append({
            "code": color_code,
            "name": info["name"],
            "count": count,
            "percentage": round(count / len(model.bricks) * 100, 1)
        })

    return {
        "total_bricks": len(model.bricks),
        "unique_colors": len(color_count),
        "colors": color_info,
        "dominant": color_info[0] if color_info else None
    }


def get_color_mapping_from_theme(original_colors: List[int], theme_name: str) -> Dict[int, int]:
    """테마 기반 색상 매핑 생성"""
    if theme_name not in COLOR_THEMES:
        print(f"[WARN] Unknown theme: {theme_name}, using 'ocean'")
        theme_name = "ocean"

    theme = COLOR_THEMES[theme_name]
    all_theme_colors = theme["primary"] + theme["secondary"] + theme["accent"]

    mapping = {}

    # 원본 색상을 빈도순으로 정렬했다고 가정
    for i, orig_color in enumerate(original_colors):
        # 순환하며 테마 색상 할당
        new_color = all_theme_colors[i % len(all_theme_colors)]
        mapping[orig_color] = new_color

    return mapping


def _find_nearest_opaque(color_code: int) -> int:
    """투명/미지원 색상 코드를 가장 가까운 불투명 색상으로 매핑"""
    # 잘 알려진 투명 색상 → 불투명 대체 매핑
    TRANSPARENT_TO_OPAQUE = {
        33: 1,    # Trans_Dark_Blue → Blue
        34: 2,    # Trans_Green → Green
        35: 10,   # Trans_Bright_Green → Bright Green
        36: 4,    # Trans_Red → Red
        37: 5,    # Trans_Dark_Pink → Dark Pink
        38: 25,   # Trans_Neon_Orange → Orange
        39: 9,    # Trans_Very_Light_Blue → Light Blue
        40: 0,    # Trans_Black → Black
        41: 73,   # Trans_Medium_Blue → Medium Blue
        42: 17,   # Trans_Neon_Green → Light Green
        43: 9,    # Trans_Light_Blue → Light Blue
        44: 85,   # Trans_Bright_Reddish_Lilac → Dark Purple
        45: 13,   # Trans_Pink → Pink
        46: 14,   # Trans_Yellow → Yellow
        47: 15,   # Trans_Clear → White
        52: 22,   # Trans_Purple → Purple
        54: 14,   # Trans_Neon_Yellow → Yellow
        57: 25,   # Trans_Orange → Orange
    }

    if color_code in TRANSPARENT_TO_OPAQUE:
        return TRANSPARENT_TO_OPAQUE[color_code]

    # 매핑에 없으면 가장 가까운 코드 찾기 (거리 기반)
    valid_codes = sorted(LDRAW_COLORS.keys())
    closest = min(valid_codes, key=lambda c: abs(c - color_code))
    return closest


def get_color_mapping_from_llm(model_analysis: Dict, prompt: str) -> Dict[int, int]:
    """LLM에게 색상 매핑 요청"""

    colors_text = "\n".join([
        f"- {c['name']} (code {c['code']}): {c['count']}개 ({c['percentage']}%)"
        for c in model_analysis["colors"]
    ])

    available_colors = "\n".join([
        f"- {code}: {info['name']}"
        for code, info in sorted(LDRAW_COLORS.items())
    ])

    llm_prompt = f"""You are a LEGO color design expert. Create a color mapping for a brick model.

## Current Model Colors
{colors_text}

## User Request
"{prompt}"

## Available LDraw Colors
{available_colors}

## Task
Create a color mapping that transforms the model according to the user's request.
Map each original color code to a new color code.

## Response Format (JSON only)
{{
  "theme_name": "테마 이름 (한글)",
  "description": "변환 설명 (한글)",
  "mapping": {{
    "원본코드1": 새코드1,
    "원본코드2": 새코드2,
    ...
  }}
}}

IMPORTANT:
- mapping의 key는 문자열, value는 숫자
- 모든 원본 색상에 대해 매핑 제공
- 비슷한 역할의 색상끼리 매핑 (주색상→주색상, 보조색상→보조색상)
- ONLY use color codes from the "Available LDraw Colors" list above. Do NOT use any other codes (especially transparent codes 33-57).
"""

    response = llm.invoke([HumanMessage(content=llm_prompt)])

    try:
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        result = json.loads(content)

        # 문자열 키를 정수로 변환
        raw_mapping = {int(k): int(v) for k, v in result["mapping"].items()}

        # 유효한 불투명 색상 코드만 허용 (투명 코드 33-57 등 방지)
        valid_codes = set(LDRAW_COLORS.keys())
        mapping = {}
        for orig, new in raw_mapping.items():
            if new in valid_codes:
                mapping[orig] = new
            else:
                # 유효하지 않은 코드 → 가장 가까운 불투명 색상으로 대체
                fallback = _find_nearest_opaque(new)
                print(f"[WARN] 색상 코드 {new}은 투명/미지원 → {fallback} ({LDRAW_COLORS[fallback]['name']})로 대체")
                mapping[orig] = fallback

        print(f"\n[LLM] 테마: {result.get('theme_name', 'N/A')}")
        print(f"[LLM] 설명: {result.get('description', 'N/A')}")

        return mapping

    except Exception as e:
        print(f"[ERROR] LLM 응답 파싱 실패: {e}")
        return {}


# ========================================
# 색상 변환 함수
# ========================================

def apply_color_mapping(model, mapping: Dict[int, int]):
    """모델에 색상 매핑 적용"""
    changed = 0

    for brick in model.bricks:
        if brick.color_code in mapping:
            new_color = mapping[brick.color_code]
            if brick.color_code != new_color:
                brick.color_code = new_color
                changed += 1

    return changed


# ========================================
# 메인 에이전트
# ========================================

def run_color_variant(ldr_path: str, theme_or_prompt: Optional[str] = None):
    """
    색상 변형 에이전트 실행

    Args:
        ldr_path: 입력 LDR 파일 경로
        theme_or_prompt: 테마 이름 또는 자유 프롬프트
                        - "sunset", "ocean" 등: 프리셋 테마
                        - 기타 문자열: LLM에게 요청
                        - None: LLM이 자동 제안
    """
    print("=" * 60)
    print("Color Variant Agent")
    print("=" * 60)

    # 모델 로드
    print("\n[1/5] 모델 로드 중...")
    parts_db = load_parts_db()
    model = ldr_to_brick_model(ldr_path)
    model.name = Path(ldr_path).stem
    print(f"  모델: {model.name}")
    print(f"  브릭: {len(model.bricks)}개")

    # 색상 분석
    print("\n[2/5] 색상 분석 중...")
    analysis = analyze_model_colors(model)
    print(f"  색상 종류: {analysis['unique_colors']}개")
    print("  색상 분포:")
    for c in analysis["colors"][:5]:  # 상위 5개만
        print(f"    - {c['name']}: {c['count']}개 ({c['percentage']}%)")

    # 색상 매핑 결정
    print("\n[3/5] 색상 매핑 생성 중...")

    original_codes = [c["code"] for c in analysis["colors"]]

    if theme_or_prompt is None:
        # LLM이 자동 제안
        prompt = "이 모델에 어울리는 새로운 색상 테마를 제안해주세요. 창의적으로!"
        mapping = get_color_mapping_from_llm(analysis, prompt)
    elif theme_or_prompt.lower() in COLOR_THEMES:
        # 프리셋 테마
        theme = theme_or_prompt.lower()
        print(f"  테마: {theme} - {COLOR_THEMES[theme]['description']}")
        mapping = get_color_mapping_from_theme(original_codes, theme)
    else:
        # 자유 프롬프트
        mapping = get_color_mapping_from_llm(analysis, theme_or_prompt)

    if not mapping:
        print("[ERROR] 색상 매핑 생성 실패")
        return None

    # 매핑 출력
    print("\n  색상 매핑:")
    for orig, new in mapping.items():
        orig_name = LDRAW_COLORS.get(orig, {}).get("name", f"Unknown({orig})")
        new_name = LDRAW_COLORS.get(new, {}).get("name", f"Unknown({new})")
        print(f"    {orig_name} ({orig}) → {new_name} ({new})")

    # 색상 적용
    print("\n[4/5] 색상 적용 중...")
    changed = apply_color_mapping(model, mapping)
    print(f"  변경된 브릭: {changed}개")

    # 저장
    print("\n[5/5] 저장 중...")
    output_path = Path(ldr_path).parent / f"{model.name}_recolor.ldr"

    ldr_content = model_to_ldr(model, parts_db, skip_validation=True, skip_physics=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(ldr_content)

    print(f"  저장 완료: {output_path}")

    # 결과 요약
    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)
    print(f"  원본: {ldr_path}")
    print(f"  결과: {output_path}")
    print(f"  변경: {changed}개 브릭")

    return str(output_path)


# ========================================
# CLI
# ========================================

def print_themes():
    """사용 가능한 테마 출력"""
    print("\n[사용 가능한 테마]")
    for name, theme in COLOR_THEMES.items():
        print(f"  - {name}: {theme['description']}")


if __name__ == "__main__":
    # Windows 콘솔 UTF-8 출력
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    if len(sys.argv) < 2:
        print("=" * 50)
        print("Color Variant Agent")
        print("=" * 50)
        print()
        print("사용법:")
        print("  python color_variant_agent.py <ldr_path> [theme]")
        print()
        print("예시:")
        print('  python color_variant_agent.py model.ldr "sunset"')
        print('  python color_variant_agent.py model.ldr "ocean"')
        print('  python color_variant_agent.py model.ldr "사이버펑크 느낌으로"')
        print('  python color_variant_agent.py model.ldr  # LLM 자동 제안')
        print_themes()
        sys.exit(0)

    ldr_path = sys.argv[1]
    theme = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(ldr_path).exists():
        print(f"Error: 파일 없음 - {ldr_path}")
        sys.exit(1)

    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)

    run_color_variant(ldr_path, theme)
