"""
Color Variant Agent - 색상 변형 에이전트

기본 위치: agent/color_variant.py
기존 모델의 형태는 유지하고 색상 팔레트만 변경
LLM이 색상 테마를 제안하거나 사용자가 지정
"""

import os
import sys
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# .env 파일 로드
load_dotenv(PROJECT_ROOT / ".env")

# 경로 설정 (데이터 격리된 위치 참조)
DATA_DIR = PROJECT_ROOT / "exporter" / "data"

# LangChain + OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 내부 모듈 (exporter 패키지 참조)
from exporter.ldr_converter import ldr_to_brick_model, model_to_ldr

# ========================================
# LDraw 색상 팔레트 (기존 내용 동일)
# ========================================
LDRAW_COLORS = {
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
    17: {"name": "Light Green", "rgb": (173, 221, 80)},
    18: {"name": "Light Yellow", "rgb": (251, 230, 150)},
    19: {"name": "Tan", "rgb": (232, 207, 161)},
    20: {"name": "Light Violet", "rgb": (215, 196, 230)},
    22: {"name": "Purple", "rgb": (129, 0, 123)},
    25: {"name": "Orange", "rgb": (255, 126, 20)},
    26: {"name": "Magenta", "rgb": (144, 31, 118)},
    27: {"name": "Lime", "rgb": (187, 233, 11)},
    28: {"name": "Dark Tan", "rgb": (149, 125, 98)},
    71: {"name": "Light Bluish Gray", "rgb": (163, 162, 165)},
    72: {"name": "Dark Bluish Gray", "rgb": (99, 95, 98)},
    85: {"name": "Dark Purple", "rgb": (63, 26, 100)},
    320: {"name": "Dark Red", "rgb": (114, 0, 18)},
    484: {"name": "Dark Orange", "rgb": (145, 80, 28)},
}

# 테마 프리셋
COLOR_THEMES = {
    "sunset": {
        "description": "노을 테마",
        "primary": [25, 4, 14],
        "secondary": [484, 320, 18],
        "accent": [15, 0],
    },
    "ocean": {
        "description": "바다 테마",
        "primary": [1, 9, 11],
        "secondary": [3, 72, 71],
        "accent": [15, 14],
    },
    "forest": {
        "description": "숲 테마",
        "primary": [2, 10, 17],
        "secondary": [6, 28, 19],
        "accent": [14, 15],
    },
    "night": {
        "description": "밤 테마",
        "primary": [0, 72, 85],
        "secondary": [22, 1, 8],
        "accent": [15, 14],
    },
    "candy": {
        "description": "캔디 테마",
        "primary": [13, 5, 20],
        "secondary": [11, 9, 17],
        "accent": [15, 14],
    },
    "monochrome": {
        "description": "흑백 테마",
        "primary": [0, 8, 7],
        "secondary": [72, 71, 15],
        "accent": [15, 0],
    },
    "fire": {
        "description": "불꽃 테마",
        "primary": [4, 25, 14],
        "secondary": [320, 484, 18],
        "accent": [0, 8],
    },
    "ice": {
        "description": "얼음 테마",
        "primary": [15, 9, 11],
        "secondary": [71, 1, 3],
        "accent": [72, 0],
    },
}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=OPENAI_API_KEY
)

_state = {
    "model": None,
    "parts_db": None,
    "ldr_path": None,
    "original_colors": {}
}

def load_parts_db():
    """파츠 DB 로드 (데이터 전용 폴더 참조)"""
    if _state["parts_db"] is not None:
        return _state["parts_db"]

    cache_file = DATA_DIR / "parts_cache.json"
    if cache_file.exists():
        with open(cache_file, 'r', encoding='utf-8') as f:
            _state["parts_db"] = json.load(f)
        print(f"[OK] Parts DB: {len(_state['parts_db'])} parts from data dir")
    else:
        # 폴백: 상위 exporter 폴더 확인
        legacy_cache = PROJECT_ROOT / "exporter" / "parts_cache.json"
        if legacy_cache.exists():
             with open(legacy_cache, 'r', encoding='utf-8') as f:
                _state["parts_db"] = json.load(f)
             print(f"[OK] Parts DB: Found in legacy exporter path")
        else:
            print(f"[WARN] parts_cache.json not found in {DATA_DIR}")
            _state["parts_db"] = {}

    return _state["parts_db"]

def analyze_model_colors(model) -> Dict:
    color_count = {}
    for brick in model.bricks:
        color = brick.color_code
        if color not in color_count:
            color_count[color] = 0
        color_count[color] += 1
    sorted_colors = sorted(color_count.items(), key=lambda x: -x[1])
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
    if theme_name not in COLOR_THEMES:
        theme_name = "ocean"
    theme = COLOR_THEMES[theme_name]
    all_theme_colors = theme["primary"] + theme["secondary"] + theme["accent"]
    mapping = {}
    for i, orig_color in enumerate(original_colors):
        mapping[orig_color] = all_theme_colors[i % len(all_theme_colors)]
    return mapping

def _find_nearest_opaque(color_code: int) -> int:
    TRANSPARENT_TO_OPAQUE = {
        33: 1, 34: 2, 35: 10, 36: 4, 37: 5, 38: 25, 39: 9, 40: 0, 
        41: 73, 42: 17, 43: 9, 44: 85, 45: 13, 46: 14, 47: 15, 52: 22, 54: 14, 57: 25,
    }
    if color_code in TRANSPARENT_TO_OPAQUE:
        return TRANSPARENT_TO_OPAQUE[color_code]
    valid_codes = sorted(LDRAW_COLORS.keys())
    return min(valid_codes, key=lambda c: abs(c - color_code))

def get_color_mapping_from_llm(model_analysis: Dict, prompt: str) -> Dict[int, int]:
    colors_text = "\n".join([f"- {c['name']} (code {c['code']}): {c['count']}개" for c in model_analysis["colors"]])
    available_colors = "\n".join([f"- {code}: {info['name']}" for code, info in sorted(LDRAW_COLORS.items())])
    llm_prompt = f"""You are a LEGO color design expert.
## Current Model Colors
{colors_text}
## User Request
"{prompt}"
## Available LDraw Colors
{available_colors}
Map original color codes to new ones (JSON format). Only use provided codes."""
    response = llm.invoke([HumanMessage(content=llm_prompt)])
    try:
        content = response.content.strip()
        if "```json" in content: content = content.split("```json")[1].split("```")[0].strip()
        result = json.loads(content)
        raw_mapping = {int(k): int(v) for k, v in result["mapping"].items()}
        valid_codes = set(LDRAW_COLORS.keys())
        mapping = {}
        for orig, new in raw_mapping.items():
            if new in valid_codes: mapping[orig] = new
            else:
                fallback = _find_nearest_opaque(new)
                mapping[orig] = fallback
        return mapping
    except Exception as e:
        print(f"[ERROR] LLM failed: {e}")
        return {}

def apply_color_mapping(model, mapping: Dict[int, int]):
    changed = 0
    for brick in model.bricks:
        if brick.color_code in mapping:
            new_color = mapping[brick.color_code]
            if brick.color_code != new_color:
                brick.color_code = new_color
                changed += 1
    return changed

def run_color_variant(ldr_path: str, theme_or_prompt: Optional[str] = None):
    print("\n[1/5] 모델 로드 중...")
    parts_db = load_parts_db()
    model = ldr_to_brick_model(ldr_path)
    model.name = Path(ldr_path).stem
    
    print("\n[2/5] 색상 분석 중...")
    analysis = analyze_model_colors(model)
    
    print("\n[3/5] 색상 매핑 생성 중...")
    original_codes = [c["code"] for c in analysis["colors"]]
    if theme_or_prompt is None: mapping = get_color_mapping_from_llm(analysis, "추천 테마 제안")
    elif theme_or_prompt.lower() in COLOR_THEMES: mapping = get_color_mapping_from_theme(original_codes, theme_or_prompt.lower())
    else: mapping = get_color_mapping_from_llm(analysis, theme_or_prompt)

    if not mapping: return None

    print("\n[4/5] 색상 적용 중...")
    changed = apply_color_mapping(model, mapping)

    print("\n[5/5] 저장 중...")
    output_path = Path(ldr_path).parent / f"{model.name}_recolor.ldr"
    ldr_content = model_to_ldr(model, parts_db, skip_validation=True, skip_physics=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(ldr_content)

    print(f"  완료! 저장 위치: {output_path}")
    return str(output_path)

if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if len(sys.argv) < 2: sys.exit(0)
    run_color_variant(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
