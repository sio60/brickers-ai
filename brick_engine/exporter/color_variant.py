"""
Color Variant - 색상 테마 변경 유틸리티

LDR 모델의 색상을 테마 프리셋 또는 LLM 기반으로 변경
"""

import os
import json
import logging
from typing import Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

# ========================================
# LDraw 색상 팔레트
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

# ========================================
# 테마 프리셋
# ========================================
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

# ========================================
# LLM (커스텀 테마용)
# ========================================
_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    return _llm


# ========================================
# 유틸 함수
# ========================================

def analyze_model_colors(model) -> Dict:
    """모델의 색상 분포 분석"""
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
    """프리셋 테마 기반 색상 매핑"""
    if theme_name not in COLOR_THEMES:
        theme_name = "ocean"
    theme = COLOR_THEMES[theme_name]
    all_theme_colors = theme["primary"] + theme["secondary"] + theme["accent"]
    mapping = {}
    for i, orig_color in enumerate(original_colors):
        mapping[orig_color] = all_theme_colors[i % len(all_theme_colors)]
    return mapping


def _find_nearest_opaque(color_code: int) -> int:
    """투명 색상 → 불투명 색상 변환"""
    TRANSPARENT_TO_OPAQUE = {
        33: 1, 34: 2, 35: 10, 36: 4, 37: 5, 38: 25, 39: 9, 40: 0,
        41: 73, 42: 17, 43: 9, 44: 85, 45: 13, 46: 14, 47: 15, 52: 22, 54: 14, 57: 25,
    }
    if color_code in TRANSPARENT_TO_OPAQUE:
        return TRANSPARENT_TO_OPAQUE[color_code]
    valid_codes = sorted(LDRAW_COLORS.keys())
    return min(valid_codes, key=lambda c: abs(c - color_code))


def get_color_mapping_from_llm(model_analysis: Dict, prompt: str) -> Dict[int, int]:
    """LLM 기반 커스텀 색상 매핑"""
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
    response = _get_llm().invoke([HumanMessage(content=llm_prompt)])
    try:
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        result = json.loads(content)
        raw_mapping = {int(k): int(v) for k, v in result["mapping"].items()}
        valid_codes = set(LDRAW_COLORS.keys())
        mapping = {}
        for orig, new in raw_mapping.items():
            if new in valid_codes:
                mapping[orig] = new
            else:
                mapping[orig] = _find_nearest_opaque(new)
        return mapping
    except Exception as e:
        logger.error("LLM color mapping failed: %s", e)
        return {}
