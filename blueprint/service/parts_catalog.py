# blueprint/service/parts_catalog.py
"""LDraw 파츠 색상/사이즈 매핑 — 아이 친화적 영어 이름"""
from __future__ import annotations

import re

# ─── LDraw 색상 코드 → 아이 친화적 영어 이름 ──────────────
LDRAW_COLOR_NAMES = {
    0: "Black",
    1: "Blue",
    2: "Green",
    3: "Teal",
    4: "Red",
    5: "Pink",
    6: "Brown",
    7: "Light Gray",
    8: "Dark Gray",
    9: "Light Blue",
    10: "Bright Green",
    11: "Cyan",
    12: "Salmon",
    13: "Pink",
    14: "Yellow",
    15: "White",
    17: "Light Green",
    18: "Light Yellow",
    19: "Tan",
    22: "Purple",
    25: "Orange",
    26: "Magenta",
    27: "Lime",
    28: "Dark Tan",
    29: "Light Purple",
    33: "Sky Blue",
    36: "Bright Orange",
    70: "Brown",
    71: "Light Gray",
    72: "Dark Gray",
    73: "Medium Blue",
    74: "Medium Green",
    78: "Light Tan",
    84: "Medium Dark Tan",
    85: "Medium Purple",
    191: "Bright Orange",
    212: "Bright Blue",
    226: "Bright Yellow",
    272: "Dark Blue",
    288: "Dark Green",
    308: "Dark Brown",
    320: "Dark Red",
    326: "Bright Yellow Green",
    330: "Olive Green",
    378: "Sand Green",
    379: "Sand Blue",
    462: "Medium Orange",
    484: "Dark Orange",
}


def get_color_name(code: int) -> str:
    return LDRAW_COLOR_NAMES.get(code, f"Color {code}")


# ─── LDraw 파츠 ID → 아이 친화적 사이즈 이름 ─────────────
PART_SIZE_NAMES = {
    # --- Bricks ---
    "3005": "1x1 Brick",
    "3004": "1x2 Brick",
    "3622": "1x3 Brick",
    "3010": "1x4 Brick",
    "3009": "1x6 Brick",
    "3008": "1x8 Brick",
    "3003": "2x2 Brick",
    "3002": "2x3 Brick",
    "3001": "2x4 Brick",
    "3007": "2x8 Brick",
    "3006": "2x10 Brick",
    "2456": "2x6 Brick",
    "6112": "1x12 Brick",
    # --- Plates ---
    "3024": "1x1 Plate",
    "3023": "1x2 Plate",
    "3623": "1x3 Plate",
    "3710": "1x4 Plate",
    "3666": "1x6 Plate",
    "3460": "1x8 Plate",
    "4477": "1x10 Plate",
    "3022": "2x2 Plate",
    "3021": "2x3 Plate",
    "3020": "2x4 Plate",
    "3795": "2x6 Plate",
    "3034": "2x8 Plate",
    "3832": "2x10 Plate",
    "2445": "2x12 Plate",
    "3036": "6x8 Plate",
    "3033": "6x10 Plate",
    "3958": "6x6 Plate",
    "3035": "4x8 Plate",
    "3032": "4x6 Plate",
    "3031": "4x4 Plate",
    "3030": "4x10 Plate",
    "3029": "4x12 Plate",
    # --- Slopes ---
    "3040": "1x2 Slope",
    "3039": "2x2 Slope",
    "3038": "2x3 Slope",
    "3037": "2x4 Slope",
    "3044": "1x3 Slope",
    "3048": "1x2 Slope",
    "3665": "1x2 Inv Slope",
    "3660": "2x2 Inv Slope",
    # --- Tiles (flat) ---
    "3070b": "1x1 Tile",
    "3069b": "1x2 Tile",
    "63864": "1x3 Tile",
    "2431": "1x4 Tile",
    "3068b": "2x2 Tile",
    "87079": "2x4 Tile",
    # --- Round ---
    "3062b": "1x1 Round Brick",
    "3941": "2x2 Round Brick",
    "4032": "2x2 Round Plate",
    "6141": "1x1 Round Plate",
    # --- Special ---
    "4070": "1x1 Headlight",
    "3794": "1x2 Jumper Plate",
    "87087": "1x2 Brick w/Studs",
    "2357": "2x2 Corner Brick",
    "3045": "2x2 Corner Slope",
}


def get_part_size(part_id: str) -> str:
    """파츠 ID에서 아이 친화적 사이즈 이름 반환"""
    clean = part_id.strip().lower()
    if clean in PART_SIZE_NAMES:
        return PART_SIZE_NAMES[clean]
    digits = re.sub(r"[^0-9a-z]", "", clean)
    if digits in PART_SIZE_NAMES:
        return PART_SIZE_NAMES[digits]
    return part_id
