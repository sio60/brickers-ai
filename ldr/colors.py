LDRAW_COLOR_NAMES = {
    0: "Black",
    1: "Blue",
    2: "Green",
    3: "Teal",
    4: "Red",
    5: "Dark Pink",
    6: "Brown",
    7: "Light Gray",
    8: "Dark Gray",
    9: "Light Blue",
    10: "Bright Green",
    11: "Light Teal",
    12: "Salmon",
    13: "Pink",
    14: "Yellow",
    15: "White",
    19: "Tan",
    25: "Orange",
    28: "Dark Green",
    70: "Reddish Brown",
    71: "Light Bluish Gray",
    72: "Dark Bluish Gray",
    89: "Dark Purple",
}

def get_color_name(code: int) -> str:
    return LDRAW_COLOR_NAMES.get(code, f"Color_{code}")
