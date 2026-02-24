# ============================================================================
# LDR 파일 파싱 및 빌드 유틸리티
# ============================================================================

from typing import Optional


def parse_ldr_line(line: str) -> Optional[dict]:
    line = line.strip()
    if not line.startswith("1 "):
        return None
    
    parts = line.split()
    if len(parts) < 15:
        return None
    
    try:
        return {
            "type": int(parts[0]),
            "color": int(parts[1]),
            "x": float(parts[2]),
            "y": float(parts[3]),
            "z": float(parts[4]),
            "matrix": [float(p) for p in parts[5:14]],
            "part": parts[14],
            "original_line": line
        }
    except (ValueError, IndexError):
        return None


def build_ldr_line(
    color: int,
    x: float, y: float, z: float,
    matrix: list,
    part: str
) -> str:
    ix, iy, iz = int(round(x)), int(round(y)), int(round(z))
    fmt_matrix = []
    for m in matrix:
        if isinstance(m, float) and m.is_integer():
            fmt_matrix.append(str(int(m)))
        else:
            fmt_matrix.append(str(m))
            
    matrix_str = " ".join(fmt_matrix)
    return f"1 {color} {ix} {iy} {iz} {matrix_str} {part}"
