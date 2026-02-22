# ============================================================================
# LDR 파일 파싱 및 빌드 유틸리티
# LDR 라인을 파싱하여 브릭 정보로 변환하거나, 브릭 정보를 LDR 라인으로 재구성
# ============================================================================

from typing import Optional


def parse_ldr_line(line: str) -> Optional[dict]:
    """
    LDR 라인을 파싱하여 브릭 정보 추출
    LDR 형식: 1 <color> <x> <y> <z> <a> <b> <c> <d> <e> <f> <g> <h> <i> <part>
    """
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
    """LDR 라인 재구성 (좌표 정수화 포함)"""
    # [COORD FIX] 최종 출력 시에도 정수 좌표 강제 (float 오차 방지)
    ix, iy, iz = int(round(x)), int(round(y)), int(round(z))
    # 매트릭스도 정수일 경우 .0 제거 (깔끔한 출력)
    fmt_matrix = []
    for m in matrix:
        if isinstance(m, float) and m.is_integer():
            fmt_matrix.append(str(int(m)))
        else:
            fmt_matrix.append(str(m))
            
    matrix_str = " ".join(fmt_matrix)
    return f"1 {color} {ix} {iy} {iz} {matrix_str} {part}"
