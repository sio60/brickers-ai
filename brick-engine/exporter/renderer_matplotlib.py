from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ai.vectordb.bbox_calc import get_bbox_doc_by_part_file, bbox_tuple_from_doc

import config


# ----------------------------
# Colors
# ----------------------------
LDRAW_COLORS = {
    0: "#05131D",    # Black
    1: "#0055BF",    # Blue
    2: "#237841",    # Green
    3: "#008F9B",    # Dark Turquoise
    4: "#C91A09",    # Red
    7: "#A0A5A9",    # Light Gray (approx)
    14: "#F2CD37",   # Yellow
    15: "#FFFFFF",   # White
    19: "#E4CD9E",   # Tan
    71: "#A0A5A9",   # Light Bluish Gray
    72: "#6C6E68",   # Dark Bluish Gray
    25: "#FE8A18",   # Orange (approx)
    70: "#582A12",   # Reddish Brown (approx)
}

def get_hex_color(code: int) -> str:
    try:
        return LDRAW_COLORS.get(int(code), "#CCCCCC")
    except Exception:
        return "#CCCCCC"

# ----------------------------
# Units (config-based)
# ----------------------------
STUD_PITCH_LDU   = float(config.STUD_PITCH_LDU)
PLATE_HEIGHT_LDU = float(config.PLATE_HEIGHT_LDU)
BRICK_HEIGHT_LDU = float(config.BRICK_HEIGHT_LDU)
FRAME_PAD_LDU    = float(config.RENDER_FRAME_PAD_LDU)

def _fallback_bbox_ldu() -> Tuple[Tuple[float,float,float], Tuple[float,float,float], Tuple[float,float,float]]:
    """
    bbox가 없는 파츠용 fallback.
    ✅ 너무 큰 큐브(20,20,20) 대신 "1x1 plate" 정도를 기본으로.
    """
    sx = STUD_PITCH_LDU
    sy = STUD_PITCH_LDU
    sz = PLATE_HEIGHT_LDU
    return ( (0.0, 0.0, 0.0), (sx, sy, sz), (sx, sy, sz) )


# ----------------------------
# Data models
# ----------------------------
@dataclass(frozen=True)
class LdrInstance:
    color: int
    x: float
    y: float
    z: float
    part_file: str


def _norm_part(p: str) -> str:
    return (p or "").strip().replace("\\", "/").split("/")[-1].lower()


def parse_type1_instances(ldr_path: str | Path) -> List[LdrInstance]:
    """
    Type-1:
    1 <color> x y z a b c d e f g h i <part.dat>
    (회전 행렬은 여기선 무시; 위치/색/part만 사용)
    """
    path = Path(ldr_path)
    if not path.exists():
        raise FileNotFoundError(f"LDR not found: {path}")

    out: List[LdrInstance] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line.startswith("1 "):
            continue
        tokens = line.split()
        if len(tokens) < 15:
            continue
        try:
            color = int(tokens[1])
            x = float(tokens[2])
            y = float(tokens[3])
            z = float(tokens[4])
        except Exception:
            continue
        part = _norm_part(tokens[-1])
        out.append(LdrInstance(color=color, x=x, y=y, z=z, part_file=part))
    return out

def _cuboid_verts(x0, y0, z0, sx, sy, sz):
    x = [x0, x0 + sx]
    y = [y0, y0 + sy]
    z = [z0, z0 + sz]
    return [
        [[x[0], y[0], z[0]], [x[1], y[0], z[0]], [x[1], y[1], z[0]], [x[0], y[1], z[0]]],  # bottom
        [[x[0], y[0], z[1]], [x[1], y[0], z[1]], [x[1], y[1], z[1]], [x[0], y[1], z[1]]],  # top
        [[x[0], y[0], z[0]], [x[1], y[0], z[0]], [x[1], y[0], z[1]], [x[0], y[0], z[1]]],  # front
        [[x[0], y[1], z[0]], [x[1], y[1], z[0]], [x[1], y[1], z[1]], [x[0], y[1], z[1]]],  # back
        [[x[0], y[0], z[0]], [x[0], y[1], z[0]], [x[0], y[1], z[1]], [x[0], y[0], z[1]]],  # left
        [[x[1], y[0], z[0]], [x[1], y[1], z[0]], [x[1], y[1], z[1]], [x[1], y[0], z[1]]],  # right
    ]


def _to_matplotlib_xyz(ldx: float, ldy: float, ldz: float) -> Tuple[float,float,float]:
    """
    LDraw (X,Y,Z) -> Matplotlib (X,Z,-Y)
    """
    return (ldx, ldz, -ldy)


def _inst_key(inst: LdrInstance, q: int = 2) -> Tuple:
    return (inst.part_file, int(inst.color), round(inst.x, q), round(inst.y, q), round(inst.z, q))


# ----------------------------
# Render
# ----------------------------
def render_ldr_to_png_matplotlib(
    ldr_path: str | Path,
    output_path: str | Path,
    parts_col,
    prev_ldr_path: str | Path | None = None,
    dpi: int = 180,
    elev: float = 25,
    azim: float = -45,
) -> str:
    instances = parse_type1_instances(ldr_path)
    if not instances:
        raise RuntimeError(f"No Type-1 bricks found in {ldr_path}")

    prev_keys: Set[Tuple] = set()
    if prev_ldr_path is not None:
        try:
            prev_insts = parse_type1_instances(prev_ldr_path)
            prev_keys = {_inst_key(i) for i in prev_insts}
        except Exception:
            prev_keys = set()

    bbox_cache: Dict[str, Tuple[Tuple[float,float,float], Tuple[float,float,float], Tuple[float,float,float]]] = {}

    boxes = []  # (is_new, facecolor, x0,y0,z0,sx,sy,sz)
    mins = [1e18, 1e18, 1e18]
    maxs = [-1e18, -1e18, -1e18]

    for inst in instances:
        pf = inst.part_file
        if pf not in bbox_cache:
            b = get_bbox_doc_by_part_file(parts_col, pf)
            if b is None:
                # ✅ config 기반 fallback (하드코딩 제거)
                b = _fallback_bbox_ldu()
            bbox_cache[pf] = b

        (mnx, mny, mnz), (_, _, _), (sx, sy, sz) = bbox_cache[pf]

        # LDraw inst 좌표는 파트 원점 → bbox.min으로 최소점 보정
        ldx0 = inst.x + mnx
        ldy0 = inst.y + mny
        ldz0 = inst.z + mnz

        # 축 변환(직립)
        x0, y0, z0 = _to_matplotlib_xyz(ldx0, ldy0, ldz0)

        is_new = (_inst_key(inst) not in prev_keys)
        fc = get_hex_color(inst.color)

        boxes.append((is_new, fc, x0, y0, z0, sx, sy, sz))

        mins[0] = min(mins[0], x0)
        mins[1] = min(mins[1], y0)
        mins[2] = min(mins[2], z0)
        maxs[0] = max(maxs[0], x0 + sx)
        maxs[1] = max(maxs[1], y0 + sy)
        maxs[2] = max(maxs[2], z0 + sz)

    z_shift = mins[2]
    cx = (mins[0] + maxs[0]) / 2.0
    cy = (mins[1] + maxs[1]) / 2.0

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()

    boxes_sorted = sorted(boxes, key=lambda t: (t[0] is True,))  # 기존 -> 신규

    for is_new, fc, x0, y0, z0, sx, sy, sz in boxes_sorted:
        x0 = x0 - cx
        y0 = y0 - cy
        z0 = z0 - z_shift

        verts = _cuboid_verts(x0, y0, z0, sx, sy, sz)

        if is_new:
            poly = Poly3DCollection(
                verts,
                facecolors=fc,
                linewidths=1.2,
                edgecolors="k",
                alpha=0.98,
            )
        else:
            poly = Poly3DCollection(
                verts,
                facecolors=fc,
                linewidths=0.25,
                edgecolors="k",
                alpha=0.45,
            )
        ax.add_collection3d(poly)

    # 축 범위 재계산
    mins2 = [1e18, 1e18, 1e18]
    maxs2 = [-1e18, -1e18, -1e18]
    for _, _, x0, y0, z0, sx, sy, sz in boxes:
        x0 = x0 - cx
        y0 = y0 - cy
        z0 = z0 - z_shift
        mins2[0] = min(mins2[0], x0)
        mins2[1] = min(mins2[1], y0)
        mins2[2] = min(mins2[2], z0)
        maxs2[0] = max(maxs2[0], x0 + sx)
        maxs2[1] = max(maxs2[1], y0 + sy)
        maxs2[2] = max(maxs2[2], z0 + sz)

    pad = FRAME_PAD_LDU
    ax.set_xlim(mins2[0] - pad, maxs2[0] + pad)
    ax.set_ylim(mins2[1] - pad, maxs2[1] + pad)
    ax.set_zlim(0.0, maxs2[2] + pad)

    try:
        ax.set_box_aspect([1, 1, 0.8])
    except Exception:
        pass

    ax.view_init(elev=elev, azim=azim)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), bbox_inches="tight", pad_inches=0.03, dpi=dpi)
    plt.close(fig)
    return str(out)
