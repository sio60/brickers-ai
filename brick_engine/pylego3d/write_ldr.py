# pylego3d/write_ldr.py
from __future__ import annotations

from typing import List, Dict, Any, Tuple
import math

STUD_LDU = 20.0
PLATE_LDU = 8.0

def _rot_y_matrix(deg: int) -> Tuple[float, float, float, float, float, float, float, float, float]:
    d = deg % 360
    if d == 0:
        return (1,0,0, 0,1,0, 0,0,1)
    if d == 90:
        return (0,0,-1, 0,1,0, 1,0,0)
    if d == 180:
        return (-1,0,0, 0,1,0, 0,0,-1)
    if d == 270:
        return (0,0,1, 0,1,0, -1,0,0)
    r = math.radians(d)
    c = math.cos(r); s = math.sin(r)
    return (c,0,-s, 0,1,0, s,0,c)

def write_ldr(
    out_path: str,
    parts: List[Dict[str, Any]],
    *,
    center: bool = True,
    step_order: str = "bottomup",  # bottomup | topdown | none
    title: str = "glb_to_ldr_quick",
    author: str = "pylego3d",
) -> None:
    if not parts:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"0 {title}\n0 Author: {author}\n")
        return

    xs = []
    zs = []
    for p in parts:
        x0 = p["x"] * STUD_LDU
        z0 = p["z"] * STUD_LDU
        x1 = (p["x"] + p["w"] - 1) * STUD_LDU
        z1 = (p["z"] + p["l"] - 1) * STUD_LDU
        xs += [x0, x1]
        zs += [z0, z1]

    cx_off = (min(xs) + max(xs)) / 2.0 if center else 0.0
    cz_off = (min(zs) + max(zs)) / 2.0 if center else 0.0

    step_order = (step_order or "bottomup").lower()
    if step_order not in ("bottomup", "topdown", "none"):
        step_order = "bottomup"

    ys = sorted({p["y"] for p in parts}, reverse=True)  # 0, -3, -6...
    if step_order == "bottomup":
        ys_sorted = ys  # 0부터
    elif step_order == "topdown":
        ys_sorted = list(reversed(ys))
    else:
        ys_sorted = [None]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"0 {title}\n")
        f.write(f"0 Author: {author}\n")

        if step_order == "none":
            for p in parts:
                _write_part_line(f, p, cx_off, cz_off)
            return

        for yi in ys_sorted:
            layer_parts = [p for p in parts if p["y"] == yi]
            layer_parts.sort(key=lambda p: (p["z"], p["x"]))
            for p in layer_parts:
                _write_part_line(f, p, cx_off, cz_off)
            f.write("0 STEP\n")

def _write_part_line(f, p: Dict[str, Any], cx_off: float, cz_off: float) -> None:
    color = p.get("color", 4)
    x = float(p["x"]); z = float(p["z"]); y_plate = float(p["y"])
    w = float(p["w"]); l = float(p["l"])
    rot = int(p.get("rot", 0))

    cx = (x + (w - 1.0) / 2.0) * STUD_LDU - cx_off
    cz = (z + (l - 1.0) / 2.0) * STUD_LDU - cz_off
    cy = y_plate * PLATE_LDU

    a,b,c,d,e,g,h,i,j = _rot_y_matrix(rot)
    part = p["part"]

    f.write(
        f"1 {color} {cx:.2f} {cy:.2f} {cz:.2f} "
        f"{a:.6f} {b:.6f} {c:.6f} {d:.6f} {e:.6f} {g:.6f} {h:.6f} {i:.6f} {j:.6f} "
        f"{part}\n"
    )