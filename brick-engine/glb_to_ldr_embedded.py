from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import trimesh

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent

for p in (_THIS_DIR, _PARENT_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from pylego3d.optimizer import optimize_bricks
from pylego3d.write_ldr import write_ldr

# -----------------------------------------------------------------------------
# 1. 내장 색상 팔레트
# -----------------------------------------------------------------------------
LDRAW_RGB = {
    0:  (33, 33, 33),     # Black
    1:  (0, 85, 191),     # Blue
    2:  (0, 133, 43),     # Green
    3:  (0, 155, 155),    # Dark Turquoise
    4:  (196, 0, 38),     # Red
    5:  (205, 98, 152),   # Dark Pink
    6:  (92, 46, 0),      # Brown
    7:  (156, 156, 156),  # Light Gray
    8:  (99, 99, 99),     # Dark Gray
    9:  (107, 171, 220),  # Light Blue
    10: (97, 189, 67),    # Bright Green
    11: (0, 174, 239),    # Bright Light Blue-ish
    14: (245, 205, 47),   # Yellow
    15: (255, 255, 255),  # White
    19: (173, 169, 142),  # Tan
    25: (245, 245, 245),  # Very Light Gray-ish
    27: (0, 0, 0),        # Black (alt)
    28: (0, 143, 155),    # Dark Teal-ish
    47: (255, 255, 255),  # Trans-Clear
    70: (105, 64, 39),    # Reddish Brown-ish
    71: (160, 165, 169),  # Light Bluish Gray
    72: (108, 110, 107),  # Dark Bluish Gray
}

def rgb_to_ldraw_id(rgb: np.ndarray) -> int:
    r = rgb.astype(np.float32)
    ids = list(LDRAW_RGB.keys())
    search_ids = [i for i in ids if i != 47]
    pal = np.array([LDRAW_RGB[i] for i in search_ids], dtype=np.float32)
    d = np.sum((pal - r[None, :]) ** 2, axis=1)
    return search_ids[int(np.argmin(d))]

# -----------------------------------------------------------------------------
# 2. GLB 로드 및 전처리
# -----------------------------------------------------------------------------
def load_glb_meshes(glb_path: str) -> List[trimesh.Trimesh]:
    scene = trimesh.load(glb_path, force="scene")
    if isinstance(scene, trimesh.Trimesh):
        return [scene]
    meshes: List[trimesh.Trimesh] = []
    if isinstance(scene, trimesh.Scene):
        dumped = scene.dump(concatenate=False)
        if isinstance(dumped, list):
            meshes = [m.copy() for m in dumped if isinstance(m, trimesh.Trimesh)]
        elif isinstance(dumped, trimesh.Trimesh):
            meshes = [dumped.copy()]
    meshes = [m for m in meshes if len(m.vertices) > 0 and len(m.faces) > 0]
    if not meshes:
        raise RuntimeError(f"Failed to load meshes from {glb_path}")
    return meshes

def preprocess_meshes(
    meshes: List[trimesh.Trimesh],
    target_studs: int,
    flipx180: bool, flipy180: bool, flipz180: bool,
) -> trimesh.Trimesh:
    meshes2 = [m.copy() for m in meshes]
    Rx = np.eye(4); Ry = np.eye(4); Rz = np.eye(4)
    if flipx180: Rx[1, 1] = -1; Rx[2, 2] = -1
    if flipy180: Ry[0, 0] = -1; Ry[2, 2] = -1
    if flipz180: Rz[0, 0] = -1; Rz[1, 1] = -1
    T = Rz @ Ry @ Rx
    if flipx180 or flipy180 or flipz180:
        for m in meshes2:
            m.apply_transform(T)

    combined = trimesh.util.concatenate(meshes2)
    bounds = combined.bounds
    size = bounds[1] - bounds[0]
    max_xz = max(float(size[0]), float(size[2]))
    if max_xz <= 1e-9:
        raise ValueError("Mesh bounds too small")

    scale = float(target_studs) / max_xz
    for m in meshes2:
        m.apply_scale(scale)

    combined2 = trimesh.util.concatenate(meshes2)
    b2 = combined2.bounds
    min_y = float(b2[0][1])

    for m in meshes2:
        m.apply_translation([0.0, -min_y, 0.0])

    for m in meshes2:
        m.vertices[:, 1] *= (20.0 / 24.0)

    return trimesh.util.concatenate(meshes2)

# -----------------------------------------------------------------------------
# 3. 복셀화
# -----------------------------------------------------------------------------
def voxelize(mesh: trimesh.Trimesh, fill: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vg = mesh.voxelized(pitch=1.0)
    if fill:
        vg = vg.fill()
    idx = vg.sparse_indices
    pts = vg.points
    if idx is None or len(idx) == 0:
        raise RuntimeError("Voxelization produced 0 voxels.")
    origin = pts[0] - (idx[0] + 0.5)
    return idx.astype(np.int32), pts.astype(np.float32), origin

def sample_voxel_colors(mesh: trimesh.Trimesh, points: np.ndarray) -> List[int]:
    target_mesh = mesh.copy()
    if hasattr(target_mesh.visual, "to_color"):
        target_mesh.visual = target_mesh.visual.to_color()

    # points에 대해 가장 가까운 삼각형 id 얻기
    _, _, triangle_ids = target_mesh.nearest.on_surface(points)

    fc = np.asarray(getattr(target_mesh.visual, "face_colors", None))
    # fc가 없거나 이상하면 solid로 대체
    if fc is None or fc.size == 0:
        return [4] * len(points)

    # ✅ 케이스 A: 단일 RGBA (shape=(4,))
    if fc.ndim == 1 and fc.shape[0] >= 3:
        rgb = fc[:3]
        cid = rgb_to_ldraw_id(np.asarray(rgb))
        return [cid] * len(points)

    # ✅ 케이스 B: RGBA 1개만 있는 (shape=(1,4))
    if fc.ndim == 2 and fc.shape[0] == 1 and fc.shape[1] >= 3:
        rgb = fc[0, :3]
        cid = rgb_to_ldraw_id(np.asarray(rgb))
        return [cid] * len(points)

    # ✅ 케이스 C: 정상 face별 색 (shape=(n_faces,4))
    if fc.ndim == 2 and fc.shape[0] >= 1 and fc.shape[1] >= 3:
        tri = np.asarray(triangle_ids, dtype=np.int64)
        # 혹시 -1 같은 값이 나오거나 범위 벗어나면 클립
        tri = np.clip(tri, 0, fc.shape[0] - 1)
        face_colors = fc[tri]
        return [rgb_to_ldraw_id(rgba[:3]) for rgba in face_colors]

    # 그 외 이상 케이스는 solid로
    return [4] * len(points)


# -----------------------------------------------------------------------------
# 4. Y 인덱스 반전
# -----------------------------------------------------------------------------
def invert_y_idx(idx: np.ndarray) -> Tuple[np.ndarray, int]:
    if idx is None or idx.size == 0:
        return idx, 0
    out = idx.copy()
    y_max = int(out[:, 1].max())
    out[:, 1] = (y_max - out[:, 1])
    return out, y_max

# -----------------------------------------------------------------------------
# 5. [핵심 물리 보정] 심지 박기
# -----------------------------------------------------------------------------
def embed_floating_parts(vox: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not vox:
        return []

    voxel_map = {(v["x"], v["y"], v["z"]): int(v["color"]) for v in vox}
    new_colors = dict(voxel_map)

    neighbors_horizontal = [(1,0,0), (-1,0,0), (0,0,1), (0,0,-1)]
    neighbors_all = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                neighbors_all.append((dx, dy, dz))

    for _ in range(2):
        updates: Dict[Tuple[int,int,int], int] = {}
        for pos, color in list(new_colors.items()):
            x, y, z = pos

            is_safe = False
            if y == 0:
                is_safe = True
            elif (x, y-1, z) in new_colors:
                is_safe = True
            elif (x, y+1, z) in new_colors:
                is_safe = True
            if is_safe:
                continue

            best_anchor: Optional[Tuple[int,int,int]] = None

            for dx, dy, dz in neighbors_horizontal:
                npos = (x+dx, y+dy, z+dz)
                if npos in new_colors:
                    nx, ny, nz = npos
                    if ny == 0 or (nx, ny-1, nz) in new_colors or (nx, ny+1, nz) in new_colors:
                        best_anchor = npos
                        break

            if not best_anchor:
                for dx, dy, dz in neighbors_all:
                    npos = (x+dx, y+dy, z+dz)
                    if npos in new_colors:
                        best_anchor = npos
                        break

            if best_anchor:
                my_color = new_colors[pos]
                if new_colors[best_anchor] != my_color:
                    updates[best_anchor] = my_color

                ax, ay, az = best_anchor
                dist = abs(x-ax) + abs(y-ay) + abs(z-az)
                if dist > 1:
                    bridge = (ax, y, z)
                    if bridge not in new_colors:
                        updates[bridge] = my_color

        if not updates:
            break
        new_colors.update(updates)

    return [{"x": x, "y": y, "z": z, "color": c} for (x, y, z), c in new_colors.items()]

# -----------------------------------------------------------------------------
# 6. STEP 레이어 재작성
# -----------------------------------------------------------------------------
def rewrite_steps_by_layer(in_path: str, out_path: str, mode: str) -> None:
    with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f]

    header: List[str] = []
    part_lines: List[str] = []

    for ln in lines:
        if ln.startswith("1 "):
            part_lines.append(ln)
        else:
            header.append(ln)

    if mode == "none":
        with open(out_path, "w", encoding="utf-8") as f:
            for ln in header + part_lines:
                f.write(ln + "\n")
        return

    ys: List[float] = []
    parsed: List[Tuple[str, float, float, float]] = []
    for ln in part_lines:
        toks = ln.split()
        if len(toks) < 6:
            continue
        x = float(toks[2]); y = float(toks[3]); z = float(toks[4])
        ys.append(y)
        parsed.append((ln, x, y, z))

    if not parsed:
        with open(out_path, "w", encoding="utf-8") as f:
            for ln in header:
                f.write(ln + "\n")
        return

    y_min = min(ys)
    y_max = max(ys)
    bottom_is_min = abs(y_min) <= abs(y_max)

    layer_groups: Dict[int, List[Tuple[str, float, float, float]]] = {}
    for ln, x, y, z in parsed:
        layer = int(round(y / 8.0))
        layer_groups.setdefault(layer, []).append((ln, x, y, z))

    layers = sorted(layer_groups.keys())
    if mode == "bottomup":
        ordered_layers = layers if bottom_is_min else list(reversed(layers))
    else:
        ordered_layers = list(reversed(layers)) if bottom_is_min else layers

    for k in layer_groups:
        layer_groups[k].sort(key=lambda t: (t[3], t[1]))

    with open(out_path, "w", encoding="utf-8") as f:
        for ln in header:
            f.write(ln + "\n")
        for layer in ordered_layers:
            for ln, *_ in layer_groups[layer]:
                f.write(ln + "\n")
            f.write("0 STEP\n")

# -----------------------------------------------------------------------------
# 7. Build Logic
# -----------------------------------------------------------------------------
def try_build(
    meshes: List[trimesh.Trimesh],
    *,
    target: int,
    flipx180: bool, flipy180: bool, flipz180: bool,
    fill: bool,
    solid_color: int,
    use_mesh_color: bool,
    invert_y: bool,
    kind: str,
    plates_per_voxel: int,
    interlock: bool,
    max_area: int,
    smart_fix: bool,
) -> Tuple[List[Dict[str, Any]], int]:

    mesh = preprocess_meshes(meshes, target, flipx180, flipy180, flipz180)
    idx_raw, _, origin = voxelize(mesh, fill)

    current_idx = idx_raw.copy()
    y_max_val = 0
    if invert_y:
        current_idx, y_max_val = invert_y_idx(current_idx)

    if use_mesh_color:
        search_idx = current_idx.copy()
    if invert_y:
        search_idx[:, 1] = y_max_val - search_idx[:, 1]
    search_pts = origin + (search_idx.astype(np.float32) + 0.5)

    try:
        current_colors = sample_voxel_colors(mesh, search_pts)
        # 혹시 길이가 안 맞으면 폴백
        if len(current_colors) != len(current_idx):
            raise RuntimeError("color length mismatch")
    except Exception:
        current_colors = [solid_color] * len(current_idx)
    else:
        current_colors = [solid_color] * len(current_idx)


    if len(current_idx) == 0:
        return [], target

    vox: List[Dict[str, Any]] = []
    min_ix = current_idx.min(axis=0)
    idx_shifted = current_idx - min_ix
    for i, (ix, iy, iz) in enumerate(idx_shifted):
        vox.append({"x": int(ix), "y": int(iy), "z": int(iz), "color": int(current_colors[i])})

    if smart_fix:
        vox = embed_floating_parts(vox)

    parts = optimize_bricks(
        vox,
        kind=kind,
        plates_per_voxel=plates_per_voxel,
        interlock=interlock,
        max_area=max_area,
    )

    return parts, target

def build_under_budget(
    meshes: List[trimesh.Trimesh],
    *,
    start_target: int, min_target: int,
    budget: int, shrink: float, search_iters: int,
    flipx180: bool, flipy180: bool, flipz180: bool,
    fill: bool,
    kind: str,
    plates_per_voxel: int,
    interlock: bool,
    max_area: int,
    solid_color: int,
    use_mesh_color: bool,
    invert_y: bool,
    smart_fix: bool,
) -> Tuple[List[Dict[str, Any]], int]:

    hi_target = start_target
    hi_parts: Optional[List[Dict[str, Any]]] = None

    target = start_target
    best_parts: Optional[List[Dict[str, Any]]] = None
    best_target: Optional[int] = None

    while target >= min_target:
        try:
            parts, t = try_build(
                meshes,
                target=target,
                flipx180=flipx180, flipy180=flipy180, flipz180=flipz180,
                fill=fill,
                solid_color=solid_color,
                use_mesh_color=use_mesh_color,
                invert_y=invert_y,
                kind=kind,
                plates_per_voxel=plates_per_voxel,
                interlock=interlock,
                max_area=max_area,
                smart_fix=smart_fix,
            )
            pc = len(parts)
            print(f"[TRY] target={t} -> parts={pc}")

            if pc <= budget:
                best_parts = parts
                best_target = t
                break

            hi_target = t
            hi_parts = parts
            target = int(max(min_target, round(target * shrink)))
            if target == hi_target:
                target = hi_target - 1

        except Exception as e:
            print(f"[WARN] Failed at target={target}: {e}")
            target = int(max(min_target, round(target * shrink)))
            if target == hi_target:
                target -= 1

    if best_parts is None or best_target is None:
        parts, t = try_build(
            meshes,
            target=min_target,
            flipx180=flipx180, flipy180=flipy180, flipz180=flipz180,
            fill=fill,
            solid_color=solid_color,
            use_mesh_color=use_mesh_color,
            invert_y=invert_y,
            kind=kind,
            plates_per_voxel=plates_per_voxel,
            interlock=interlock,
            max_area=max_area,
            smart_fix=smart_fix,
        )
        return parts, t

    if hi_parts is None:
        return best_parts, best_target

    lo = best_target
    hi = hi_target
    best = (best_parts, best_target)

    for _ in range(search_iters):
        mid = (lo + hi) // 2
        if mid == lo or mid == hi:
            break
        try:
            parts_mid, tmid = try_build(
                meshes,
                target=mid,
                flipx180=flipx180, flipy180=flipy180, flipz180=flipz180,
                fill=fill,
                solid_color=solid_color,
                use_mesh_color=use_mesh_color,
                invert_y=invert_y,
                kind=kind,
                plates_per_voxel=plates_per_voxel,
                interlock=interlock,
                max_area=max_area,
                smart_fix=smart_fix,
            )
            pc = len(parts_mid)
            print(f"[SEARCH] target={tmid} -> parts={pc}")

            if pc <= budget:
                best = (parts_mid, tmid)
                lo = mid
            else:
                hi = mid
        except Exception:
            hi = mid

    return best

# -----------------------------------------------------------------------------
# 8. API 엔트리
# -----------------------------------------------------------------------------
def convert_glb_to_ldr(
    glb_path: str,
    out_ldr_path: str,
    *,
    budget: int = 100,
    target: int = 60,
    min_target: int = 5,
    shrink: float = 0.85,
    search_iters: int = 6,
    flipx180: bool = False,
    flipy180: bool = False,
    flipz180: bool = False,
    kind: str = "brick",
    plates_per_voxel: int = 3,
    interlock: bool = True,
    max_area: int = 20,
    solid_color: int = 4,
    use_mesh_color: bool = True,
    invert_y: bool = False,
    smart_fix: bool = True,
    step_order: str = "bottomup",

    # ✅ 새로 추가: 타이틀 지정(기존 호출 안 깨짐)
    title: Optional[str] = None,

    # ✅ 넘어와도 안 죽게 받기만
    solid: bool = False,
    fill: bool = False,
    extend_catalog: bool = True,
    max_len: int = 8,

    # ✅ 기타 호환용(무시)
    span: int = 4,
    max_new_voxels: int = 12000,
    refine_iters: int = 8,
    ensure_connected: bool = True,
    min_embed: int = 2,
    erosion_iters: int = 1,
    fast_search: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    _ = (
        solid, fill, extend_catalog, max_len,
        span, max_new_voxels, refine_iters, ensure_connected, min_embed, erosion_iters, fast_search,
        kwargs,
    )

    meshes = load_glb_meshes(glb_path)

    parts, final_target = build_under_budget(
        meshes,
        start_target=int(target),
        min_target=int(min_target),
        budget=int(budget),
        shrink=float(shrink),
        search_iters=int(search_iters),
        flipx180=bool(flipx180), flipy180=bool(flipy180), flipz180=bool(flipz180),
        fill=False,  # embedded 모드: 내부 채움 X
        kind=str(kind),
        plates_per_voxel=int(plates_per_voxel),
        interlock=bool(interlock),
        max_area=int(max_area),
        solid_color=int(solid_color),
        use_mesh_color=bool(use_mesh_color),
        invert_y=bool(invert_y),
        smart_fix=bool(smart_fix),
    )

    os.makedirs(os.path.dirname(out_ldr_path) or ".", exist_ok=True)

    tmp = out_ldr_path + ".nostep.tmp.ldr"
    write_ldr(
        tmp,
        parts,
        center=True,
        step_order="none",
        title=(title or f"{os.path.basename(glb_path)} (tgt={final_target})"),
        author="glb_to_ldr_embedded",
    )
    rewrite_steps_by_layer(tmp, out_ldr_path, step_order)

    try:
        os.remove(tmp)
    except OSError:
        pass

    return {"out": out_ldr_path, "parts": len(parts), "final_target": int(final_target)}

# -----------------------------------------------------------------------------
# 9. CLI main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glb", help="input .glb path")
    ap.add_argument("--out", default="out/result.ldr")
    ap.add_argument("--title", default=None)
    ap.add_argument("--target", type=int, default=60)
    ap.add_argument("--min-target", type=int, default=5)
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--shrink", type=float, default=0.85)
    ap.add_argument("--search-iters", type=int, default=6)
    ap.add_argument("--flipx180", action="store_true")
    ap.add_argument("--flipy180", action="store_true")
    ap.add_argument("--flipz180", action="store_true")
    ap.add_argument("--kind", choices=["brick", "plate"], default="brick")
    ap.add_argument("--plates-per-voxel", type=int, default=3)
    ap.add_argument("--no-interlock", action="store_true")
    ap.add_argument("--max-area", type=int, default=20)
    ap.add_argument("--solid", action="store_true", help="(호환용) fill 내부 채움. embedded 모드에서는 무시됨")
    ap.add_argument("--step-order", choices=["bottomup", "topdown", "none"], default="bottomup")
    ap.add_argument("--color", type=int, default=4, help="Solid color ID")
    ap.add_argument("--no-color", action="store_true", help="Disable mesh color")
    ap.add_argument("--invert-y", action="store_true", help="Force invert Y axis")
    ap.add_argument("--smart-fix", action="store_true", default=True)

    args = ap.parse_args()
    use_mesh_color = not args.no_color

    result = convert_glb_to_ldr(
        args.glb,
        args.out,
        title=args.title,
        budget=args.budget,
        target=args.target,
        min_target=args.min_target,
        shrink=args.shrink,
        search_iters=args.search_iters,
        flipx180=args.flipx180,
        flipy180=args.flipy180,
        flipz180=args.flipz180,
        kind=args.kind,
        plates_per_voxel=args.plates_per_voxel,
        interlock=(not args.no_interlock),
        max_area=args.max_area,
        solid_color=args.color,
        use_mesh_color=use_mesh_color,
        invert_y=args.invert_y,
        smart_fix=args.smart_fix,
        step_order=args.step_order,
        solid=bool(args.solid),
    )

    print(f"[DONE] {result}")

if __name__ == "__main__":
    main()
