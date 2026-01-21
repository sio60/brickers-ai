# glb_to_ldr_quick.py
from __future__ import annotations

import argparse
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import trimesh

# pylego3d 라이브러리 의존성 제거 -> 로컬 파일 import
from optimizer import optimize_bricks
from write_ldr import write_ldr

# -----------------------------------------------------------------------------
# 1. 내장 색상 팔레트 & 변환 함수
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
    70: (105, 64, 39),    # Reddish Brown-ish
    71: (160, 165, 169),  # Light Bluish Gray
    72: (108, 110, 107),  # Dark Bluish Gray
}

def rgb_to_ldraw_id(rgb: np.ndarray) -> int:
    """RGB (0~255) 값을 받아 가장 가까운 LDraw ID를 반환"""
    r = rgb.astype(np.float32)
    ids = list(LDRAW_RGB.keys())
    pal = np.array([LDRAW_RGB[i] for i in ids], dtype=np.float32)
    d = np.sum((pal - r[None, :]) ** 2, axis=1)
    return ids[int(np.argmin(d))]


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


def apply_flip(meshes: List[trimesh.Trimesh], flipx180: bool, flipy180: bool, flipz180: bool) -> None:
    Rx = np.eye(4); Ry = np.eye(4); Rz = np.eye(4)
    if flipx180: Rx[1,1]=-1; Rx[2,2]=-1
    if flipy180: Ry[0,0]=-1; Ry[2,2]=-1
    if flipz180: Rz[0,0]=-1; Rz[1,1]=-1
    T = Rz @ Ry @ Rx
    if flipx180 or flipy180 or flipz180:
        for m in meshes:
            m.apply_transform(T)


def preprocess_meshes(
    meshes: List[trimesh.Trimesh],
    *,
    target_studs: int,
    flipx180: bool,
    flipy180: bool,
    flipz180: bool,
) -> trimesh.Trimesh:
    meshes2 = [m.copy() for m in meshes]
    apply_flip(meshes2, flipx180, flipy180, flipz180)

    combined = trimesh.util.concatenate(meshes2)
    bounds = combined.bounds
    size = bounds[1] - bounds[0]
    max_xz = max(float(size[0]), float(size[2]))
    if max_xz <= 1e-9:
        raise ValueError("Mesh bounds too small / invalid")

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
# 3. 복셀화 및 색상 추출
# -----------------------------------------------------------------------------
def voxelize(mesh: trimesh.Trimesh, *, fill: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns: idx(indices), pts(centers), origin(calculated)
    """
    vg = mesh.voxelized(pitch=1.0)
    if fill:
        vg = vg.fill()
    
    idx = vg.sparse_indices
    pts = vg.points 
    
    if idx is None or len(idx) == 0:
        raise RuntimeError("Voxelization produced 0 voxels.")
        
    # Origin 역산
    origin = pts[0] - (idx[0] + 0.5)
    
    return idx.astype(np.int32), pts.astype(np.float32), origin


def sample_voxel_colors(mesh: trimesh.Trimesh, points: np.ndarray) -> List[int]:
    """
    points: (N, 3) 좌표
    Return: 각 포인트에 해당하는 LDraw Color ID 리스트
    """
    # 원본 메쉬 보존을 위해 copy 후 변환
    target_mesh = mesh.copy()
    
    # 텍스처 -> Vertex/Face Color 변환 및 Visual 재할당
    if hasattr(target_mesh.visual, 'to_color'):
        target_mesh.visual = target_mesh.visual.to_color()

    # 가장 가까운 메쉬 표면의 face index 찾기
    _, _, triangle_ids = target_mesh.nearest.on_surface(points)

    # 색상 추출
    face_colors = target_mesh.visual.face_colors[triangle_ids]

    ldraw_ids = []
    for rgba in face_colors:
        rgb = rgba[:3]
        c_id = rgb_to_ldraw_id(rgb)
        ldraw_ids.append(c_id)
    
    return ldraw_ids


# -----------------------------------------------------------------------------
# 4. 필터링 유틸
# -----------------------------------------------------------------------------
def invert_y_idx(idx: np.ndarray) -> Tuple[np.ndarray, int]:
    if idx is None or idx.size == 0:
        return idx, 0
    out = idx.copy()
    y_max = int(out[:, 1].max())
    out[:, 1] = (y_max - out[:, 1])
    return out, y_max


def keep_only_connected_to_ground(idx: np.ndarray) -> np.ndarray:
    min_ix = idx.min(axis=0)
    g = idx - min_ix
    
    vox_set = {tuple(v) for v in g.tolist()}
    if not vox_set:
        return idx

    seeds = [v for v in vox_set if v[1] == 0]
    if not seeds:
        return idx 

    from collections import deque
    q = deque(seeds)
    seen = set(seeds)

    neigh = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    while q:
        x,y,z = q.popleft()
        for dx,dy,dz in neigh:
            nv = (x+dx, y+dy, z+dz)
            if nv in vox_set and nv not in seen:
                seen.add(nv)
                q.append(nv)

    kept = np.array(list(seen), dtype=np.int32)
    kept = kept + min_ix
    return kept


def build_voxel_list(idx: np.ndarray, colors: List[int]) -> List[Dict[str, Any]]:
    min_ix = idx.min(axis=0)
    idx2 = idx - min_ix
    vox = []
    for i, (ix, iy, iz) in enumerate(idx2):
        vox.append({
            "x": int(ix), "y": int(iy), "z": int(iz), 
            "color": int(colors[i])
        })
    return vox


# -----------------------------------------------------------------------------
# 5. Build Logic
# -----------------------------------------------------------------------------
def try_build(
    meshes: List[trimesh.Trimesh],
    *,
    target: int,
    flipx180: bool, flipy180: bool, flipz180: bool,
    fill: bool,
    keep_connected: bool,
    kind: str,
    plates_per_voxel: int,
    interlock: bool,
    max_area: int,
    solid_color: int,
    use_mesh_color: bool, 
    invert_y: bool,
) -> Tuple[List[Dict[str, Any]], int]:

    # 1. 전처리
    mesh = preprocess_meshes(
        meshes, target_studs=target,
        flipx180=flipx180, flipy180=flipy180, flipz180=flipz180,
    )

    # 2. 복셀화 (원본 Y-up 상태)
    idx_raw, _, origin = voxelize(mesh, fill=fill)
    
    # 3. Y 반전 및 필터링 수행
    current_idx = idx_raw.copy()
    y_max_val = 0
    
    if invert_y:
        current_idx, y_max_val = invert_y_idx(current_idx)
    
    if keep_connected:
        current_idx = keep_only_connected_to_ground(current_idx)

    # 4. 색상 샘플링
    if use_mesh_color:
        search_idx = current_idx.copy()
        if invert_y:
            search_idx[:, 1] = y_max_val - search_idx[:, 1]
        
        search_pts = origin + (search_idx.astype(np.float32) + 0.5)
        
        final_colors = sample_voxel_colors(mesh, search_pts)
    else:
        final_colors = [solid_color] * len(current_idx)

    # 5. 최적화기 실행
    vox = build_voxel_list(current_idx, final_colors)
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
    start_target: int,
    min_target: int,
    budget: int,
    shrink: float,
    search_iters: int,
    flipx180: bool, flipy180: bool, flipz180: bool,
    fill: bool,
    keep_connected: bool,
    kind: str,
    plates_per_voxel: int,
    interlock: bool,
    max_area: int,
    solid_color: int,
    use_mesh_color: bool,
    invert_y: bool,
) -> Tuple[List[Dict[str, Any]], int]:

    kwargs = {
        "flipx180": flipx180, "flipy180": flipy180, "flipz180": flipz180,
        "fill": fill, "keep_connected": keep_connected,
        "kind": kind, "plates_per_voxel": plates_per_voxel,
        "interlock": interlock, "max_area": max_area,
        "solid_color": solid_color, "use_mesh_color": use_mesh_color,
        "invert_y": invert_y
    }

    hi_target = start_target
    hi_parts: Optional[List[Dict[str, Any]]] = None

    target = start_target
    best_parts = None
    best_target = None

    # 1. Budget 만족하는 최대 크기 찾기
    while target >= min_target:
        parts, t = try_build(meshes, target=target, **kwargs)
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

    if best_parts is None:
        parts, t = try_build(meshes, target=min_target, **kwargs)
        return parts, t

    if hi_parts is None:
        return best_parts, best_target

    # 2. Binary Search
    lo = best_target
    hi = hi_target
    best = (best_parts, best_target)

    for _ in range(search_iters):
        mid = (lo + hi) // 2
        if mid == lo or mid == hi:
            break
        parts_mid, tmid = try_build(meshes, target=mid, **kwargs)
        pc = len(parts_mid)
        print(f"[SEARCH] target={tmid} -> parts={pc}")
        if pc <= budget:
            best = (parts_mid, tmid)
            lo = mid
        else:
            hi = mid

    return best


# -----------------------------------------------------------------------------
# 6. Main CLI
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glb", help="input .glb path")
    ap.add_argument("--out", default="out/result.ldr")

    ap.add_argument("--target", type=int, default=60)
    
    # [설정] 기본 5 studs
    ap.add_argument("--min-target", type=int, default=5)
    
    # [설정] 기본 100개 이하
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

    ap.add_argument("--solid", action="store_true", help="Fill internal voxels")
    ap.add_argument("--keep-floating", action="store_true")
    ap.add_argument("--step-order", choices=["bottomup", "topdown", "none"], default="bottomup")

    ap.add_argument("--color", type=int, default=4, help="Solid color ID if mesh color unused")
    
    # [수정됨] 기본이 '색상 적용' 상태임. 끄고 싶으면 --no-color 사용
    ap.add_argument("--no-color", action="store_true", help="Disable mesh color sampling")
    
    ap.add_argument("--kids", action="store_true")
    ap.add_argument("--invert-y", action="store_true", help="Force invert Y axis")

    args = ap.parse_args()

    if args.kids:
        args.step_order = "bottomup"
        args.no_interlock = False

    meshes = load_glb_meshes(args.glb)

    # 로직 반전: no_color가 False일 때 use_mesh_color는 True
    use_mesh_color = not args.no_color

    parts, final_target = build_under_budget(
        meshes,
        start_target=args.target,
        min_target=args.min_target,
        budget=args.budget,
        shrink=args.shrink,
        search_iters=args.search_iters,
        flipx180=args.flipx180,
        flipy180=args.flipy180,
        flipz180=args.flipz180,
        fill=bool(args.solid),
        keep_connected=(not args.keep_floating),
        kind=args.kind,
        plates_per_voxel=args.plates_per_voxel,
        interlock=(not args.no_interlock),
        max_area=args.max_area,
        solid_color=args.color,
        use_mesh_color=use_mesh_color, # 전달
        invert_y=args.invert_y,
    )

    write_ldr(
        args.out,
        parts,
        center=True,
        step_order=args.step_order,
        title=f"{args.glb} (tgt={final_target})",
        author="glb_to_ldr_quick",
    )

    print(f"[DONE] out={args.out} parts={len(parts)} final_target={final_target} "
          f"color_mode={'MESH' if use_mesh_color else 'SOLID'}")


if __name__ == "__main__":
    main()