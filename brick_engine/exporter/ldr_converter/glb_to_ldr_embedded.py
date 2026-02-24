"""
glb_to_ldr_embedded.py - Enhanced Brickify Engine
- automatic stability detection and repair
- floating brick pre-repair (structural reinforcement)
- expanded LDraw color palette (30+)
- SSE real-time logs (brick only, no plates)
"""

import logging

logger = logging.getLogger("brick_engine.glb_to_ldr")

import os
import sys
import argparse
import numpy as np
import trimesh
from scipy.spatial import KDTree
from scipy.ndimage import binary_dilation, binary_erosion
from typing import Dict, List, Any, Optional, Tuple, Literal, Set, Deque
from collections import deque
from pathlib import Path

# Add project root to path for pylego3d imports
# Now located in exporter/ldr_converter/, so root is two levels up
_curr_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_curr_dir, "..", ".."))

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from pylego3d.optimizer import optimize_bricks
    from pylego3d.write_ldr import write_ldr
except ImportError:
    sys.path.append(_project_root)
    from pylego3d.optimizer import optimize_bricks
    from pylego3d.write_ldr import write_ldr

# =============================================================================
# Expanded LDraw color palette (30+)
# =============================================================================
LDRAW_COLORS: Dict[int, Tuple[int, int, int, str]] = {
    0:  (33, 33, 33, "Black"),
    1:  (0, 85, 191, "Blue"),
    2:  (0, 123, 40, "Green"),
    3:  (0, 131, 138, "Teal"),
    4:  (180, 0, 0, "Red"),
    5:  (171, 67, 183, "Dark Pink"),
    6:  (91, 28, 12, "Brown"),
    7:  (156, 146, 145, "Light Gray"),
    8:  (99, 95, 82, "Dark Gray"),
    9:  (107, 171, 220, "Light Blue"),
    10: (97, 189, 76, "Bright Green"),
    11: (0, 170, 164, "Light Turquoise"),
    12: (255, 99, 71, "Salmon"),
    13: (255, 148, 194, "Pink"),
    14: (255, 220, 0, "Yellow"),
    15: (255, 255, 255, "White"),
    17: (173, 221, 80, "Lime"),
    18: (251, 171, 24, "Light Orange"),
    19: (215, 197, 153, "Tan"),
    25: (245, 134, 36, "Orange"),
    26: (202, 31, 123, "Magenta"),
    27: (159, 195, 65, "Bright Light Green"),
    28: (33, 55, 23, "Dark Green"),
    70: (89, 47, 14, "Reddish Brown"),
    71: (175, 181, 199, "Light Bluish Gray"),
    72: (108, 110, 104, "Dark Bluish Gray"),
    73: (117, 142, 220, "Medium Blue"),
    74: (183, 212, 37, "Medium Lime"),
    78: (254, 186, 189, "Light Pink"),
    84: (170, 125, 85, "Medium Dark Flesh"),
    85: (89, 39, 115, "Dark Purple"),
    110: (255, 187, 0, "Bright Light Orange"),
    226: (242, 206, 46, "Bright Light Yellow"),
    320: (120, 27, 33, "Dark Red"),
    378: (163, 193, 173, "Sand Green"),
    484: (179, 62, 0, "Dark Orange"),
}

_COLOR_IDS = list(LDRAW_COLORS.keys())
_COLOR_RGB = np.array([LDRAW_COLORS[i][:3] for i in _COLOR_IDS], dtype=np.float32)
_COLOR_TREE = KDTree(_COLOR_RGB)


def match_ldraw_color(rgb: Tuple[float, float, float]) -> int:
    """Map an RGB value to the nearest LDraw color ID."""
    v = np.array(rgb, dtype=np.float32)
    if v.max() <= 1.0:
        v *= 255.0
    _, idx = _COLOR_TREE.query(v)
    return _COLOR_IDS[int(idx)]


# =============================================================================
# Floating brick repair (structural reinforcement)
# =============================================================================
def _horizontal_neighbors(pos: Tuple[int, int, int]) -> Tuple[Tuple[int, int, int], ...]:
    x, y, z = pos
    return (
        (x + 1, y, z),
        (x - 1, y, z),
        (x, y, z + 1),
        (x, y, z - 1),
    )


def _neighbors6(pos: Tuple[int, int, int]) -> Tuple[Tuple[int, int, int], ...]:
    x, y, z = pos
    return (
        (x + 1, y, z),
        (x - 1, y, z),
        (x, y + 1, z),
        (x, y - 1, z),
        (x, y, z + 1),
        (x, y, z - 1),
    )


def _is_vertically_supported(
    pos: Tuple[int, int, int],
    occupied: Set[Tuple[int, int, int]],
) -> bool:
    x, y, z = pos
    return y == 0 or (x, y - 1, z) in occupied


def _compute_stable_positions(occupied: Set[Tuple[int, int, int]]) -> Set[Tuple[int, int, int]]:
    """
    Stable voxels:
    - direct support from below, or
    - same-layer horizontal connection to another stable voxel.
    """
    stable: Set[Tuple[int, int, int]] = set()
    queue: Deque[Tuple[int, int, int]] = deque()

    for pos in occupied:
        if _is_vertically_supported(pos, occupied):
            stable.add(pos)
            queue.append(pos)

    while queue:
        cur = queue.popleft()
        for nxt in _horizontal_neighbors(cur):
            if nxt in occupied and nxt not in stable:
                stable.add(nxt)
                queue.append(nxt)

    return stable


def _build_horizontal_path(
    start: Tuple[int, int, int],
    end: Tuple[int, int, int],
) -> List[Tuple[int, int, int]]:
    """
    Manhattan path on the same y level (x first, then z).
    Returns moved-to cells (start excluded, end included).
    """
    sx, sy, sz = start
    ex, _, ez = end
    x, z = sx, sz
    path: List[Tuple[int, int, int]] = []

    while x != ex:
        x += 1 if ex > x else -1
        path.append((x, sy, z))
    while z != ez:
        z += 1 if ez > z else -1
        path.append((x, sy, z))

    return path


def _connected_components_3d(
    occupied: Set[Tuple[int, int, int]],
) -> List[Set[Tuple[int, int, int]]]:
    if not occupied:
        return []

    visited: Set[Tuple[int, int, int]] = set()
    comps: List[Set[Tuple[int, int, int]]] = []

    for seed in occupied:
        if seed in visited:
            continue
        q: Deque[Tuple[int, int, int]] = deque([seed])
        visited.add(seed)
        comp: Set[Tuple[int, int, int]] = set()

        while q:
            cur = q.popleft()
            comp.add(cur)
            for nxt in _neighbors6(cur):
                if nxt in occupied and nxt not in visited:
                    visited.add(nxt)
                    q.append(nxt)

        comps.append(comp)

    comps.sort(key=len, reverse=True)
    return comps


def add_vertical_connector_plugs(
    vox: List[Dict[str, Any]],
    *,
    max_passes: int = 2,
) -> List[Dict[str, Any]]:
    """
    Connect detached components to the largest component by inserting a single
    voxel vertically (same x/z, one-cell y gap) when possible.
    """
    if not vox:
        return []

    voxel_map = {(v["x"], v["y"], v["z"]): int(v["color"]) for v in vox}
    total_added = 0

    for _ in range(max_passes):
        occupied = set(voxel_map.keys())
        comps = _connected_components_3d(occupied)
        if len(comps) <= 1:
            break

        main = comps[0]
        added_this_pass = 0

        for comp in comps[1:]:
            bridge_pos: Optional[Tuple[int, int, int]] = None
            bridge_color: Optional[int] = None

            # Prefer a one-voxel vertical plug (same x/z, y gap == 2).
            for (x, y, z) in comp:
                for dy in (-2, 2):
                    target = (x, y + dy, z)
                    mid = (x, y + (dy // 2), z)
                    if target in main and mid not in occupied:
                        bridge_pos = mid
                        bridge_color = voxel_map.get((x, y, z), voxel_map.get(target, 4))
                        break
                if bridge_pos is not None:
                    break

            if bridge_pos is None:
                continue

            voxel_map[bridge_pos] = int(bridge_color if bridge_color is not None else 4)
            occupied.add(bridge_pos)
            added_this_pass += 1

        total_added += added_this_pass
        if added_this_pass == 0:
            break

    if total_added > 0:
        logger.info(f"[Step] Vertical connector plugs added: {total_added}")

    return [{"x": x, "y": y, "z": z, "color": c} for (x, y, z), c in voxel_map.items()]


def embed_floating_parts(
    vox: List[Dict[str, Any]],
    max_iters: int = 3,
    max_bridge_len: int = 6,
) -> List[Dict[str, Any]]:
    """
    Connect floating parts with horizontal bridges before optimization.
    """
    if not vox:
        return []

    voxel_map = {(v["x"], v["y"], v["z"]): int(v["color"]) for v in vox}
    new_colors = dict(voxel_map)

    for _ in range(max_iters):
        occupied = set(new_colors.keys())
        stable = _compute_stable_positions(occupied)
        unstable = occupied - stable
        if not unstable:
            break

        stable_by_y: Dict[int, List[Tuple[int, int, int]]] = {}
        for s in stable:
            stable_by_y.setdefault(s[1], []).append(s)

        updates: Dict[Tuple[int, int, int], int] = {}
        visited: Set[Tuple[int, int, int]] = set()

        for seed in unstable:
            if seed in visited:
                continue

            queue = deque([seed])
            visited.add(seed)
            component: List[Tuple[int, int, int]] = []

            # Connected floating component in the same layer.
            while queue:
                cur = queue.popleft()
                component.append(cur)
                for nxt in _horizontal_neighbors(cur):
                    if nxt in unstable and nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)

            y = seed[1]
            anchors = stable_by_y.get(y, [])

            # Fallback: project from stable voxels on the layer below.
            if not anchors and y > 0:
                anchors = [(ax, y, az) for (ax, _, az) in stable_by_y.get(y - 1, [])]
            if not anchors:
                continue

            best_pair: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None
            best_dist: Optional[int] = None

            for pos in component:
                px, _, pz = pos
                for anchor in anchors:
                    ax, _, az = anchor
                    dist = abs(px - ax) + abs(pz - az)
                    if dist == 0:
                        continue
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best_pair = (pos, anchor)

            if best_pair is None:
                continue
            if best_dist is not None and best_dist > max_bridge_len:
                continue

            src, anchor = best_pair
            src_color = int(new_colors[src])
            for bridge in _build_horizontal_path(src, anchor):
                if bridge not in new_colors:
                    updates[bridge] = src_color

        if not updates:
            break
        new_colors.update(updates)

    return [{"x": x, "y": y, "z": z, "color": c} for (x, y, z), c in new_colors.items()]


# =============================================================================
# Color smoothing (noise reduction)
# =============================================================================
def smooth_colors(vox: List[Dict[str, Any]], passes: int = 1) -> List[Dict[str, Any]]:
    """Reduce color noise by majority vote from neighboring voxels."""
    if passes <= 0 or not vox:
        return vox
    
    voxel_map = {(v["x"], v["y"], v["z"]): v["color"] for v in vox}
    
    neighbors = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    
    for _ in range(passes):
        new_map = dict(voxel_map)
        for (x, y, z), color in voxel_map.items():
            neighbor_colors = []
            for dx, dy, dz in neighbors:
                npos = (x+dx, y+dy, z+dz)
                if npos in voxel_map:
                    neighbor_colors.append(voxel_map[npos])
            
            if neighbor_colors:
                # Replace with the most frequent neighbor color
                vals, counts = np.unique(neighbor_colors, return_counts=True)
                most_common = vals[np.argmax(counts)]
                # Replace only when a strict neighborhood majority exists
                if counts.max() > len(neighbor_colors) // 2:
                    new_map[(x, y, z)] = int(most_common)
        
        voxel_map = new_map
    
    return [{"x": x, "y": y, "z": z, "color": c} for (x, y, z), c in voxel_map.items()]


def prune_detached_fragments(
    vox: List[Dict[str, Any]],
    *,
    min_component_size: int = 12,
    max_component_centroid_ratio: float = 1.8,
) -> List[Dict[str, Any]]:
    """
    Remove small detached voxel components that are far from the model core.
    This avoids long artifact lines when support fixing is disabled or limited.
    """
    if not vox:
        return []

    voxel_map = {(v["x"], v["y"], v["z"]): int(v["color"]) for v in vox}
    occupied = set(voxel_map.keys())
    if not occupied:
        return []

    pts = np.array(list(occupied), dtype=np.float32)
    global_centroid = pts.mean(axis=0)
    dists = np.linalg.norm(pts - global_centroid, axis=1)
    core_radius = float(np.percentile(dists, 95))
    if core_radius <= 1e-6:
        core_radius = 1.0

    neighbors = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    visited: Set[Tuple[int, int, int]] = set()
    keep: Set[Tuple[int, int, int]] = set()

    for seed in occupied:
        if seed in visited:
            continue
        queue = deque([seed])
        visited.add(seed)
        component: List[Tuple[int, int, int]] = []

        while queue:
            cx, cy, cz = queue.popleft()
            component.append((cx, cy, cz))
            for dx, dy, dz in neighbors:
                nxt = (cx + dx, cy + dy, cz + dz)
                if nxt in occupied and nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)

        comp_size = len(component)
        comp_pts = np.array(component, dtype=np.float32)
        comp_centroid = comp_pts.mean(axis=0)
        comp_dist = float(np.linalg.norm(comp_centroid - global_centroid))

        if comp_size >= min_component_size or comp_dist <= core_radius * max_component_centroid_ratio:
            keep.update(component)

    return [{"x": x, "y": y, "z": z, "color": voxel_map[(x, y, z)]} for (x, y, z) in keep]


# =============================================================================
# Main conversion function
# =============================================================================
def _single_conversion(
    combined: trimesh.Trimesh,
    out_ldr_path: str,
    target: int,
    kind: str,
    plates_per_voxel: int,
    interlock: bool,
    max_area: int,
    solid_color: int,
    use_mesh_color: bool,
    step_order: str,
    glb_path: str,
    smart_fix: bool = True,
    color_smooth: int = 0,
    **kwargs: Any
) -> Tuple[int, List[Dict]]:
    """
    Single conversion pass: voxelization + optimization. Returns (brick_count, optimized_bricks).
    """
    import time
    start_t = time.time()
    mesh = combined.copy()
    
    # Scale to target studs
    bounds = mesh.bounds
    size = bounds[1] - bounds[0]
    max_xz = max(size[0], size[2])
    scale = target / max_xz
    mesh.apply_scale(scale)
    
    # Ground the model
    mesh.apply_translation([0, -mesh.bounds[0][1], 0])
    
    # Brick height correction (20 LDU width / 24 LDU height ratio)
    mesh.vertices[:, 1] *= (20.0 / 24.0)

    # Voxelize
    kwargs = kwargs.copy()
    pitch = kwargs.pop("pitch", 1.0)
    surface_dilate_iters = int(kwargs.pop("surface_dilate_iters", 0))
    surface_only = bool(kwargs.pop("surface_only", False))
    solid_fill = bool(kwargs.pop("solid", True))
    logger.info(f"[Step] Voxelizing (Target: {target}, Pitch: {pitch})...")
    v_start = time.time()
    vg = mesh.voxelized(pitch=pitch)
    if solid_fill:
        vg = vg.fill()
    dense = vg.matrix.astype(bool)
    if surface_dilate_iters > 0:
        dense = binary_dilation(
            dense,
            structure=np.ones((3, 3, 3), dtype=bool),
            iterations=surface_dilate_iters,
        )
    if surface_only:
        eroded = binary_erosion(dense, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)
        shell = dense & ~eroded
        if np.any(shell):
            dense = shell
    vg = trimesh.voxel.base.VoxelGrid(dense, transform=vg.transform)
    v_end = time.time()
    logger.info(f"[Step] Voxelization Done: {v_end - v_start:.2f}s")
    
    indices = vg.sparse_indices
    if indices is None or len(indices) == 0:
        return 0, []
        
    logger.info(f"[Step] Voxel count: {len(indices)}")
    
    # Voxel threshold check (memory guard).
    voxel_threshold = kwargs.get("max_new_voxels", 50000)
    max_pitch = kwargs.get("max_pitch", 3.0)
    
    if len(indices) > voxel_threshold:
        if pitch < max_pitch:
            new_pitch = pitch + 0.5
            logger.warning(f"[Warning] Voxels ({len(indices)}) > threshold ({voxel_threshold})")
            logger.info(f"[Retry] Lowering resolution: pitch {pitch} -> {new_pitch}")
            return _single_conversion(
                combined, out_ldr_path, target, kind, plates_per_voxel,
                interlock, max_area, solid_color, use_mesh_color, step_order, glb_path,
                smart_fix=smart_fix, color_smooth=color_smooth,
                pitch=new_pitch, **kwargs
            )
        else:
            logger.error(f"[Error] Pitch at max ({max_pitch}), still {len(indices)} voxels > {voxel_threshold}")
            return -1, []

    # Color sampling (KDTree-based speed optimization)
    logger.info("[Step] Color Sampling...")
    c_start = time.time()
    centers = vg.points
    if use_mesh_color:
        # Convert texture/visual information to per-vertex colors (once)
        if hasattr(mesh.visual, 'to_color'):
            mesh.visual = mesh.visual.to_color()
        
        # Sample color from nearest mesh vertex for each voxel center
        v_tree = KDTree(mesh.vertices)
        _, v_indices = v_tree.query(centers)
        colors_raw = mesh.visual.vertex_colors[v_indices][:, :3].astype(np.float32)
        
        # Brightness/saturation correction (avoid over-saturation)
        # Defaults are intentionally mild to preserve source colors
        brightness = kwargs.get("color_brightness", 1.05)
        saturation = kwargs.get("color_saturation", 1.05)
        
        if brightness != 1.0:
            colors_raw = np.clip(colors_raw * brightness, 0, 255)
            
        if saturation != 1.0:
            avg = np.mean(colors_raw, axis=-1, keepdims=True)
            colors_raw = np.clip(avg + (colors_raw - avg) * saturation, 0, 255)
    else:
        colors_raw = np.tile([200, 200, 200], (len(centers), 1))
    c_end = time.time()
    logger.info(f"[Step] Color Sampling Done: {c_end - c_start:.2f}s")

    # Build voxel list
    bricks_data = []
    for i in range(len(indices)):
        c_id = solid_color if not use_mesh_color else match_ldraw_color(tuple(colors_raw[i]))
        bricks_data.append({
            "x": int(indices[i][0]),
            "y": int(indices[i][1]),
            "z": int(indices[i][2]),
            "color": c_id
        })

    # Color smoothing
    if color_smooth > 0:
        bricks_data = smooth_colors(bricks_data, passes=color_smooth)

    if kwargs.get("prune_fragments", False):
        bricks_data = prune_detached_fragments(
            bricks_data,
            min_component_size=int(kwargs.get("prune_min_component_size", 12)),
            max_component_centroid_ratio=float(kwargs.get("prune_max_centroid_ratio", 1.8)),
        )
    
    # Floating brick repair
    if smart_fix:
        logger.info("[Step] Embedding floating parts...")
        bricks_data = embed_floating_parts(
            bricks_data,
            max_iters=int(kwargs.get("support_repair_iters", 3)),
            max_bridge_len=int(kwargs.get("max_bridge_len", 6)),
        )
        if bool(kwargs.get("vertical_connector_plug", True)):
            bricks_data = add_vertical_connector_plugs(
                bricks_data,
                max_passes=int(kwargs.get("vertical_connector_passes", 2)),
            )

    # Optimize (Greedy Packing)
    logger.info("[Step] Optimization (Greedy Packing) starting...")
    o_start = time.time()
    keep_unanchored_default = False
    optimized = optimize_bricks(
        bricks_data,
        support_direction=str(kwargs.get("support_direction", "topdown")),
        kind=kind,
        plates_per_voxel=plates_per_voxel,
        interlock=interlock,
        max_area=max_area,
        cross_color_bridge_cells=int(kwargs.get("cross_color_bridge_cells", 1)),
        detail_max_area=kwargs.get("detail_max_area"),
        keep_unanchored_voxels=bool(kwargs.get("keep_unanchored_voxels", keep_unanchored_default)),
        avoid_1x1=bool(kwargs.get("avoid_1x1", False)),
    )
    o_end = time.time()
    logger.info(f"[Step] Optimization Done: {o_end - o_start:.2f}s")
    
    return len(optimized), optimized


def convert_glb_to_ldr(
    glb_path: str,
    out_ldr_path: str,
    *,
    target: int = 80,
    budget: int = 100,
    shrink: float = 0.85,
    search_iters: int = 3,
    flipx180: bool = False,
    flipy180: bool = False,
    flipz180: bool = False,
    kind: str = "brick",
    plates_per_voxel: int = 3,
    interlock: bool = True,
    max_area: int = 32,
    solid_color: int = 4,
    use_mesh_color: bool = True,
    invert_y: bool = False,
    smart_fix: bool = True,
    step_order: str = "bottomup",
    color_smooth: int = 0,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Convert GLB to LDR (with budget-seeking loop).
    
    Features:
    - automatic stability detection and repair
    - floating brick pre-repair
    - expanded color palette and SSE log callback
    """
    # SSE log callback
    _log_cb = kwargs.pop("log_callback", None)
    def _log(step: str, message: str):
        if _log_cb:
            try:
                _log_cb(step, message)
            except Exception:
                pass

    logger.info(f"[Engine] Starting conversion: {glb_path} -> {out_ldr_path}")
    logger.info(f"[Engine] Target: {target} studs, Budget: {budget} bricks")

    _log("brickify", "Computing a stable brickification strategy...")

    # 1. Load meshes
    scene = trimesh.load(glb_path, force='scene')
    if isinstance(scene, trimesh.Trimesh):
        meshes = [scene]
    else:
        meshes = [m for m in scene.dump(concatenate=False) if isinstance(m, trimesh.Trimesh)]
    
    if not meshes:
        raise RuntimeError("No meshes found in GLB")

    # 2. Combine & Initial Orientation
    combined = trimesh.util.concatenate(meshes)
    Rx = np.eye(4); Ry = np.eye(4); Rz = np.eye(4)
    if flipx180: Rx[1,1]=-1; Rx[2,2]=-1
    if flipy180: Ry[0,0]=-1; Ry[2,2]=-1
    if flipz180: Rz[0,0]=-1; Rz[1,1]=-1
    combined.apply_transform(Rz @ Ry @ Rx)

    # 3. Budget-Seeking Loop
    curr_target = float(target)
    final_optimized = []
    
    for i in range(search_iters):
        logger.info(f"[Engine] SEARCH ITERATION {i+1}/{search_iters}")
        logger.info(f"[Engine] Current Target Studs: {int(curr_target)}")
        _log("brickify", f"Tuning brick count... ({i+1}/{search_iters})")
        
        parts_count, optimized = _single_conversion(
            combined=combined,
            out_ldr_path=out_ldr_path,
            target=int(curr_target),
            kind=kind,
            plates_per_voxel=plates_per_voxel,
            interlock=interlock,
            max_area=max_area,
            solid_color=solid_color,
            use_mesh_color=use_mesh_color,
            step_order=step_order,
            glb_path=glb_path,
            smart_fix=smart_fix,
            color_smooth=color_smooth,
            **kwargs
        )
        
        if parts_count < 0:
            logger.warning(f"[Engine] Iter {i+1} Result: VOXEL_THRESHOLD EXCEEDED")
        else:
            final_optimized = optimized
            logger.info(f"[Engine] Iter {i+1} Result: {parts_count} bricks (Budget: {budget})")
            
            if parts_count <= budget:
                logger.info(f"[Engine] SUCCESS: Budget met! ({parts_count} <= {budget})")
                _log("brickify", f"Budget met with {parts_count} bricks.")
                break
        
        if i < search_iters - 1:
            curr_target *= shrink
            if curr_target < 5:
                curr_target = 5
            logger.info(f"[Engine] Budget EXCEEDED. Shrinking target to {curr_target:.1f}")
            _log("brickify", "Too many bricks; retrying with lower resolution...")
        else:
            logger.warning(f"[Engine] Failed to meet budget after {search_iters} iters.")

    # 4. Write LDR
    if not final_optimized:
        raise RuntimeError("Failed to generate any bricks")

    _log("brickify", "Preparing build steps...")

    write_ldr(
        out_ldr_path,
        final_optimized,
        step_order=step_order,
        title=Path(glb_path).stem
    )

    _log("brickify", "Brick model generation complete.")

    return {
        "parts": len(final_optimized),
        "final_target": int(curr_target),
        "out": out_ldr_path
    }


# Compatibility shims
def convert_glb_to_ldr_v3_inline(*args, **kwargs):
    """Compatibility wrapper."""
    return convert_glb_to_ldr(*args, **kwargs)


def embed_voxels_downwards(grid):
    """Compatibility stub."""
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("glb")
    parser.add_argument("--out", default="output.ldr")
    parser.add_argument("--target", type=int, default=80)
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--smart-fix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--color-smooth", type=int, default=0)
    parser.add_argument("--support-direction", choices=["topdown", "bottomup"], default="topdown")
    args = parser.parse_args()
    
    convert_glb_to_ldr(
        args.glb, args.out, 
        target=args.target, 
        budget=args.budget,
        smart_fix=args.smart_fix,
        color_smooth=args.color_smooth,
        solid=True,
        surface_only=False,
        support_direction=args.support_direction,
    )
