# pylego3d/optimizer.py
from __future__ import annotations

from typing import Dict, Tuple, List, Optional, Any
import numpy as np

# Canonical Orientation:
# key = (w, l) where w is the longer side in LDraw default orientation (rot=0).
# pylego3d/optimizer.py

PLATE_PARTS: Dict[Tuple[int, int], str] = {
    (1, 1): "3024.dat",
    (2, 1): "3023.dat",
    (2, 2): "3022.dat",
    (3, 2): "3021.dat",
    (4, 2): "3020.dat",
    (4, 1): "3710.dat",
}

BRICK_PARTS: Dict[Tuple[int, int], str] = {
    (1, 1): "3005.dat",  # 1x1
    (2, 1): "3004.dat",  # 1x2
    (3, 1): "3622.dat",  # 1x3
    (4, 1): "3010.dat",  # 1x4
    (6, 1): "3009.dat",  # 1x6
    (2, 2): "3003.dat",  # 2x2
    (3, 2): "3002.dat",  # 2x3
    (4, 2): "3001.dat",  # 2x4
}


def _get_part(kind: str, w: int, l: int) -> Optional[Tuple[str, int]]:
    catalog = PLATE_PARTS if kind == "plate" else BRICK_PARTS
    if (w, l) in catalog:
        return catalog[(w, l)], 0
    if (l, w) in catalog:
        return catalog[(l, w)], 90
    return None

def _candidate_sizes(kind: str, max_area: Optional[int] = None, avoid_1x1: bool = False) -> List[Tuple[int, int]]:
    catalog = PLATE_PARTS if kind == "plate" else BRICK_PARTS
    uniq = set()
    for (w, l) in catalog.keys():
        if avoid_1x1 and (w == 1 and l == 1):
            continue
        a, b = sorted((w, l))
        if max_area is not None and a * b > max_area:
            continue
        uniq.add((a, b))
    return sorted(list(uniq), key=lambda t: (t[0] * t[1], t[1]), reverse=True)

def _count_edge_crossings_patch(prev_ids: np.ndarray, x: int, z: int, w: int, l: int) -> int:
    patch = prev_ids[z:z+l, x:x+w]
    if patch.size == 0:
        return 0
    if patch.max() == -1:
        return 0
    v_edges = (patch[:, :-1] != -1) & (patch[:, 1:] != -1) & (patch[:, :-1] != patch[:, 1:])
    h_edges = (patch[:-1, :] != -1) & (patch[1:, :] != -1) & (patch[:-1, :] != patch[1:, :])
    return int(v_edges.sum() + h_edges.sum())

def _tile_one_layer(
    occ: np.ndarray,
    color_grid: np.ndarray,
    *,
    kind: str,
    layer_index: int,
    plates_per_voxel: int,
    interlock: bool,
    prev_ids: Optional[np.ndarray],
    max_area: Optional[int],
    cross_color_bridge_cells: int = 1,
    detail_max_area: Optional[int] = None,
    keep_unanchored_voxels: bool = False,
    avoid_1x1: bool = False,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    H, W = occ.shape
    used = np.zeros((H, W), dtype=bool)
    sizes = _candidate_sizes(kind, max_area=max_area, avoid_1x1=avoid_1x1)

    parts: List[Dict[str, Any]] = []
    ids = np.full((H, W), -1, dtype=np.int32)
    one_by_one = _get_part(kind, 1, 1)
    if one_by_one is None:
        raise RuntimeError("Catalog missing 1x1 part.")

    detail_mask: Optional[np.ndarray] = None
    if detail_max_area is not None:
        detail_mask = np.zeros((H, W), dtype=bool)
        for z in range(H):
            for x in range(W):
                if not occ[z, x]:
                    continue
                c = color_grid[z, x]
                for nx, nz in ((x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)):
                    if nx < 0 or nz < 0 or nx >= W or nz >= H:
                        detail_mask[z, x] = True
                        break
                    if not occ[nz, nx] or color_grid[nz, nx] != c:
                        detail_mask[z, x] = True
                        break

    def best_fit_at(cell_x: int, cell_z: int):
        anchor_color = int(color_grid[cell_z, cell_x])
        if anchor_color < 0:
            return None

        best = None
        best_key = None
        for (a, b) in sizes:
            for (w, l) in ((b, a), (a, b)):
                got = _get_part(kind, w, l)
                if got is None:
                    continue
                part, rot = got

                for off_z in range(l):
                    for off_x in range(w):
                        x = cell_x - off_x
                        z = cell_z - off_z
                        if x < 0 or z < 0 or x + w > W or z + l > H:
                            continue
                        if not occ[z:z+l, x:x+w].all():
                            continue
                        if used[z:z+l, x:x+w].any():
                            continue

                        area = w * l
                        if detail_max_area is not None and area > detail_max_area and detail_mask is not None:
                            if detail_mask[z:z+l, x:x+w].any():
                                continue
                        
                        color_patch = color_grid[z:z+l, x:x+w]
                        same_color_count = int(np.count_nonzero(color_patch == anchor_color))
                        cross_color_count = area - same_color_count

                        if cross_color_count > cross_color_bridge_cells:
                            continue

                        if layer_index == 0 or prev_ids is None:
                            support_count = area
                        else:
                            support_patch = prev_ids[z:z+l, x:x+w]
                            support_count = int(np.count_nonzero(support_patch != -1))

                        # Interlock requires vertical stud coupling only.
                        if layer_index > 0 and support_count == 0:
                            continue

                        cross = 0
                        if interlock and prev_ids is not None and layer_index > 0:
                            cross = _count_edge_crossings_patch(prev_ids, x, z, w, l)

                        anchor_supported = 1 if (layer_index == 0 or (prev_ids is not None and prev_ids[cell_z, cell_x] != -1)) else 0

                        weighted_area = area * 10
                        if cross > 0:
                            weighted_area += 1000

                        key = (
                            support_count > 0,
                            weighted_area,
                            anchor_supported,
                            support_count,
                            same_color_count,
                            cross,
                            -cross_color_count,
                        )
                        if best is None or key > best_key:
                            best = (x, z, w, l, part, rot, anchor_color)
                            best_key = key
        return best

    progress = True
    while progress:
        progress = False
        
        # Process unsupported edge cells first to force cantilever style bridging.
        candidates = []
        for z in range(H):
            for x in range(W):
                if occ[z, x] and not used[z, x] and color_grid[z, x] >= 0:
                    is_supp = 1 if (layer_index == 0 or (prev_ids is not None and prev_ids[z, x] != -1)) else 0
                    candidates.append((is_supp, x, z))
        
        candidates.sort(key=lambda item: item[0])

        for (is_supp, x, z) in candidates:
            if used[z, x]:
                continue

            anchor_color = int(color_grid[z, x])
            bf = best_fit_at(x, z)
            
            if bf is None:
                if avoid_1x1:
                    used[z, x] = True
                    continue

                part, rot = one_by_one
                x0 = x
                z0 = z
                w = l = 1
                color = anchor_color
                
                # Drop unsupported residual 1x1 cells above the first layer.
                if layer_index > 0:
                    vertical_support = 1 if (prev_ids is not None and prev_ids[z, x] != -1) else 0
                    if vertical_support == 0:
                        used[z, x] = True
                        continue
            else:
                x0, z0, w, l, part, rot, color = bf

            bid = len(parts)
            used[z0:z0+l, x0:x0+w] = True
            ids[z0:z0+l, x0:x0+w] = bid

            y = -layer_index * plates_per_voxel
            parts.append({
                "x": int(x0), "z": int(z0), "y": int(y),
                "w": int(w), "l": int(l),
                "rot": int(rot),
                "color": color,
                "part": part,
            })
            progress = True
            break

    if keep_unanchored_voxels:
        part_1x1, rot_1x1 = one_by_one
        ysigned = -layer_index * plates_per_voxel
        for z in range(H):
            for x in range(W):
                if not occ[z, x] or used[z, x]:
                    continue
                color = int(color_grid[z, x])
                if color < 0:
                    continue

                if layer_index > 0:
                    vertical_support = prev_ids is not None and prev_ids[z, x] != -1
                    if not vertical_support:
                        continue

                bid = len(parts)
                used[z, x] = True
                ids[z, x] = bid
                parts.append({
                    "x": int(x), "z": int(z), "y": int(ysigned),
                    "w": 1, "l": 1,
                    "rot": int(rot_1x1),
                    "color": color,
                    "part": part_1x1,
                })

    return parts, ids

def optimize_bricks(
    bricks: List[Dict[str, Any]],
    *,
    mode: str = "voxel",
    support_direction: str = "topdown",
    kind: str = "brick",
    plates_per_voxel: int = 3,
    interlock: bool = True,
    max_area: Optional[int] = 32,
    cross_color_bridge_cells: int = 1,
    detail_max_area: Optional[int] = None,
    keep_unanchored_voxels: bool = False,
    avoid_1x1: bool = False,
) -> List[Dict[str, Any]]:
    if mode != "voxel":
        raise ValueError("Only mode='voxel' supported.")
    if not bricks:
        return []

    xs = np.array([b["x"] for b in bricks], dtype=np.int32)
    ys = np.array([b["y"] for b in bricks], dtype=np.int32)
    zs = np.array([b["z"] for b in bricks], dtype=np.int32)

    min_x, min_y, min_z = int(xs.min()), int(ys.min()), int(zs.min())
    nx, ny, nz = xs - min_x, ys - min_y, zs - min_z

    W = int(nx.max()) + 1
    H = int(nz.max()) + 1
    L = int(ny.max()) + 1

    occ_layers: List[np.ndarray] = [np.zeros((H, W), dtype=bool) for _ in range(L)]
    color_layers: List[np.ndarray] = [np.full((H, W), -1, dtype=np.int32) for _ in range(L)]
    for i, b in enumerate(bricks):
        x = int(nx[i]); y = int(ny[i]); z = int(nz[i])
        c = int(b.get("color", 4))
        occ_layers[y][z, x] = True
        color_layers[y][z, x] = c

    out: List[Dict[str, Any]] = []
    prev_ids_global: Optional[np.ndarray] = None

    if support_direction == "topdown":
        layer_sequence = list(range(L - 1, -1, -1))
    elif support_direction == "bottomup":
        layer_sequence = list(range(L))
    else:
        raise ValueError("support_direction must be 'topdown' or 'bottomup'.")

    for processing_step, layer_index in enumerate(layer_sequence):
        ids_global = np.full((H, W), -1, dtype=np.int32)
        parts, ids = _tile_one_layer(
            occ_layers[layer_index],
            color_layers[layer_index],
            kind=kind,
            layer_index=processing_step if support_direction == "topdown" else layer_index,
            plates_per_voxel=plates_per_voxel,
            interlock=interlock,
            prev_ids=prev_ids_global,
            max_area=max_area,
            cross_color_bridge_cells=cross_color_bridge_cells,
            detail_max_area=detail_max_area,
            keep_unanchored_voxels=keep_unanchored_voxels,
            avoid_1x1=avoid_1x1,
        )

        for p in parts:
            p["x"] += min_x
            p["z"] += min_z
            # Keep absolute Y coordinates aligned with the original layer index.
            p["y"] = -layer_index * plates_per_voxel
            out.append(p)

        base = len(out) - len(parts)
        m = ids >= 0
        ids_global[m] = ids[m] + base

        prev_ids_global = ids_global

    return out
