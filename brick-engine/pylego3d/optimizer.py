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
    (1, 1): "3005.dat",
    (2, 1): "3004.dat",
    (2, 2): "3003.dat",
    (3, 2): "3002.dat",
    (4, 2): "3001.dat",
    (4, 1): "3010.dat",
}


def _get_part(kind: str, w: int, l: int) -> Optional[Tuple[str, int]]:
    catalog = PLATE_PARTS if kind == "plate" else BRICK_PARTS
    if (w, l) in catalog:
        return catalog[(w, l)], 0
    if (l, w) in catalog:
        return catalog[(l, w)], 90
    return None

def _candidate_sizes(kind: str, max_area: Optional[int] = None) -> List[Tuple[int, int]]:
    catalog = PLATE_PARTS if kind == "plate" else BRICK_PARTS
    uniq = set()
    for (w, l) in catalog.keys():
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

def _tile_one_color(
    occ: np.ndarray,
    color: Any,
    *,
    kind: str,
    layer_index: int,
    plates_per_voxel: int,
    interlock: bool,
    prev_ids: Optional[np.ndarray],
    max_area: Optional[int],
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    H, W = occ.shape
    used = np.zeros((H, W), dtype=bool)
    sizes = _candidate_sizes(kind, max_area=max_area)

    parts: List[Dict[str, Any]] = []
    ids = np.full((H, W), -1, dtype=np.int32)

    def best_fit_at(x: int, z: int):
        best = None
        best_key = None  # (cross, area)
        for (a, b) in sizes:
            for (w, l) in ((b, a), (a, b)):
                if x + w > W or z + l > H:
                    continue
                if not occ[z:z+l, x:x+w].all():
                    continue
                if used[z:z+l, x:x+w].any():
                    continue
                got = _get_part(kind, w, l)
                if got is None:
                    continue
                part, rot = got
                cross = 0
                if interlock and prev_ids is not None and layer_index > 0:
                    cross = _count_edge_crossings_patch(prev_ids, x, z, w, l)
                area = w * l
                key = (cross, area)
                if best is None or key > best_key:
                    best = (w, l, part, rot)
                    best_key = key
        return best

    for z in range(H):
        for x in range(W):
            if not occ[z, x] or used[z, x]:
                continue

            bf = best_fit_at(x, z)
            if bf is None:
                got = _get_part(kind, 1, 1)
                if got is None:
                    raise RuntimeError("Catalog missing 1x1 part.")
                part, rot = got
                w = l = 1
            else:
                w, l, part, rot = bf

            bid = len(parts)
            used[z:z+l, x:x+w] = True
            ids[z:z+l, x:x+w] = bid

            y = -layer_index * plates_per_voxel
            parts.append({
                "x": int(x), "z": int(z), "y": int(y),
                "w": int(w), "l": int(l),
                "rot": int(rot),
                "color": color,
                "part": part,
            })

    return parts, ids

def optimize_bricks(
    bricks: List[Dict[str, Any]],
    *,
    mode: str = "voxel",
    kind: str = "brick",
    plates_per_voxel: int = 3,
    interlock: bool = True,
    max_area: Optional[int] = 20,
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

    layers: List[Dict[Any, np.ndarray]] = [dict() for _ in range(L)]
    for i, b in enumerate(bricks):
        x = int(nx[i]); y = int(ny[i]); z = int(nz[i])
        c = b.get("color", 4)
        d = layers[y]
        if c not in d:
            d[c] = np.zeros((H, W), dtype=bool)
        d[c][z, x] = True

    out: List[Dict[str, Any]] = []
    prev_ids_global: Optional[np.ndarray] = None

    for layer_index in range(L):
        color_masks = layers[layer_index]
        ids_global = np.full((H, W), -1, dtype=np.int32)

        for color, occ in color_masks.items():
            parts, ids = _tile_one_color(
                occ, color,
                kind=kind,
                layer_index=layer_index,
                plates_per_voxel=plates_per_voxel,
                interlock=interlock,
                prev_ids=prev_ids_global,
                max_area=max_area,
            )

            for p in parts:
                p["x"] += min_x
                p["z"] += min_z
                out.append(p)

            base = len(out) - len(parts)
            m = ids >= 0
            ids_global[m] = ids[m] + base

        prev_ids_global = ids_global

    return out
