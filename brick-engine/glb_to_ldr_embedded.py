from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional, Literal, Callable

import numpy as np
import trimesh
from scipy.spatial import KDTree


import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent
for p in (_THIS_DIR, _PARENT_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from pylego3d.optimizer import optimize_bricks
import pylego3d.optimizer as optmod
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

# =============================================================================
# V3 INLINE (ported from exporter/glb_to_ldr_v3.py)
# =============================================================================
LDU_PER_STUD = 20
LDU_BRICK_H = 24
LDU_PLATE_H = 8

ROT_0 = "1 0 0 0 1 0 0 0 1"
ROT_90 = "0 0 1 0 1 0 -1 0 0"
ROT_180 = "-1 0 0 0 1 0 0 0 -1"
ROT_270 = "0 0 -1 0 1 0 1 0 0"

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
    20: (215, 240, 215, "Light Violet"),
    25: (245, 134, 36, "Orange"),
    26: (202, 31, 123, "Magenta"),
    27: (159, 195, 65, "Bright Light Green"),
    28: (33, 55, 23, "Dark Green"),
    29: (160, 188, 172, "Light Bluish Gray"),
    70: (89, 47, 14, "Reddish Brown"),
    71: (175, 181, 199, "Light Bluish Gray"),
    72: (108, 110, 104, "Dark Bluish Gray"),
    73: (117, 142, 220, "Medium Blue"),
    74: (183, 212, 37, "Medium Lime"),
    78: (254, 186, 189, "Light Pink"),
    84: (170, 125, 85, "Medium Dark Flesh"),
    85: (89, 39, 115, "Dark Purple"),
    320: (120, 27, 33, "Dark Red"),
    378: (163, 193, 173, "Sand Green"),
    484: (179, 62, 0, "Dark Orange"),
}

_COLOR_IDS = list(LDRAW_COLORS.keys())
_COLOR_RGB = np.array([LDRAW_COLORS[i][:3] for i in _COLOR_IDS], dtype=np.float32)
_COLOR_TREE = KDTree(_COLOR_RGB)


def match_ldraw_color(rgb: Tuple[float, float, float]) -> Tuple[int, str]:
    v = np.array(rgb, dtype=np.float32)
    if v.max() <= 1.0:
        v *= 255.0
    _, idx = _COLOR_TREE.query(v)
    code = _COLOR_IDS[int(idx)]
    return code, LDRAW_COLORS[code][3]


@dataclass
class BrickPart:
    ldraw_id: str
    name: str
    width: int
    depth: int
    height: int
    is_slope: bool = False
    min_age: int = 4


PARTS_PRO = [
    BrickPart("3009.dat", "Brick 1x6", 6, 1, 3),
    BrickPart("3007.dat", "Brick 2x8", 8, 2, 3),
    BrickPart("3006.dat", "Brick 2x10", 10, 2, 3),
    BrickPart("3001.dat", "Brick 2x4", 4, 2, 3),
    BrickPart("3002.dat", "Brick 2x3", 3, 2, 3),
    BrickPart("3003.dat", "Brick 2x2", 2, 2, 3),
    BrickPart("3004.dat", "Brick 1x2", 2, 1, 3),
    BrickPart("3005.dat", "Brick 1x1", 1, 1, 3),
    BrickPart("3020.dat", "Plate 2x4", 4, 2, 1),
    BrickPart("3021.dat", "Plate 2x3", 3, 2, 1),
    BrickPart("3022.dat", "Plate 2x2", 2, 2, 1),
    BrickPart("3023.dat", "Plate 1x2", 2, 1, 1),
    BrickPart("3024.dat", "Plate 1x1", 1, 1, 1),
    BrickPart("3040.dat", "Slope 45 2x1", 1, 2, 3, is_slope=True),
    BrickPart("3039.dat", "Slope 45 2x2", 2, 2, 3, is_slope=True),
]

PARTS_KIDS = [
    BrickPart("3007.dat", "Brick 2x8", 8, 2, 3, min_age=7),
    BrickPart("3001.dat", "Brick 2x4", 4, 2, 3, min_age=4),
    BrickPart("3002.dat", "Brick 2x3", 3, 2, 3, min_age=4),
    BrickPart("3003.dat", "Brick 2x2", 2, 2, 3, min_age=4),
    BrickPart("3004.dat", "Brick 1x2", 2, 1, 3, min_age=7),
]

CAP_PLATES = [
    BrickPart("3020.dat", "Plate 2x4", 4, 2, 1),
    BrickPart("3021.dat", "Plate 2x3", 3, 2, 1),
    BrickPart("3022.dat", "Plate 2x2", 2, 2, 1),
    BrickPart("3710.dat", "Plate 1x4", 4, 1, 1),
    BrickPart("3623.dat", "Plate 1x3", 3, 1, 1),
    BrickPart("3023.dat", "Plate 1x2", 2, 1, 1),
    BrickPart("3024.dat", "Plate 1x1", 1, 1, 1),
]

CAP_PLATES_KIDS = [
    BrickPart("3020.dat", "Plate 2x4", 4, 2, 1),
    BrickPart("3021.dat", "Plate 2x3", 3, 2, 1),
    BrickPart("3022.dat", "Plate 2x2", 2, 2, 1),
    BrickPart("3023.dat", "Plate 1x2", 2, 1, 1),
]


@dataclass
class VoxelGrid:
    occupied: np.ndarray
    color_ids: np.ndarray
    pitch: float
    origin: np.ndarray

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.occupied.shape


@dataclass
class PlacedBrick:
    part: BrickPart
    x: int
    y: int
    z: int
    rotation: int
    color_code: int
    color_name: str


@dataclass
class ConversionResult:
    placements: List[PlacedBrick]
    grid_shape: Tuple[int, int, int]
    total_bricks: int
    brick_counts: Dict[str, int]
    stability_score: float
    color_summary: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def load_glb_scene(path: Path) -> trimesh.Scene:
    obj = trimesh.load(path.as_posix(), force="scene")
    if isinstance(obj, trimesh.Scene):
        return obj
    scene = trimesh.Scene()
    scene.add_geometry(obj)
    return scene


def get_meshes(scene: trimesh.Scene) -> List[trimesh.Trimesh]:
    meshes: List[trimesh.Trimesh] = []
    try:
        dumped = scene.dump()
    except Exception:
        dumped = []

    for geom in dumped:
        if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0:
            meshes.append(geom)

    if meshes:
        return meshes

    for _, geom in scene.geometry.items():
        if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0:
            meshes.append(geom)
    return meshes


def calculate_auto_pitch(scene: trimesh.Scene, target_studs: int) -> float:
    meshes = get_meshes(scene)
    if not meshes:
        return 1.0
    combined = trimesh.util.concatenate(meshes)
    size = combined.bounds[1] - combined.bounds[0]
    max_dim = float(np.max(size))
    return max_dim / target_studs if max_dim > 0 else 1.0


def extract_face_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Robustly extract colors from a trimesh object.
    Uses to_color() to resolve textures/vertex colors into per-face RGB.
    """
    try:
        # Convert any visual state (textured, vertex-colored, etc.) to face-based colors
        face_colors = mesh.visual.to_color().face_colors
        return face_colors[:, :3].astype(np.float32)
    except Exception:
        # Fallback to gray (71)
        return np.tile([160, 160, 160], (len(mesh.faces), 1)).astype(np.float32)


def voxelize_scene(
    scene: trimesh.Scene,
    pitch: float,
    solid: bool = True,
    use_mesh_color: bool = True,
    solid_color: int = 4,
) -> VoxelGrid:
    meshes = get_meshes(scene)
    if not meshes:
        raise ValueError("메시 없음")

    combined = trimesh.util.concatenate(meshes)
    origin = combined.bounds[0]

    voxelized = trimesh.voxel.creation.voxelize(combined, pitch=pitch)
    if solid:
        voxelized = voxelized.fill()
    if voxelized.points is None or len(voxelized.points) == 0:
        raise ValueError("복셀화 실패")

    occupied = voxelized.matrix.astype(bool)
    nx, ny, nz = occupied.shape

    filled_idx = np.array(voxelized.sparse_indices, dtype=int)
    centers = np.asarray(voxelized.points, dtype=np.float32)

    if use_mesh_color:
        face_colors = extract_face_colors(combined)
        face_centers = combined.triangles_center.astype(np.float32)
        face_tree = KDTree(face_centers)
        _, face_idx = face_tree.query(centers)
        rgb_values = face_colors[face_idx]
        color_ids = np.array([
            match_ldraw_color(tuple(rgb_values[i]))[0]
            for i in range(len(rgb_values))
        ], dtype=np.int16)
    else:
        color_ids = np.full((len(centers),), int(solid_color), dtype=np.int16)

    color_grid = np.full((nx, ny, nz), -1, dtype=np.int16)
    color_grid[filled_idx[:, 0], filled_idx[:, 1], filled_idx[:, 2]] = color_ids

    return VoxelGrid(
        occupied=occupied,
        color_ids=color_grid,
        pitch=pitch,
        origin=origin
    )


def smooth_colors(grid: VoxelGrid, passes: int = 1) -> None:
    if passes <= 0:
        return
    occ, cid = grid.occupied, grid.color_ids
    nx, ny, nz = occ.shape
    for _ in range(passes):
        new_cid = cid.copy()
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if not occ[x, y, z]:
                        continue
                    neigh = []
                    for dx, dy, dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
                        nx2, ny2, nz2 = x+dx, y+dy, z+dz
                        if 0 <= nx2 < nx and 0 <= ny2 < ny and 0 <= nz2 < nz:
                            if cid[nx2, ny2, nz2] >= 0:
                                neigh.append(int(cid[nx2, ny2, nz2]))
                    if neigh:
                        vals, counts = np.unique(neigh, return_counts=True)
                        new_cid[x, y, z] = int(vals[np.argmax(counts)])
        cid[:] = new_cid


def detect_symmetry(grid: VoxelGrid) -> Optional[Literal["x", "z"]]:
    occ = grid.occupied
    nx, ny, nz = occ.shape
    if nx < 4 and nz < 4:
        return None

    def sym_score_x():
        score = 0
        for x in range(nx // 2):
            mx = nx - 1 - x
            score += np.sum(occ[x] == occ[mx])
        return score

    def sym_score_z():
        score = 0
        for z in range(nz // 2):
            mz = nz - 1 - z
            score += np.sum(occ[:, :, z] == occ[:, :, mz])
        return score

    sx = sym_score_x()
    sz = sym_score_z()
    if max(sx, sz) == 0:
        return None
    return "x" if sx >= sz else "z"


def enforce_symmetry(grid: VoxelGrid, axis: Literal["x", "z"]) -> None:
    occ, cid = grid.occupied, grid.color_ids
    nx, ny, nz = occ.shape
    if axis == "x":
        for x in range(nx // 2):
            mx = nx - 1 - x
            combined = occ[x] | occ[mx]
            occ[x] = combined
            occ[mx] = combined
            for y in range(ny):
                for z in range(nz):
                    if combined[y, z]:
                        if cid[x, y, z] >= 0:
                            cid[mx, y, z] = cid[x, y, z]
                        elif cid[mx, y, z] >= 0:
                            cid[x, y, z] = cid[mx, y, z]
    else:
        for z in range(nz // 2):
            mz = nz - 1 - z
            combined = occ[:, :, z] | occ[:, :, mz]
            occ[:, :, z] = combined
            occ[:, :, mz] = combined
            for x in range(nx):
                for y in range(ny):
                    if combined[x, y]:
                        if cid[x, y, z] >= 0:
                            cid[x, y, mz] = cid[x, y, z]
                        elif cid[x, y, mz] >= 0:
                            cid[x, y, z] = cid[x, y, mz]


def greedy_pack_bricks(
    grid: VoxelGrid,
    parts: List[BrickPart],
    optimize_bonds: bool = True,
    support_ratio: float = 0.3,
    pre_used: Optional[np.ndarray] = None,
    allow_side_contact: bool = False
) -> List[PlacedBrick]:
    occ, cid = grid.occupied, grid.color_ids
    nx, ny, nz = occ.shape
    used = pre_used.copy() if pre_used is not None else np.zeros_like(occ, dtype=bool)
    placements: List[PlacedBrick] = []

    def dominant_color(x: int, y: int, z: int, w: int, d: int) -> Tuple[int, str]:
        block = cid[x:x+w, y, z:z+d]
        valid = block[block >= 0]
        if len(valid) == 0:
            return 71, "Light Bluish Gray"
        vals, counts = np.unique(valid, return_counts=True)
        code = int(vals[np.argmax(counts)])
        return code, LDRAW_COLORS.get(code, (0,0,0,"Unknown"))[3]

    def can_place(
        x: int,
        y: int,
        z: int,
        w: int,
        d: int,
        support_ratio: float = 0.3,
        allow_side_contact: bool = False
    ) -> bool:
        if x + w > nx or z + d > nz:
            return False
        region_occ = occ[x:x+w, y, z:z+d]
        region_used = used[x:x+w, y, z:z+d]
        if not region_occ.all() or region_used.any():
            return False
        if y > 0:
            below_used = used[x:x+w, y-1, z:z+d]
            below_mean = float(np.mean(below_used)) if below_used.size else 0.0
            if below_mean <= 0.0:
                return False
            if below_mean < support_ratio:
                if not allow_side_contact:
                    return False
                supported = used[:, y, :] & (used[:, y-1, :] if y > 0 else True)
                if x > 0 and supported[x-1, z:z+d].any():
                    return True
                if x + w < nx and supported[x+w, z:z+d].any():
                    return True
                if z > 0 and supported[x:x+w, z-1].any():
                    return True
                if z + d < nz and supported[x:x+w, z+d].any():
                    return True
                return False
        return True

    def bond_score(x: int, y: int, z: int, w: int, d: int) -> float:
        if y == 0:
            return 1.0
        below_used = used[x:x+w, y-1, z:z+d]
        overlap_ratio = np.mean(below_used)
        if 0.3 <= overlap_ratio <= 0.7:
            return 1.5
        elif overlap_ratio > 0:
            return 1.0
        return 0.5

    for y in range(ny):
        offset = 1 if (optimize_bonds and y % 2 == 1) else 0
        for z in range(nz):
            x_start = offset if offset < nx else 0
            for x in range(x_start, nx):
                if not occ[x, y, z] or used[x, y, z]:
                    continue

                placed = False
                best_placement = None
                best_score = -1

                for part in parts:
                    orientations = [(part.width, part.depth, 0)]
                    if part.width != part.depth:
                        orientations.append((part.depth, part.width, 90))

                    for w, d, rot in orientations:
                        if can_place(
                            x, y, z, w, d,
                            support_ratio=support_ratio,
                            allow_side_contact=allow_side_contact
                        ):
                            score = bond_score(x, y, z, w, d) * (w * d)
                            if score > best_score:
                                best_score = score
                                color_code, color_name = dominant_color(x, y, z, w, d)
                                best_placement = (part, x, y, z, rot, w, d, color_code, color_name)
                            placed = True
                            break
                    if placed and not optimize_bonds:
                        break

                if best_placement:
                    part, px, py, pz, rot, w, d, cc, cn = best_placement
                    used[px:px+w, py, pz:pz+d] = True
                    placements.append(PlacedBrick(
                        part=part,
                        x=px, y=py, z=pz,
                        rotation=rot,
                        color_code=cc,
                        color_name=cn
                    ))

    return placements


def _find_top_surfaces(grid: VoxelGrid, used: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    occ = grid.occupied
    nx, ny, nz = occ.shape
    y_top = np.full((nx, nz), -1, dtype=int)
    exposed = np.zeros((nx, nz), dtype=bool)
    for x in range(nx):
        for z in range(nz):
            ys = np.where(occ[x, :, z])[0]
            if ys.size == 0:
                continue
            yt = int(ys.max())
            y_top[x, z] = yt
            if yt == ny - 1 or not occ[x, yt + 1, z]:
                exposed[x, z] = True
    return y_top, exposed


def _greedy_2d_pack(
    mask: np.ndarray,
    color_grid: np.ndarray,
    parts: List[BrickPart]
) -> List[Tuple[BrickPart, int, int, int, int, str]]:
    nx, nz = mask.shape
    used = np.zeros_like(mask, dtype=bool)
    placements = []

    def can_place(x: int, z: int, w: int, d: int) -> bool:
        if x + w > nx or z + d > nz:
            return False
        region = mask[x:x+w, z:z+d]
        if not region.all():
            return False
        if used[x:x+w, z:z+d].any():
            return False
        return True

    def dominant_color(x: int, z: int, w: int, d: int) -> Tuple[int, str]:
        block = color_grid[x:x+w, z:z+d]
        valid = block[block >= 0]
        if len(valid) == 0:
            return 71, "Light Bluish Gray"
        vals, counts = np.unique(valid, return_counts=True)
        code = int(vals[np.argmax(counts)])
        return code, LDRAW_COLORS.get(code, (0,0,0,"Unknown"))[3]

    for z in range(nz):
        for x in range(nx):
            if not mask[x, z] or used[x, z]:
                continue

            placed = False
            for part in parts:
                orientations = [(part.width, part.depth, 0)]
                if part.width != part.depth:
                    orientations.append((part.depth, part.width, 90))

                for w, d, rot in orientations:
                    if can_place(x, z, w, d):
                        cc, cn = dominant_color(x, z, w, d)
                        used[x:x+w, z:z+d] = True
                        placements.append((part, x, z, rot, cc, cn))
                        placed = True
                        break
                if placed:
                    break

            if not placed:
                used[x, z] = True

    return placements


def cap_with_plates(
    grid: VoxelGrid,
    brick_placements: List[PlacedBrick],
    cap_mode: Literal["off", "top", "all"] = "all",
    mode: Literal["pro", "kids"] = "pro",
    min_patch_area: int = 2
) -> List[PlacedBrick]:
    if cap_mode == "off":
        return []

    occ, cid = grid.occupied, grid.color_ids
    nx, ny, nz = occ.shape
    used = np.zeros_like(occ, dtype=bool)
    for p in brick_placements:
        w = p.part.width if p.rotation == 0 else p.part.depth
        d = p.part.depth if p.rotation == 0 else p.part.width
        used[p.x:p.x+w, p.y, p.z:p.z+d] = True

    y_top, exposed = _find_top_surfaces(grid, used)

    if cap_mode == "top":
        valid_ys = y_top[exposed]
        if valid_ys.size == 0:
            return []
        target_y = int(valid_ys.max())
        layers = [target_y]
    else:
        layers = sorted(set(int(v) for v in y_top[exposed].ravel() if v >= 0))

    cap_parts = CAP_PLATES_KIDS if mode == "kids" else CAP_PLATES
    plate_placements: List[PlacedBrick] = []

    for yl in layers:
        layer_mask = exposed & (y_top == yl)
        if layer_mask.sum() < min_patch_area:
            continue

        color_grid = np.full((nx, nz), -1, dtype=np.int16)
        for x in range(nx):
            for z in range(nz):
                if layer_mask[x, z]:
                    color_grid[x, z] = cid[x, yl, z]

        packed = _greedy_2d_pack(layer_mask, color_grid, cap_parts)

        for part, x, z, rot, cc, cn in packed:
            plate_placements.append(PlacedBrick(
                part=part,
                x=x,
                y=yl,
                z=z,
                rotation=rot,
                color_code=cc,
                color_name=cn
            ))

    return plate_placements


def calculate_stability(placements: List[PlacedBrick], grid_shape: Tuple[int, int, int]) -> float:
    if not placements:
        return 0.0
    nx, ny, nz = grid_shape
    scores = []
    base_count = sum(1 for p in placements if p.y == 0)
    total = len(placements)
    support_ratio = base_count / total if total > 0 else 0
    scores.append(min(support_ratio * 2, 1.0) * 30)
    cx = sum(p.x for p in placements) / total
    cz = sum(p.z for p in placements) / total
    center_x, center_z = nx / 2, nz / 2
    dist = np.sqrt((cx - center_x)**2 + (cz - center_z)**2)
    max_dist = np.sqrt(center_x**2 + center_z**2)
    center_score = (1 - dist / max_dist) if max_dist > 0 else 1
    scores.append(center_score * 30)
    large_bricks = sum(1 for p in placements if p.part.width * p.part.depth >= 4)
    large_ratio = large_bricks / total if total > 0 else 0
    scores.append(large_ratio * 40)
    return min(sum(scores), 100.0)


def generate_ldr(
    placements: List[PlacedBrick],
    grid_shape: Tuple[int, int, int],
    title: str = "Model",
    mode: str = "pro",
    step_mode: Literal["none", "layer", "brick"] = "layer",
    cap_placements: Optional[List[PlacedBrick]] = None
) -> str:
    nx, ny, nz = grid_shape
    cx = (nx - 1) / 2.0
    cz = (nz - 1) / 2.0

    lines = [
        f"0 {title}",
        f"0 Name: {title}.ldr",
        "0 Author: Brick CoScientist v3",
        f"0 Mode: {mode}",
        f"0 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    placements_by_layer = {}
    for p in placements:
        placements_by_layer.setdefault(p.y, []).append(p)

    for layer in sorted(placements_by_layer.keys()):
        if step_mode != "none":
            lines.append(f"0 // Layer {layer}")

        for p in placements_by_layer[layer]:
            w = p.part.width if p.rotation == 0 else p.part.depth
            d = p.part.depth if p.rotation == 0 else p.part.width

            sx = (p.x + (w - 1) / 2.0) - cx
            sz = (p.z + (d - 1) / 2.0) - cz

            ldu_x = int(round(sx * LDU_PER_STUD))
            ldu_z = int(round(sz * LDU_PER_STUD))
            ldu_y = int(round(-p.y * LDU_BRICK_H))

            rot = {0: ROT_0, 90: ROT_90, 180: ROT_180, 270: ROT_270}.get(p.rotation, ROT_0)

            lines.append(f"1 {p.color_code} {ldu_x} {ldu_y} {ldu_z} {rot} {p.part.ldraw_id}")

            if step_mode == "brick":
                lines.append("0 STEP")

        if step_mode == "layer":
            lines.append("0 STEP")
            lines.append("")

    if cap_placements:
        lines.append("0 // Cap Plates (top finish)")
        for p in cap_placements:
            w = p.part.width if p.rotation == 0 else p.part.depth
            d = p.part.depth if p.rotation == 0 else p.part.width
            sx = (p.x + (w - 1) / 2.0) - cx
            sz = (p.z + (d - 1) / 2.0) - cz
            ldu_x = int(round(sx * LDU_PER_STUD))
            ldu_z = int(round(sz * LDU_PER_STUD))
            ldu_y = int(round(-p.y * LDU_BRICK_H - LDU_BRICK_H))
            rot = {0: ROT_0, 90: ROT_90, 180: ROT_180, 270: ROT_270}.get(p.rotation, ROT_0)
            lines.append(f"1 {p.color_code} {ldu_x} {ldu_y} {ldu_z} {rot} {p.part.ldraw_id}")

        if step_mode != "none":
            lines.append("0 STEP")

    return "\n".join(lines)


def convert_glb_to_ldr_v3_inline(
    input_path: str,
    output_path: str,
    mode: Literal["pro", "kids"] = "pro",
    target_studs: int = 20,
    symmetry: Literal["off", "auto", "x", "z"] = "auto",
    color_smooth: int = 1,
    optimize_bonds: bool = True,
    support_ratio: float = 0.3,
    solid: bool = True,
    bricks_only: bool = True,
    small_side_contact: bool = False,
    step_mode: Literal["none", "layer", "brick"] = "layer",
    cap_mode: Literal["off", "top", "all"] = "all",
    use_mesh_color: bool = True,
    solid_color: int = 4,
    smart_fix: bool = True,
    **kwargs: Any,
) -> ConversionResult:
    in_path = Path(input_path).resolve()
    out_path = Path(output_path).resolve()

    scene = load_glb_scene(in_path)
    pitch = calculate_auto_pitch(scene, max(1, int(target_studs)))
    grid = voxelize_scene(scene, pitch, solid=solid, use_mesh_color=use_mesh_color, solid_color=solid_color)
    smooth_colors(grid, passes=color_smooth)

    if symmetry != "off":
        if symmetry == "auto":
            axis = detect_symmetry(grid)
            if axis:
                enforce_symmetry(grid, axis)
        else:
            enforce_symmetry(grid, symmetry)

    # Smart Fix (Embedding floating parts)
    if smart_fix:
        nx, ny, nz = grid.occupied.shape
        vox_list = []
        for x, y, z in zip(*np.where(grid.occupied)):
            vox_list.append({"x": int(x), "y": int(y), "z": int(z), "color": int(grid.color_ids[x,y,z])})
        fixed_voxels = embed_floating_parts(vox_list)
        grid.occupied.fill(False)
        grid.color_ids.fill(-1)
        for v in fixed_voxels:
            x, y, z = v["x"], v["y"], v["z"]
            if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
                grid.occupied[x, y, z] = True
                grid.color_ids[x, y, z] = v["color"]

    parts = PARTS_KIDS if mode == "kids" else PARTS_PRO

    if bricks_only:
        brick_parts = [p for p in parts if p.height == 3]
        cap_mode = "off"

        large_parts = [p for p in brick_parts if p.width * p.depth >= 4]
        primary_support = max(support_ratio, 0.35)
        placements = greedy_pack_bricks(
            grid, large_parts, optimize_bonds,
            support_ratio=primary_support
        )

        small_parts = [p for p in brick_parts if p.width * p.depth < 4]
        used = np.zeros_like(grid.occupied, dtype=bool)
        for p in placements:
            w = p.part.width if p.rotation == 0 else p.part.depth
            d = p.part.depth if p.rotation == 0 else p.part.width
            used[p.x:p.x+w, p.y, p.z:p.z+d] = True

        detail_support = min(support_ratio, 0.2)
        detail = greedy_pack_bricks(
            grid, small_parts, optimize_bonds=False,
            support_ratio=detail_support,
            pre_used=used,
            allow_side_contact=small_side_contact
        )
        placements = placements + detail
    else:
        placements = greedy_pack_bricks(grid, parts, optimize_bonds, support_ratio=support_ratio)

    if not placements:
        raise ValueError("배치된 브릭 없음")

    cap_placements = cap_with_plates(grid, placements, cap_mode, mode)
    all_placements = placements + cap_placements

    brick_counts: Dict[str, int] = {}
    color_summary: Dict[str, int] = {}
    for p in all_placements:
        brick_counts[p.part.name] = brick_counts.get(p.part.name, 0) + 1
        color_summary[p.color_name] = color_summary.get(p.color_name, 0) + 1

    stability = calculate_stability(placements, grid.shape)

    title = in_path.stem
    ldr_content = generate_ldr(placements, grid.shape, title, mode, step_mode, cap_placements)
    out_path.write_text(ldr_content, encoding="utf-8")

    return ConversionResult(
        placements=all_placements,
        grid_shape=grid.shape,
        total_bricks=len(all_placements),
        brick_counts=brick_counts,
        stability_score=stability,
        color_summary=color_summary,
        warnings=[]
    )

# -----------------------------------------------------------------------------
# [핵심 물리 보정] 심지 박기 (Embed Inwards)
# -----------------------------------------------------------------------------
def embed_floating_parts(vox_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not vox_list:
        return []

    voxel_map = {(v["x"], v["y"], v["z"]): int(v["color"]) for v in vox_list}
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


def convert_glb_to_ldr(
    glb_path: str,
    out_ldr_path: str,
    *,
    budget: int = 100,
    target: int = 60,
    min_target: int = 5,
    shrink: float = 0.85,
    search_iters: int = 6,
    kind: str = "brick",
    solid: bool = True,
    use_mesh_color: bool = True,
    solid_color: int = 4,
    smart_fix: bool = True,
    step_order: str = "bottomup",
    callback: Optional[Callable[[str], None]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    
    step_mode = "none" if step_order == "none" else "layer"
    bricks_only = (kind != "plate")
    cap_mode = "off" if bricks_only else "all"

    target = max(1, int(target))
    min_target = max(1, int(min_target))
    budget_int = int(budget) if budget is not None else 0
    shrink_val = float(shrink) if 0.0 < float(shrink) < 1.0 else 0.85
    search_iters = max(1, int(search_iters))

    def _run_v3(t: int):
        res = convert_glb_to_ldr_v3_inline(
            input_path=glb_path,
            output_path=out_ldr_path,
            mode="pro",
            target_studs=max(1, int(t)),
            symmetry="auto",
            color_smooth=1,
            optimize_bonds=True,
            support_ratio=0.3,
            solid=bool(solid),
            bricks_only=bricks_only,
            small_side_contact=False,
            step_mode=step_mode,
            cap_mode=cap_mode,
            use_mesh_color=bool(use_mesh_color),
            solid_color=int(solid_color),
            smart_fix=smart_fix,
        )
        return res, int(res.total_bricks), int(t)

    result, parts, last_target = _run_v3(target)
    best_target = last_target
    best_parts = parts
    best_under = parts <= budget_int if budget_int > 0 else False

    if budget_int > 0 and parts > budget_int:
        hi_target = last_target
        t = last_target
        for _ in range(search_iters):
            t = int(max(min_target, round(t * shrink_val)))
            if t >= hi_target:
                t = hi_target - 1
            if t < min_target:
                t = min_target

            result, parts, last_target = _run_v3(t)
            if parts < best_parts:
                best_parts = parts
                best_target = t
            if parts <= budget_int:
                best_target = t
                best_parts = parts
                best_under = True
                break

            hi_target = t
            if t <= min_target:
                break

    return {"out": out_ldr_path, "parts": int(best_parts), "final_target": int(best_target)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glb", help="input .glb path")
    ap.add_argument("--out", default="out/result.ldr")
    ap.add_argument("--target", type=int, default=60)
    ap.add_argument("--budget", type=int, default=100)
    args = ap.parse_args()
    print(convert_glb_to_ldr(args.glb, args.out, target=args.target, budget=args.budget))

if __name__ == "__main__":
    main()
