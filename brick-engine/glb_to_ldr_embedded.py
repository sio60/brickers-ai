from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional, Literal

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


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """RGB (0-255) -> CIE LAB 변환 (perceptual color space)"""
    # sRGB -> linear RGB
    rgb_norm = rgb / 255.0
    mask = rgb_norm > 0.04045
    linear = np.where(mask, ((rgb_norm + 0.055) / 1.055) ** 2.4, rgb_norm / 12.92)

    # linear RGB -> XYZ (D65)
    mat = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = linear @ mat.T

    # XYZ -> Lab
    ref = np.array([0.95047, 1.0, 1.08883])  # D65
    xyz_n = xyz / ref
    epsilon = 0.008856
    kappa = 903.3
    mask2 = xyz_n > epsilon
    f = np.where(mask2, np.cbrt(xyz_n), (kappa * xyz_n + 16.0) / 116.0)

    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


# LAB 색공간 기반 KDTree (perceptually uniform)
_COLOR_LAB = _rgb_to_lab(_COLOR_RGB)
_COLOR_TREE = KDTree(_COLOR_LAB)


def match_ldraw_color(rgb: Tuple[float, float, float]) -> Tuple[int, str]:
    v = np.array(rgb, dtype=np.float32).reshape(1, 3)
    if v.max() <= 1.0:
        v *= 255.0
    lab = _rgb_to_lab(v)
    _, idx = _COLOR_TREE.query(lab[0])
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
    mat = getattr(mesh.visual, "material", None)
    img = None
    if mat is not None:
        if hasattr(mat, "image") and mat.image is not None:
            img = mat.image
        elif hasattr(mat, "baseColorTexture") and mat.baseColorTexture is not None:
            bct = mat.baseColorTexture
            if hasattr(bct, "image") and bct.image is not None:
                img = bct.image
            else:
                img = bct

    if img is not None:
        uv = getattr(mesh.visual, "uv", None)
        if uv is not None:
            uv_face = uv[mesh.faces].mean(axis=1)
            try:
                rgba = trimesh.visual.color.uv_to_color(uv_face, img)
                return rgba[:, :3].astype(np.float32)
            except Exception:
                pass

    vc = getattr(mesh.visual, "vertex_colors", None)
    if vc is not None and len(vc) > 0:
        vc = np.asarray(vc, dtype=np.float32)
        if vc.shape[1] >= 3:
            rgb = vc[:, :3]
            if rgb.max() <= 1.0:
                rgb *= 255.0
            return rgb[mesh.faces].mean(axis=1)

    if mat and hasattr(mat, "baseColorFactor"):
        bc = mat.baseColorFactor
        if bc is not None and len(bc) >= 3:
            col = np.array([bc[0], bc[1], bc[2]], dtype=np.float32)
            if col.max() <= 1.0:
                col *= 255.0
            return np.tile(col, (len(mesh.faces), 1))

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


def embed_voxels_downwards(grid: VoxelGrid) -> None:
    """공중에 떠 있는 복셀(다리 등)을 바닥이나 인접 덩어리로 연결하여 절단 방지"""
    occ, cid = grid.occupied, grid.color_ids
    nx, ny, nz = occ.shape
    
    # 1. 접지(Grounded) 상태 확인 (바닥 y=0부터 시작)
    grounded = np.zeros_like(occ, dtype=bool)
    grounded[:, 0, :] = occ[:, 0, :]
    
    # 연결된 덩어리 전파 (최대 높이만큼 반복)
    for _ in range(ny):
        prev = grounded.copy()
        # 수직 연결 (아래가 접지면 위도 접지 가능)
        grounded[:, 1:, :] |= (occ[:, 1:, :] & grounded[:, :-1, :])
        # 수평 연결 (사이드가 접지면 옆도 접지 가능)
        grounded[1:, :, :] |= (occ[1:, :, :] & grounded[:-1, :, :])
        grounded[:-1, :, :] |= (occ[:-1, :, :] & grounded[1:, :, :])
        grounded[:, :, 1:] |= (occ[:, :, 1:] & grounded[:, :, :-1])
        grounded[:, :, :-1] |= (occ[:, :, :-1] & grounded[:, :, 1:])
        
        if np.array_equal(prev, grounded):
            break
            
    # 2. 떠 있는(Floating) 복셀 처리
    floating = occ & (~grounded)
    if not np.any(floating):
        return

    # 바닥 레이어부터 순차적으로 기둥(Pillar) 생성
    for y in range(ny):
        fx, fz = np.where(floating[:, y, :])
        for x, z in zip(fx, fz):
            # 현재 위치가 여전히 접지되지 않았다면
            if not grounded[x, y, z]:
                # 해당 복셀 색상 추출
                color = int(cid[x, y, z])
                if color < 0: color = 71
                
                # 아래로 내려가며 채움
                for dy in range(y - 1, -1, -1):
                    if occ[x, dy, z] and grounded[x, dy, z]:
                        break # 접지된 곳에 닿음
                    occ[x, dy, z] = True
                    cid[x, dy, z] = color
                    grounded[x, dy, z] = True
                    
                    # 주변으로도 전파 (덩어리 인식)
                    # 이 시점부터는 해당 Column이 접지된 것으로 간주됨


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
) -> ConversionResult:
    in_path = Path(input_path).resolve()
    out_path = Path(output_path).resolve()

    scene = load_glb_scene(in_path)
    pitch = calculate_auto_pitch(scene, max(1, int(target_studs)))
    grid = voxelize_scene(scene, pitch, solid=solid, use_mesh_color=use_mesh_color, solid_color=solid_color)
    smooth_colors(grid, passes=color_smooth)
    # [Fix] 떠 있는 부분(다리 등) 절단 방지 보정
    embed_voxels_downwards(grid)

    if symmetry != "off":
        if symmetry == "auto":
            axis = detect_symmetry(grid)
            if axis:
                enforce_symmetry(grid, axis)
        else:
            enforce_symmetry(grid, symmetry)

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
    _, _, triangle_ids = target_mesh.nearest.on_surface(points)
    face_colors = target_mesh.visual.face_colors[triangle_ids]
    return [rgb_to_ldraw_id(rgba[:3]) for rgba in face_colors]

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
# 5. [핵심 물리 보정] 심지 박기 (Embed Inwards)
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
# 6. STEP을 "레이어 단위"로 재작성
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
        current_colors = sample_voxel_colors(mesh, search_pts)
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
    
    # Voxelize with adaptive pitch
    pitch = kwargs.get("pitch", 1.0)
    print(f"      [Step] Voxelizing (Target: {target}, Pitch: {pitch})...")
    v_start = time.time()
    vg = mesh.voxelized(pitch=pitch)
    if kwargs.get("solid", True):
        vg = vg.fill()
    v_end = time.time()
    print(f"      [Step] Voxelization Done: {v_end - v_start:.2f}s")
    
    indices = vg.sparse_indices
    if indices is None or len(indices) == 0:
        return 0, []
        
    print(f"      [Step] Voxel count: {len(indices)}")
    # Kids 모드에서는 3만 개 이상의 보셀은 연산이 너무 느리므로 조기에 해상도를 낮추거나 타겟을 줄임
    voxel_threshold = kwargs.get("voxel_threshold", 30000)
    max_pitch = kwargs.get("max_pitch", 2.5)
    
    if len(indices) > voxel_threshold:
        if pitch < max_pitch:
            new_pitch = pitch + 0.5
            print(f"      [Warning] Voxels ({len(indices)}) > threshold ({voxel_threshold})")
            print(f"      [Retry] Lowering resolution: pitch {pitch} -> {new_pitch}")
            # Recursive retry with higher pitch (lower resolution)
            return _single_conversion(
                combined, out_ldr_path, target, kind, plates_per_voxel,
                interlock, max_area, solid_color, use_mesh_color, step_order, glb_path,
                pitch=new_pitch, **kwargs
            )
        else:
            print(f"      [Error] Pitch at max ({max_pitch}), still {len(indices)} voxels > {voxel_threshold}")
            return -1, []

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
# 8. API 엔트리 (호환용 키워드 방탄)
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
    small_side_contact: bool = False,

    # ?? ???? (v3??? ???)
    solid: bool = True,
    fill: bool = False,
    extend_catalog: bool = True,
    max_len: int = 8,

    span: int = 4,
    max_new_voxels: int = 12000,
    refine_iters: int = 8,
    ensure_connected: bool = True,
    min_embed: int = 2,
    erosion_iters: int = 1,
    fast_search: bool = True,
    support_ratio: float = 0.3, # [Added] To fix NameError in _run_v3
    **kwargs: Any,
) -> Dict[str, Any]:
    _ = (
        budget, min_target, shrink, search_iters, flipx180, flipy180, flipz180,
        plates_per_voxel, interlock, max_area, invert_y, smart_fix,
        fill, extend_catalog, max_len, span, max_new_voxels, refine_iters, ensure_connected,
        min_embed, erosion_iters, fast_search, support_ratio, kwargs,
    )

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
            mode=kwargs.get("mode", "kids"), # Use kids mode by default for kids_render
            target_studs=max(1, int(t)),
            symmetry="auto",
            color_smooth=1,
            optimize_bonds=True,
            support_ratio=max(0.05, float(support_ratio) - 0.1), # Lower support for detail
            solid=bool(solid),
            bricks_only=bricks_only,
            small_side_contact=bool(small_side_contact),
            step_mode=step_mode,
            cap_mode=cap_mode,
            use_mesh_color=bool(use_mesh_color),
            solid_color=int(solid_color),
        )
        return res, int(res.total_bricks), int(t)

    result, parts, last_target = _run_v3(target)
    best_target = last_target
    best_parts = parts
    best_under = parts <= budget_int if budget_int > 0 else False

    if budget_int > 0:
        # tighter budget convergence
        if shrink_val > 0.75:
            shrink_val = 0.75
        if search_iters < 10:
            search_iters = 10
        if parts > budget_int:
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

            if best_under and best_target < hi_target:
                lo = best_target
                hi = hi_target
                for _ in range(search_iters):
                    mid = (lo + hi) // 2
                    if mid == lo or mid == hi:
                        break
                    res_mid, parts_mid, last_target = _run_v3(mid)
                    if parts_mid <= budget_int:
                        result = res_mid
                        best_target = mid
                        best_parts = parts_mid
                        best_under = True
                        lo = mid
                    else:
                        hi = mid
        else:
            best_target = last_target
            best_parts = parts
            best_under = True
            hi = last_target
            for _ in range(search_iters):
                cand = int(round(hi / shrink_val))
                if cand <= hi:
                    cand = hi + 1
                res_cand, parts_cand, last_target = _run_v3(cand)
                if parts_cand <= budget_int:
                    result = res_cand
                    best_target = cand
                    best_parts = parts_cand
                    best_under = True
                    hi = cand
                else:
                    lo = best_target
                    hi = cand
                    for _ in range(search_iters):
                        mid = (lo + hi) // 2
                        if mid == lo or mid == hi:
                            break
                        res_mid, parts_mid, last_target = _run_v3(mid)
                        if parts_mid <= budget_int:
                            result = res_mid
                            best_target = mid
                            best_parts = parts_mid
                            best_under = True
                            lo = mid
                        else:
                            hi = mid
                    break

    if budget_int > 0 and best_parts > budget_int:
        # Aggressive fallback: shrink target based on budget ratio to enforce hard cap.
        ratio = max(0.01, float(budget_int) / float(max(1, best_parts)))
        t = int(round(best_target * (ratio ** 0.5)))
        t = max(min_target, min(best_target - 1, t))
        for _ in range(search_iters):
            if t < min_target:
                break
            result, parts, last_target = _run_v3(t)
            if parts < best_parts:
                best_parts = parts
                best_target = t
            if parts <= budget_int:
                best_parts = parts
                best_target = t
                break
            t_next = int(round(t * shrink_val))
            if t_next >= t:
                t_next = t - 1
            t = max(min_target, t_next)
        if best_parts > budget_int:
            print(f"[WARN] Budget not met: parts={best_parts} > budget={budget_int} (target={best_target})")

    if best_target != last_target:
        result, parts, last_target = _run_v3(best_target)
        best_parts = parts

    return {"out": out_ldr_path, "parts": int(best_parts), "final_target": int(best_target)}

# -----------------------------------------------------------------------------
# 9. CLI main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glb", help="input .glb path")
    ap.add_argument("--out", default="out/result.ldr")
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
    ap.add_argument("--solid", action="store_true")
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
