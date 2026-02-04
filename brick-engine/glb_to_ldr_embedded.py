from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional, Literal
from pathlib import Path
import sys

import numpy as np
import trimesh
from scipy.spatial import KDTree

# Add project root to path
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
# 1. Colors & Palette
# -----------------------------------------------------------------------------
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
    if v.max() <= 1.0: v *= 255.0
    _, idx = _COLOR_TREE.query(v)
    code = _COLOR_IDS[int(idx)]
    return code, LDRAW_COLORS[code][3]

def rgb_to_ldraw_id(rgb: np.ndarray) -> int:
    return match_ldraw_color(tuple(rgb))[0]

# -----------------------------------------------------------------------------
# 2. Part Definitions
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 3. Data Classes
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 4. Geometry & Voxelization
# -----------------------------------------------------------------------------
def load_glb_scene(path: Path) -> trimesh.Scene:
    obj = trimesh.load(path.as_posix(), force="scene")
    if isinstance(obj, trimesh.Scene): return obj
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
    if meshes: return meshes
    for _, geom in scene.geometry.items():
        if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0:
            meshes.append(geom)
    return meshes

def calculate_auto_pitch(scene: trimesh.Scene, target_studs: int) -> float:
    meshes = get_meshes(scene)
    if not meshes: return 1.0
    combined = trimesh.util.concatenate(meshes)
    size = combined.bounds[1] - combined.bounds[0]
    max_dim = float(np.max(size))
    return max_dim / target_studs if max_dim > 0 else 1.0

def extract_face_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    mat = getattr(mesh.visual, "material", None)
    img = None
    if mat is not None:
        if hasattr(mat, "image") and mat.image is not None: img = mat.image
        elif hasattr(mat, "baseColorTexture") and mat.baseColorTexture is not None:
            bct = mat.baseColorTexture
            img = getattr(bct, "image", bct)

    if img is not None:
        uv = getattr(mesh.visual, "uv", None)
        if uv is not None:
            uv_face = uv[mesh.faces].mean(axis=1)
            try:
                rgba = trimesh.visual.color.uv_to_color(uv_face, img)
                return rgba[:, :3].astype(np.float32)
            except Exception: pass

    vc = getattr(mesh.visual, "vertex_colors", None)
    if vc is not None and len(vc) > 0:
        vc = np.asarray(vc, dtype=np.float32)
        if vc.shape[1] >= 3:
            rgb = vc[:, :3]
            if rgb.max() <= 1.0: rgb *= 255.0
            return rgb[mesh.faces].mean(axis=1)

    if mat and hasattr(mat, "baseColorFactor"):
        bc = mat.baseColorFactor
        if bc is not None and len(bc) >= 3:
            col = np.array([bc[0], bc[1], bc[2]], dtype=np.float32)
            if col.max() <= 1.0: col *= 255.0
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
    if not meshes: raise ValueError("Mesh not found")
    combined = trimesh.util.concatenate(meshes)
    origin = combined.bounds[0]
    voxelized = trimesh.voxel.creation.voxelize(combined, pitch=pitch)
    if solid: voxelized = voxelized.fill()
    if voxelized.points is None or len(voxelized.points) == 0:
        raise ValueError("Voxelization failed")
    occupied = voxelized.matrix.astype(bool)
    filled_idx = np.array(voxelized.sparse_indices, dtype=int)
    centers = np.asarray(voxelized.points, dtype=np.float32)
    if use_mesh_color:
        face_colors = extract_face_colors(combined)
        face_tree = KDTree(combined.triangles_center.astype(np.float32))
        _, face_idx = face_tree.query(centers)
        rgb_values = face_colors[face_idx]
        color_ids = np.array([match_ldraw_color(tuple(rgb_values[i]))[0] for i in range(len(rgb_values))], dtype=np.int16)
    else:
        color_ids = np.full((len(centers),), int(solid_color), dtype=np.int16)
    nx, ny, nz = occupied.shape
    color_grid = np.full((nx, ny, nz), -1, dtype=np.int16)
    color_grid[filled_idx[:, 0], filled_idx[:, 1], filled_idx[:, 2]] = color_ids
    return VoxelGrid(occupied=occupied, color_ids=color_grid, pitch=pitch, origin=origin)

def smooth_colors(grid: VoxelGrid, passes: int = 1) -> None:
    if passes <= 0: return
    occ, cid = grid.occupied, grid.color_ids
    nx, ny, nz = occ.shape
    for _ in range(passes):
        new_cid = cid.copy()
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if not occ[x, y, z]: continue
                    neigh = []
                    for dx, dy, dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
                        nx2, ny2, nz2 = x+dx, y+dy, z+dz
                        if 0 <= nx2 < nx and 0 <= ny2 < ny and 0 <= nz2 < nz:
                            if cid[nx2, ny2, nz2] >= 0: neigh.append(int(cid[nx2, ny2, nz2]))
                    if neigh:
                        vals, counts = np.unique(neigh, return_counts=True)
                        new_cid[x, y, z] = int(vals[np.argmax(counts)])
        cid[:] = new_cid

# -----------------------------------------------------------------------------
# 5. Symmetry
# -----------------------------------------------------------------------------
def detect_symmetry(grid: VoxelGrid) -> Optional[Literal["x", "z"]]:
    occ = grid.occupied
    nx, ny, nz = occ.shape
    if nx < 4 and nz < 4: return None
    def score(axis):
        s = 0
        if axis == "x":
            for x in range(nx // 2): s += np.sum(occ[x] == occ[nx-1-x])
        else:
            for z in range(nz // 2): s += np.sum(occ[:,:,z] == occ[:,:,nz-1-z])
        return s
    sx, sz = score("x"), score("z")
    if max(sx, sz) == 0: return None
    return "x" if sx >= sz else "z"

def enforce_symmetry(grid: VoxelGrid, axis: Literal["x", "z"]) -> None:
    occ, cid = grid.occupied, grid.color_ids
    nx, ny, nz = occ.shape
    if axis == "x":
        for x in range(nx // 2):
            mx = nx-1-x; combined = occ[x] | occ[mx]; occ[x] = combined; occ[mx] = combined
            for y in range(ny):
                for z in range(nz):
                    if combined[y,z]:
                        if cid[x,y,z] >= 0: cid[mx,y,z] = cid[x,y,z]
                        elif cid[mx,y,z] >= 0: cid[x,y,z] = cid[mx,y,z]
    else:
        for z in range(nz // 2):
            mz = nz-1-z; combined = occ[:,:,z] | occ[:,:,mz]; occ[:,:,z] = combined; occ[:,:,mz] = combined
            for x in range(nx):
                for y in range(ny):
                    if combined[x,y]:
                        if cid[x,y,z] >= 0: cid[x,y,mz] = cid[x,y,z]
                        elif cid[x,y,mz] >= 0: cid[x,y,z] = cid[x,y,mz]

# -----------------------------------------------------------------------------
# 6. Physical Corrections (Floating Parts / Protrusions)
# -----------------------------------------------------------------------------
def embed_floating_parts(vox_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Connects floating voxels to the main body to ensure they are packed."""
    if not vox_list: return []
    voxel_map = {(v["x"], v["y"], v["z"]): int(v["color"]) for v in vox_list}
    new_colors = dict(voxel_map)
    neighbors_horizontal = [(1,0,0), (-1,0,0), (0,0,1), (0,0,-1)]
    neighbors_all = [(dx, dy, dz) for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1) if not (dx==dy==dz==0)]
    
    for _ in range(2):
        updates = {}
        for pos, color in list(new_colors.items()):
            x, y, z = pos
            if y == 0 or (x, y-1, z) in new_colors or (x, y+1, z) in new_colors: continue
            
            best_anchor = None
            for dx, dy, dz in neighbors_horizontal:
                npos = (x+dx, y+dy, z+dz)
                if npos in new_colors:
                    nx, ny, nz = npos
                    if ny == 0 or (nx, ny-1, nz) in new_colors or (nx, ny+1, nz) in new_colors:
                        best_anchor = npos; break
            if not best_anchor:
                for dx, dy, dz in neighbors_all:
                    npos = (x+dx, y+dy, z+dz)
                    if npos in new_colors: best_anchor = npos; break
            
            if best_anchor:
                updates[best_anchor] = color
                ax, ay, az = best_anchor
                if abs(x-ax)+abs(y-ay)+abs(z-az) > 1:
                    bridge = (ax, y, z)
                    if bridge not in new_colors: updates[bridge] = color
        if not updates: break
        new_colors.update(updates)
    return [{"x": x, "y": y, "z": z, "color": c} for (x, y, z), c in new_colors.items()]

# -----------------------------------------------------------------------------
# 7. Packing
# -----------------------------------------------------------------------------
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

    def dominant_color(x, y, z, w, d):
        block = cid[x:x+w, y, z:z+d]
        valid = block[block >= 0]
        if len(valid) == 0: return 71, "Light Bluish Gray"
        vals, counts = np.unique(valid, return_counts=True)
        code = int(vals[np.argmax(counts)])
        return code, LDRAW_COLORS.get(code, (0,0,0,"Unknown"))[3]

    def can_place(x, y, z, w, d, support_ratio, allow_side_contact):
        if x + w > nx or z + d > nz: return False
        if not occ[x:x+w, y, z:z+d].all() or used[x:x+w, y, z:z+d].any(): return False
        if y > 0:
            below = used[x:x+w, y-1, z:z+d]
            if np.mean(below) < support_ratio:
                if not allow_side_contact: return False
                # Side contact logic
                supported = used[:, y, :] & (used[:, y-1, :] if y > 0 else True)
                if x > 0 and supported[x-1, z:z+d].any(): return True
                if x+w < nx and supported[x+w, z:z+d].any(): return True
                if z > 0 and supported[x:x+w, z-1].any(): return True
                if z+d < nz and supported[x:x+w, z+d].any(): return True
                return False
        return True

    for y in range(ny):
        offset = 1 if (optimize_bonds and y % 2 == 1) else 0
        for z in range(nz):
            for x in range(offset if offset < nx else 0, nx):
                if not occ[x, y, z] or used[x, y, z]: continue
                best = None; best_score = -1
                for part in parts:
                    for w, d, rot in [(part.width, part.depth, 0), (part.depth, part.width, 90)] if part.width != part.depth else [(part.width, part.depth, 0)]:
                        if can_place(x, y, z, w, d, support_ratio, allow_side_contact):
                            score = (w * d) * (1.5 if y > 0 and 0.3 <= np.mean(used[x:x+w, y-1, z:z+d]) <= 0.7 else 1.0)
                            if score > best_score:
                                cc, cn = dominant_color(x, y, z, w, d)
                                best = (part, x, y, z, rot, w, d, cc, cn); best_score = score
                            if not optimize_bonds: break
                    if best and not optimize_bonds: break
                if best:
                    p, px, py, pz, rot, w, d, cc, cn = best
                    used[px:px+w, py, pz:pz+d] = True
                    placements.append(PlacedBrick(part=p, x=px, y=py, z=pz, rotation=rot, color_code=cc, color_name=cn))
    return placements

def cap_with_plates(grid: VoxelGrid, brick_placements: List[PlacedBrick], cap_mode: str, mode: str) -> List[PlacedBrick]:
    if cap_mode == "off": return []
    occ, cid = grid.occupied, grid.color_ids
    nx, ny, nz = occ.shape
    used = np.zeros_like(occ, dtype=bool)
    for p in brick_placements:
        w, d = (p.part.width, p.part.depth) if p.rotation == 0 else (p.part.depth, p.part.width)
        used[p.x:p.x+w, p.y, p.z:p.z+d] = True
    
    y_top = np.full((nx, nz), -1, dtype=int); exposed = np.zeros((nx, nz), dtype=bool)
    for x in range(nx):
        for z in range(nz):
            ys = np.where(occ[x, :, z])[0]
            if ys.size > 0:
                yt = int(ys.max()); y_top[x, z] = yt
                if yt == ny - 1 or not occ[x, yt+1, z]: exposed[x, z] = True
    
    layers = [int(y_top[exposed].max())] if cap_mode == "top" else sorted(set(int(v) for v in y_top[exposed].ravel() if v >= 0))
    cap_parts = CAP_PLATES_KIDS if mode == "kids" else CAP_PLATES
    plate_placements = []
    for yl in layers:
        layer_mask = exposed & (y_top == yl)
        if layer_mask.sum() < 2: continue
        l_used = np.zeros_like(layer_mask)
        for z in range(nz):
            for x in range(nx):
                if not layer_mask[x, z] or l_used[x, z]: continue
                for p in cap_parts:
                    for w, d, rot in [(p.width, p.depth, 0), (p.depth, p.width, 90)] if p.width != p.depth else [(p.width, p.depth, 0)]:
                        if x+w <= nx and z+d <= nz and layer_mask[x:x+w, z:z+d].all() and not l_used[x:x+w, z:z+d].any():
                            block = cid[x:x+w, yl, z:z+d]; valid = block[block >= 0]
                            cc = int(np.unique(valid)[0]) if valid.size > 0 else 71
                            cn = LDRAW_COLORS.get(cc, (0,0,0,"Unknown"))[3]
                            l_used[x:x+w, z:z+d] = True
                            plate_placements.append(PlacedBrick(part=p, x=x, y=yl, z=z, rotation=rot, color_code=cc, color_name=cn))
                            break
                    if l_used[x, z]: break
    return plate_placements

# -----------------------------------------------------------------------------
# 8. Output & Stability
# -----------------------------------------------------------------------------
def calculate_stability(placements: List[PlacedBrick], shape: Tuple[int, int, int]) -> float:
    if not placements: return 0.0
    total = len(placements)
    base = sum(1 for p in placements if p.y == 0) / total
    large = sum(1 for p in placements if p.part.width * p.part.depth >= 4) / total
    return min((base * 30 + large * 70) * 1.5, 100.0)

def generate_ldr(placements, shape, title, mode, step_mode, cap_placements=None) -> str:
    nx, ny, nz = shape
    cx, cz = (nx-1)/2.0, (nz-1)/2.0
    lines = [f"0 {title}", f"0 Name: {title}.ldr", "0 Author: BrickEngine", f"0 Mode: {mode}", ""]
    p_by_y = {}; all_p = placements + (cap_placements or [])
    for p in all_p: p_by_y.setdefault(p.y, []).append(p)
    LDU_S, LDU_H = 20, 24
    for y in sorted(p_by_y.keys()):
        for p in p_by_y[y]:
            w, d = (p.part.width, p.part.depth) if p.rotation == 0 else (p.part.depth, p.part.width)
            lx, lz = int(round(((p.x + (w-1)/2.0) - cx) * LDU_S)), int(round(((p.z + (d-1)/2.0) - cz) * LDU_S))
            ly = int(round(-p.y * LDU_H)) if p.part.height == 3 else int(round(-p.y * LDU_H - 16)) # offset for plates on top
            rot = {0: "1 0 0 0 1 0 0 0 1", 90: "0 0 1 0 1 0 -1 0 0"}.get(p.rotation, "1 0 0 0 1 0 0 0 1")
            lines.append(f"1 {p.color_code} {lx} {ly} {lz} {rot} {p.part.ldraw_id}")
            if step_mode == "brick": lines.append("0 STEP")
        if step_mode == "layer": lines.append("0 STEP")
    return "\n".join(lines)

# -----------------------------------------------------------------------------
# 9. Unified API Entry
# -----------------------------------------------------------------------------
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
    smart_fix: bool = True, # ✅ Connect the overhang fix
    **kwargs: Any,
) -> ConversionResult:
    in_path = Path(input_path).resolve()
    out_path = Path(output_path).resolve()
    scene = load_glb_scene(in_path)
    
    # 1. Scaling (Well-working logic)
    pitch = calculate_auto_pitch(scene, max(1, int(target_studs)))
    grid = voxelize_scene(scene, pitch, solid=solid, use_mesh_color=use_mesh_color, solid_color=solid_color)
    smooth_colors(grid, passes=color_smooth)
    if symmetry != "off":
        axis = detect_symmetry(grid) if symmetry == "auto" else symmetry
        if axis: enforce_symmetry(grid, axis)

    # 2. Overhang Fix (Ensures turtle head appears)
    if smart_fix:
        nx, ny, nz = grid.occupied.shape
        voxels = []
        for x, y, z in zip(*np.where(grid.occupied)):
            voxels.append({"x": int(x), "y": int(y), "z": int(z), "color": int(grid.color_ids[x,y,z])})
        fixed_voxels = embed_floating_parts(voxels)
        # Update grid with fixed voxels
        grid.occupied.fill(False); grid.color_ids.fill(-1)
        for v in fixed_voxels:
            x, y, z = v["x"], v["y"], v["z"]
            if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
                grid.occupied[x, y, z] = True; grid.color_ids[x, y, z] = v["color"]

    # 3. Packing
    parts = PARTS_KIDS if mode == "kids" else PARTS_PRO
    if bricks_only:
        parts = [p for p in parts if p.height == 3]
        cap_mode = "off"

    placements = greedy_pack_bricks(grid, parts, optimize_bonds, support_ratio, allow_side_contact=small_side_contact)
    cap_placements = cap_with_plates(grid, placements, cap_mode, mode)
    all_placements = placements + cap_placements
    
    brick_counts = {}
    for p in all_placements: brick_counts[p.part.name] = brick_counts.get(p.part.name, 0)+1
    
    ldr_content = generate_ldr(placements, grid.shape, in_path.stem, mode, step_mode, cap_placements)
    out_path.write_text(ldr_content, encoding="utf-8")

    return ConversionResult(
        placements=all_placements, grid_shape=grid.shape, total_bricks=len(all_placements),
        brick_counts=brick_counts, stability_score=calculate_stability(placements, grid.shape)
    )

def convert_glb_to_ldr(
    glb_path: str, out_ldr_path: str, *,
    budget: int = 100, target: int = 60, min_target: int = 5,
    shrink: float = 0.85, search_iters: int = 6,
    smart_fix: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    # Budget finding loop
    current_target = max(1, int(target))
    min_t = max(1, int(min_target))
    
    best_res, best_parts, best_t = None, 999999, current_target
    
    for _ in range(search_iters):
        res = convert_glb_to_ldr_v3_inline(
            glb_path, out_ldr_path, target_studs=current_target, smart_fix=smart_fix, **kwargs
        )
        if res.total_bricks <= budget:
            best_res, best_parts, best_t = res, res.total_bricks, current_target
            break
        best_res, best_parts, best_t = res, res.total_bricks, current_target
        current_target = int(max(min_t, round(current_target * shrink)))
        if current_target >= best_t: current_target -= 1
        if current_target < min_t: break
    
    return {"out": out_ldr_path, "parts": int(best_parts), "final_target": int(best_t)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glb"); ap.add_argument("--out", default="out/result.ldr")
    ap.add_argument("--target", type=int, default=60); ap.add_argument("--budget", type=int, default=100)
    args = ap.parse_args()
    print(convert_glb_to_ldr(args.glb, args.out, target=args.target, budget=args.budget))

if __name__ == "__main__": main()
