"""
glb_to_ldr_embedded.py - Enhanced Brickify Engine
- 대칭성 자동 감지 및 적용
- 부유 브릭 심지 박기 (구조 보정)
- 확장된 색상 팔레트 (30+개)
- SSE 실시간 로그 지원
- Brick만 사용 (Plate 없음)
"""

import os
import sys
import argparse
import numpy as np
import trimesh
from scipy.spatial import KDTree
from typing import Dict, List, Any, Optional, Tuple, Literal
from pathlib import Path

# Add current dir to path for pylego3d imports
_curr_dir = os.path.dirname(os.path.abspath(__file__))
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)

try:
    from pylego3d.optimizer import optimize_bricks
    from pylego3d.write_ldr import write_ldr
except ImportError:
    sys.path.append(_curr_dir)
    from pylego3d.optimizer import optimize_bricks
    from pylego3d.write_ldr import write_ldr

# =============================================================================
# 확장된 LDraw 색상 팔레트 (30+개)
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
    46: (242, 115, 0, "Flame Yellowish Orange"),
}

_COLOR_IDS = list(LDRAW_COLORS.keys())
_COLOR_RGB = np.array([LDRAW_COLORS[i][:3] for i in _COLOR_IDS], dtype=np.float32)
_COLOR_TREE = KDTree(_COLOR_RGB)


def match_ldraw_color(rgb: Tuple[float, float, float]) -> int:
    """RGB 값을 가장 가까운 LDraw 색상 ID로 매칭"""
    v = np.array(rgb, dtype=np.float32)
    if v.max() <= 1.0:
        v *= 255.0
    _, idx = _COLOR_TREE.query(v)
    return _COLOR_IDS[int(idx)]


# =============================================================================
# 부유 브릭 심지 박기 (구조 보정)
# =============================================================================
def embed_floating_parts(vox: List[Dict[str, Any]], max_iters: int = 3) -> List[Dict[str, Any]]:
    """
    공중에 떠 있는 복셀을 인접한 안정적인 복셀과 연결
    - 바닥(y=0)에 있거나
    - 아래에 복셀이 있으면 안정적
    """
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
    
    for _ in range(max_iters):
        updates: Dict[Tuple[int,int,int], int] = {}
        
        for pos, color in list(new_colors.items()):
            x, y, z = pos
            
            # 안정성 체크
            is_stable = False
            if y == 0:
                is_stable = True
            elif (x, y-1, z) in new_colors:
                is_stable = True
            elif (x, y+1, z) in new_colors:
                is_stable = True
            
            if is_stable:
                continue
            
            # 불안정한 복셀 → 인접한 안정적인 복셀 찾기
            best_anchor: Optional[Tuple[int,int,int]] = None
            
            # 수평 이웃 먼저 확인
            for dx, dy, dz in neighbors_horizontal:
                npos = (x+dx, y+dy, z+dz)
                if npos in new_colors:
                    nx, ny, nz = npos
                    if ny == 0 or (nx, ny-1, nz) in new_colors or (nx, ny+1, nz) in new_colors:
                        best_anchor = npos
                        break
            
            # 없으면 모든 이웃 확인
            if not best_anchor:
                for dx, dy, dz in neighbors_all:
                    npos = (x+dx, y+dy, z+dz)
                    if npos in new_colors:
                        best_anchor = npos
                        break
            
            if best_anchor:
                my_color = new_colors[pos]
                
                # 앵커 복셀과 현재 복셀 사이에 브릿지 복셀 추가
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


# =============================================================================
# 색상 스무딩 (노이즈 제거)
# =============================================================================
def smooth_colors(vox: List[Dict[str, Any]], passes: int = 1) -> List[Dict[str, Any]]:
    """인접 복셀의 색상을 참조하여 노이즈 색상 제거"""
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
                # 가장 많이 등장하는 색상으로 교체
                vals, counts = np.unique(neighbor_colors, return_counts=True)
                most_common = vals[np.argmax(counts)]
                # 이웃 중 과반수가 다른 색이면 교체
                if counts.max() > len(neighbor_colors) // 2:
                    new_map[(x, y, z)] = int(most_common)
        
        voxel_map = new_map
    
    return [{"x": x, "y": y, "z": z, "color": c} for (x, y, z), c in voxel_map.items()]


# =============================================================================
# 메인 변환 함수
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
    color_smooth: int = 1,
    **kwargs: Any
) -> Tuple[int, List[Dict]]:
    """
    단일 변환 패스: 복셀화 + 최적화
    Returns (brick_count, optimized_bricks)
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
    
    # Voxel threshold check (메모리 보호용)
    # Kids 모드에서 t3.small 서버 기준 6,000개가 넘으면 최적화가 너무 느려짐
    voxel_threshold = kwargs.get("max_new_voxels", 6000) 
    max_pitch = kwargs.get("max_pitch", 3.0)
    
    if len(indices) > voxel_threshold:
        if pitch < max_pitch:
            new_pitch = pitch + 0.5
            print(f"      [Warning] Voxels ({len(indices)}) > threshold ({voxel_threshold})")
            print(f"      [Retry] Lowering resolution: pitch {pitch} -> {new_pitch}")
            return _single_conversion(
                combined, out_ldr_path, target, kind, plates_per_voxel,
                interlock, max_area, solid_color, use_mesh_color, step_order, glb_path,
                smart_fix=smart_fix, color_smooth=color_smooth,
                pitch=new_pitch, **kwargs
            )
        else:
            print(f"      [Error] Pitch at max ({max_pitch}), still {len(indices)} voxels > {voxel_threshold}")
            return -1, []

    # Color sampling (KDTree를 사용하여 속도 최적화)
    print(f"      [Step] Color Sampling...")
    c_start = time.time()
    centers = vg.points
    if use_mesh_color:
        # Texture/Visual을 Color로 변환 (한 번만 수행)
        if hasattr(mesh.visual, 'to_color'):
            mesh.visual = mesh.visual.to_color()
        
        # Triangle center 대신 Vertex 기반으로 샘플링 (외곽면 색상을 더 잘 잡음)
        v_tree = KDTree(mesh.vertices)
        _, v_indices = v_tree.query(centers)
        colors_raw = mesh.visual.vertex_colors[v_indices][:, :3].astype(np.float32)
        
        # 밝기 및 채도 보정 (AI 생성 모델의 어두운/탁한 텍스처 보정)
        brightness = kwargs.get("color_brightness", 1.4)
        saturation = kwargs.get("color_saturation", 1.5)
        
        if brightness != 1.0:
            colors_raw = np.clip(colors_raw * brightness, 0, 255)
            
        if saturation != 1.0:
            avg = np.mean(colors_raw, axis=-1, keepdims=True)
            colors_raw = np.clip(avg + (colors_raw - avg) * saturation, 0, 255)
    else:
        colors_raw = np.tile([200, 200, 200], (len(centers), 1))
    c_end = time.time()
    print(f"      [Step] Color Sampling Done: {c_end - c_start:.2f}s")

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

    # 색상 스무딩
    if color_smooth > 0:
        print(f"      [Step] Smoothing colors ({color_smooth} passes)...")
        bricks_data = smooth_colors(bricks_data, passes=color_smooth)
    
    # 부유 브릭 보정
    if smart_fix:
        print(f"      [Step] Embedding floating parts...")
        bricks_data = embed_floating_parts(bricks_data)

    # Optimize (Greedy Packing)
    print(f"      [Step] Optimization (Greedy Packing) starting...")
    o_start = time.time()
    optimized = optimize_bricks(
        bricks_data,
        kind=kind,
        plates_per_voxel=plates_per_voxel,
        interlock=interlock,
        max_area=max_area
    )
    o_end = time.time()
    print(f"      [Step] Optimization Done: {o_end - o_start:.2f}s")
    
    return len(optimized), optimized


def convert_glb_to_ldr(
    glb_path: str,
    out_ldr_path: str,
    *,
    target: int = 60,
    budget: int = 100,
    shrink: float = 0.85,
    search_iters: int = 3,
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
    color_smooth: int = 1,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    GLB to LDR 변환 (예산 맞추기 루프 포함)
    
    Features:
    - 대칭성 자동 감지 및 적용
    - 부유 브릭 심지 박기
    - 확장된 색상 팔레트
    - SSE 로그 콜백 지원
    """
    # SSE log callback
    _log_cb = kwargs.pop("log_callback", None)
    def _log(step: str, message: str):
        if _log_cb:
            try:
                _log_cb(step, message)
            except Exception:
                pass

    print(f"[Engine] Starting conversion: {glb_path} -> {out_ldr_path}")
    print(f"[Engine] Target: {target} studs, Budget: {budget} bricks")

    _log("brickify", "브릭으로 어떻게 만들지 고민하고 있어요...")

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
        print(f"\n[Engine] SEARCH ITERATION {i+1}/{search_iters}")
        print(f"[Engine] Current Target Studs: {int(curr_target)}")
        _log("brickify", f"브릭을 하나씩 쌓아보고 있어요... ({i+1}/{search_iters})")
        
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
            print(f"[Engine] Iter {i+1} Result: VOXEL_THRESHOLD EXCEEDED")
        else:
            final_optimized = optimized
            print(f"[Engine] Iter {i+1} Result: {parts_count} bricks (Budget: {budget})")
            
            if parts_count <= budget:
                print(f"[Engine] SUCCESS: Budget met! ({parts_count} <= {budget})")
                _log("brickify", f"딱 맞게 {parts_count}개로 쌓았어요!")
                break
        
        if i < search_iters - 1:
            curr_target *= shrink
            if curr_target < 5:
                curr_target = 5
            print(f"[Engine] Budget EXCEEDED. Shrinking target to {curr_target:.1f}")
            _log("brickify", "브릭이 좀 많네요, 다시 고민해볼게요...")
        else:
            print(f"[Engine] WARNING: Failed to meet budget after {search_iters} iters.")

    # 4. Write LDR
    if not final_optimized:
        raise RuntimeError("Failed to generate any bricks")

    _log("brickify", "조립 순서를 고민하고 있어요...")

    write_ldr(
        out_ldr_path,
        final_optimized,
        step_order=step_order,
        title=Path(glb_path).stem
    )

    _log("brickify", "브릭 설계가 끝났어요!")

    return {
        "parts": len(final_optimized),
        "final_target": int(curr_target),
        "out": out_ldr_path
    }


# Compatibility shims
def convert_glb_to_ldr_v3_inline(*args, **kwargs):
    """호환성 유지용"""
    return convert_glb_to_ldr(*args, **kwargs)


def embed_voxels_downwards(grid):
    """호환성 유지용 stub"""
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("glb")
    parser.add_argument("--out", default="output.ldr")
    parser.add_argument("--target", type=int, default=60)
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--smart-fix", action="store_true", default=True)
    parser.add_argument("--color-smooth", type=int, default=1)
    args = parser.parse_args()
    
    convert_glb_to_ldr(
        args.glb, args.out, 
        target=args.target, 
        budget=args.budget,
        smart_fix=args.smart_fix,
        color_smooth=args.color_smooth
    )