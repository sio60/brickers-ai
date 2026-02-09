import os
import sys
import argparse
import numpy as np
import trimesh
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
    # Fallback if pylego3d is not in path
    sys.path.append(_curr_dir)
    from pylego3d.optimizer import optimize_bricks
    from pylego3d.write_ldr import write_ldr

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
    **kwargs: Any
) -> Tuple[int, List[Dict]]:
    """
    Internal helper: run one voxelization + optimization pass at a given target scale.
    Returns (brick_count, optimized_bricks).
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

    # Color sampling
    print(f"      [Step] Color Sampling...")
    c_start = time.time()
    centers = vg.points
    if use_mesh_color:
        if hasattr(mesh.visual, 'to_color'):
            temp_mesh = mesh.copy()
            temp_mesh.visual = temp_mesh.visual.to_color()
            _, _, face_indices = temp_mesh.nearest.on_surface(centers)
            colors_raw = temp_mesh.visual.face_colors[face_indices][:, :3]
        else:
            _, _, face_indices = mesh.nearest.on_surface(centers)
            colors_raw = mesh.visual.face_colors[face_indices][:, :3]
    else:
        colors_raw = np.tile([200, 200, 200], (len(centers), 1))
    c_end = time.time()
    print(f"      [Step] Color Sampling Done: {c_end - c_start:.2f}s")

    # Build points list for optimizer
    bricks_data = []
    LDRAW_RGB = {
        0: (33,33,33), 1: (0,85,191), 2: (0,133,43), 4: (196,0,38), 
        14: (245,205,47), 15:(255,255,255), 19: (173,169,142), 71: (160,165,169)
    }
    palette_ids = list(LDRAW_RGB.keys())
    palette_colors = np.array([LDRAW_RGB[i] for i in palette_ids])
    
    def get_color_id(rgb):
        dists = np.sum((palette_colors - rgb)**2, axis=1)
        return palette_ids[np.argmin(dists)]

    for i in range(len(indices)):
        c_id = solid_color if not use_mesh_color else get_color_id(colors_raw[i])
        bricks_data.append({
            "x": int(indices[i][0]),
            "y": int(indices[i][1]),
            "z": int(indices[i][2]),
            "color": c_id
        })

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
    **kwargs: Any
) -> Dict[str, Any]:
    """
    GLB to LDR conversion with budget-seeking loop.
    If the resulting brick count exceeds the budget, shrink target and retry.
    """
    # --- SSE log callback ---
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
            **kwargs
        )
        
        if parts_count < 0:
            print(f"[Engine] Iter {i+1} Result: VOXEL_THRESHOLD EXCEEDED (Skipping optimization)")
        else:
            final_optimized = optimized
            print(f"[Engine] Iter {i+1} Result: {parts_count} bricks (Budget: {budget})")
            
            if parts_count <= budget:
                print(f"[Engine] SUCCESS: Budget met! ({parts_count} <= {budget})")
                _log("brickify", f"딱 맞게 {parts_count}개로 쌓았어요!")
                break
        
        if i < search_iters - 1:
            curr_target *= shrink
            if curr_target < 5: # Don't go below 5 studs
                curr_target = 5
            print(f"[Engine] Budget EXCEEDED or Threshold Hit. Shrinking target to {curr_target:.1f}")
            _log("brickify", "브릭이 좀 많네요, 다시 고민해볼게요...")
        else:
            print(f"[Engine] WARNING: Failed to meet budget after {search_iters} iters.")

    # 4. Write LDR (Best attempt)
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

def convert_glb_to_ldr_v3_inline(*args, **kwargs):
    """Compatibility shim for newer code expecting v3 signature"""
    # Simply call the original stable version
    return convert_glb_to_ldr(*args, **kwargs)

def embed_voxels_downwards(grid):
    """Stub for compatibility"""
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("glb")
    parser.add_argument("--out", default="output.ldr")
    parser.add_argument("--target", type=int, default=60)
    parser.add_argument("--budget", type=int, default=100)
    args = parser.parse_args()
    
    convert_glb_to_ldr(args.glb, args.out, target=args.target, budget=args.budget)
