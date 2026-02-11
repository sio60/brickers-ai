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
# 부유 브릭 결합 보정 (BFS 연결성 분석 + 심지 박기)
# =============================================================================
from collections import deque

_NEIGHBORS_6 = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]


def _find_stable_set(voxel_map: Dict[Tuple[int,int,int], int]) -> set:
    """BFS로 바닥(y=0)에서 연결된 모든 안정 복셀을 찾는다."""
    ground = {pos for pos in voxel_map if pos[1] == 0}
    if not ground:
        min_y = min(pos[1] for pos in voxel_map)
        ground = {pos for pos in voxel_map if pos[1] == min_y}

    stable = set(ground)
    queue = deque(ground)
    while queue:
        pos = queue.popleft()
        for dx, dy, dz in _NEIGHBORS_6:
            npos = (pos[0]+dx, pos[1]+dy, pos[2]+dz)
            if npos in voxel_map and npos not in stable:
                stable.add(npos)
                queue.append(npos)
    return stable


def _find_components(positions: set) -> List[set]:
    """6-연결 기준으로 연결 컴포넌트를 그룹화한다."""
    visited: set = set()
    components: List[set] = []
    for pos in positions:
        if pos in visited:
            continue
        comp: set = set()
        queue = deque([pos])
        while queue:
            p = queue.popleft()
            if p in visited:
                continue
            visited.add(p)
            comp.add(p)
            for dx, dy, dz in _NEIGHBORS_6:
                npos = (p[0]+dx, p[1]+dy, p[2]+dz)
                if npos in positions and npos not in visited:
                    queue.append(npos)
        components.append(comp)
    return components


def _bridge_component_to_stable(
    comp: set,
    stable: set,
    voxel_map: Dict[Tuple[int,int,int], int],
    max_dist: int,
) -> List[Tuple[int,int,int]]:
    """
    떠 있는 컴포넌트에서 안정 구조까지 BFS 최단 경로를 찾고,
    경로 위의 브릿지 복셀 목록을 반환한다.
    """
    # BFS: 컴포넌트 표면에서 바깥으로 확장
    parent: Dict[Tuple[int,int,int], Optional[Tuple[int,int,int]]] = {}
    depth: Dict[Tuple[int,int,int], int] = {}
    queue: deque = deque()

    # 컴포넌트의 모든 복셀을 시작점으로 (depth=0)
    for pos in comp:
        depth[pos] = 0
        parent[pos] = None

    # 컴포넌트 경계에서 1-hop 이웃 중 이미 stable인게 있으면 바로 연결
    for pos in comp:
        for dx, dy, dz in _NEIGHBORS_6:
            npos = (pos[0]+dx, pos[1]+dy, pos[2]+dz)
            if npos in stable:
                return []  # 이미 인접, 브릿지 불필요

    # 컴포넌트 경계에서 바깥으로 BFS 확장
    for pos in comp:
        for dx, dy, dz in _NEIGHBORS_6:
            npos = (pos[0]+dx, pos[1]+dy, pos[2]+dz)
            if npos not in comp and npos not in depth and npos[1] >= 0:
                depth[npos] = 1
                parent[npos] = pos
                queue.append(npos)

    target: Optional[Tuple[int,int,int]] = None
    while queue:
        pos = queue.popleft()
        d = depth[pos]

        # stable에 도달하면 종료
        if pos in stable:
            target = pos
            break

        if d >= max_dist:
            continue

        for dx, dy, dz in _NEIGHBORS_6:
            npos = (pos[0]+dx, pos[1]+dy, pos[2]+dz)
            if npos in stable:
                parent[npos] = pos
                target = npos
                break
            if npos in depth or npos in comp or npos[1] < 0:
                continue
            depth[npos] = d + 1
            parent[npos] = pos
            queue.append(npos)

        if target is not None:
            break

    if target is None:
        return []

    # 경로 역추적 — 컴포넌트/stable에 이미 속한 복셀은 제외
    bridge: List[Tuple[int,int,int]] = []
    p: Optional[Tuple[int,int,int]] = target
    while p is not None:
        if p not in comp and p not in stable and p not in voxel_map:
            bridge.append(p)
        p = parent[p]
    return bridge


def embed_floating_parts(
    vox: List[Dict[str, Any]],
    max_bridge_short: int = 2,
    max_bridge_long: int = 6,
) -> List[Dict[str, Any]]:
    """
    공중에 떠 있는 복셀을 안정 구조에 접합한다.

    전략 (2단계):
      1) 인접 결합 (Adjacent Merging) — 거리 ≤ max_bridge_short
         짧은 브릿지로 가까운 빈 칸을 채워서 연결
      2) 내부 심지 박기 (Internal Anchoring) — 거리 ≤ max_bridge_long
         더 먼 곳까지 BFS 경로를 탐색하여 연결
    """
    if not vox:
        return []

    voxel_map = {(v["x"], v["y"], v["z"]): int(v["color"]) for v in vox}

    for max_dist in (max_bridge_short, max_bridge_long):
        stable = _find_stable_set(voxel_map)
        floating = set(voxel_map.keys()) - stable

        if not floating:
            break

        components = _find_components(floating)
        # 큰 컴포넌트부터 처리 (중요한 부분 우선)
        components.sort(key=len, reverse=True)

        for comp in components:
            bridge = _bridge_component_to_stable(
                comp, stable, voxel_map, max_dist
            )
            if bridge:
                # 브릿지 색상: 인접한 기존 복셀의 최빈 색상 사용
                neighbor_colors = []
                for bpos in bridge:
                    for dx, dy, dz in _NEIGHBORS_6:
                        npos = (bpos[0]+dx, bpos[1]+dy, bpos[2]+dz)
                        if npos in voxel_map:
                            neighbor_colors.append(voxel_map[npos])
                if neighbor_colors:
                    vals, counts = np.unique(neighbor_colors, return_counts=True)
                    bridge_color = int(vals[np.argmax(counts)])
                else:
                    # fallback: 컴포넌트의 최빈 색
                    comp_colors = [voxel_map[p] for p in comp]
                    vals, counts = np.unique(comp_colors, return_counts=True)
                    bridge_color = int(vals[np.argmax(counts)])

                for bpos in bridge:
                    voxel_map[bpos] = bridge_color

                # 연결된 컴포넌트를 stable로 편입
                stable.update(comp)
                stable.update(bridge)

    # 최종 상태 로그
    final_stable = _find_stable_set(voxel_map)
    still_floating = set(voxel_map.keys()) - final_stable
    if still_floating:
        print(f"      [Warning] {len(still_floating)} voxels still floating after fix")
    else:
        print(f"      [OK] All voxels connected to ground")

    return [{"x": x, "y": y, "z": z, "color": c} for (x, y, z), c in voxel_map.items()]


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
    color_smooth: int = 1,
    avoid_1x1: bool = False,
    log_fn: Optional[Any] = None,
    smart_fix: bool = True,
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
                pitch=new_pitch, log_fn=log_fn, **kwargs
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
        if log_fn:
            log_fn("brickify", "공중에 뜬 부분이 있으면 인접 브릭에 결합하고, 내부 심지를 박아 보강하고 있어요.")
        bricks_data = embed_floating_parts(bricks_data)

    # Optimize (Greedy Packing)
    print(f"      [Step] Optimization (Greedy Packing) starting...")
    o_start = time.time()
    optimized = optimize_bricks(
        bricks_data,
        kind=kind,
        plates_per_voxel=plates_per_voxel,
        interlock=interlock,
        max_area=max_area,
        avoid_1x1=avoid_1x1
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
    avoid_1x1: bool = False,
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

    _log("brickify", "여러 설계 중에서 가장 균형 잡힌 안을 찾고 있어요.")

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

    # 3. Budget-Seeking Loop (Binary Search for optimal target)
    low_target = 5.0
    high_target = float(target)
    curr_target = high_target
    
    final_optimized = []
    best_count = 0
    
    # search_iters 만큼 반복하여 예산에 근접한 최적의 target을 찾음
    for i in range(search_iters):
        print(f"\n[Engine] SEARCH ITERATION {i+1}/{search_iters}")
        print(f"[Engine] Range: [{low_target:.1f}, {high_target:.1f}] -> Testing: {int(curr_target)}")
        _log("brickify", f"가장 정밀한 설계를 엔진이 계산하고 있어요. ({i+1}/{search_iters})")
        
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
            avoid_1x1=avoid_1x1,
            log_fn=_log,
            **kwargs
        )
        
        if parts_count < 0 or parts_count > budget:
            # 예산 초과: 범위를 아래로 좁힘
            if parts_count < 0:
                print(f"[Engine] Iter {i+1}: VOXEL_THRESHOLD EXCEEDED")
            else:
                print(f"[Engine] Iter {i+1}: {parts_count} bricks (EXCEEDED {budget})")
            
            high_target = curr_target
        else:
            # 예산 충족: 더 높은 품질을 시도해보기 위해 범위를 위로 좁힘
            print(f"[Engine] Iter {i+1}: {parts_count} bricks (OK, Budget: {budget})")
            final_optimized = optimized
            best_count = parts_count
            low_target = curr_target
            
            # 충분히 예산에 근접했다면 (예: 예산의 95% 이상) 조기 종료 가능
            if parts_count >= budget * 0.95:
                print(f"[Engine] Close enough to budget ({parts_count}/{budget}).")
                break
        
        # 다음 시도 target 계산 (이진 탐색)
        curr_target = (low_target + high_target) / 2.0
        
        # Target 변화폭이 너무 작으면 종료
        if abs(high_target - low_target) < 1.0:
            break

    if not final_optimized:
        print(f"[Engine] CRITICAL: Could not meet budget even at min target. Using last attempt.")
        # if we never hit a success, we might have no final_optimized. 
        # In that case, we fall back to whatever was the last 'successful' if any, 
        # but the loop logic above ensures final_optimized is only set on success.
    else:
        print(f"[Engine] Final Selected: {best_count} bricks at ~{int(low_target)} target studs.")

    # 4. Write LDR
    if not final_optimized:
        raise RuntimeError("Failed to generate any bricks")

    _log("brickify", "조립 순서를 재정렬해서 더 자연스럽게 만들고 있어요.")

    write_ldr(
        out_ldr_path,
        final_optimized,
        step_order=step_order,
        title=Path(glb_path).stem
    )

    _log("brickify", "완성되었어요. 브릭 설계를 마무리할게요.")

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