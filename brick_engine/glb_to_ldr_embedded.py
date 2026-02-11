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
def smooth_colors(
    vox: List[Dict[str, Any]], 
    passes: int = 1,
    protect_top: float = 0.0
) -> List[Dict[str, Any]]:
    """인접 복셀의 색상을 참조하여 노이즈 색상 제거 (protect_top 비율만큼 상단 보호)"""
    if passes <= 0 or not vox:
        return vox
    
    # 상단 보호 임계값 계산
    threshold_y = -float('inf')
    if protect_top > 0:
        max_y = max(v["y"] for v in vox)
        min_y = min(v["y"] for v in vox)
        threshold_y = max_y - (max_y - min_y) * protect_top

    voxel_map = {(v["x"], v["y"], v["z"]): v["color"] for v in vox}
    
    neighbors = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    
    for _ in range(passes):
        new_map = dict(voxel_map)
        for (x, y, z), color in voxel_map.items():
            # 상단 보존 영역이면 패스
            if y >= threshold_y:
                continue

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
# 내부를 비우고 껍데기만 남기기 (성벽 쌓듯이)
# =============================================================================
def make_hollow(
    vox: List[Dict[str, Any]],
    thickness: int = 2
) -> List[Dict[str, Any]]:
    """
    내부를 비우되 'thickness' 만큼의 두께를 가진 껍데기를 남긴다.
    thickness=1: 6-이웃 중 하나라도 비어있으면 남김 (매우 얇음, 1x1 발생 많음)
    thickness=2: 반경 1 이내에 빈 공간이 있으면 남김 (버거움 방지, 결합력 강화)
    """
    if not vox:
        return []

    voxel_map = {(v["x"], v["y"], v["z"]) for v in vox}
    
    # 껍데기 판별: 특정 반경(thickness-1) 내에 빈 공간이 하나라도 있으면 표면(껍데기)
    radius = thickness - 1
    surface_voxels = []
    
    for v in vox:
        x, y, z = v["x"], v["y"], v["z"]
        is_inner_core = True
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if (x+dx, y+dy, z+dz) not in voxel_map:
                        is_inner_core = False
                        break
                if not is_inner_core: break
            if not is_inner_core: break
        
        if not is_inner_core:
            surface_voxels.append(v)
            
    return surface_voxels


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
    hollow: bool = False,
    hollow_thickness: int = 2,
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
    # 하지만 Hollow 모드에서는 표면만 남기므로 최적화 부담이 적어 훨씬 더 높은 해상도를 허용함
    base_threshold = kwargs.get("max_new_voxels", 6000)
    voxel_threshold = base_threshold * 10 if hollow else base_threshold
    max_pitch = kwargs.get("max_pitch", 3.0)
    
    if len(indices) > voxel_threshold:
        if pitch < max_pitch:
            new_pitch = pitch + 0.1
            print(f"      [Warning] Voxels ({len(indices)}) > threshold ({voxel_threshold})")
            print(f"      [Retry] Lowering resolution: pitch {pitch} -> {new_pitch}")
            return _single_conversion(
                combined, out_ldr_path, target, kind, plates_per_voxel,
                interlock, max_area, solid_color, use_mesh_color, step_order, glb_path,
                smart_fix=smart_fix, color_smooth=color_smooth,
                pitch=new_pitch, log_fn=log_fn, hollow=hollow, 
                hollow_thickness=hollow_thickness, **kwargs
            )
        else:
            print(f"      [Error] Pitch at max ({max_pitch}), still {len(indices)} voxels > {voxel_threshold}")
            return -1, []

    # Color sampling (Extreme: 27-point grid with Feature Priority)
    print(f"      [Step] Color Sampling (Extreme 27-point)...")
    c_start = time.time()
    centers = vg.points
    indices = vg.sparse_indices
    
    # 상단 40% (얼굴) 보호 구역 계산
    max_y_idx = np.max(indices[:, 1])
    min_y_idx = np.min(indices[:, 1])
    head_threshold = max_y_idx - (max_y_idx - min_y_idx) * 0.4

    if use_mesh_color:
        if hasattr(mesh.visual, 'to_color'):
            mesh.visual = mesh.visual.to_color()
        v_tree = KDTree(mesh.vertices)
        
        # 27점 가중치 샘플링 (3x3x3 격자)
        # 중심부와 주요 축 방향에 더 높은 가중치를 부여함
        offsets = [-0.35, 0, 0.35]
        grid = [[dx * pitch, dy * pitch, dz * pitch] for dx in offsets for dy in offsets for dz in offsets]
        
        # 가중치 맵 생성 (중심: 3, 축: 2, 대각선: 1)
        weights = []
        for d in grid:
            dist_sq = sum(v**2 for v in d)
            if dist_sq == 0: weights.append(3.0) # 중심
            elif dist_sq <= (0.35 * pitch)**2 + 1e-6: weights.append(2.0) # 축방향
            else: weights.append(1.0) # 대각선
            
        FEATURE_COLORS = {0, 6, 8, 70, 72, 85, 320}
        
        final_colors = []
        for i, center in enumerate(centers):
            y_idx = indices[i, 1]
            is_head = y_idx >= head_threshold
            
            # 모든 샘플의 색상 ID와 해당 지점의 가중치 수집
            sample_weighted_counts = {}
            feature_weighted_counts = {}
            
            for d, w in zip(grid, weights):
                _, v_idx = v_tree.query(center + d)
                rgb = tuple(mesh.visual.vertex_colors[v_idx][:3])
                sid = match_ldraw_color(rgb)
                
                sample_weighted_counts[sid] = sample_weighted_counts.get(sid, 0.0) + w
                if sid in FEATURE_COLORS:
                    feature_weighted_counts[sid] = feature_weighted_counts.get(sid, 0.0) + w
            
            # 일반적인 최빈값 (가중치 적용)
            most_common = max(sample_weighted_counts, key=sample_weighted_counts.get)
            
            # [Feature Priority] 얼굴 영역에서는 특징 색상의 가중치 합이 
            # 일정 수준(전체의 약 15% 이상)을 넘으면 우선시함
            if is_head and feature_weighted_counts:
                best_feature = max(feature_weighted_counts, key=feature_weighted_counts.get)
                # 가중치 합이 5.0 이상이면 채택 (노이즈 방지턱 상향)
                if feature_weighted_counts[best_feature] >= 5.0:
                    most_common = best_feature
            
            final_colors.append(int(most_common))
        colors_final = np.array(final_colors)
    else:
        colors_final = np.full(len(centers), solid_color)
    c_end = time.time()
    print(f"      [Step] Color Sampling Done: {c_end - c_start:.2f}s")

    # Build voxel list
    bricks_data = []
    for i in range(len(indices)):
        bricks_data.append({
            "x": int(indices[i][0]),
            "y": int(indices[i][1]),
            "z": int(indices[i][2]),
            "color": int(colors_final[i])
        })

    # 색상 스무딩 (상단 40% 보호)
    if color_smooth > 0:
        print(f"      [Step] Smoothing colors ({color_smooth} passes, Head Protect: 40%)...")
        bricks_data = smooth_colors(bricks_data, passes=color_smooth, protect_top=0.4)
    
    # 부유 브릭 보정
    if smart_fix:
        print(f"      [Step] Embedding floating parts...")
        if log_fn:
            log_fn("brickify", "공중에 뜬 부분이 있으면 인접 브릭에 결합하고, 내부 심지를 박아 보강하고 있어요.")
        bricks_data = embed_floating_parts(bricks_data)

    # 내부 비우기 (Hollow) - 성벽 쌓기 모드
    if hollow:
        print(f"      [Step] Hollowing out internal voxels...")
        if log_fn:
            log_fn("brickify", "성벽을 쌓듯이 내부를 비우고 튼튼한 외벽만 남기고 있어요.")
        bricks_data = make_hollow(bricks_data, thickness=hollow_thickness)

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
    hollow: bool = True,
    hollow_thickness: int = 2,
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

    # 3. Budget-Seeking Loop
    curr_target = float(target)
    final_optimized = []
    
    for i in range(search_iters):
        print(f"\n[Engine] SEARCH ITERATION {i+1}/{search_iters}")
        print(f"[Engine] Current Target Studs: {int(curr_target)}")
        _log("brickify", f"가설을 바탕으로 브릭을 쌓아보는 중이에요. ({i+1}/{search_iters})")
        
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
            hollow=hollow,
            hollow_thickness=hollow_thickness,
            log_fn=_log,
            **kwargs
        )
        
        if parts_count < 0:
            print(f"[Engine] Iter {i+1} Result: VOXEL_THRESHOLD EXCEEDED")
        else:
            final_optimized = optimized
            print(f"[Engine] Iter {i+1} Result: {parts_count} bricks (Budget: {budget})")
            
            if parts_count <= budget:
                print(f"[Engine] SUCCESS: Budget met! ({parts_count} <= {budget})")
                _log("brickify", f"구조적으로 안정적이에요. 총 {parts_count}개의 브릭으로 구성됐어요.")
                break
        
        if i < search_iters - 1:
            curr_target *= shrink
            if curr_target < 5:
                curr_target = 5
            print(f"[Engine] Budget EXCEEDED. Shrinking target to {curr_target:.1f}")
            _log("brickify", "부품 수가 목표보다 많네요. 핵심 형태는 유지하면서 단순화하는 방향으로 다시 시도해 볼게요.")
        else:
            print(f"[Engine] WARNING: Failed to meet budget after {search_iters} iters.")

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
    parser.add_argument("--solid-color", type=int, default=4)
    parser.add_argument("--smart-fix", action="store_true", default=True)
    parser.add_argument("--color-smooth", type=int, default=1)
    parser.add_argument("--avoid-1x1", action="store_true", default=False)
    parser.add_argument("--no-hollow", action="store_false", dest="hollow", help="Generate solid model (fill interior)")
    parser.add_argument("--hollow-thickness", type=int, default=2)
    parser.set_defaults(hollow=True, avoid_1x1=False)
    args = parser.parse_args()
    
    convert_glb_to_ldr(
        args.glb, args.out, 
        target=args.target, 
        budget=args.budget,
        solid_color=args.solid_color,
        smart_fix=args.smart_fix,
        color_smooth=args.color_smooth,
        avoid_1x1=args.avoid_1x1,
        hollow=args.hollow,
        hollow_thickness=args.hollow_thickness
    )