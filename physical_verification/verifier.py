# ============================================================================
# 물리적 구조 검증 모듈
# 이 파일은 레고 브릭 구조의 물리적 건전성을 검증하는 핵심 로직을 담당합니다.
# 브릭 간의 물리적 연결 그래프를 구축하고, 부동(floating) 브릭, 구조적 안정성,
# 연결 강도(stud-tube), 오버행(overhang) 및 브릭 간 충돌 여부를 종합적으로
# 검사합니다. 모든 검증 결과는 `VerificationResult` 객체에 집계됩니다.
# ============================================================================
import networkx as nx
import numpy as np
from scipy.spatial import ConvexHull
from typing import List, Dict, Set
try:
    from .models import Brick, BrickPlan, VerificationResult, Evidence
    from .part_library import get_part_dims, get_part_description, get_part_geometry
except ImportError:
    from models import Brick, BrickPlan, VerificationResult, Evidence
    from part_library import get_part_dims, get_part_description, get_part_geometry

class PhysicalVerifier:
    def __init__(self, plan: BrickPlan):
        self.plan = plan
        self.graph = nx.Graph()
        self._build_graph()

    def _build_graph(self):
        """브릭을 노드로, 물리적 접촉을 에지로 하는 연결 그래프를 구축합니다."""
        bricks = self.plan.get_all_bricks()
        for b in bricks:
            self.graph.add_node(b.id, brick=b)
            
        # 연결성 검사 (필요 시 나중에 공간 해시로 최적화 가능)
        # 브릭을 축 정렬 상자(AABB)로 가정하고 검사합니다.
        for i, b1 in enumerate(bricks):
            for b2 in bricks[i+1:]:
                if self._are_connected(b1, b2):
                    self.graph.add_edge(b1.id, b2.id)

    def _are_connected(self, b1: Brick, b2: Brick) -> bool:
        """두 브릭이 서로 닿아 있는지 확인합니다."""
        TOL = 0.05

        # 0. 교차 검사 (부품이 서로 내부에 있는 경우, 예: 림 위의 타이어)
        # 부피를 공유하는지 확인합니다.
        intersect_x = max(0, min(b1.x + b1.width, b2.x + b2.width) - max(b1.x, b2.x))
        intersect_y = max(0, min(b1.y + b1.depth, b2.y + b2.depth) - max(b1.y, b2.y))
        intersect_z = max(0, min(b1.z + b1.height, b2.z + b2.height) - max(b1.z, b2.z))
        
        # 유의미한 부피 교차가 있다면 연결된 것으로 간주합니다.
        if intersect_x > TOL and intersect_y > TOL and intersect_z > TOL:
             return True

        # 1. 수직 접촉 (Z축 적층)
        vertical_touch = (abs((b1.z + b1.height) - b2.z) < TOL) or (abs((b2.z + b2.height) - b1.z) < TOL)
        if vertical_touch:
             if self._compute_overlap_area(b1, b2) > 0:
                 return True

        # 2. 측면 접촉 (수평 연결)
        # 먼저 Z축 겹침을 확인합니다.
        z_overlap_start = max(b1.z, b2.z)
        z_overlap_end = min(b1.z + b1.height, b2.z + b2.height)
        z_overlap = z_overlap_end - z_overlap_start
        
        if z_overlap > TOL: # 유의미한 Z축 겹침 (> 0.05)
             # X축 접촉 확인 (좌/우 면이 닿았는지)
             # Y축 겹침이 있는지 확인
             y_overlap_start = max(b1.y, b2.y)
             y_overlap_end = min(b1.y + b1.depth, b2.y + b2.depth)
             if (y_overlap_end - y_overlap_start) > TOL:
                 # X축으로 닿아있는지?
                 x_touch = (abs((b1.x + b1.width) - b2.x) < TOL) or (abs((b2.x + b2.width) - b1.x) < TOL)
                 if x_touch: 
                     return True

             # Y축 접촉 확인 (앞/뒤 면이 닿았는지)
             # X축 겹침이 있는지 확인
             x_overlap_start = max(b1.x, b2.x)
             x_overlap_end = min(b1.x + b1.width, b2.x + b2.width)
             if (x_overlap_end - x_overlap_start) > TOL:
                # Y축으로 닿아있는지?
                if (abs((b1.y + b1.depth) - b2.y) < TOL) or (abs((b2.y + b2.depth) - b1.y) < TOL):
                    return True
                     
        return False

    def _compute_overlap_area(self, b1: Brick, b2: Brick) -> float:
        """XY 평면상에서 b1(아래)과 b2(위) 사이의 겹침 영역 넓이를 반환합니다."""
        # 사각형 교차
        dx = min(b1.x + b1.width, b2.x + b2.width) - max(b1.x, b2.x)
        dy = min(b1.y + b1.depth, b2.y + b2.depth) - max(b1.y, b2.y)
        if dx > 0 and dy > 0:
            return dx * dy
        return 0.0

    def verify_floating(self, result: VerificationResult):
        """지면(z=0)에 연결되지 않은 브릭이 있는지 검사합니다."""
        ground_nodes = [
            bid for bid, attr in self.graph.nodes(data=True)
            if attr['brick'].z < 0.1 # 지면 검사 완화 (기존 0.0 근접에서 변경)
        ]
        
        if not ground_nodes:
            result.add_hard_fail(Evidence(
                type="FLOATING", 
                severity="CRITICAL", 
                brick_ids=[], 
                message="지면 레이어(z=0)에서 브릭을 찾을 수 없습니다."
            ))
            return

        # 지면에 연결된 모든 노드를 찾습니다.
        connected_to_ground = set()
        for g_node in ground_nodes:
             # BFS/DFS를 사용하여 연결 성분을 찾습니다.
             component = nx.node_connected_component(self.graph, g_node)
             connected_to_ground.update(component)
             
        all_nodes = set(self.graph.nodes())
        floating_nodes = all_nodes - connected_to_ground
        
        if floating_nodes:
            print("\n--- 부동 브릭 진단 (FLOATING BRICK DIAGNOSTICS) ---")
            bricks = self.plan.get_all_bricks()
            for fid in floating_nodes:
                fb = self.plan.bricks[fid]
                print(f"\n[진단] 브릭 {fid} 위치: ({fb.x:.2f}, {fb.y:.2f}, {fb.z:.2f}) 높이={fb.height:.2f}")
                
                # 주변 브릭 확인
                for other in bricks:
                     if other.id == fid: continue
                     
                     # 수직 간격 확인
                     gap_below = fb.z - (other.z + other.height)
                     if abs(gap_below) < 1.0:
                         print(f"  -> 아래쪽 근처: {other.id} 상단={other.z + other.height:.2f} (간격={gap_below:.4f})")
                         
                         # 수평 겹침 확인
                         dx = min(fb.x + fb.width, other.x + other.width) - max(fb.x, other.x)
                         dy = min(fb.y + fb.depth, other.y + other.depth) - max(fb.y, other.y)
                         if dx > 0.05 and dy > 0.05:
                             print(f"     [수평 겹침 확인 OK] dx={dx:.2f}, dy={dy:.2f}")
                             # 간격이 매우 작은데 왜 연결되지 않았는가?
                             if abs(gap_below) < 0.05:
                                 print(f"     [수직 접촉 확인 OK] 연결되어야 합니다! 그래프 에지가 누락되었나요?")
                             else:
                                 print(f"     [수직 간격 오류] 간격 {gap_below:.4f} > 0.05")
                         else:
                             print(f"     [수평 겹침 오류] dx={dx:.2f}, dy={dy:.2f}")
                     
                     # 교차 확인
                     ix = min(fb.x + fb.width, other.x + other.width) - max(fb.x, other.x)
                     iy = min(fb.y + fb.depth, other.y + other.depth) - max(fb.y, other.y)
                     iz = min(fb.z + fb.height, other.z + other.height) - max(fb.z, other.z)
                     
                     if ix > 0.05 and iy > 0.05 and iz > 0.05:
                         print(f"  -> 교차(INTERSECT): {other.id}와 ({ix:.2f}, {iy:.2f}, {iz:.2f})만큼 겹침")
            print("----------------------------------\n")
            result.add_hard_fail(Evidence(
                type="FLOATING",
                severity="CRITICAL",
                brick_ids=list(floating_nodes),
                message=f"{len(floating_nodes)}개의 부동 브릭이 발견되었습니다. 상세: " + ", ".join(
                    [f"{bid}(위치:{self.plan.bricks[bid].x:.2f},{self.plan.bricks[bid].y:.2f},{self.plan.bricks[bid].z:.2f} 크기:{self.plan.bricks[bid].width:.2f}x{self.plan.bricks[bid].depth:.2f}x{self.plan.bricks[bid].height:.2f})" 
                     for bid in floating_nodes]
                )
            ))

    def verify_stability(self, result: VerificationResult, strict_mode: bool = False):
        """
        각 연결된 성분에 대해 무게중심(COM)을 계산하고, 그것이 접촉 바닥면의 
        Convex Hull 내부에 있는지 확인하여 안정성을 검사합니다.
        상단에서 하단으로의 로직 또는 성분 기반 로직을 사용합니다.
        """
        # 단순화: 전체 구조의 무게중심과 바닥 Convex Hull 비교 (전역적 안정성)
        # 실제 구현에서는 하위 구조에 대한 재귀적 검사가 필요합니다.
        
        # 1. 전역 무게중심(Global COM) 계산
        bricks = self.plan.get_all_bricks()
        total_mass = sum(b.mass for b in bricks)
        if total_mass == 0: return

        weighted_pos = np.zeros(3)
        for b in bricks:
            weighted_pos += b.center_of_mass * b.mass
        
        global_com = weighted_pos / total_mass
        
        # 2. 지면 접점(Base) 가져오기
        ground_bricks = [b for b in bricks if np.isclose(b.z, 0.0)]
        if not ground_bricks:
            return # 부동 브릭 검사에서 이미 처리됨
            
        points = []
        for b in ground_bricks:
            points.extend(b.footprint_poly)
        
        points = np.array(points)
        if len(points) < 3:
            # 1개 또는 2개의 점으로는 Hull을 형성할 수 없으며, 질량이 선상에 정확히 있지 않으면 매우 불안정합니다.
            # 지면의 1x1 브릭에 대한 단순화된 체크:
            # 무게중심이 지면 브릭의 바운딩 박스 내부에 있는지 확인합니다.
            min_x, min_y = np.min(points, axis=0)
            max_x, max_y = np.max(points, axis=0)
            stable = (min_x <= global_com[0] <= max_x) and (min_y <= global_com[1] <= max_y)
            if not stable:
                 result.add_penalty(Evidence("UNSTABLE", "WARNING", [], "전역 무게중심이 바닥 경계를 벗어났습니다."), 50)
            return

        # 3. Convex Hull 검사
        try:
            hull = ConvexHull(points)
            # 무게중심(x, y)이 Hull 내부에 있는지 확인합니다.
            # 최적화: hull.equations를 사용하여 평면까지의 거리를 확인합니다.
            # 모든 평면에 대해 dot(normal, point) + offset <= 0 이면 내부입니다.
            
            # Scipy ConvexHull은 입력이 2D이면 2D 결과를 생성합니다.
            # points가 (x, y) 목록이므로 정상 작동합니다.
            
            in_hull = True
            for eq in hull.equations:
                # eq는 [a, b, offset]이며, a*x + b*y + offset <= 0인지 확인합니다.
                if eq[0]*global_com[0] + eq[1]*global_com[1] + eq[2] > 1e-6: # 오차 허용 범위(Epsilon)
                    in_hull = False
                    break
            
            if not in_hull:
                result.add_hard_fail(Evidence(
                    type="UNSTABLE",
                    severity="CRITICAL",
                    brick_ids=[],
                    message="전역 무게중심이 지지 바닥면 다각형 밖에 있습니다. 모델이 넘어질 수 있습니다."
                ))
                
        except Exception as e:
            # 평면 기하학 형태이거나 오류 발생 시
            print(f"Hull 오류: {e}")

    def verify_connection_strength(self, result: VerificationResult, strict_mode: bool = False):
        """
        약한 연결(특히 1-스터드 연결)을 검사합니다.
        strict_mode (Kids Mode): 1-스터드 연결에 대해 Hard Fail 처리합니다.
        Normal Mode: 경고(Warning)를 제공합니다.
        """
        bricks = self.plan.get_all_bricks()
        for i, b1 in enumerate(bricks):
            for b2 in bricks[i+1:]:
                # 수직으로 연결되어 있는지 확인 (하나가 다른 하나 위에 있는지)
                # b2가 b1 위에 있거나, b1이 b2 위에 있는 경우
                is_b2_on_b1 = abs((b1.z + b1.height) - b2.z) < 0.1
                is_b1_on_b2 = abs((b2.z + b2.height) - b1.z) < 0.1
                
                if not (is_b2_on_b1 or is_b1_on_b2):
                    continue
                    
                overlap_area = self._compute_overlap_area(b1, b2)
                
                # 1x1 스터드 면적을 약 1.0 unit²으로 가정 (단위 스케일에 따라 조정 필요)
                if 0 < overlap_area <= 1.0 + 1e-6: # 1 스터드 이하인 경우
                    msg = f"{b1.id}와 {b2.id} 사이의 약한 연결(1-스터드)이 감지되었습니다."
                    
                    # 로직: 1-스터드 연결인 경우, 그것이 상단 브릭의 '유일한' 연결인지 확인해야 할 수도 있습니다.
                    # 이상적으로는 상단 브릭의 모든 연결을 확인해야 하지만,
                    # 여기서는 '국소적 약점(local weakness)'을 찾는 좋은 시작점으로 쌍 단위 체크를 수행합니다.
                    
                    evidence = Evidence(
                        type="WEAK_CONNECTION",
                        severity="CRITICAL" if strict_mode else "WARNING",
                        brick_ids=[b1.id, b2.id],
                        message=msg,
                        layer=int(max(b1.z, b2.z))
                    )
                    
                    if strict_mode:
                        result.add_hard_fail(evidence)
                    else:
                        result.add_penalty(evidence, 10.0)

    def verify_overhang(self, result: VerificationResult, mode: str = "ADULT"):
        """
        브릭이 아래쪽 브릭들에 의해 충분히 지지되고 있는지 확인합니다.
        Kids Mode: 50% 이상의 지지가 필요합니다.
        Adult Mode: 30% 이상의 지지가 필요합니다.
        """
        threshold_ratio = 0.5 if mode == "KIDS" else 0.3
        
        bricks = self.plan.get_all_bricks()
        # z축 기준으로 정렬하여 상향식으로 처리 (개별 체크에는 순서가 엄격하게 중요하진 않음)
        bricks_sorted = sorted(bricks, key=lambda b: b.z)
        
        for top_brick in bricks_sorted:
            if np.isclose(top_brick.z, 0.0):
                continue # 지면 브릭은 100% 지지되는 것으로 간주
                
            # 이 브릭 바로 아래에 있는 모든 브릭 찾기
            supporting_bricks = []
            for potential_support in bricks:
                if np.isclose(potential_support.z + potential_support.height, top_brick.z):
                    if self._compute_overlap_area(potential_support, top_brick) > 0:
                        supporting_bricks.append(potential_support)
            
            # 총 지지 면적 계산
            total_supported_area = 0.0
            for support in supporting_bricks:
                total_supported_area += self._compute_overlap_area(support, top_brick)
                
            # 지지율 계산 (volume/height = 바닥 면적)
            brick_area = top_brick.width * top_brick.depth
            if brick_area > 0:
                support_ratio = total_supported_area / brick_area
            else:
                support_ratio = 0 # 발생해서는 안 되는 케이스
                
            if support_ratio < threshold_ratio:
                 # 부동(floating) 상태인지 아니면 단순 오버행인지 확인
                 if support_ratio < 1e-6:
                     pass # 이미 부동 브릭 검사에서 개념적으로 처리되었지만, 오버행 로직에서도 표시 가능
                          # 기존의 부동 브릭 검사는 그래프 기반 연결성 검사입니다.
                          # 이것은 면적 기반 검사입니다. 여기서는 부분 오버행에 집중합니다.
                 else:
                     msg = f"위험한 오버행: 브릭 {top_brick.id}의 지지율이 {support_ratio*100:.1f}%에 불과합니다 (필요 지지율 > {threshold_ratio*100}%)."
                     result.add_penalty(
                         Evidence("OVERHANG", "WARNING", [top_brick.id], msg, layer=int(top_brick.z)), 
                         20.0
                     )
                     if mode == "KIDS": # 키즈 모드에서는 더 엄격하게 처리하거나 실패로 간주
                         result.add_hard_fail(Evidence("OVERHANG", "CRITICAL", [top_brick.id], msg, layer=int(top_brick.z)))

    def run_collision_check(self, tolerance: float = 0) -> VerificationResult:
        """PyBulletVerifier 호환 인터페이스 - 충돌 검사만 수행"""
        result = VerificationResult()
        self.verify_collision(result)
        return result

    def run_stability_check(self, duration: float = 2.0, auto_close: bool = True) -> VerificationResult:
        """PyBulletVerifier 호환 인터페이스 - 안정성 종합 검사

        Args:
            duration: PyBullet 호환용 (무시됨 - 시뮬레이션 불필요)
            auto_close: PyBullet 호환용 (무시됨)
        """
        result = VerificationResult()
        self.verify_floating(result)
        self.verify_stability(result)
        self.verify_connection_strength(result)
        self.verify_overhang(result)
        self._compute_stability_grade(result)
        return result

    def _compute_stability_grade(self, result: VerificationResult):
        """검증 결과를 기반으로 안정성 등급과 점수를 계산합니다."""
        has_floating = False
        has_unstable_com = False
        has_overhang = False
        has_weak_connection = False

        for ev in result.evidence:
            if ev.type == "FLOATING":
                has_floating = True
            elif ev.type == "UNSTABLE":
                has_unstable_com = True
            elif ev.type == "OVERHANG":
                has_overhang = True
            elif ev.type == "WEAK_CONNECTION":
                has_weak_connection = True

        # 점수 계산
        if has_floating or has_unstable_com:
            # 치명적 문제 → UNSTABLE (0-39)
            result.stability_grade = "UNSTABLE"
            result.score = min(result.score, 39.0)
            result.max_drift = 999.0  # 시뮬레이션 없이 최악의 변위로 표시
        elif has_overhang or has_weak_connection:
            # 경고 수준 → MEDIUM (40-89)
            result.stability_grade = "MEDIUM"
            result.score = min(result.score, 89.0)
            result.max_drift = 0.5
        else:
            # 문제 없음 → STABLE (90-100)
            result.stability_grade = "STABLE"
            result.score = max(result.score, 90.0)
            result.max_drift = 0.0

    def run_all_checks(self, mode: str = "ADULT") -> VerificationResult:
        result = VerificationResult()
        
        # 1. 부동 브릭 검사 (필수)
        self.verify_floating(result)
        if not result.is_valid: 
            return result # 부동 브릭이 있으면 중단 (치명적)
            
        # 2. 충돌 검사 (불가능한 겹침 확인)
        self.verify_collision(result)
        if not result.is_valid:
            print("\n[중단] 치명적인 충돌이 감지되었습니다.")
            return result

        # 3. 안정성 검사 (전역 중력 평형)
        self.verify_stability(result)

        # 4. 연결 강도 및 오버행 검사 (선택적 경고)
        self.verify_connection_strength(result, strict_mode=(mode == "KIDS"))
        self.verify_overhang(result, mode=mode)

        return result

    def verify_collision(self, result: VerificationResult):
        """
        브릭들이 유의미하게 겹치는지 확인합니다 (물리적으로 불가능한 배치).
        스터드나 구멍 등을 위한 미세한 겹침은 허용합니다 (약 10-15% 부피 또는 절대 임계값).
        """
        bricks = self.plan.get_all_bricks()
        collision_count = 0
        
        # 최적화: N^2 검사는 느리지만 1000개 미만의 브릭에서는 수용 가능합니다.
        for i, b1 in enumerate(bricks):
            for b2 in bricks[i+1:]:
                # 빠른 AABB 검사
                if (b1.x + b1.width <= b2.x or b2.x + b2.width <= b1.x or
                    b1.y + b1.depth <= b2.y or b2.y + b2.depth <= b1.y or
                    b1.z + b1.height <= b2.z or b2.z + b2.height <= b1.z):
                    continue
                
                # 교차 부피 계산
                ix = min(b1.x + b1.width, b2.x + b2.width) - max(b1.x, b2.x)
                iy = min(b1.y + b1.depth, b2.y + b2.depth) - max(b1.y, b2.y)
                iz = min(b1.z + b1.height, b2.z + b2.height) - max(b1.z, b2.z)
                
                if ix > 0.05 and iy > 0.05 and iz > 0.05:
                    # 스터드 고려 계산:
                    # LDraw에서 스터드 높이는 4 LDU입니다. 모델 높이 단위인 24 LDU 기준입니다.
                    # 4/24 = 0.1666... 
                    # 수직 적층인 경우 수직 교차 부분에서 스터드 높이만큼 제외해줍니다.
                    # 수직으로 스터드 높이보다 더 많이 겹칠 때만 실제 충돌로 간주합니다.
                    
                    adjusted_iz = max(0, iz - 0.17) # 스터드 높이(약 0.17) 차감
                    intersect_vol = ix * iy * adjusted_iz
                    
                    vol1 = b1.width * b1.depth * b1.height
                    vol2 = b2.width * b2.depth * b2.height
                    
                    # 더 작은 브릭 대비 비율 계산
                    min_vol = min(vol1, vol2)
                    ratio = intersect_vol / min_vol if min_vol > 0 else 0
                    
                    # 스마트 키워드 기반 예외 처리
                    desc1 = get_part_description(b1.id)
                    desc2 = get_part_description(b2.id)
                    
                    # 기계적 결합(구멍 속의 핀/축 등)을 암시하는 키워드
                    mech_keywords = [
                        "wheel", "tyre", "tire", "rim", 
                        "hinge", "plate modified", "holder", 
                        "clip", "propeller", "rotor", "fan",
                        "technic pin", "axle", "cylinder", 
                        "gear", "turntable", "steering",
                        "minifig", "helmet", "hair", "head", "hat", "cap" # 착용용 부품
                    ]
                    
                    is_mech = False
                    for kw in mech_keywords:
                        if kw in desc1.lower() or kw in desc2.lower():
                            is_mech = True
                            break
                    
                    allowed_ratio = 0.10 # 기본 10%
                    if is_mech:
                        allowed_ratio = 0.85 # 기계적 결합 부품은 85%까지 겹침 허용
                        
                    if ratio > allowed_ratio: 
                         collision_count += 1
                         msg = f"{b1.id}({desc1})와 {b2.id}({desc2}) 사이의 충돌 감지 (겹침 비율: {ratio*100:.1f}%)"
                         # 스팸 방지를 위해 초기 20개까지만 상세 기록
                         if collision_count <= 20: 
                             result.add_hard_fail(Evidence("COLLISION", "CRITICAL", [b1.id, b2.id], msg))
        
        if collision_count > 0:
            msg = f"{collision_count}개의 중대한 충돌이 발견되었습니다. 물리적으로 불가능한 모델입니다."
            result.add_hard_fail(Evidence("COLLISION", "CRITICAL", [], msg))
