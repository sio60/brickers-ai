# ============================================================================
# PyBullet 기반 물리 검증 모듈
# 이 파일은 PyBullet 물리 엔진을 활용하여 레고 브릭 구조의 물리적 충돌 및
# 안정성을 검증하는 기능을 제공합니다. LDraw 모델 데이터를 PyBullet 환경으로
# 변환하고, 중력 시뮬레이션을 통해 구조물의 붕괴 여부를 확인하며,
# 상세한 검증 결과를 생성합니다.
# ============================================================================
import pybullet as p
import pybullet_data
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import List, Dict, Set, Tuple
import sys
import os

# 프로젝트 루트 경로를 path에 추가 (config 모듈 인식을 위해 필수)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from .models import Brick, BrickPlan, VerificationResult, Evidence
    from .lego_physics import check_stud_tube_connection, find_floating_bricks, find_all_connections, get_brick_mass_kg, get_brick_studs_count, STUD_SPACING, BRICK_HEIGHT, PLATE_HEIGHT
    from .part_library import get_part_geometry
except ImportError:
    from models import Brick, BrickPlan, VerificationResult, Evidence
    from lego_physics import check_stud_tube_connection, find_floating_bricks, find_all_connections, get_brick_mass_kg, get_brick_studs_count, STUD_SPACING, BRICK_HEIGHT, PLATE_HEIGHT
    from part_library import get_part_geometry

class PyBulletVerifier:
    SCALE = 0.01  # LDraw 단위를 PyBullet 카메라에서 다루기 쉬운 범위로 축소
    
    def __init__(self, plan: BrickPlan = None, gui: bool = False):
        self.plan = plan
        self.gui = gui
        self.physicsClient = None
        self.brick_bodies = {} # brick_id -> body_id 매핑
        self.cached_shapes = {} # part_file -> collision_shape_id 캐싱

    def _init_simulation(self):
        if self.physicsClient is None:
            # GUI 모드는 디버깅에 도움이 되며, DIRECT 모드는 속도가 더 빠릅니다.
            mode = p.GUI if self.gui else p.DIRECT
            self.physicsClient = p.connect(mode)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        p.resetSimulation()
        # 중력 설정: 스케일링된 환경이므로 1 유닛은 현재 40mm에 해당함. 실제 중력값 사용.
        p.setGravity(0, 0, -9.8)
        
        # 고품질 물리 설정
        p.setPhysicsEngineParameter(
            numSolverIterations=100,  # 기본값 50보다 높게 설정하여 제약 조건의 안정성 확보
            numSubSteps=4,            # 프레임당 서브스텝 수 증가
            erp=0.1,                  # 오차 감소 파라미터 (제약 조건을 타이트하게 유지)
            contactERP=0.1
        )

    def _close_simulation(self):
        if self.physicsClient is not None:
            p.disconnect()
            self.physicsClient = None

    def _get_collision_shape(self, part_file: str):
        """안정성을 위해 단순화된 상자(BOX) 모양의 충돌 형상을 생성합니다."""
        # 파일명 정리
        part_file = part_file.lower().strip()
        
        if part_file in self.cached_shapes:
            return self.cached_shapes[part_file]

        # 라이브러리에서 치수 정보 가져오기
        try:
             # 필요한 설정값들을 lego_physics에서 가져옴
             from lego_physics import get_brick_studs_count, STUD_SPACING, BRICK_HEIGHT, PLATE_HEIGHT
             studs_x, studs_z, is_plate = get_brick_studs_count(part_file)
             height = PLATE_HEIGHT if is_plate else BRICK_HEIGHT
             
             # 절반 범위 계산 (PyBullet은 half-extents 요구)
             # X 전체 = studs_x * 20
             # Y 전체 = 높이 (24 또는 8)
             # Z 전체 = studs_z * 20
             
             # 스케일 적용
             # 수직 인접 브릭 간의 마찰 간섭을 피하기 위해 1%(0.99) 정도 작게 설정
             
             safe_factor = 0.99
             x_half = (studs_x * STUD_SPACING * self.SCALE * safe_factor) / 2.0
             y_half = (height * self.SCALE * safe_factor) / 2.0  # LDraw Y는 높이임
             z_half = (studs_z * STUD_SPACING * self.SCALE * safe_factor) / 2.0
             
             # PyBullet GEOM_BOX는 halfExtents를 매개변수로 받음
             # LDraw 로컬 좌표계: X=너비, Y=높이, Z=깊이
             
             colShapeId = p.createCollisionShape(
                 p.GEOM_BOX, 
                 halfExtents=[x_half, y_half, z_half]
             )
             self.cached_shapes[part_file] = colShapeId
             return colShapeId
             
        except Exception as e:
            print(f"[WARN] {part_file}의 박스 생성 실패: {e}")
            # 대체값 (Fallback)
            colShapeId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
            self.cached_shapes[part_file] = colShapeId
            return colShapeId

    def load_bricks(self, plan: BrickPlan = None):
        """브릭들을 PyBullet에 고정체(Static Bodies)로 로드합니다."""
        if plan:
            self.plan = plan
        
        self._init_simulation()
        
        # 이전 세션의 브릭 바디 매핑 초기화 (수정된 LDR 파일 로드 시 필수)
        self.brick_bodies = {}
        
        bricks = self.plan.get_all_bricks()
        
        # 사전 처리: Z축 위치를 계산하고 지면에 맞추기 위한 최소 오프셋 찾기
        # LDraw에서는 Y가 아래 방향입니다. PyBullet에서는 Z가 위 방향입니다.
        # LDraw에서 브릭의 원점은 보통 상단 표면에 위치합니다.
        # 따라서 브릭의 아래면이 지면에 닿도록 브릭 높이만큼 위로 오프셋을 주어야 합니다.
        
        BRICK_HEIGHT_LDU = 24.0  # LDU 단위의 표준 브릭 높이
        
        # PyBullet 좌표계에서 가장 낮은 점 찾기 (브릭 바닥 기준)
        z_positions = []
        for b in bricks:
            if b.origin is not None:
                # pb_z = -ldr_y * SCALE, 그다음 절반 높이를 빼서 바닥을 구함
                pb_z = -b.origin[1] * self.SCALE
                # LDraw 원점은 상단에 있으므로, 바닥은 pb_z - 높이임
                bottom_z = pb_z - (BRICK_HEIGHT_LDU * self.SCALE)
                z_positions.append(bottom_z)
        
        # 가장 낮은 바닥이 0이 되도록 모든 브릭을 들어올리는 오프셋
        if z_positions:
            min_z = min(z_positions)
            z_offset = -min_z  # 각 Z 위치에 더할 값
        else:
            z_offset = 0
            
        print(f"[PyBullet] Z 오프셋 적용됨: {z_offset:.4f} (모델을 지면에 배치하기 위함)")
        
        for b in bricks:
            # 원본 데이터 확인
            if b.part_file is None or b.origin is None or b.matrix is None:
                print(f"[경고] 브릭 {b.id}의 원본 LDraw 데이터가 누락되었습니다. PyBullet 로드를 건너뜁니다.")
                continue
                
            shape_id = self._get_collision_shape(b.part_file)
            
            # 회전 행렬 (3x3) -> 쿼터니언 변환
            # b.matrix는 3x3 넘파이 배열임
            try:
                r = R.from_matrix(b.matrix)
                # PyBullet 순서: x, y, z, w
                quat = r.as_quat() 
            except Exception as e:
                print(f"[오류] {b.id}에 대한 행렬 변환 실패: {e}")
                quat = [0, 0, 0, 1]

            # 실제 질량을 사용하여 바디 생성
            # 부피로부터 실제 브릭 무게 계산 (2x4 브릭 ≈ 2.3g)
            brick_mass = get_brick_mass_kg(b.part_file)
            
            # 시각화를 위해 무작위 색상 추가
            import random
            col = [random.random(), random.random(), random.random(), 1.0]

            # 좌표 변환: LDraw (X, Y-아래, Z) -> PyBullet (X, Z, -Y)
            # LDraw: Y는 수직(아래가 양수), Z는 깊이
            # PyBullet: Z는 수직(위가 양수), Y는 깊이
            ldr_x, ldr_y, ldr_z = b.origin[0], b.origin[1], b.origin[2]
            pb_x = ldr_x * self.SCALE
            pb_y = ldr_z * self.SCALE
            pb_z = -ldr_y * self.SCALE + z_offset  # 지면에 배치하기 위해 오프셋 적용
            
            # 상자 기본도형 중심 조정
            # LDraw 원점: 상단 표면 중심 (Y=0)
            # 상자 기본도형 원점: 기하학적 중심 (Y=높이/2)
            # So if we place Box at (0,0,0), its top is at -Height/2, bottom at +Height/2? No.
            # 상자가 0에 위치하면 -H/2에서 +H/2까지 뻗습니다.
            # 우리는 모델 상단이 0에 오길 원하므로, 중심은 +H/2에 있어야 합니다.
            # LDraw는 Y축이 아래 방향입니다. 
            #   상단 = 원점 Y
            #   하단 = 원점 Y + 높이
            #   중심 = 원점 Y + 높이/2
            
            # PyBullet 설정:
            #   pb_x = ldr_x * SCALE
            #   pb_y = ldr_z * SCALE 
            #   pb_z = -ldr_y * SCALE + z_offset (원점을 정확한 Z 높이에 둡니다)
            
            # 하지만 p.createMultiBody는 질량중심(COM)이나 링크 프레임을 배치합니다.
            # 만약 시각/충돌 형상이 중앙 정렬된 상자이고 바디를 pb_z(전역 Z의 상단 표면)에 배치하면,
            # 상자의 절반은 위로, 절반은 아래로 튀어나오게 됩니다.
            # 우리는 상자가 pb_z로부터 아래로 뻗어나가길 원하므로,
            # 상자 중심(Box CENTER)을 PyBullet Z 기준 절반 높이만큼 아래로 내려야 합니다.
            # PyBullet Z는 위 방향이므로 "아래"는 -Z입니다.
            
            studs_x, studs_z, is_plate = get_brick_studs_count(b.part_file)
            height_val = PLATE_HEIGHT if is_plate else BRICK_HEIGHT
            half_h_scaled = (height_val * self.SCALE) / 2.0
            
            # 상자의 "상단"이 바디 원점과 정렬되도록 Z를 절반 높이만큼 내립니다.
            pb_z -= half_h_scaled
            
            # 좌표 변환: LDraw (X, Y-아래, Z) -> PyBullet (X, Z, -Y)
            # 이미 기본 pb_x, pb_y를 계산했습니다.
            # 회전은 까다롭습니다. 
            # LDraw 매트릭스는 벡터를 회전시킵니다.
            # LDraw (1,0,0) -> X, (0,1,0) -> 아래, (0,0,1) -> Z
            # 우리의 상자는 로컬 축 X=너비, Y=높이, Z=깊이로 정의됩니다.
            # 로컬 Y(높이)를 글로벌 아래(-Z)로 매핑해야 합니다.
            # 그리고 로컬 X/Z를 글로벌 X/Y로 매핑합니다.
            
            # 표준 좌표 변환 로직에 의존합니다:
            coord_convert = R.from_euler('x', -90, degrees=True)
            try:
                brick_rotation = R.from_matrix(b.matrix)
                final_rotation = coord_convert * brick_rotation
                quat = final_rotation.as_quat()  # x, y, z, w
            except:
                quat = coord_convert.as_quat()
            
            body_id = p.createMultiBody(
                baseMass=brick_mass,  # 실제 질량 (2x4 ≈ 0.0023kg)
                baseCollisionShapeIndex=shape_id,
                baseVisualShapeIndex=-1, 
                basePosition=[pb_x, pb_y, pb_z],
                baseOrientation=quat
            )
            p.changeVisualShape(body_id, -1, rgbaColor=col)
            
            self.brick_bodies[b.id] = body_id
            
        # 3. 자동 카메라 설정 - 줌 아웃
        if self.gui and bricks:
            all_pos = [b.origin for b in bricks if b.origin is not None]
            if all_pos:
                all_pos = np.array(all_pos) * self.SCALE
                min_b = np.min(all_pos, axis=0)
                max_b = np.max(all_pos, axis=0)
                center = (min_b + max_b) / 2.0
                extent = np.linalg.norm(max_b - min_b)
                
                # 거리: 모든 것을 볼 수 있을 만큼 멀리 줌 아웃함
                cam_dist = max(extent * 3.0, 10.0)  # 최소 10 유닛 뒤로
                p.resetDebugVisualizerCamera(
                    cameraDistance=cam_dist, 
                    cameraYaw=45, 
                    cameraPitch=-30, 
                    cameraTargetPosition=center
                )

    def run_collision_check(self, tolerance: float = -0.05) -> VerificationResult:
        """
        전역 접촉점(Global Contact Points)을 사용하여 충돌 감지를 실행합니다 (빠름).
        """
        self.load_bricks()
        result = VerificationResult()
        
        # 1. 전역 충돌 검사 (Broadphase + Narrowphase)
        # performCollisionDetection은 getContactPoints에 암시되어 있지만 명시적으로 호출하는 것이 좋음
        p.performCollisionDetection()
        points = p.getContactPoints()
        
        collisions = []
        checked_pairs = set()

        if points:
            for pt in points:
                # pt[1] = bodyUniqueIdA, pt[2] = bodyUniqueIdB
                b1, b2 = pt[1], pt[2]
                
                # 중복 방지 (A-B 와 B-A)
                if b1 > b2: b1, b2 = b2, b1
                if (b1, b2) in checked_pairs: continue
                checked_pairs.add((b1, b2))
                
                # pt[8] = contactDistance (접촉 거리)
                dist = pt[8]
                
                # 심각한 침투(Penetration) 필터링
                if dist < tolerance:
                     bid1 = [k for k, v in self.brick_bodies.items() if v == b1][0]
                     bid2 = [k for k, v in self.brick_bodies.items() if v == b2][0]
                     
                     msg = f"Mesh Collision: {bid1} <-> {bid2} (Depth: {abs(dist):.2f} LDU)"
                     collisions.append(msg)
                     result.add_hard_fail(Evidence(
                         type="COLLISION",
                         severity="CRITICAL",
                         brick_ids=[bid1, bid2],
                         message=msg
                     ))

        # 시뮬레이션을 여기서 닫지 않음. 안정성 검사가 필요할 수 있으므로 열어둠
        # self._close_simulation() 
        
        if not collisions:
            print("PyBullet 검증 통과 (충돌 없음)")
            result.score = 100
        else:
            result.is_valid = False
            result.score = 0
            
        return result

    def run_stability_check(self, duration: float = 2.0, auto_close: bool = True) -> VerificationResult:
        """
        중력 시뮬레이션을 실행하여 안정성을 확인합니다.
        접촉점(Contact Points)을 사용하여 제약 조건(Glue)을 자동 생성합니다.
        """
        print("안정성 시뮬레이션 초기화 중...")
        result = VerificationResult()  # 시작 시 결과 초기화
        
        # 시뮬레이션이 유효한지 확인. run_collision_check가 호출되었다면 열려 있음.
        # 아니라면 초기화.
        if self.physicsClient is None:
            self._init_simulation()
            self.load_bricks()
             
        # 안정성을 위해 중력 리셋 (축소된 세계이므로 실제 중력 사용)
        p.setGravity(0, 0, -9.8) 
        
        # 1. 지면(Ground Plane) (없으면 로드)
        # 이미 로드되었는지 확인? 그냥 로드해도 무방함.
        try:
            planeId = p.loadURDF("plane.urdf")
        except:
             pass # 이미 로드되었거나 파일이 없을 수 있음. Plane은 보통 내장됨.

        # 2. 동적 질량 & 제약 조건 (Dynamic Mass & Constraints)
        # 바디를 동적으로 전환해야 함? mass=0으로 생성되었었음.
        # PyBullet: changeDynamics로 질량 변경 가능!
        
        brick_bodies = self.brick_bodies
        brick_ids = list(brick_bodies.keys())
        original_positions = {}
        
        # 지면 임계값 결정 (가장 낮은 Z)
        # 이미 오프셋을 적용해서 최저점이 0이지만, 안전하게 다시 확인.
        all_z = []
        for body_id in brick_bodies.values():
             pos, _ = p.getBasePositionAndOrientation(body_id)
             all_z.append(pos[2])
        
        min_z = min(all_z) if all_z else 0.0
        ground_threshold = min_z + 0.05 # 5cm(축소) 또는 0.2스터드 이내
        
        for bid, body_id in brick_bodies.items():
            pos, orn = p.getBasePositionAndOrientation(body_id)
            original_positions[body_id] = (pos, orn)
            
            # 동적 바디 안정화 (DYNAMIC BODIES STABILIZATION)
            # 모든 브릭이 물리 시뮬레이션에 참여 (무한 질량 앵커(Anchor) 없음)
            # 이를 통해 전체적인 불안정성(넘어짐/기울어짐)을 확인할 수 있음
            # 높은 마찰력은 미끄러짐을 방지하지만 구르거나 넘어지는 것은 허용함.
            p.changeDynamics(
                body_id, 
                -1, 
                mass=0.1,  # 모든 브릭에 질량 부여
                lateralFriction=0.9,     # 지면 접지력을 위한 높은 마찰계수
                rollingFriction=0.1,
                spinningFriction=0.1,
                restitution=0.0,         # 튕김(Bouncing) 없음
                linearDamping=0.5,       # 공기 저항 등가
                angularDamping=0.5
            )

        # 3. 스터드-튜브(STUD-TUBE) 연결 로직을 이용한 제약 조건 생성
        # 스터드-튜브 정렬을 통해 올바르게 연결된 브릭들만 연결
        constraints_count = 0
        
        # 모든 브릭을 가져와 적절한 연결 찾기
        bricks = self.plan.get_all_bricks()
        print(f"[Stability] {len(bricks)}개 브릭에 대한 STUD-TUBE 연결 확인 중...")
        
        # lego_physics 모듈을 사용하여 적절한 연결 찾기
        connections = find_all_connections(bricks)
        print(f"[Stability] {len(connections)}개의 스터드-튜브 연결 발견.")
        
        # Create id -> body_id mapping
        id_to_body = brick_bodies
        
        # Create constraints only for properly connected bricks
        for brick_id_a, brick_id_b in connections:
            if brick_id_a not in id_to_body or brick_id_b not in id_to_body:
                continue
                
            body_a = id_to_body[brick_id_a]
            body_b = id_to_body[brick_id_b]
            
            # 초기 오프셋 보존을 위한 상대적 변환(Relative Transform) 계산
            # 현재의 상대적 위치에서 A와 B를 고정(Lock)함.
            # A의 중심(로컬 A = [0,0,0])을 기준으로 피벗 설정.
            # 로컬 B 좌표계에서 본 A의 중심 위치가 필요함.
            
            pos_a, orn_a = p.getBasePositionAndOrientation(body_a)
            pos_b, orn_b = p.getBasePositionAndOrientation(body_b)
            
            # P_a를 B의 로컬 프레임으로 변환
            # 로컬 위치 = 회전_역행렬(전역_위치 - 바디_위치)
            
            # B의 역회전 구하기
            inv_orn_b = p.invertTransform([0,0,0], orn_b)[1] # 오직 역회전 정보만 사용
            
            # B에서 A로 향하는 벡터
            diff_pos = np.array(pos_a) - np.array(pos_b)
            
            # B의 프레임으로 회전
            # multiplyTransforms가 쉬운 방법이지만 diff_pos는 벡터임.
            # multiplyTransforms 사용 시:
            # T_world_to_b = (pos_b, orn_b)^-1
            # P_a_in_b = T_world_to_b * P_a
            
            # PyBullet 헬퍼 사용:
            # invertTransform은 (invPos, invOrn)을 반환함
            invPosB, invOrnB = p.invertTransform(pos_b, orn_b)
            localPosA_in_B, localOrnA_in_B = p.multiplyTransforms(invPosB, invOrnB, pos_a, orn_a)
            
            p.createConstraint(
                parentBodyUniqueId=body_a,
                parentLinkIndex=-1,
                childBodyUniqueId=body_b,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0],     # A 중심을 피벗으로 설정
                childFramePosition=localPosA_in_B, # B에 대한 상대적 피벗 위치
                parentFrameOrientation=[0,0,0,1],  # 단위 행렬 (A의 정렬 유지)
                childFrameOrientation=localOrnA_in_B # B 내에서의 A의 상대적 방향
            )
            # 중요: 연결된 브릭 간의 충돌 감지 비활성화!
            # LDraw 지오메트리는 겹치는 부분이 있어(스터드가 튜브 안으로 들어감), 비활성화하지 않으면 물리 엔진이 폭발할 수 있음.
            p.setCollisionFilterPair(body_a, body_b, -1, -1, enableCollision=0)
            constraints_count += 1
        
        # 부동(Floating) 브릭 확인 (아무것에도 연결되지 않고 지면에도 닿지 않음)
        floating = find_floating_bricks(bricks)
        if floating:
            print(f"[Stability] 경고: {len(floating)}개의 부동(Floating) 브릭 감지됨: {floating[:5]}...")
            for fid in floating:
                result.evidence.append(Evidence(
                    type="FLOATING_BRICK",
                    severity="CRITICAL",
                    brick_ids=[fid],
                    message=f"Brick {fid} is not connected to any structure"
                ))
        
        print(f"[안정성] {constraints_count}개의 제약 조건(스터드-튜브 연결)이 생성되었습니다.")
        
        # 4. 시뮬레이션 실행
        steps = int(240 * duration)
        print(f"[안정성] {duration}초 간 시뮬레이션 진행 중 ({steps} 스텝)...")
        
        first_failure_id = None
        first_failure_step = -1
        
        # 실시간 모니터링 루프
        frame_skip = 10 if not self.gui else 1 # 헤드리스 모드에서는 속도를 위해 10프레임마다 확인, GUI 모드는 매 프레임 확인
        
        print(f"[안정성] {steps} 스텝에 대한 시뮬레이션 루프 시작...")
        
        for step in range(steps):
            p.stepSimulation()
            
            # 몇 단계마다 붕괴 여부 확인
            if step % 10 == 0:
                current_max_drift = 0.0
                worst_brick = None
                
                for bid, body_id in brick_bodies.items():
                    current_pos, _ = p.getBasePositionAndOrientation(body_id)
                    start_pos, _ = original_positions[body_id]
                    dist = np.linalg.norm(np.array(current_pos) - np.array(start_pos))
                    
                    if dist > current_max_drift:
                        current_max_drift = dist
                        worst_brick = bid
                
                # 60스텝(0.25초)마다 또는 이동 거리가 0.1 이상일 때 디버그 출력
                if step % 60 == 0 or current_max_drift > 0.1:
                    # 유의미한 이동이 있을 때만 출력
                    if current_max_drift > 0.05:
                        print(f"   [스텝 {step}] 최대 이동: {current_max_drift:.2f} (브릭 {worst_brick})")

                # 임계값: 0.5 (약 50 LDU = 2.5 스터드 변위)
                # 브릭이 2.5 스터드 이상 움직였다면 확실히 떨어지는 것으로 간주함
                fail_threshold_val = 0.5 
                
                if current_max_drift > fail_threshold_val and first_failure_id is None:
                        first_failure_id = worst_brick
                        first_failure_step = step
                        print(f"[안정성] 실패 감지 - 스텝 {step} ({step/240:.2f}초): {worst_brick}이 {current_max_drift:.2f}만큼 이동함")
                        # 즉시 중단하여 파이프라인 속도 향상
                        # GUI 모드에서는 붕괴 과정을 끝까지 보여주기 위해 계속 진행
                        # 자동화(CI) 모드에서는 빠른 결과를 위해 즉시 중단
                        if not self.gui:
                            break
            
            if self.gui:
                import time
                time.sleep(1./240.)
                
        # 5. 변위 확인 및 리포트
        # (결과 객체 초기화는 시작 부분에서 수행됨)
        result.is_valid = not bool(first_failure_id) # 브릭이 떨어지지 않은 경우에만 유효
        failed_bricks = list() # 호환성을 위해 리스트 사용
        max_drift = 0.0
        drift_threshold = 0.5 # 최종 확인을 위한 동일 임계값

        # 최초 실패 증거 추가
        if first_failure_id:
            result.evidence.append(Evidence(
                type="FIRST_FAILURE",
                severity="CRITICAL",
                brick_ids=[first_failure_id],
                message=f"구조적 붕괴가 {first_failure_id}에서 시작됨 (t={first_failure_step/240:.2f}초)"
            ))

        for bid, body_id in brick_bodies.items():
            current_pos, _ = p.getBasePositionAndOrientation(body_id)
            start_pos, _ = original_positions[body_id]
            dist = np.linalg.norm(np.array(current_pos) - np.array(start_pos))
            max_drift = max(max_drift, dist)

            if dist > drift_threshold:
                failed_bricks.append(bid)
                # 중복 방지를 위해 최초 실패가 아닌 경우에만 상세 증거 추가
                if bid != first_failure_id:
                    result.evidence.append(Evidence(
                        type="COLLAPSE_AFTERMATH",
                        severity="ERROR",
                        brick_ids=[bid],
                        message=f"붕괴 시작 후 브릭이 {dist:.1f} 유닛만큼 이동함"
                    ))

        # ============================================================
        # 3단계 안정성 등급 (Stability Grade)
        # ============================================================
        # STABLE (안정):   냅뒀을 때 잘 서있음 - max_drift < 0.05
        # MEDIUM (중간):   냅뒀을 때 기우는 정도 - 0.05 <= max_drift < 0.5
        # UNSTABLE (불안정): 냅뒀을 때 무너질 가능성 - max_drift >= 0.5 또는 붕괴 감지
        STABLE_DRIFT = 0.05   # 거의 움직이지 않음 (~1 LDU)
        MEDIUM_DRIFT = 0.5    # 기울어지지만 무너지지 않음 (~10 LDU)

        if first_failure_id is not None or max_drift >= MEDIUM_DRIFT:
            # 불안정: 붕괴 발생 또는 큰 변위
            stability_grade = "UNSTABLE"
            clamped = min(max_drift, 2.0)
            score = max(0, int(39 * (1 - clamped / 2.0)))
        elif max_drift >= STABLE_DRIFT:
            # 중간: 약간 기울어짐
            stability_grade = "MEDIUM"
            ratio = (max_drift - STABLE_DRIFT) / (MEDIUM_DRIFT - STABLE_DRIFT)
            score = int(89 - ratio * 49)  # 89 ~ 40
        else:
            # 안정: 거의 움직이지 않음
            stability_grade = "STABLE"
            ratio = max_drift / STABLE_DRIFT if STABLE_DRIFT > 0 else 0
            score = int(100 - ratio * 10)  # 100 ~ 90

        # 공중부양 브릭이 있으면 등급 하향 (최대 MEDIUM)
        if floating and stability_grade == "STABLE":
            stability_grade = "MEDIUM"
            score = min(score, 70)

        result.stability_grade = stability_grade
        result.score = score
        result.max_drift = max_drift

        if failed_bricks:
            result.is_valid = False
            print(f"[안정성] 검증 실패. 최대 변위: {max_drift:.2f}")
        else:
            print(f"[안정성] 검증 통과. 최대 변위: {max_drift:.2f}")

        # 공중부양 존재 시 is_valid = False
        if floating:
            result.is_valid = False

        # --- 결과 리포트 (REPORT CARD) ---
        GRADE_LABEL = {"STABLE": "안정", "MEDIUM": "중간", "UNSTABLE": "불안정"}
        GRADE_EMOJI = {"STABLE": "🟢", "MEDIUM": "🟡", "UNSTABLE": "🔴"}

        print("\n" + "="*40)
        print(" 🏭 물리 검증 리포트 (Physics Report)")
        print("="*40)
        print(f" - 🧱 총 브릭 수: {len(brick_bodies)}")
        print(f" - 🔗 연결 상태: {constraints_count}개 본드 결합 완료")

        # 리포트를 위해 부동 브릭 재평가
        if floating: # 이전 체크에서 계산된 'floating' 변수 재사용
             print(f" - ⚠️ 위험 요소: 공중 부양 브릭(Floating Brick) {len(floating)}개 발견! (주의)")
        else:
             print(f" - ✨ 구조 상태: 모든 브릭이 잘 연결됨")

        print("-" * 40)
        print(f" [시뮬레이션 결과]")
        print(f" - 🕒 진행 시간: {duration:.1f}초")
        print(f" - 📏 최대 이동(Drift): {max_drift:.2f} (허용치: {drift_threshold})")
        print(f" - {GRADE_EMOJI[stability_grade]} 안정성 등급: {GRADE_LABEL[stability_grade]} ({stability_grade})")
        print(f" - 📊 점수: {score}/100")
        print("-" * 40)

        if stability_grade == "STABLE" and not floating:
            print(" ✅ 최종 판정: [안정] (STABLE)")
            print("    \"이 모델은 튼튼합니다!\"")
        elif stability_grade == "MEDIUM" or (stability_grade == "STABLE" and floating):
            reason = f"공중부양 {len(floating)}개" if floating else f"기울어짐 (drift: {max_drift:.2f})"
            print(f" 🟡 최종 판정: [중간] (MEDIUM - {reason})")
            if floating:
                print(f"    \"모델은 무너지지 않았지만, {len(floating)}개의 브릭이 허공에 떠 있어 불완전합니다.\"")
                print(f"    - 공중부양 브릭: {floating[:5]}...")
            else:
                print(f"    \"모델이 약간 기울어집니다. 구조 보강이 필요합니다.\"")
        else:
            print(" 🔴 최종 판정: [불안정] (UNSTABLE)")
            # 원인 분석
            culprit = "알 수 없음"
            for ev in result.evidence:
                if ev.type == "FIRST_FAILURE" and ev.brick_ids:
                    culprit = ev.brick_ids[0]
                    break
            print(f"    💥 최초 붕괴 시작점: {culprit}")

            # 다른 피해 브릭 목록
            victims = []
            for ev in result.evidence:
                if ev.type == "COLLAPSE_AFTERMATH" and ev.brick_ids:
                    victims.append(ev.brick_ids[0])

            if victims:
                print(f"    📉 추가 붕괴 ({len(victims)}개): {', '.join(victims[:5])}" + (f"...외 {len(victims)-5}개" if len(victims)>5 else ""))

            print("    \"구조가 불안정하여 무너졌습니다.\"")
        print("="*40 + "\n")
        
        # GUI 모드인 경우 사용자가 확인할 수 있도록 창을 열어둠
        if self.gui:
            print("[PyBullet] 시뮬레이션 종료. 창을 닫으려면 엔터를 누르세요...")
            input()

        if auto_close:
            self._close_simulation()
        return result

# ============================================================================
# 실행 스크립트 (CLI)
# ============================================================================
def main():
    import argparse
    import os
    
    # Imports
    try:
        from physical_verification.ldr_loader import LdrLoader
    except ImportError:
         from ldr_loader import LdrLoader

    parser = argparse.ArgumentParser(description="PyBullet Physical Verification Runner")
    parser.add_argument("file", help="Path to the LDR file to verify")
    parser.add_argument("--gui", action="store_true", help="Enable GUI visualization")
    parser.add_argument("--time", type=float, default=5.0, help="Simulation duration in seconds (default: 60.0)")
    args = parser.parse_args()

    target_file = args.file
    if not os.path.exists(target_file):
        # 상대 경로로 시도 (프로젝트 루트 기준)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        target_file = os.path.join(project_root, args.file)
        if not os.path.exists(target_file):
            print(f"❌ 에러: 파일을 찾을 수 없습니다: {args.file}")
            return

    print(f"🚀 PyBullet 물리 검증 시작: {target_file}")
    
    # 1. LDR 로드
    loader = LdrLoader()
    try:
        plan = loader.load_from_file(target_file)
        print(f"✅ 모델 로드 완료: 브릭 {len(plan.bricks)}개")
    except Exception as e:
        print(f"❌ 로드 실패: {e}")
        return

    # 2. PyBullet Verifier 초기화
    verifier = PyBulletVerifier(plan, gui=args.gui)
    
    # 3. 충돌 검사 (Collision Check)
    print("\n[1/2] 정밀 충돌 검사 실행 중...")
    col_result = verifier.run_collision_check()
    if not col_result.is_valid:
        print("⚠️ 충돌 감지됨!")

    # 4. 안정성 검사 (Stability Check)
    print(f"\n[2/2] 구조적 안정성(중력) 시뮬레이션 ({args.time}초)...")
    stab_result = verifier.run_stability_check(duration=args.time)
    
    print("\n" + "="*40)
    if col_result.is_valid and stab_result.is_valid:
        print("🎉 최종 결과: [PASS] 모든 검증 통과!")
    else:
        print("🚫 최종 결과: [FAIL] 검증 실패")
        if not col_result.is_valid: print(" - 사유: 부품 간 충돌 발생")
        if not stab_result.is_valid: print(" - 사유: 구조적 붕괴 발생")
    print("="*40)

if __name__ == "__main__":
    main()
