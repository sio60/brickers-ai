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
    SCALE = 0.01  # LDU 단위를 PyBullet 카메라 범위에 맞게 축소 (1/100)
    
    def __init__(self, plan: BrickPlan = None, gui: bool = False):
        self.plan = plan
        self.gui = gui
        self.physicsClient = None
        self.brick_bodies = {} # brick_id -> body_id 매핑
        self.cached_shapes = {} # part_file -> collision_shape_id 캐싱

    def _init_simulation(self):
        if self.physicsClient is None:
            # GUI 모드는 디버깅용, DIRECT 모드는 속도가 빠름
            mode = p.GUI if self.gui else p.DIRECT
            self.physicsClient = p.connect(mode)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        p.resetSimulation()
        # 중력 설정: 축소된 세계이므로 실제 중력값을 사용해도 무방 (1 unit = 40mm)
        p.setGravity(0, 0, -9.8)
        
        # 고품질 물리 엔진 설정
        p.setPhysicsEngineParameter(
            numSolverIterations=100,  # 기본값 50, 높을수록 제약 조건이 더 안정적
            numSubSteps=4,            # 프레임당 서브 스텝 수 증가
            erp=0.1,                  # 에러 감소 파라미터 (제약 조건 강화)
            contactERP=0.1
        )

    def _close_simulation(self):
        if self.physicsClient is not None:
            p.disconnect()
            self.physicsClient = None

    def _get_collision_shape(self, part_file: str):
        """안정성을 위해 단순화된 BOX 충돌 모양(Collision Shape)을 생성합니다."""
        # 파일명 정리
        part_file = part_file.lower().strip()
        
        if part_file in self.cached_shapes:
            return self.cached_shapes[part_file]

        # 라이브러리에서 치수 가져오기
        try:
             # 딕셔너리가 있는지 확인하기 위해 동적 임포트
             from lego_physics import get_brick_studs_count, STUD_SPACING, BRICK_HEIGHT, PLATE_HEIGHT
             studs_x, studs_z, is_plate = get_brick_studs_count(part_file)
             height = PLATE_HEIGHT if is_plate else BRICK_HEIGHT
             
             # Half Extents 계산 (PyBullet은 절반 크기를 원함)
             # X 전체 = studs_x * 20
             # Y 전체 = height (24 또는 8)
             # Z 전체 = studs_z * 20
             
             # 스케일 적용
             # 수직 이웃과의 마찰을 피하기 위해 1% 축소할지?
             # 아니면 연결된 부분의 충돌을 비활성화하므로 그냥 1.0을 쓸지?
             # 수평 이웃과의 간섭을 피하기 위해 0.99 사용
             
             safe_factor = 0.99
             x_half = (studs_x * STUD_SPACING * self.SCALE * safe_factor) / 2.0
             y_half = (height * self.SCALE * safe_factor) / 2.0  # LDraw Y는 높이(Height)
             z_half = (studs_z * STUD_SPACING * self.SCALE * safe_factor) / 2.0
             
             # PyBullet GEOM_BOX는 halfExtents를 인자로 받음
             # 참고: 나중에 배치할 때 Y/Z를 교환하지만, 여기서는 단순히 박스를 생성함
             # 로컬 좌표계에서 너비(X), 높이(Y), 깊이(Z)
             # 잠깐, LDraw 로컬 좌표계:
             # X는 너비 (Width)
             # Y는 높이 (Height)
             # Z는 깊이 (Depth)
             # 따라서 박스 크기는 [x, y, z] 순서여야 함
             
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
        """PyBullet에 브릭들을 정적 바디(Static Body)로 로드합니다."""
        if plan:
            self.plan = plan
        
        self._init_simulation()
        
        bricks = self.plan.get_all_bricks()
        
        # 사전 패스(Pre-pass): Z 위치를 계산하고 최소값을 찾아 지면에 맞춤
        # LDraw에서 Y는 아래쪽 방향입니다. PyBullet에서 Z는 위쪽 방향입니다.
        # 브릭의 LDraw 원점은 일반적으로 '윗면'에 있습니다.
        # 따라서 바닥면이 지면에 닿으려면 브릭 높이만큼 위로 올려야 합니다.
        
        BRICK_HEIGHT_LDU = 24.0  # LDU 표준 브릭 높이
        
        # PyBullet 좌표계에서 가장 낮은 지점을 찾음 (브릭 바닥 기준)
        z_positions = []
        for b in bricks:
            if b.origin is not None:
                # pb_z = -ldr_y * SCALE, 그리고 바닥면을 구하기 위해 높이를 뺌
                pb_z = -b.origin[1] * self.SCALE
                # LDraw 원점이 위쪽이므로 바닥은 pb_z - height
                bottom_z = pb_z - (BRICK_HEIGHT_LDU * self.SCALE)
                z_positions.append(bottom_z)
        
        # 모든 브릭을 들어 올려 가장 낮은 바닥면이 0이 되도록 오프셋 설정
        if z_positions:
            min_z = min(z_positions)
            z_offset = -min_z  # 각 Z 위치에 더할 값
        else:
            z_offset = 0
            
        print(f"[PyBullet] Z 오프셋 적용: {z_offset:.4f} (모델을 지면에 배치)")
        
        for b in bricks:
            # 원본 데이터 확인
            if b.part_file is None or b.origin is None or b.matrix is None:
                print(f"[WARN] 브릭 {b.id}의 LDraw 원본 데이터 누락. 로드 건너뜀.")
                continue
                
            shape_id = self._get_collision_shape(b.part_file)
            
            # 회전 행렬 (3x3) -> 쿼터니언 (Quaternion)
            # b.matrix는 3x3 numpy 배열
            # Scipy rotation 사용
            try:
                r = R.from_matrix(b.matrix)
                # PyBullet 순서: x, y, z, w
                quat = r.as_quat() 
            except Exception as e:
                print(f"[ERR] {b.id}의 매트릭스 변환 실패: {e}")
                quat = [0, 0, 0, 1]

            # 현실적인 질량(Mass)으로 바디 생성
            # 부피 기반 실제 무게 계산 (2x4 브릭 ≈ 2.3g)
            brick_mass = get_brick_mass_kg(b.part_file)
            
            # 가시성을 위해 무작위 색상 추가
            import random
            col = [random.random(), random.random(), random.random(), 1.0]

            # 좌표 변환: LDraw (X, Y-down, Z) -> PyBullet (X, Z, -Y)
            # LDraw: Y가 수직(아래쪽이 양수), Z가 깊이
            # PyBullet: Z가 수직(위쪽이 양수), Y가 깊이
            ldr_x, ldr_y, ldr_z = b.origin[0], b.origin[1], b.origin[2]
            pb_x = ldr_x * self.SCALE
            pb_y = ldr_z * self.SCALE
            pb_z = -ldr_y * self.SCALE + z_offset  # 지면에 놓기 위해 오프셋 적용
            
            # 박스 프리미티브 중심 보정 (Box Primitive Center Adjustment)
            # LDraw 원점: 윗면 중심 (Y=0)
            # 박스 프리미티브 원점: 기하학적 중심 (Y=Height/2)
            # 만약 박스를 `pb_z`(전역 Z, 윗면)에 배치하면 박스는 위로 절반, 아래로 절반 튀어나옴.
            # 우리는 박스가 `pb_z`에서 아래로 뻗어나가길 원함.
            # 따라서 박스 중심을 PyBullet Z 축 아래로 절반 높이만큼 이동시켜야 함.
            # PyBullet Z는 위쪽이 양수이므로 "아래"는 -Z 방향.
            
            studs_x, studs_z, is_plate = get_brick_studs_count(b.part_file)
            height_val = PLATE_HEIGHT if is_plate else BRICK_HEIGHT
            half_h_scaled = (height_val * self.SCALE) / 2.0
            
            # 바디 원점과 박스 윗면("Top")을 맞추기 위해 Z를 절반 높이만큼 내림
            pb_z -= half_h_scaled
            
            # 좌표 변환: LDraw (X, Y-down, Z) -> PyBullet (X, Z, -Y)
            # 기본 위치(bp_x, bp_y)는 이미 계산함
            # 회전은 까다로움. 
            # LDraw 매트릭스는 벡터를 회전시킴.
            # LDraw (1,0,0) -> X, (0,1,0) -> Down, (0,0,1) -> Z
            # 우리 박스는 로컬 축 X=너비, Y=높이, Z=깊이로 정의됨.
            # 로컬 Y(높이)를 전역 Down(-Z)에 매핑해야 함.
            # 그리고 로컬 X/Z를 전역 X/Y에 매핑해야 함.
            
            # 표준 좌표 변환 로직 사용:
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
            
        # 3. 자동 카메라 설정 - 멀리 줌아웃(ZOOM WAY OUT)
        # if self.gui and bricks:
        #     all_pos = [b.origin for b in bricks if b.origin is not None]
        #     if all_pos:
        #         all_pos = np.array(all_pos) * self.SCALE
        #         min_b = np.min(all_pos, axis=0)
        #         max_b = np.max(all_pos, axis=0)
        #         center = (min_b + max_b) / 2.0
        #         extent = np.linalg.norm(max_b - min_b)
                
        #         # 거리: 전체를 볼 수 있을 만큼 조정 (이전보다 가깝게)
        #         cam_dist = max(extent * 1.5, 3.0)  # 배율 3.0 -> 1.5로 축소, 최소 거리 10 -> 3
        #         p.resetDebugVisualizerCamera(
        #             cameraDistance=cam_dist, 
        #             cameraYaw=45, 
        #             cameraPitch=-30, 
        #             cameraTargetPosition=center
        #         )

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

    def run_stability_check(self, duration: float = 2.0) -> VerificationResult:
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
            
            # 상대 변환(Relative Transform)을 계산하여 초기 오프셋 유지
            # A를 B에 현재 상대 위치 그대로 고정하고 싶음.
            # A의 중심(Local A = [0,0,0])을 피벗으로 설정.
            # A의 중심을 B의 로컬 좌표계로 표현해야 함.
            
            pos_a, orn_a = p.getBasePositionAndOrientation(body_a)
            pos_b, orn_b = p.getBasePositionAndOrientation(body_b)
            
            # P_a를 B의 로컬 프레임으로 변환
            # Local_Pos = Rotate_Inv(World_Pos - Body_Pos)
            
            # B의 회전 역행렬
            inv_orn_b = p.invertTransform([0,0,0], orn_b)[1] # 회전 역행렬만 필요
            
            # B에서 A로 가는 벡터
            diff_pos = np.array(pos_a) - np.array(pos_b)
            
            # B의 프레임으로 회전
            # p.multiplyTransforms가 쉬운 방법
            # 하지만 diff_pos는 벡터임.
            # multiplyTransforms 활용:
            # T_world_to_b = (pos_b, orn_b)^-1
            # P_a_in_b = T_world_to_b * P_a
            
            # PyBullet 헬퍼 사용:
            # invertTransform은 (invPos, invOrn) 반환
            invPosB, invOrnB = p.invertTransform(pos_b, orn_b)
            localPosA_in_B, localOrnA_in_B = p.multiplyTransforms(invPosB, invOrnB, pos_a, orn_a)
            
            p.createConstraint(
                parentBodyUniqueId=body_a,
                parentLinkIndex=-1,
                childBodyUniqueId=body_b,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0],     # A 중심에서 피벗
                childFramePosition=localPosA_in_B, # B에 상대적인 피벗
                parentFrameOrientation=[0,0,0,1],  # 항등원 (A를 자신과 정렬 유지)
                childFrameOrientation=localOrnA_in_B # B 내에서 A의 상대적 오리엔테이션
            )
            # 중요: 연결된 브릭 간의 충돌 비활성화!
            # LDraw 형상은 겹쳐 있음(스터드가 튜브 내부로 들어감). 비활성화 안 하면 물리 폭발 발생.
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
        
        print(f"[Stability] {constraints_count}개의 제약 조건 생성 완료 (스터드-튜브 연결).")
        
        # 4. 시뮬레이션 실행
        steps = int(240 * duration)
        print(f"[Stability] {duration}초 간 시뮬레이션 ({steps} 스텝)...")
        
        first_failure_id = None
        first_failure_step = -1
        
        # 실시간 모니터링 루프
        frame_skip = 10 if not self.gui else 1 # Headless는 속도를 위해 덜 자주 확인, GUI는 매 프레임? 아니 10도 괜찮음.
        
        print(f"[Stability] {steps} 스텝 루프 시작...")
        
        for step in range(steps):
            p.stepSimulation()
            
            # 일정 스텝마다 실패 여부 확인
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
                
                # 디버그 출력: 60스탭(0.25초)마다
                if step % 60 == 0:
                    # 유의미한 경우에만 출력 (0.05 이하는 무시)
                    if current_max_drift > 0.05:
                        print(f"   [Step {step}] 최대 이동: {current_max_drift:.2f} (브릭 {worst_brick})")

                # 임계값: 0.5 (약 50 LDU = 2.5 스터드 변위)
                # 만약 브릭이 2.5 스터드 이상 움직이면 확실히 떨어지는 중임.
                fail_threshold_val = 0.5 
                
                if current_max_drift > fail_threshold_val and first_failure_id is None:
                        first_failure_id = worst_brick
                        first_failure_step = step
                        print(f"[Stability] 단계 {step}에서 실패 ({step/240:.2f}초): {worst_brick}이(가) {current_max_drift:.2f}만큼 이동함")
                        
                        # GUI 모드에서는 붕괴 과정을 끝까지 보여주기 위해 계속 진행
                        # 자동화(CI) 모드에서는 빠른 결과를 위해 즉시 중단
                        if not self.gui:
                            break
            
            if self.gui:
                import time
                time.sleep(1./240.)
                
        # 5. 변위 확인 및 리포트
        # (결과 초기화는 시작 부분으로 이동함)
        result.is_valid = not bool(first_failure_id) # 브릭이 하나도 안 떨어져야 유효
        failed_bricks = list() # 호환성을 위해 리스트 사용
        max_drift = 0.0
        drift_threshold = 0.5 # 최종 확인에도 동일한 임계값 적용
        
        # 첫 번째 실패가 감지되면 증거 추가
        if first_failure_id:
            result.evidence.append(Evidence(
                type="FIRST_FAILURE",
                severity="CRITICAL",
                brick_ids=[first_failure_id],
                message=f"구조적 붕괴 시작점: {first_failure_id} (시간={first_failure_step/240:.2f}초)"
            ))
        
        for bid, body_id in brick_bodies.items():
            current_pos, _ = p.getBasePositionAndOrientation(body_id)
            start_pos, _ = original_positions[body_id]
            dist = np.linalg.norm(np.array(current_pos) - np.array(start_pos))
            max_drift = max(max_drift, dist)
            
            if dist > drift_threshold:
                failed_bricks.append(bid)
                # 첫 번째 실패가 아닌 경우 상세 증거 추가 (중복 방지)
                if bid != first_failure_id:
                    result.evidence.append(Evidence(
                        type="COLLAPSE_AFTERMATH",
                        severity="ERROR",
                        brick_ids=[bid],
                        message=f"붕괴 시작 후 브릭이 {dist:.1f}만큼 이동함"
                    ))

        if failed_bricks:
            result.is_valid = False
            result.score = 0
            print(f"[Stability] 실패. 최대 이동: {max_drift:.2f}")
        else:
            print(f"[Stability] 통과. 최대 이동: {max_drift:.2f}")
            result.score = 100
        
        # --- REPORT CARD ---
        print("\n" + "="*40)
        print(" 🏭 물리 검증 리포트 (Physics Report)")
        print("="*40)
        print(f" - 🧱 총 브릭 수: {len(brick_bodies)}") # Changed self.brick_bodies to brick_bodies
        print(f" - 🔗 연결 상태: {constraints_count}개 본드 결합 완료") # Changed self.constraints to constraints_count
        
        # Re-evaluate floating bricks for report, using ground_threshold from earlier
        # connected_bricks needs to be derived from constraints
        connected_brick_ids = set()
        for brick_id_a, brick_id_b in connections:
            connected_brick_ids.add(brick_id_a)
            connected_brick_ids.add(brick_id_b)

        # Assuming 'bricks' is a list of brick IDs from self.plan.get_all_bricks()
        # And 'brick_bodies' maps brick IDs to PyBullet body IDs
        # brick_plans와 brick_ids를 알아야 부동 확인을 위한 위치 정보를 얻을 수 있음
        # 이 범위(Scope)에서는 해당 정보가 직접적으로 없음.
        # 일단은 이미 계산된 'floating' 변수를 사용하자.
        # 원래 'floating' 확인이 충분했다면 재사용 가능.
        # 원래 'floating' 확인: floating = find_floating_bricks(bricks)
        # 이 변수는 이미 사용 가능함.
        
        if floating: # 이전 검사에서의 'floating' 변수 재사용
             print(f" - ⚠️ 위험 요소: Floating Brick {len(floating)}개 발견! (주의)")
        else:
             print(f" - ✨ 구조 상태: 모든 브릭이 잘 연결됨")
             
        print("-" * 40)
        print(f" [시뮬레이션 결과]")
        print(f" - 🕒 진행 시간: {duration:.1f}초")
        print(f" - 📏 최대 이동(Drift): {max_drift:.2f} (허용치: {drift_threshold})") # Changed threshold to drift_threshold
        print("-" * 40)
        
        if result.score == 100: # Changed score to result.score
            print(" ✅ 최종 판정: [합격] (SUCCESS)")
            print("    \"이 모델은 튼튼합니다!\"")
        else:
            print(" ❌ 최종 판정: [불합격] (FAIL)")
            # 원인 찾기
            culprit = "알 수 없음"
            for ev in result.evidence:
                if ev.type == "FIRST_FAILURE" and ev.brick_ids:
                    culprit = ev.brick_ids[0]
                    break
            print(f"    💥 최초 붕괴: {culprit}")
            
            # 다른 피해 브릭들 나열
            victims = []
            for ev in result.evidence:
                if ev.type == "COLLAPSE_AFTERMATH" and ev.brick_ids:
                    victims.append(ev.brick_ids[0])
            
            if victims:
                print(f"    📉 추가 붕괴 ({len(victims)}개): {', '.join(victims[:5])}" + (f"...외 {len(victims)-5}개" if len(victims)>5 else ""))
                
            print("    \"구조가 불안정하여 무너졌습니다.\"")
        print("="*40 + "\n")
        
        # GUI인 경우, 사용자가 볼 수 있도록 창 유지
        if self.gui:
            print("[PyBullet] 시뮬레이션 종료. 창을 닫으려면 Enter 키를 누르세요...")
            input()

        self._close_simulation()
        return result

# Simple Test
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
