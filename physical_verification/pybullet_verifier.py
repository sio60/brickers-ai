# 이 파일은 PyBullet 물리 엔진을 사용하여 레고 모델의 조립 가능성 및 구조적 안정성을 검증하는 핵심 검증기입니다.
import pybullet as p
import pybullet_data
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import List, Dict, Set, Tuple

try:
    from .models import Brick, BrickPlan, VerificationResult, Evidence
    from .lego_physics import check_stud_tube_connection, find_floating_bricks, find_all_connections, get_brick_mass_kg, get_brick_studs_count, STUD_SPACING, BRICK_HEIGHT, PLATE_HEIGHT
    from .part_library import get_part_geometry
except ImportError:
    from models import Brick, BrickPlan, VerificationResult, Evidence
    from lego_physics import check_stud_tube_connection, find_floating_bricks, find_all_connections, get_brick_mass_kg, get_brick_studs_count, STUD_SPACING, BRICK_HEIGHT, PLATE_HEIGHT
    from part_library import get_part_geometry

class PyBulletVerifier:
    SCALE = 0.01  # PyBullet 카메라를 위해 LDU 단위를 축소 (1 LDU = 0.01 단위)
    
    def __init__(self, plan: BrickPlan = None, gui: bool = False):
        self.plan = plan
        self.gui = gui
        self.physicsClient = None
        self.brick_bodies = {} # brick_id -> body_id 매핑
        self.cached_shapes = {} # part_file -> collision_shape_id 캐시

    def _init_simulation(self):
        if self.physicsClient is None:
            # GUI 모드는 디버깅에 유용하며, DIRECT 모드는 속도가 빠름
            mode = p.GUI if self.gui else p.DIRECT
            self.physicsClient = p.connect(mode)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        p.resetSimulation()
        # 중력 설정: 스케일된 월드 (1 단위 = 40mm) 이지만 실제 중력 가속도 사용
        p.setGravity(0, 0, -9.8)
        
        # 고품질 물리 설정
        p.setPhysicsEngineParameter(
            numSolverIterations=100,  # 기본값 50, 높을수록 제약 조건이 안정적
            numSubSteps=4,            # 프레임당 서브 스텝 수 증가
            erp=0.1,                  # 오류 감소 매개변수 (제약 조건을 단단하게 유지)
            contactERP=0.1
        )

    def _close_simulation(self):
        if self.physicsClient is not None:
            p.disconnect()
            self.physicsClient = None

    def _get_collision_shape(self, part_file: str):
        """안정성을 위해 단순화된 BOX 충돌 형태를 생성합니다."""
        # 파일명 정리
        part_file = part_file.lower().strip()
        
        if part_file in self.cached_shapes:
            return self.cached_shapes[part_file]

        # 라이브러리에서 치수 가져오기
        try:
             # 딕셔너리가 로드되었는지 확인하기 위해 동적 임포트
             from lego_physics import get_brick_studs_count, STUD_SPACING, BRICK_HEIGHT, PLATE_HEIGHT
             studs_x, studs_z, is_plate = get_brick_studs_count(part_file)
             height = PLATE_HEIGHT if is_plate else BRICK_HEIGHT
             
             # 반(Half) 크기 계산 (PyBullet은 half-extent를 사용)
             # X 전체 = studs_x * 20
             # Y 전체 = height (24 또는 8)
             # Z 전체 = studs_z * 20
             
             # 스케일 적용
             # 수직 이웃과의 마찰을 피하기 위해 1%를 줄임 (0.99)
             # 연결된 브릭 간 충돌은 비활성화하므로 수평 이웃에 대한 안전장치로 0.99 사용
             
             safe_factor = 0.99
             x_half = (studs_x * STUD_SPACING * self.SCALE * safe_factor) / 2.0
             y_half = (height * self.SCALE * safe_factor) / 2.0  # LDraw Y는 높이
             z_half = (studs_z * STUD_SPACING * self.SCALE * safe_factor) / 2.0
             
             # PyBullet GEOM_BOX는 halfExtents를 인자로 받음
             # 참고: 배치 시 Y/Z를 교환하지만, 여기서는 단순히 상자를 생성
             
             colShapeId = p.createCollisionShape(
                 p.GEOM_BOX, 
                 halfExtents=[x_half, y_half, z_half]
             )
             self.cached_shapes[part_file] = colShapeId
             return colShapeId
             
        except Exception as e:
            print(f"[WARN] {part_file}에 대한 박스 생성 실패: {e}")
            # 대체값 (Fallback)
            colShapeId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
            self.cached_shapes[part_file] = colShapeId
            return colShapeId

    def load_bricks(self, plan: BrickPlan = None):
        """브릭들을 PyBullet에 정적 바디(Static Body)로 로드합니다."""
        if plan:
            self.plan = plan
        
        self._init_simulation()
        
        bricks = self.plan.get_all_bricks()
        
        # 사전 처리: Z 위치를 계산하고 지면에 맞추기 위한 최소값을 찾음
        # LDraw에서 Y는 아래쪽 방향입니다. PyBullet에서 Z는 위쪽 방향입니다.
        # 브릭의 LDraw 원점은 일반적으로 상단 표면에 위치합니다.
        # 따라서 바닥이 지면에 닿도록 브릭 높이만큼 위로 올려야 합니다.
        
        BRICK_HEIGHT_LDU = 24.0  # 표준 브릭 높이 (LDU)
        
        # PyBullet 좌표계에서 가장 낮은 지점 찾기 (브릭 바닥 고려)
        z_positions = []
        for b in bricks:
            if b.origin is not None:
                # pb_z = -ldr_y * SCALE, 그리고 바닥면을 얻기 위해 반 높이를 뺌
                pb_z = -b.origin[1] * self.SCALE
                # LDraw 원점은 상단이므로, 바닥은 pb_z - height
                bottom_z = pb_z - (BRICK_HEIGHT_LDU * self.SCALE)
                z_positions.append(bottom_z)
        
        # 가장 낮은 바닥이 0이 되도록 모든 브릭을 들어올리는 오프셋
        if z_positions:
            min_z = min(z_positions)
            z_offset = -min_z  # 각 Z 위치에 더할 값
        else:
            z_offset = 0
            
        print(f"[PyBullet] Z 오프셋 적용됨: {z_offset:.4f} (모델을 지면에 배치)")
        
        for b in bricks:
            # 원본 데이터가 있는지 확인
            if b.part_file is None or b.origin is None or b.matrix is None:
                print(f"[WARN] 브릭 {b.id}에 원본 LDraw 데이터가 없습니다. PyBullet 로드를 건너뜁니다.")
                continue
                
            shape_id = self._get_collision_shape(b.part_file)
            
            # 회전 행렬 (3x3) -> 쿼터니언 변환
            try:
                r = R.from_matrix(b.matrix)
                # PyBullet 순서: x, y, z, w
                quat = r.as_quat() 
            except Exception as e:
                print(f"[ERR] {b.id}에 대한 행렬 변환 실패: {e}")
                quat = [0, 0, 0, 1]

            # 현실적인 질량으로 바디 생성
            # 부피 기반 실제 브릭 무게 계산 (2x4 브릭 ≈ 2.3g)
            brick_mass = get_brick_mass_kg(b.part_file)
            
            # 가시성을 위해 무작위 색상 추가
            import random
            col = [random.random(), random.random(), random.random(), 1.0]

            # 좌표 변환: LDraw (X, Y-down, Z) -> PyBullet (X, Z, -Y)
            # LDraw: Y는 수직(아래로 양수), Z는 깊이
            # PyBullet: Z는 수직(위로 양수), Y는 깊이
            ldr_x, ldr_y, ldr_z = b.origin[0], b.origin[1], b.origin[2]
            pb_x = ldr_x * self.SCALE
            pb_y = ldr_z * self.SCALE
            pb_z = -ldr_y * self.SCALE + z_offset  # 지면에 앉히기 위해 오프셋 적용
            
            # 박스 프리미티브 중심 조정
            # LDraw 원점: 상단 표면 중심 (Y=0)
            # 박스 프리미티브 원점: 기하학적 중심 (Y=Height/2)
            # p.createMultiBody는 COM/링크 프레임을 배치합니다.
            # 시각적/충돌 형태가 중심에 있는 박스이고, 바디를 `pb_z`(상단 표면)에 배치하면
            # 박스는 위로 반, 아래로 반 튀어나옵니다.
            # 우리는 박스가 `pb_z`에서 아래로 확장되기를 원합니다.
            # 따라서 박스 중심을 PyBullet Z축에서 반 높이만큼 내려야 합니다.
            
            studs_x, studs_z, is_plate = get_brick_studs_count(b.part_file)
            height_val = PLATE_HEIGHT if is_plate else BRICK_HEIGHT
            half_h_scaled = (height_val * self.SCALE) / 2.0
            
            # 박스의 "상단"이 바디 원점과 일치하도록 Z를 반 높이만큼 내림
            pb_z -= half_h_scaled
            
            # 좌표 변환 로직
            # 이미 기본 (pb_x, pb_y)는 계산됨
            # 회전은 까다로움. LDraw 행렬은 벡터를 회전시킵니다.
            # 표준 좌표 변환 로직에 의존:
            coord_convert = R.from_euler('x', -90, degrees=True)
            try:
                brick_rotation = R.from_matrix(b.matrix)
                final_rotation = coord_convert * brick_rotation
                quat = final_rotation.as_quat()  # x, y, z, w
            except:
                quat = coord_convert.as_quat()
            
            body_id = p.createMultiBody(
                baseMass=brick_mass,  # 현실적인 질량 (2x4 ≈ 0.0023kg)
                baseCollisionShapeIndex=shape_id,
                baseVisualShapeIndex=-1, 
                basePosition=[pb_x, pb_y, pb_z],
                baseOrientation=quat
            )
            p.changeVisualShape(body_id, -1, rgbaColor=col)
            
            self.brick_bodies[b.id] = body_id
            
        # 3. 자동 카메라 설정 - 멀리 줌 아웃
        if self.gui and bricks:
            all_pos = [b.origin for b in bricks if b.origin is not None]
            if all_pos:
                all_pos = np.array(all_pos) * self.SCALE
                min_b = np.min(all_pos, axis=0)
                max_b = np.max(all_pos, axis=0)
                center = (min_b + max_b) / 2.0
                extent = np.linalg.norm(max_b - min_b)
                
                # 거리: 모든 것을 볼 수 있도록 충분히 멀리 줌 아웃
                cam_dist = max(extent * 3.0, 10.0)  # 최소 10 단위 뒤로
                p.resetDebugVisualizerCamera(
                    cameraDistance=cam_dist, 
                    cameraYaw=45, 
                    cameraPitch=-30, 
                    cameraTargetPosition=center
                )

    def run_collision_check(self, tolerance: float = -0.05) -> VerificationResult:
        """
        글로벌 접촉 포인트(Fast)를 사용하여 충돌 감지를 실행합니다.
        """
        self.load_bricks()
        result = VerificationResult()
        
        # 1. 글로벌 충돌 체크 (Broadphase + Narrowphase)
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
                
                # 심각한 관통에 대해 필터링
                if dist < tolerance:
                     bid1 = [k for k, v in self.brick_bodies.items() if v == b1][0]
                     bid2 = [k for k, v in self.brick_bodies.items() if v == b2][0]
                     
                     msg = f"메쉬 충돌 감지: {bid1} <-> {bid2} (깊이: {abs(dist):.2f} LDU)"
                     collisions.append(msg)
                     result.add_hard_fail(Evidence(
                         type="COLLISION",
                         severity="CRITICAL",
                         brick_ids=[bid1, bid2],
                         message=msg
                     ))

        # 여기서는 시뮬레이션을 닫지 않고, 안정성 검사에 필요할 경우 유지함
        
        if not collisions:
            print("PyBullet 검증 통과 (충돌 없음)")
            result.score = 100
        else:
            result.is_valid = False
            result.score = 0
            
        return result

    def run_stability_check(self, duration: float = 2.0) -> VerificationResult:
        """
        안정성 확인을 위해 중력 시뮬레이션을 실행합니다.
        접촉 포인트를 사용하여 제약 조건(Glue)을 자동 생성합니다.
        """
        print("안정성 시뮬레이션 초기화 중...")
        result = VerificationResult()  # 결과 초기화
        
        # 시뮬레이션이 유효한지 확인. 충돌 체크가 실행되지 않았다면 초기화.
        if self.physicsClient is None:
            self._init_simulation()
            self.load_bricks()
             
        # 안정성을 위해 중력 재설정 (스케일된 월드이므로 실제 중력 사용)
        p.setGravity(0, 0, -9.8) 
        
        # 1. 지면(Ground Plane) (없으면 로드)
        try:
            planeId = p.loadURDF("plane.urdf")
        except:
             pass # 이미 로드되었거나 파일이 없을 수 있음. Plane은 보통 내장됨.

        # 2. 동적 질량 및 제약 조건
        # 바디를 동적(Dynamic)으로 전환해야 함. 초기 생성 시 mass=0일 수 있음.
        
        brick_bodies = self.brick_bodies
        brick_ids = list(brick_bodies.keys())
        original_positions = {}
        
        # 지면 임계값 결정 (가장 낮은 Z)
        all_z = []
        for body_id in brick_bodies.values():
             pos, _ = p.getBasePositionAndOrientation(body_id)
             all_z.append(pos[2])
        
        min_z = min(all_z) if all_z else 0.0
        ground_threshold = min_z + 0.05 # 5cm 이내 (스케일된 값)
        
        for bid, body_id in brick_bodies.items():
            pos, orn = p.getBasePositionAndOrientation(body_id)
            original_positions[body_id] = (pos, orn)
            
            # 동적 바디 안정화
            # 모든 브릭이 물리 시뮬레이션에 참여 (무한 질량 앵커 없음)
            # 이를 통해 전체적인 불안정성(넘어짐/기우뚱)을 확인할 수 있음
            p.changeDynamics(
                body_id, 
                -1, 
                mass=0.1,                # 모든 브릭에 질량 부여
                lateralFriction=0.9,     # 지면 마찰력 높임
                rollingFriction=0.1,
                spinningFriction=0.1,
                restitution=0.0,         # 튕김 없음
                linearDamping=0.5,       # 공기 저항
                angularDamping=0.5
            )

        # 3. 스터드-튜브 연결 로직을 사용한 제약 조건 생성
        # 스터드와 튜브가 제대로 정렬된 브릭만 연결
        constraints_count = 0
        
        # 모든 브릭을 가져오고 적절한 연결 찾기
        bricks = self.plan.get_all_bricks()
        print(f"[Stability] {len(bricks)}개 브릭의 스터드-튜브 연결 확인 중...")
        
        # lego_physics 모듈을 사용하여 연결 찾기
        connections = find_all_connections(bricks)
        print(f"[Stability] {len(connections)}개의 스터드-튜브 연결 발견.")
        
        # id -> body_id 매핑 생성
        id_to_body = brick_bodies
        
        # 연결된 브릭에 대해서만 제약 조건 생성
        for brick_id_a, brick_id_b in connections:
            if brick_id_a not in id_to_body or brick_id_b not in id_to_body:
                continue
                
            body_a = id_to_body[brick_id_a]
            body_b = id_to_body[brick_id_b]
            
            # 초기 오프셋을 유지하기 위해 상대 변환 계산
            # 현재 상대 위치에서 A를 B에 고정하고 싶음.
            
            pos_a, orn_a = p.getBasePositionAndOrientation(body_a)
            pos_b, orn_b = p.getBasePositionAndOrientation(body_b)
            
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
                parentFramePosition=[0, 0, 0],     # A 중심 기준
                childFramePosition=localPosA_in_B, # B 기준 상대 위치
                parentFrameOrientation=[0,0,0,1],  # 항등 행렬 (A 자체 정렬 유지)
                childFrameOrientation=localOrnA_in_B # B 내에서의 A 상대 회전
            )
            # 중요: 연결된 브릭 간 충돌 비활성화!
            # LDraw 형상은 겹치므로(튜브 내 스터드) 비활성화하지 않으면 물리 엔진 폭발
            p.setCollisionFilterPair(body_a, body_b, -1, -1, enableCollision=0)
            constraints_count += 1
        
        # 공중 부양 브릭 확인 (아무것도 연결되지 않고 지면에도 닿지 않음)
        floating = find_floating_bricks(bricks)
        if floating:
            print(f"[Stability] 경고: {len(floating)}개의 공중 부양 브릭 발견: {floating[:5]}...")
            for fid in floating:
                result.evidence.append(Evidence(
                    type="FLOATING_BRICK",
                    severity="CRITICAL",
                    brick_ids=[fid],
                    message=f"브릭 {fid}는 어떤 구조물에도 연결되지 않았습니다."
                ))
        
        print(f"[Stability] {constraints_count}개의 제약 조건 생성 완료 (스터드-튜브 연결).")
        
        # 4. 시뮬레이션 실행
        steps = int(240 * duration)
        print(f"[Stability] {duration}초 동안 시뮬레이션 ({steps} 스텝)...")
        
        first_failure_id = None
        first_failure_step = -1
        
        # 실시간 모니터링 루프
        frame_skip = 10 if not self.gui else 1 
        
        print(f"[Stability] {steps} 스텝 시뮬레이션 루프 시작...")
        
        for step in range(steps):
            p.stepSimulation()
            
            # 몇 스텝마다 실패 확인
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
                
                # 디버그 출력 (매 60스텝 또는 큰 이동 발생 시)
                if step % 60 == 0 or current_max_drift > 0.1:
                    if current_max_drift > 0.05:
                        print(f"   [Step {step}] 최대 이동: {current_max_drift:.2f} (브릭 {worst_brick})")

                # 임계값: 0.5 (약 50 LDU = 2.5 스터드 변위)
                # 브릭이 2.5 스터드 이상 움직이면 떨어지는 것으로 간주
                fail_threshold_val = 0.5 
                
                if current_max_drift > fail_threshold_val and first_failure_id is None:
                        first_failure_id = worst_brick
                        first_failure_step = step
                        print(f"[Stability] 실패 감지 (스텝 {step}, {step/240:.2f}초): {worst_brick} 이동 거리 {current_max_drift:.2f}")
                        # 파이프라인 속도를 위해 즉시 중단
                        break
            
            if self.gui:
                import time
                time.sleep(1./240.)
                
        # 5. 변위 확인 및 보고
        result.is_valid = not bool(first_failure_id) # 떨어진 브릭이 없으면 유효
        failed_bricks = list() 
        max_drift = 0.0
        drift_threshold = 0.5 # 최종 확인용 임계값
        
        # 감지된 경우 최초 실패 증거 추가
        if first_failure_id:
            result.evidence.append(Evidence(
                type="FIRST_FAILURE",
                severity="CRITICAL",
                brick_ids=[first_failure_id],
                message=f"구조 붕괴 시작: {first_failure_id} (시간={first_failure_step/240:.2f}s)"
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
                        message=f"붕괴 시작 후 브릭이 {dist:.1f}만큼 이동함"
                    ))

        if failed_bricks:
            result.is_valid = False
            result.score = 0
            print(f"[Stability] 실패. 최대 이동: {max_drift:.2f}")
        else:
            print(f"[Stability] 통과. 최대 이동: {max_drift:.2f}")
            result.score = 100
        
        # --- 성적표 ---
        print("\n" + "="*40)
        print(" 🏭 물리 검증 리포트 (Physics Report)")
        print("="*40)
        print(f" - 🧱 총 브릭 수: {len(brick_bodies)}") 
        print(f" - 🔗 연결 상태: {constraints_count}개 본드 결합 완료") 
        
        if floating: 
             print(f" - ⚠️ 위험 요소: 공중 부양 브릭 {len(floating)}개 발견! (주의)")
        else:
             print(f" - ✨ 구조 상태: 모든 브릭이 잘 연결됨")
             
        print("-" * 40)
        print(f" [시뮬레이션 결과]")
        print(f" - 🕒 진행 시간: {duration:.1f}초")
        print(f" - 📏 최대 이동(Drift): {max_drift:.2f} (허용치: {drift_threshold})") 
        print("-" * 40)
        
        if result.score == 100: 
            print(" ✅ 최종 판정: [합격] (SUCCESS)")
            print("    \"이 모델은 튼튼합니다!\"")
        else:
            print(" ❌ 최종 판정: [불합격] (FAIL)")
            # 원인 제공자 찾기
            culprit = "알 수 없음"
            for ev in result.evidence:
                if ev.type == "FIRST_FAILURE" and ev.brick_ids:
                    culprit = ev.brick_ids[0]
                    break
            print(f"    💥 최초 붕괴: {culprit}")
            
            # 다른 피해자 나열
            victims = []
            for ev in result.evidence:
                if ev.type == "COLLAPSE_AFTERMATH" and ev.brick_ids:
                    victims.append(ev.brick_ids[0])
            
            if victims:
                print(f"    📉 추가 붕괴 ({len(victims)}개): {', '.join(victims[:5])}" + (f"...외 {len(victims)-5}개" if len(victims)>5 else ""))
                
            print("    \"구조가 불안정하여 무너졌습니다.\"")
        print("="*40 + "\n")
        
        # GUI인 경우, 사용자가 확인할 수 있도록 창 유지
        if self.gui:
            print("[PyBullet] 시뮬레이션 종료. 창을 닫으려면 Enter를 누르세요...")
            input()

        self._close_simulation()
        return result

# 간단 테스트
if __name__ == "__main__":
    pass
