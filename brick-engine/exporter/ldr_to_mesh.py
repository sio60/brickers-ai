# ============================================================================
# LDR 파일을 3D 메쉬로 변환하는 파일
# 이 파일은 LDR(레고 조립도) 파일을 읽어서 Trimesh Scene 객체로 변환하고,
# 선택적으로 PyBullet 물리 검증을 수행하여 불안정한 브릭을 빨간색으로 표시합니다.
# 사용법: python ldr_to_mesh.py <ldr_file_path> [--verify]
# ============================================================================

import trimesh
import numpy as np
import os
import sys
import argparse

# 프로젝트 루트를 경로에 추가 (config.py 등 일반 모듈 용)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# physical_verification 경로 추가
pv_path = os.path.join(project_root, 'brickers-ai', 'physical_verification')
if pv_path not in sys.path:
    sys.path.append(pv_path)

try:
    import part_library
    from ldr_loader import LdrLoader
    from verifier import PhysicalVerifier
    from pybullet_verifier import PyBulletVerifier
except ImportError as e:
    print(f"물리 검증 모듈 임포트 오류: {e}")
    # 경로 디버깅을 위해 현재 sys.path 출력
    # print(f"현재 sys.path: {sys.path}")
    sys.exit(1)

# LDraw 색상 ID -> RGB (0-255)
LDRAW_COLORS = {
    0:  (33, 33, 33),
    1:  (0, 85, 191),
    2:  (0, 123, 40),
    3:  (0, 131, 138),
    4:  (180, 0, 0),
    5:  (171, 67, 183),
    6:  (91, 28, 12),
    7:  (156, 146, 145),
    8:  (99, 95, 82),
    9:  (107, 171, 220),
    10: (97, 189, 76),
    11: (0, 170, 164),
    12: (255, 99, 71),
    13: (255, 148, 194),
    14: (255, 220, 0),
    15: (255, 255, 255),
    17: (173, 221, 80),
    18: (251, 171, 24),
    19: (215, 197, 153),
    20: (215, 240, 215),
    21: (255, 240, 60), # 밝은 노랑
    22: (88, 42, 18), # 진한 보라
    23: (0, 72, 150), # 중간 파랑
    25: (245, 134, 36),
    26: (202, 31, 123),
    27: (159, 195, 65),
    28: (33, 55, 23),
    29: (160, 188, 172),
    31: (208, 208, 208), # 중간 라벤더
    33: (252, 252, 252, 128), # 투명-투명
    34: (35, 120, 65, 128),   # 투명-초록
    35: (50, 205, 50, 128),   # 투명-밝은 초록
    36: (200, 50, 50, 128),   # 투명-빨강
    37: (100, 100, 200, 128), # 투명-진한 파랑
    38: (255, 100, 0, 128),   # 투명-네온 주황
    39: (252, 252, 252, 128), # 투명-투명 (구버전/car.ldr용 커스텀 매핑)
    40: (100, 100, 100, 128), # 투명-검정 (연기)
    41: (150, 200, 255, 128), # 투명-연한 파랑
    42: (200, 255, 50, 128),  # 투명-네온 초록
    43: (200, 220, 255, 128), # 투명-매우 연한 파랑
    46: (255, 220, 0, 128),   # 투명-노랑
    47: (255, 255, 255, 128), # 투명-투명 (구버전)
    52: (150, 50, 200, 128),  # 투명-보라
    54: (255, 255, 50, 128),  # 투명-네온 노랑
    57: (255, 150, 0, 128),   # 투명-주황
    70: (89, 47, 14),
    71: (175, 181, 199),
    72: (108, 110, 104),
    73: (117, 142, 220),
    74: (183, 212, 37),
    77: (249, 164, 199), # 연한 분홍
    78: (254, 186, 189),
    84: (170, 125, 85),
    85: (89, 39, 115),
    89: (44, 21, 119), # 청보라
    92: (208, 208, 208), # 살색
    100: (254, 196, 1), # 연한 노랑
    110: (67, 84, 163), # 보라
    112: (104, 116, 159), # 중간 청보라
    114: (100, 100, 100), # 중간 회색
    115: (0, 204, 0), # 중간 초록
    118: (179, 215, 209), # 연한 아쿠아
    120: (147, 185, 60), # 연한 라임
    125: (249, 164, 199), # 연한 분홍
    129: (166, 202, 240), # 옅은 파랑
    134: (158, 163, 20), # 올리브 초록
    135: (91, 93, 58), # 진주 회색
    137: (223, 102, 149), # 중간 누가
    142: (179, 139, 23), # 금색
    148: (99, 95, 97), # 진한 진주 회색
    151: (95, 117, 140), # 모래 파랑
    178: (140, 0, 255), # 평면 진한 금색
    179: (204, 204, 204), # 평면 은색
    183: (255, 255, 255), # 진주 흰색
    191: (248, 187, 61),
    212: (175, 217, 255), # 연한 로얄 블루
    216: (174, 164, 111), # 녹
    226: (255, 255, 153), # 쿨 옐로우
    232: (135, 192, 234), # 하늘색
    272: (46, 85, 197), # 진한 파랑
    288: (30, 90, 168), # 진한 초록
    308: (50, 48, 47), # 진한 갈색
    313: (206, 233, 242), # 머스크 블루
    320: (120, 27, 33),
    321: (64, 90, 155), # 진한 하늘색
    322: (85, 140, 244), # 중간 하늘색
    323: (206, 234, 243), # 연한 하늘색
    326: (231, 242, 167), # 봄 황록색
    330: (128, 8, 27), # 올리브 초록
    335: (223, 223, 102), # 모래 빨강
    351: (247, 133, 177), # 중간 진한 분홍
    353: (220, 96, 174), # 산호색
    366: (178, 116, 145), # 흙 주황
    373: (178, 190, 197), # 모래 보라
    378: (163, 193, 173),
    379: (208, 219, 97), # 모래 파랑
    450: (250, 213, 166), # 파불랜드 갈색
    462: (211, 211, 101), # 중간 주황
    484: (179, 62, 0),
}

def ldr_to_mesh(file_path, verify=False):
    """
    LDR 파일을 로드하여 단일 Trimesh Scene 객체로 변환합니다.
    'part_library'를 사용하여 형상을 가져옵니다 (DB 캐시 또는 로컬 파일).
    
    Args:
        file_path (str): LDR 파일 경로.
        verify (bool): True일 경우 PyBullet 물리 검증을 실행하고 불안정한 브릭을 강조 표시합니다.
    """

    # 1. 파일 존재 확인
    if not os.path.exists(file_path):
        print(f"파일을 찾을 수 없습니다: {file_path}")
        return None

    print(f"로딩 중: {file_path}...")

    # ---------------------------------------------------------
    # [물리 검증] (선택 사항, --verify 플래그 사용 시)
    # ---------------------------------------------------------
    failed_brick_ids = set() # 실패한 브릭 강조용 집합
    
    if verify:
        print("물리 검증 실행 중 (PyBullet)...")
        try:
            loader = LdrLoader()
            plan = loader.load_from_file(file_path)

            # [PyBullet 검사] 정밀 메쉬 충돌 확인
            print("PyBullet 정밀 충돌 검사 실행 중 (GUI 모드)...")
            pb_verifier = PyBulletVerifier(plan, gui=True)
            # 허용 오차 0 = 최대 민감도 (모든 접촉이 충돌로 간주됨)
            pb_result = pb_verifier.run_collision_check(tolerance=0) 

            if not pb_result.is_valid:
                print(f"PyBullet 충돌 감지!")
                for ev in pb_result.evidence:
                    print(f"  [PyBullet] {ev.message}")
                    for bid in ev.brick_ids:
                        failed_brick_ids.add(bid)

            # [PyBullet 검사] 안정성 확인 (중력 시뮬레이션)
            print("PyBullet 안정성 검증 실행 중 (중력) - 2초 동안...")
            stab_result = pb_verifier.run_stability_check(duration=2.0)

            if not stab_result.is_valid:
                 print(f"PyBullet 불안정성 감지!")
                 for ev in stab_result.evidence:
                     print(f"  [Stability] {ev.message}")
                     for bid in ev.brick_ids:
                         failed_brick_ids.add(bid)

            if not failed_brick_ids:
                print("물리 검증 통과 (모든 브릭 안정적)!")
            else:
                 print(f"강조 표시할 실패 브릭 총합: {len(failed_brick_ids)}")
                 print(f"실패 ID 예시: {list(failed_brick_ids)[:10]}...")

        except Exception as e:
            print(f"오류로 인해 검증이 취소되었습니다: {e}")
            import traceback
            traceback.print_exc()

    # ---------------------------------------------------------
    # [시각화] 씬 구축
    # ---------------------------------------------------------
    scene = trimesh.Scene()
    brick_counter = 0 

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('0'): continue

            parts = line.split()
            if not parts: continue

            line_type = parts[0]

            # 라인 타입 1: 서브파일 (부품)
            if line_type == '1':
                # 형식: 1 <색상> x y z a b c d e f g h <파일>
                try:
                    # 위치 및 회전 파싱
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    a, b, c = float(parts[5]), float(parts[6]), float(parts[7])
                    d, e, f = float(parts[8]), float(parts[9]), float(parts[10])
                    g, h, i = float(parts[11]), float(parts[12]), float(parts[13])

                    part_id = " ".join(parts[14:]) 

                    # 검증을 위한 매핑용 ID
                    verifier_id = f"{parts[14]}_{brick_counter}"
                    brick_counter += 1

                    # ---------------------------------------------------------
                    # [핵심 로직] 형상 정보 가져오기
                    # ---------------------------------------------------------
                    triangles = part_library.get_part_geometry(part_id)

                    if not triangles:
                        continue

                    # 삼각형 데이터를 Trimesh 객체로 변환
                    vertices = np.array([p for tri in triangles for p in tri])
                    faces = np.arange(len(vertices)).reshape(-1, 3)
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

                    # ---------------------------------------------------------
                    # [변환] LDraw 위치 및 회전 행렬 적용
                    # ---------------------------------------------------------
                    # 1. LDraw 4x4 변환 행렬 구성
                    ldraw_matrix = np.eye(4)
                    ldraw_matrix[0, :3] = [a, b, c]
                    ldraw_matrix[1, :3] = [d, e, f]
                    ldraw_matrix[2, :3] = [g, h, i]
                    ldraw_matrix[0, 3] = x
                    ldraw_matrix[1, 3] = y
                    ldraw_matrix[2, 3] = z

                    # 2. 좌표계 변환 행렬 (LDraw -> 모델 뷰어 표준)
                    scale = 1/20.0
                    conv_matrix = np.array([
                        [scale, 0,     0,     0],
                        [0,     0,     scale, 0],
                        [0,     -scale, 0,     0], 
                        [0,     0,     0,     1]
                    ])

                    # 변환 행렬 곱 적용
                    final_matrix = conv_matrix @ ldraw_matrix
                    mesh.apply_transform(final_matrix)

                    # ---------------------------------------------------------
                    # [색상 적용] 물리 검증 실패 시 빨간색 표시
                    # ---------------------------------------------------------
                    if verifier_id in failed_brick_ids:
                        print(f"!!! 일치하는 브릭 발견: 실패한 브릭 강조 - {verifier_id}")
                        final_color = [255, 0, 0, 255] # 빨강
                    else:
                        color_id = int(parts[1])
                        rgb_or_rgba = LDRAW_COLORS.get(color_id, (128, 128, 128)) # 기본 회색
                        if len(rgb_or_rgba) == 4:
                             final_color = list(rgb_or_rgba)
                        else:
                             final_color = [*rgb_or_rgba, 255]

                    mesh.visual.face_colors = final_color
                    scene.add_geometry(mesh)

                except Exception as e:
                    print(f"라인 파싱 오류: {line}\n{e}")
                    continue

    return scene

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LDR to Mesh 변환기 및 뷰어")
    parser.add_argument("file", help="LDR 파일 경로")
    parser.add_argument("--verify", action="store_true", help="물리 검증을 실행하고 불안정한 브릭을 강조 표시합니다.")
    
    args = parser.parse_args()
    
    target_ldr = args.file
    # 절대 경로가 아닌 경우 프로젝트 루트 기준으로 확인 시도
    if not os.path.exists(target_ldr):
         target_ldr = os.path.join(project_root, args.file)

    print(f"대상 LDR: {target_ldr}")
    print(f"검증 모드: {'켜짐' if args.verify else '꺼짐'}")

    # 변환 및 씬 생성 실행
    scene = ldr_to_mesh(target_ldr, verify=args.verify)

    if scene:
        print("성공! 씬을 로드했습니다.")
        print("3D 뷰어를 여는 중...")
        scene.show()
    else:
        print("씬 로드에 실패했습니다.")
