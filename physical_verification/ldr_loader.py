# ============================================================================
# LDR 파일 로더 모듈
# 이 파일은 LDraw(.ldr) 형식의 레고 모델 파일을 파싱하여 
# Brick 객체 리스트로 변환하는 역할을 합니다.
# LDraw 좌표계를 모델 좌표계로 변환하고, 회전 행렬을 적용하여
# 각 브릭의 월드 좌표와 크기를 계산합니다.
# ============================================================================

import os
import numpy as np
import re
try:
    from .models import Brick, BrickPlan
    from .part_library import get_part_dims
except ImportError:
    from models import Brick, BrickPlan
    from part_library import get_part_dims

class LdrLoader:
    """LDR 파일을 로드하고 파싱하는 클래스"""
    
    def __init__(self):
        pass

    def load_from_file(self, file_path: str) -> BrickPlan:
        """
        LDR 파일을 읽어서 BrickPlan 객체로 변환합니다.
        
        Args:
            file_path: 로드할 LDR 파일 경로
            
        Returns:
            BrickPlan: 파싱된 브릭 정보를 담은 객체
            
        Raises:
            FileNotFoundError: 파일을 찾을 수 없는 경우
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"LDR file not found: {file_path}")
            
        bricks = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('0'):  # 주석 라인
                    continue
                    
                parts = line.split()
                if not parts: continue
                
                line_type = parts[0]
                
                # 라인 타입 1: 서브파일 참조 (브릭)
                # 형식: 1 <색상> x y z a b c d e f g h <파일>
                if line_type == '1':
                    # 기본 정보 파싱
                    # color = parts[1]  # 아직 물리 검증에서 사용하지 않음
                    
                    # LDraw 좌표: x, y, z
                    # LDraw Y는 수직 (아래가 양수). X, Z는 수평면.
                    ldraw_x = float(parts[2])
                    ldraw_y = float(parts[3])
                    ldraw_z = float(parts[4])
                    
                    # 회전 행렬 (a b c / d e f / g h i)
                    # a=5, b=6, c=7, d=8, e=9, f=10, g=11, h=12, i=13
                    rot_matrix = np.array([
                        [float(parts[5]), float(parts[6]), float(parts[7])],
                        [float(parts[8]), float(parts[9]), float(parts[10])],
                        [float(parts[11]), float(parts[12]), float(parts[13])]
                    ])
                    
                    part_id = parts[14]
                    
                    dims = get_part_dims(part_id)
                    if not dims:
                        # 기본값 1x1x1 사용 (20x24x20 LDU, 원점은 상단 중앙 근처)
                        dims = (-10.0, 0.0, -10.0, 10.0, 24.0, 10.0)

                    # part_library에서 6-튜플 언팩
                    min_x_ldu, min_y_ldu, min_z_ldu, max_x_ldu, max_y_ldu, max_z_ldu = dims

                    # 1. 부품 로컬 공간에서 코너 정의
                    # LDraw 축: X=오른쪽, Y=아래, Z=앞
                    corners = [
                        np.array([min_x_ldu, min_y_ldu, min_z_ldu]),
                        np.array([min_x_ldu, min_y_ldu, max_z_ldu]),
                        np.array([min_x_ldu, max_y_ldu, min_z_ldu]),
                        np.array([min_x_ldu, max_y_ldu, max_z_ldu]),
                        np.array([max_x_ldu, min_y_ldu, min_z_ldu]),
                        np.array([max_x_ldu, min_y_ldu, max_z_ldu]),
                        np.array([max_x_ldu, max_y_ldu, min_z_ldu]),
                        np.array([max_x_ldu, max_y_ldu, max_z_ldu]),
                    ]
                    
                    # 2. 코너를 글로벌 LDraw 공간으로 변환 (회전 + 이동)
                    final_corners_ldraw = []
                    pos_vec = np.array([ldraw_x, ldraw_y, ldraw_z])
                    
                    for c in corners:
                         # 회전 적용 (행렬 * 벡터)
                         rc = rot_matrix.dot(c)
                         # 이동 적용
                         final_corners_ldraw.append(rc + pos_vec)

                    # 3. 글로벌 경계 찾기
                    g_min_x = min(c[0] for c in final_corners_ldraw)
                    g_max_x = max(c[0] for c in final_corners_ldraw)
                    g_min_y = min(c[1] for c in final_corners_ldraw)
                    g_max_y = max(c[1] for c in final_corners_ldraw)
                    g_min_z = min(c[2] for c in final_corners_ldraw)
                    g_max_z = max(c[2] for c in final_corners_ldraw)
                    
                    # 4. 모델 좌표계로 변환
                    # LDraw X -> 모델 X (1/20)
                    # LDraw Z -> 모델 Y (깊이) (1/20)
                    # LDraw Y -> 모델 Z (높이) (-1/24)
                    
                    model_min_x = g_min_x / 20.0
                    model_max_x = g_max_x / 20.0
                    
                    model_min_y = g_min_z / 20.0
                    model_max_y = g_max_z / 20.0
                    
                    model_min_z = -g_max_y / 24.0  # 반전
                    model_max_z = -g_min_y / 24.0
                    
                    final_width = model_max_x - model_min_x
                    final_depth = model_max_y - model_min_y
                    final_height = model_max_z - model_min_z
                    
                    brick = Brick(
                        id=f"{part_id}_{len(bricks)}",
                        x=model_min_x,
                        y=model_min_y,
                        z=model_min_z,
                        width=final_width,
                        depth=final_depth,
                        height=final_height,
                        origin=(ldraw_x, ldraw_y, ldraw_z),
                        part_file=part_id,
                        matrix=rot_matrix
                    )
                    bricks.append(brick)
        
        # Z 정규화: 모델이 지면(Z=0)에 놓이도록 이동
        if bricks:
            min_model_z = min(b.z for b in bricks)
            if min_model_z > 0.001 or min_model_z < -0.001:
                print(f"Z 정규화: 모델을 {-min_model_z:.2f} 단위만큼 이동합니다.")
                for b in bricks:
                    b.z -= min_model_z
                    
        return BrickPlan(bricks)
