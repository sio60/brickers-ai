# 이 파일은 LDraw (.ldr) 파일을 읽어 내부 데이터 구조(BrickPlan)로 변환하는 로더입니다.
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
    def __init__(self):
        pass

    def load_from_file(self, file_path: str) -> BrickPlan:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"LDR file not found: {file_path}")
            
        bricks = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('0'): # 주석(Comment)
                    continue
                    
                parts = line.split()
                if not parts: continue
                
                line_type = parts[0]
                
                # 라인 타입 1: 서브 파일 참조 (브릭)
                # 형식: 1 <색상> x y z a b c d e f g h <파일>
                if line_type == '1':
                    # 기본 정보 파싱
                    # color = parts[1] # 물리 연산에는 아직 사용되지 않음
                    
                    # LDraw 좌표: x, y, z
                    # LDraw Y는 수직(아래쪽이 양수). X, Z는 수평면.
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
                        # 기본값 1x1x1 사용 (20x24x20 LDU, 원점은 상단 중앙 부근)
                        dims = (-10.0, 0.0, -10.0, 10.0, 24.0, 10.0)

                    # part_library에서 6-튜플 언팩
                    min_x_ldu, min_y_ldu, min_z_ldu, max_x_ldu, max_y_ldu, max_z_ldu = dims

                    # 1. 부품의 로컬 공간에서 모서리 정의
                    # LDraw 축: X=오른쪽, Y=아래, Z=전방
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
                    
                    # 2. 모서리를 글로벌 LDraw 공간으로 변환 (회전 + 이동)
                    final_corners_ldraw = []
                    pos_vec = np.array([ldraw_x, ldraw_y, ldraw_z])
                    
                    for c in corners:
                         # 회전 적용 (행렬 * 벡터)
                         rc = rot_matrix.dot(c)
                         # 이동 적용
                         final_corners_ldraw.append(rc + pos_vec)

                    # 3. 글로벌 범위(Extents) 찾기
                    g_min_x = min(c[0] for c in final_corners_ldraw)
                    g_max_x = max(c[0] for c in final_corners_ldraw)
                    g_min_y = min(c[1] for c in final_corners_ldraw)
                    g_max_y = max(c[1] for c in final_corners_ldraw)
                    g_min_z = min(c[2] for c in final_corners_ldraw)
                    g_max_z = max(c[2] for c in final_corners_ldraw)
                    
                    # 4. 모델 시스템으로 변환
                    # LDraw X -> 모델 X (1/20)
                    # LDraw Z -> 모델 Y (깊이) (1/20)
                    # LDraw Y -> 모델 Z (높이) (-1/24)
                    
                    model_min_x = g_min_x / 20.0
                    model_max_x = g_max_x / 20.0
                    
                    model_min_y = g_min_z / 20.0
                    model_max_y = g_max_z / 20.0
                    
                    model_min_z = -g_max_y / 24.0 # 반전됨
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
                        matrix=rot_matrix,
                        origin=np.array([ldraw_x, ldraw_y, ldraw_z]),
                        part_file=part_id
                    )
                    bricks.append(brick)
        
        # Z-정규화: 모델이 지면(Z=0)에 놓이도록 이동
        if bricks:
            min_model_z = min(b.z for b in bricks)
            if min_model_z > 0.001 or min_model_z < -0.001:
                print(f"Normalizing Z: Shifting model by {-min_model_z:.2f} units.")
                for b in bricks:
                    b.z -= min_model_z
                    
        return BrickPlan(bricks)
