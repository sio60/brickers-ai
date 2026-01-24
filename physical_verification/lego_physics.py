# 이 파일은 레고 브릭의 물리적 특성(스터드/튜브 연결, 질량, 치수 등)을 정의하고 계산하는 모듈입니다.
"""
LEGO 물리 모듈 - 스터드/튜브 연결 로직

레고 브릭은 스터드(Stud, 상단 돌기)와 튜브(Tube, 하단 홈)의 
마찰력(Clutch Power)으로 결합됩니다.

LDU 단위:
- X/Z 그리드: 20 LDU (1 스터드 간격)
- Y (높이): 24 LDU (일반 브릭), 8 LDU (판 브릭)
- 스터드 높이: 약 4 LDU
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
import re

# LDU 상수
STUD_SPACING = 20.0  # X/Z 그리드 간격
BRICK_HEIGHT = 24.0  # 일반 브릭 높이
PLATE_HEIGHT = 8.0   # 플레이트 높이
STUD_HEIGHT = 4.0    # 스터드 돌출 높이

# 브릭 크기 데이터베이스 (part_id -> (studs_x, studs_z, is_plate))
# studs_x * studs_z = 스터드 개수
BRICK_SIZES = {
    # 기본 브릭 (높이 = 24 LDU)
    # LDraw에서 X는 일반적으로 긴 축입니다!
    "3001.dat": (4, 2, False),  # 2x4 브릭 (X축 4개, Z축 2개 스터드)
    "3002.dat": (3, 2, False),  # 2x3 브릭
    "3003.dat": (2, 2, False),  # 2x2 브릭
    "3004.dat": (2, 1, False),  # 1x2 브릭
    "3005.dat": (1, 1, False),  # 1x1 브릭
    "3006.dat": (10, 2, False), # 2x10 브릭
    "3007.dat": (8, 2, False),  # 2x8 브릭
    "3008.dat": (8, 1, False),  # 1x8 브릭
    "3009.dat": (6, 1, False),  # 1x6 브릭
    "3010.dat": (4, 1, False),  # 1x4 브릭
    "2456.dat": (6, 2, False),  # 2x6 브릭
    
    # 플레이트 (높이 = 8 LDU)
    "3020.dat": (4, 2, True),   # 2x4 플레이트
    "3021.dat": (3, 2, True),   # 2x3 플레이트
    "3022.dat": (2, 2, True),   # 2x2 플레이트
    "3023.dat": (2, 1, True),   # 1x2 플레이트
    "3024.dat": (1, 1, True),   # 1x1 플레이트
    "3795.dat": (6, 2, True),   # 2x6 플레이트
    "3034.dat": (8, 2, True),   # 2x8 플레이트
    "3832.dat": (10, 2, True),  # 2x10 플레이트
    
    # 추가된 부품 (Stair 모델용)
    "30072.dat": (12, 6, False), # 12x6 브릭 (가정) - 바닥/플랫폼 역할 추정
    "2465.dat": (16, 1, False),  # 1x16 브릭
}

# 실제 질량 계산
# 참고: 2x4 브릭 = 19200 LDU³ 부피, 무게 2.3그램
# 밀도 상수: 2.3 / 19200 ≈ 0.00012 g/LDU³
BRICK_DENSITY = 2.3 / (40 * 20 * 24)  # g/LDU³ (약 0.00012)

def get_brick_mass_kg(part_file: str) -> float:
    """
    브릭의 실제 질량을 킬로그램(kg) 단위로 반환합니다.
    PyBullet은 SI 단위(kg, 미터, 초)를 사용합니다.
    
    예시:
    - 2x4 브릭: ~2.3g = 0.0023 kg
    - 1x1 브릭: ~0.4g = 0.0004 kg
    """
    studs_x, studs_z, is_plate = get_brick_studs_count(part_file)
    height = PLATE_HEIGHT if is_plate else BRICK_HEIGHT
    
    # 부피 (LDU³)
    volume = (studs_x * STUD_SPACING) * (studs_z * STUD_SPACING) * height
    
    # 그램(g) 단위 질량 계산 후 kg 변환
    mass_grams = volume * BRICK_DENSITY
    return mass_grams / 1000.0  # kg 단위로 변환

def get_brick_studs_count(part_file: str) -> Tuple[int, int, bool]:
    """
    주어진 부품에 대한 (studs_x, studs_z, is_plate) 정보를 반환합니다.
    알려지지 않은 부품의 경우 part ID 파싱을 시도합니다.
    """
    part_file = part_file.lower().strip()
    
    if part_file in BRICK_SIZES:
        return BRICK_SIZES[part_file]
    
    # 기본값
    # return (2, 4, False)

    # 2. DB / Part Library에서 치수 가져오기
    try:
        import part_library
        bbox = part_library.get_part_dims(part_file)
        
        if bbox:
            min_x, min_y, min_z, max_x, max_y, max_z = bbox
            width = max_x - min_x
            height = max_y - min_y
            depth = max_z - min_z
            
            # LDU -> Studs 변환 (20 LDU = 1 Stud)
            studs_x = max(1, round(width / 20.0))
            studs_z = max(1, round(depth / 20.0))
            
            # 높이 확인 (Plate vs Brick)
            # Plate: ~8, Brick: ~24
            is_plate = height < 20.0
            
            # 캐싱 (선택 사항)
            BRICK_SIZES[part_file] = (studs_x, studs_z, is_plate)
            
            return (studs_x, studs_z, is_plate)
            
    except Exception as e:
        print(f"[WARN] DB Lookup failed for {part_file}: {e}")

    # 3. 정말 아무것도 안 될 때의 최후의 수단 (Fallback)
    match = re.match(r'^(\d+)\.dat$', part_file)
    if match:
         return (2, 4, False) # 2x4 Brick
         
    return (2, 4, False)


def get_stud_positions_local(part_file: str) -> List[Tuple[float, float, float]]:
    """
    로컬 좌표계(부품 원점 0,0,0)에서의 스터드 중심 위치 목록을 반환합니다.
    스터드는 브릭의 상단(LDraw에서 Y=0, Y축이 아래를 향하므로)에 위치합니다.
    
    LDraw 좌표계: Y는 수직(아래쪽이 양수)
    따라서 스터드는 Y=0(윗면), 튜브는 Y=height(아랫면)에 있습니다.
    """
    studs_x, studs_z, is_plate = get_brick_studs_count(part_file)
    
    # 원점을 중심으로 스터드 위치 계산
    # 2x4 브릭의 경우: X = [-30, -10, 10, 30] (실제 계산 필요)
    # 실제: X 범위 = studs_x * 20 LDU, 0을 중심으로 배치
    # X축 2개 스터드: X = [-10, 10]
    # Z축 4개 스터드: Z = [-30, -10, 10, 30]
    
    positions = []
    
    # 시작 위치 계산 (중앙 정렬)
    start_x = -((studs_x - 1) * STUD_SPACING) / 2.0
    start_z = -((studs_z - 1) * STUD_SPACING) / 2.0
    
    for i in range(studs_x):
        for j in range(studs_z):
            x = start_x + i * STUD_SPACING
            z = start_z + j * STUD_SPACING
            y = 0.0  # LDraw 상단 (Y-down)
            positions.append((x, y, z))
    
    return positions


def get_tube_positions_local(part_file: str) -> List[Tuple[float, float, float]]:
    """
    로컬 좌표계에서의 튜브 위치 목록을 반환합니다.
    튜브는 브릭의 하단(LDraw에서 Y=height)에 위치합니다.
    """
    studs_x, studs_z, is_plate = get_brick_studs_count(part_file)
    height = PLATE_HEIGHT if is_plate else BRICK_HEIGHT
    
    # 튜브는 스터드 위치와 대칭되지만 Y=height에 위치
    positions = []
    start_x = -((studs_x - 1) * STUD_SPACING) / 2.0
    start_z = -((studs_z - 1) * STUD_SPACING) / 2.0
    
    for i in range(studs_x):
        for j in range(studs_z):
            x = start_x + i * STUD_SPACING
            z = start_z + j * STUD_SPACING
            y = height  # LDraw 하단 (Y-down)
            positions.append((x, y, z))
    
    return positions


def transform_positions(positions: List[Tuple[float, float, float]], 
                        matrix: np.ndarray, 
                        origin: np.ndarray) -> List[np.ndarray]:
    """
    로컬 위치를 글로벌 LDraw 좌표로 변환합니다.
    
    Args:
        positions: 로컬 좌표 (x, y, z) 목록
        matrix: 3x3 회전 행렬
        origin: [x, y, z] 이동 벡터
    
    Returns:
        글로벌 좌표 [x, y, z] 목록
    """
    result = []
    for pos in positions:
        local = np.array(pos)
        rotated = matrix @ local
        global_pos = rotated + origin
        result.append(global_pos)
    return result


def check_stud_tube_connection(brick_a, brick_b, tolerance: float = 5.0) -> bool:
    """
    단순화된 연결성 검사:
    바운딩 박스를 기준으로 브릭 A가 브릭 B 바로 위에 있는지(또는 그 반대) 확인합니다.
    
    규칙:
    1. Y축 차이가 브릭 높이와 일치 (적층됨)
    2. X/Z 바운딩 박스가 겹침 (수평적으로 맞닿음)
    """
    if brick_a.origin is None or brick_b.origin is None:
        return False
    
    # 1. 글로벌 바운딩 박스 가져오기 (스터드/치수로부터 근사값 계산)
    # 원점은 상단 중앙 (LDraw 표준)
    # 높이: 24 (브릭) 또는 8 (플레이트)
    # 가로/세로는 BRICK_SIZES에서 가져옴
    
    def get_bbox(brick):
        studs_x, studs_z, is_plate = get_brick_studs_count(brick.part_file or "3001.dat")
        height = PLATE_HEIGHT if is_plate else BRICK_HEIGHT
        
        # 원점은 상단 중앙. 
        # LDraw Y-down: 상단=Y, 하단=Y+height
        y_top = brick.origin[1]
        y_bottom = brick.origin[1] + height
        
        # 로컬 가로/깊이
        w = studs_x * STUD_SPACING
        d = studs_z * STUD_SPACING
        
        # 글로벌 경계 (회전 0/90 단순화 가정)
        # 단순화를 위해 중심 거리를 사용하여 겹침 확인
        return {
            "y_top": y_top,
            "y_bottom": y_bottom,
            "xz_center": brick.origin[[0, 2]],
            "xz_dims": np.array([w, d])
        }

    bb_a = get_bbox(brick_a)
    bb_b = get_bbox(brick_b)
    
    # 수직 적층 확인
    # 케이스 1: A가 B 위에 있음 (A.bottom ≈ B.top)
    a_on_b = abs(bb_a["y_bottom"] - bb_b["y_top"]) < tolerance
    # 케이스 2: B가 A 위에 있음 (B.bottom ≈ A.top)
    b_on_a = abs(bb_b["y_bottom"] - bb_a["y_top"]) < tolerance
    
    if not (a_on_b or b_on_a):
        return False
        
    # 수평 겹침 확인
    # [단순 접근]: 중심 간 거리가 치수의 합보다 작은지 확인
    
    dist_x = abs(bb_a["xz_center"][0] - bb_b["xz_center"][0])
    dist_z = abs(bb_a["xz_center"][1] - bb_b["xz_center"][1])
    
    # 겹침 임계값: (width_a + width_b) / 2
    # 회전을 정확히 파악하기 어려우므로 최대 치수를 사용하여 안전하게(관대하게) 처리
    max_dim_a = max(bb_a["xz_dims"])
    max_dim_b = max(bb_b["xz_dims"])
    
    # 관대하게 처리: X와 Z 모두 충분히 가까우면 겹침으로 간주
    limit_x = (bb_a["xz_dims"][0] + bb_b["xz_dims"][0])/2.0
    limit_z = (bb_a["xz_dims"][1] + bb_b["xz_dims"][1])/2.0
    
    is_overlapping = (dist_x < limit_x - 0.1) and (dist_z < limit_z - 0.1)
    
    return is_overlapping


def find_all_connections(bricks: list) -> List[Tuple[str, str]]:
    """
    브릭 간의 모든 스터드-튜브 연결을 찾습니다.
    
    Args:
        bricks: Brick 객체 리스트
    
    Returns:
        연결된 쌍의 (brick_id_a, brick_id_b) 튜플 리스트
    """
    connections = []
    
    print(f"[DEBUG] {len(bricks)}개 브릭의 연결 상태를 확인 중...")
    
    for i, brick_a in enumerate(bricks):
        for brick_b in bricks[i+1:]:
            connected, reason = check_stud_tube_connection_debug(brick_a, brick_b)
            if connected:
                # 개별 연결 로그 비활성화 (너무 많으면 주석 해제해서 디버그)
                # print(f"[DEBUG] 연결됨: {brick_a.id} <-> {brick_b.id} | {reason}")
                connections.append((brick_a.id, brick_b.id))
            else:
                # 연결되지 않은 이유 디버깅 출력 (필요 시 주석 해제)
                pass
    
    print(f"[DEBUG] 총 연결 발견: {len(connections)}개")
    return connections


def check_stud_tube_connection_debug(brick_a, brick_b, tolerance: float = 5.0) -> Tuple[bool, str]:
    """check_stud_tube_connection의 디버그 버전. 연결 이유를 반환합니다."""
    if brick_a.origin is None or brick_b.origin is None:
        return False, ""
    if brick_a.matrix is None or brick_b.matrix is None:
        return False, ""
    if brick_a.part_file is None or brick_b.part_file is None:
        return False, ""
    
    # 글로벌 좌표계에서 스터드 및 튜브 위치 계산
    studs_a = transform_positions(
        get_stud_positions_local(brick_a.part_file),
        brick_a.matrix, brick_a.origin
    )
    tubes_a = transform_positions(
        get_tube_positions_local(brick_a.part_file),
        brick_a.matrix, brick_a.origin
    )
    
    studs_b = transform_positions(
        get_stud_positions_local(brick_b.part_file),
        brick_b.matrix, brick_b.origin
    )
    tubes_b = transform_positions(
        get_tube_positions_local(brick_b.part_file),
        brick_b.matrix, brick_b.origin
    )
    
    # A의 스터드가 B의 튜브와 연결되는지 확인 (정렬 시 y_diff는 거의 0이어야 함)
    for stud in studs_a:
        for tube in tubes_b:
            y_diff = abs(tube[1] - stud[1])
            xz_dist = np.sqrt((stud[0] - tube[0])**2 + (stud[2] - tube[2])**2)
            
            if y_diff < tolerance and xz_dist < tolerance:
                return True, f"A_stud({stud[0]:.1f},{stud[1]:.1f},{stud[2]:.1f})->B_tube({tube[0]:.1f},{tube[1]:.1f},{tube[2]:.1f}) y_diff={y_diff:.1f} xz_dist={xz_dist:.1f}"
    
    # B의 스터드가 A의 튜브와 연결되는지 확인
    for stud in studs_b:
        for tube in tubes_a:
            y_diff = abs(tube[1] - stud[1])
            xz_dist = np.sqrt((stud[0] - tube[0])**2 + (stud[2] - tube[2])**2)
            
            if y_diff < tolerance and xz_dist < tolerance:
                return True, f"B_stud({stud[0]:.1f},{stud[1]:.1f},{stud[2]:.1f})->A_tube({tube[0]:.1f},{tube[1]:.1f},{tube[2]:.1f}) y_diff={y_diff:.1f} xz_dist={xz_dist:.1f}"
    
    return False, ""


def find_floating_bricks(bricks: list, ground_y: float = 0.0, tolerance: float = 5.0) -> List[str]:
    """
    어떤 것과도 연결되지 않고 지면에도 닿지 않은 공중 부양 브릭을 찾습니다.
    
    Args:
        bricks: Brick 객체 리스트
        ground_y: 지면의 Y 좌표 (LDraw Y-down이므로 최대 Y가 지면)
        tolerance: 지면 접촉 허용 오차
    
    Returns:
        공중 부양 브릭 ID 리스트
    """
    if not bricks:
        return []
    
    # 연결 상태 확인
    connections = find_all_connections(bricks)
    connected_bricks = set()
    for a, b in connections:
        connected_bricks.add(a)
        connected_bricks.add(b)
    
    # 지면 레벨 브릭 찾기 (최대 Y 값 = 브릭 바닥)
    ground_bricks = set()
    max_y = max(b.origin[1] for b in bricks if b.origin is not None)
    
    for brick in bricks:
        if brick.origin is not None:
            # 바닥 Y 좌표 계산 (LDraw Y-down: 원점 Y + 높이)
            _, _, is_plate = get_brick_studs_count(brick.part_file or "3001.dat")
            height = PLATE_HEIGHT if is_plate else BRICK_HEIGHT
            bottom_y = brick.origin[1] + height
            
            # 지면에 닿아있는지 확인 (최대 Y 허용 오차 내)
            if abs(bottom_y - max_y) < tolerance:
                ground_bricks.add(brick.id)
    
    # 지면에서부터 연결성 전파
    visited = set(ground_bricks)
    to_visit = list(ground_bricks)
    
    while to_visit:
        current = to_visit.pop()
        for a, b in connections:
            if a == current and b not in visited:
                visited.add(b)
                to_visit.append(b)
            elif b == current and a not in visited:
                visited.add(a)
                to_visit.append(a)
    
    # 방문되지 않음 = 공중 부양
    floating = [b.id for b in bricks if b.id not in visited]
    return floating
