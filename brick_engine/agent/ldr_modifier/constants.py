# ============================================================================
# LDR 브릭 상수 테이블 (통합 버전)
# 중복 선언되던 상수들을 단일 파일로 관리
# ============================================================================

import logging

logger = logging.getLogger(__name__)

# LDU 상수 (브릭 병합용)
STUD_SPACING = 20.0  # X/Z 그리드 간격
BRICK_HEIGHT = 24.0  # 일반 브릭 높이
PLATE_HEIGHT = 8.0   # 플레이트 높이

# 1x1 브릭/플레이트 부품 번호
SMALL_BRICK_PARTS = {"3005.dat", "3024.dat"}

# 병합 대상 브릭 매핑 (스터드 수 → 파트 번호)
# 단순 병합 시에는 최대 1x4, 구조적 병합 시에는 1x8까지 사용
MERGE_TARGET_BRICKS = {
    2: "3004.dat",   # 1x2 브릭
    3: "3622.dat",   # 1x3 브릭
    4: "3010.dat",   # 1x4 브릭
    6: "3009.dat",   # 1x6 브릭
    8: "3008.dat",   # 1x8 브릭
}

# 플레이트 병합 대상
PLATE_MERGE_TARGETS = {
    2: "3023.dat",   # 1x2 Plate
    3: "3623.dat",   # 1x3 Plate
    4: "3710.dat",   # 1x4 Plate
    6: "3666.dat",   # 1x6 Plate
    8: "3460.dat",   # 1x8 Plate
}

# 브릭 크기별 스터드 수 (역매핑용)
BRICK_STUD_COUNT = {
    # Bricks
    "3005.dat": 1, "3004.dat": 2, "3622.dat": 3, "3010.dat": 4, 
    "3009.dat": 6, "3008.dat": 8, "3007.dat": 10, "3006.dat": 12,
    "3003.dat": 4,   # 2x2 Brick
    "3002.dat": 6,   # 2x3 Brick
    "3001.dat": 8,   # 2x4 Brick
    
    # Plates
    "3024.dat": 1, "3023.dat": 2, "3623.dat": 3, "3710.dat": 4,
    "3666.dat": 6, "3460.dat": 8, "3022.dat": 4, # 2x2 Plate
    "3021.dat": 6, # 2x3 Plate
    "3020.dat": 8, # 2x4 Plate
}

# 브릭 치수 정보 (Part → (Rows, Cols))
# 앞뒤(Z)가 Rows, 좌우(X)가 Cols (LDraw 기본 방향 기준)
BRICK_DIMENSIONS = {
    # 1xN Bricks
    "3005.dat": (1, 1), "3004.dat": (1, 2), "3622.dat": (1, 3), "3010.dat": (1, 4), 
    "3009.dat": (1, 6), "3008.dat": (1, 8), "3007.dat": (1, 10), "3006.dat": (1, 12),
    "3062.dat": (1, 1), "3062b.dat": (1, 1), # Round Bricks
    
    # 2xN Bricks
    "3003.dat": (2, 2), "3002.dat": (2, 3), "3001.dat": (2, 4), "3000.dat": (2, 2), 
    
    # 1xN Plates
    "3024.dat": (1, 1), "3023.dat": (1, 2), "3623.dat": (1, 3), "3710.dat": (1, 4),
    "3666.dat": (1, 6), "3460.dat": (1, 8), 
    "3070.dat": (1, 1), "3069.dat": (1, 2), # Tiles
    "3070b.dat": (1, 1), "3069b.dat": (1, 2),
    "4073.dat": (1, 1), "6141.dat": (1, 1), # Round Plates
    
    # 2xN Plates
    "3022.dat": (2, 2), "3021.dat": (2, 3), "3020.dat": (2, 4), "3795.dat": (2, 6)
}

# 2x2 병합 매핑 (입력 1x2 → 출력 2x2)
BRICK_2X2_MAPPING = {
    "3004.dat": "3003.dat", # 1x2 Brick → 2x2 Brick
    "3023.dat": "3022.dat"  # 1x2 Plate → 2x2 Plate
}

# 2xN 병합 매핑 (Target Size → Target Part)
TARGET_RECT_SIZES = {
    (2, 2): "3003.dat", # 2x2 Brick
    (2, 3): "3002.dat", # 2x3 Brick
    (2, 4): "3001.dat", # 2x4 Brick
}

PLATE_RECT_SIZES = {
    (2, 2): "3022.dat",
    (2, 3): "3021.dat",
    (2, 4): "3020.dat",
    (2, 6): "3795.dat"
}

# 플레이트 부품 집합 (판별용)
PLATE_PARTS = {
    "3024.dat", "3023.dat", "3623.dat", "3710.dat", 
    "3666.dat", "3460.dat", "3022.dat", "3021.dat", "3020.dat"
}


def is_plate(part_name: str) -> bool:
    """부품 번호를 기준으로 플레이트 여부 판별"""
    if part_name in PLATE_PARTS:
        return True
    if part_name.startswith("302"):  # 302x 계열은 대부분 플레이트
        return True
    return False
