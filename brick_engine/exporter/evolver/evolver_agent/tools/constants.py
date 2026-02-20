"""상수 및 설정 클래스 정의"""
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from ..config import get_config

# ============================================================================
# 모듈 레벨 경로 설정
# ============================================================================
_PROJECT_ROOT = Path(__file__).resolve().parents[5]  # brickers-ai
_PHYS_PATH = _PROJECT_ROOT / "physical_verification"
_AGENT_PATH = _PROJECT_ROOT / "brick_engine" / "agent"
_EXPORTER_PATH = Path(__file__).resolve().parents[2]  # exporter

for _p in [_PROJECT_ROOT, _PHYS_PATH, _AGENT_PATH, _EXPORTER_PATH]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# =============================================================================
# 상수 정의
# =============================================================================

BRICK_HEIGHT = 24  # LDU
PLATE_HEIGHT = 8   # LDU
LDU_PER_STUD = 20  # 1 스터드 = 20 LDU
CELL_SIZE = LDU_PER_STUD  # 점유 맵 셀 크기

# 검증 관련 상수
GROUND_TOLERANCE = 2  # 바닥 연결 판정 허용 오차 (LDU)
MAX_SUPPORT_PER_BRICK = 20  # 브릭당 최대 지지대 수
DEFAULT_MAX_REMOVE = 5  # 기본 최대 삭제 개수

# Kids Mode 안전 기준
KIDS_MIN_PART_SIZE = "3003"  # 2x2 이상만 허용
ADULT_DEFAULT_PART = "3005"  # 1x1 브릭

# 배치 검증 상수
DEFAULT_SUPPORT_RATIO = 0.3  # 최소 30% 지지 필요
OPTIMAL_BOND_MIN = 0.3  # 최적 겹침 범위
OPTIMAL_BOND_MAX = 0.7

# 대칭 분석 상수
SYMMETRY_TOLERANCE = 5  # 위치 매칭 허용 오차 (LDU)
SYMMETRY_CENTER_MARGIN = 5  # 중앙 영역 판정 마진
SPARSE_LAYER_THRESHOLD = 2  # 이 개수 이하면 희소 레이어

# 플레이트 파츠 ID (높이 8 LDU)
PLATE_PART_IDS = frozenset([
    "3024", "3023", "3022", "3020", "3021", "3710", "3666"
])


class LDrawColor:
    """LDraw 색상 코드"""
    BLACK = 0
    BLUE = 1
    GREEN = 2
    RED = 4
    BROWN = 6
    LIGHT_GRAY = 7
    DARK_GRAY = 8
    YELLOW = 14
    WHITE = 15
    LIGHT_BLUISH_GRAY = 71
    DARK_BLUISH_GRAY = 72


class SupportParts:
    """지지대용 파츠 ID"""
    BRICK_1X1 = '3005'
    BRICK_1X2 = '3004'
    BRICK_1X4 = '3010'
    BRICK_2X2 = '3003'
    BRICK_2X4 = '3001'
    PLATE_1X1 = '3024'
    PLATE_1X2 = '3023'
    PLATE_2X2 = '3022'
    PLATE_2X4 = '3020'


def _get_parts_db() -> Dict[str, Any]:
    """파츠 DB 가져오기 (Configuration 사용)"""
    config = get_config()
    return config.parts_db if config.is_initialized else {}


def _get_exporter_dir() -> Optional[Path]:
    """exporter 디렉토리 경로 가져오기 (Configuration 사용)"""
    config = get_config()
    return config.exporter_dir
