"""Agent Constants - 중앙화된 상수 정의

핵심 목적: 형태 진화 (브릭을 올바른 위치/방향으로 재배치)
지지대 추가는 최후의 수단
"""
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from .config import get_config

# ============================================================================
# 모듈 레벨 경로 설정
# ============================================================================
_PROJECT_ROOT = Path(__file__).resolve().parents[4]  # brickers-ai
_PHYS_PATH = _PROJECT_ROOT / "physical_verification"
_AGENT_PATH = _PROJECT_ROOT / "brick_engine" / "agent"
_EXPORTER_PATH = Path(__file__).resolve().parents[1]  # exporter

for _p in [_PROJECT_ROOT, _PHYS_PATH, _AGENT_PATH, _EXPORTER_PATH]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ============================================================================
# 제한값
# ============================================================================
MAX_REMOVAL_RATIO = 0.10  # 최대 10% 브릭 삭제 허용
FIXED_ITERATIONS = 1      # 1회만 실행, 나빠지면 롤백

# ============================================================================
# LDraw 단위
# ============================================================================
BRICK_HEIGHT = 24   # 브릭 높이 (LDU)
PLATE_HEIGHT = 8    # 플레이트 높이 (LDU)
LDU_PER_STUD = 20   # 1 스터드 = 20 LDU
CELL_SIZE = LDU_PER_STUD  # 점유 맵 셀 크기

# ============================================================================
# 검증 관련
# ============================================================================
GROUND_TOLERANCE = 2  # 바닥 연결 판정 허용 오차 (LDU)
MAX_SUPPORT_PER_BRICK = 20  # 브릭당 최대 지지대 수
DEFAULT_MAX_REMOVE = 5  # 기본 최대 삭제 개수

# ============================================================================
# 파츠 관련
# ============================================================================
KIDS_MIN_PART_SIZE = "3003"  # 2x2 이상만 허용
ADULT_DEFAULT_PART = "3005"  # 1x1 브릭

# 플레이트 파츠 ID (높이 8 LDU)
PLATE_PART_IDS = frozenset([
    "3024", "3023", "3022", "3020", "3021", "3710", "3666"
])

# ============================================================================
# 배치 검증
# ============================================================================
DEFAULT_SUPPORT_RATIO = 0.3  # 최소 30% 지지 필요
OPTIMAL_BOND_MIN = 0.3  # 최적 겹침 범위
OPTIMAL_BOND_MAX = 0.7

# ============================================================================
# 대칭 분석
# ============================================================================
SYMMETRY_TOLERANCE = 20      # 위치 매칭 허용 오차 (LDU) - 1 stud
SYMMETRY_CENTER_MARGIN = 20  # 중앙 영역 판정 마진 - 1 stud
SPARSE_LAYER_THRESHOLD = 2   # 희소 레이어 기준
SKIP_SYMMETRY_TYPES = frozenset(['plant'])  # 비대칭이 자연스러운 모형

# ============================================================================
# LLM 설정
# ============================================================================
LLM_MODEL = "gpt-4o"         # 기본 LLM 모델
LLM_TIMEOUT = 180.0          # LLM 호출 타임아웃 (초)
SUMMARIZE_TIMEOUT = 60.0     # 메모리 요약 타임아웃 (초)
LESSONS_THRESHOLD = 10       # lessons 요약 임계값

# ============================================================================
# 전략 리스트
# ============================================================================
MUTATION_STRATEGIES = [
    "RELOCATE",      # 브릭 위치 이동
    "ROTATE",        # 브릭 회전
    "ADD_SUPPORT",   # 지지대 추가
    "SYMMETRY_FIX",  # 대칭 맞추기 (최후순위)
]

SHAPE_STRATEGIES = [
    "RELOCATE",      # 1순위: 브릭 위치 이동
    "ROTATE",        # 2순위: 브릭 회전
]

PHYSICAL_STRATEGIES = [
    "SELECTIVE_REMOVE",  # 1순위: 불필요한 부유 브릭 삭제
    "BRIDGE",            # 2순위: 부유 클러스터 연결
    "ADD_SUPPORT",       # 3순위 (최후의 수단): 지지대 추가
    "ROLLBACK",          # 비상용: 원본 복원
]

VISION_STRATEGIES = SHAPE_STRATEGIES  # 하위 호환용 alias

# ============================================================================
# DEBATE 에이전트 및 가중치
# ============================================================================
AGENTS = ["FIDELITY", "CREATIVE"]

AGENT_WEIGHTS = {
    "FIDELITY": 0.5,  # 충실도 (원본 유지)
    "CREATIVE": 0.5,  # 창의성 (개선)
}


# ============================================================================
# 색상 / 파츠 클래스
# ============================================================================
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


# ============================================================================
# 헬퍼 함수
# ============================================================================
def _get_parts_db() -> Dict[str, Any]:
    """파츠 DB 가져오기 (Configuration 사용)"""
    config = get_config()
    return config.parts_db if config.is_initialized else {}


def _get_exporter_dir() -> Optional[Path]:
    """exporter 디렉토리 경로 가져오기 (Configuration 사용)"""
    config = get_config()
    return config.exporter_dir
