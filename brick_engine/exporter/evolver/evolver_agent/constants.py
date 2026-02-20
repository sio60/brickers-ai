"""Agent Constants - 중앙화된 상수 정의

핵심 목적: 형태 진화 (브릭을 올바른 위치/방향으로 재배치)
지지대 추가는 최후의 수단
"""

# 제한값
MAX_REMOVAL_RATIO = 0.10  # 최대 10% 브릭 삭제 허용
FIXED_ITERATIONS = 1      # 1회만 실행, 나빠지면 롤백

# LDraw 단위
BRICK_HEIGHT = 24   # 브릭 높이 (LDU)
PLATE_HEIGHT = 8    # 플레이트 높이 (LDU)
LDU_PER_STUD = 20   # 1 스터드 = 20 LDU

# 대칭 분석
SYMMETRY_TOLERANCE = 20      # 위치 매칭 허용 오차 (LDU) - 1 stud
SYMMETRY_CENTER_MARGIN = 20  # 중앙 영역 판정 마진 - 1 stud
SPARSE_LAYER_THRESHOLD = 2   # 희소 레이어 기준
SKIP_SYMMETRY_TYPES = frozenset(['plant'])  # 비대칭이 자연스러운 모형 (animal 제거 - 다리는 대칭 필요)

# LLM 설정
LLM_MODEL = "gpt-4o-mini"    # 기본 LLM 모델 (gpt-4o, gpt-4o-mini, gpt-4-turbo 등)
LLM_TIMEOUT = 180.0          # LLM 호출 타임아웃 (초)
SUMMARIZE_TIMEOUT = 60.0     # 메모리 요약 타임아웃 (초)
LESSONS_THRESHOLD = 10       # lessons 요약 임계값

# 변이 에이전트 전략 (형태 진화 중심)
# SYMMETRY_FIX는 ADD_SUPPORT 직전, 후순위로 배치
MUTATION_STRATEGIES = [
    "RELOCATE",      # 브릭 위치 이동
    "ROTATE",        # 브릭 회전
    "REBUILD",       # 잘못된 부분 삭제 + 재배치
    "ADD_SUPPORT",   # 지지대 추가
    "SYMMETRY_FIX",  # 대칭 맞추기 (최후순위)
]

# 형태 개선 전략 (Vision 분석 기반) - 우선순위 순
# SYMMETRY_FIX는 후순위로 분리됨 (물리 문제 해결 후 처리)
SHAPE_STRATEGIES = [
    "RELOCATE",      # 1순위: 브릭 위치 이동
    "ROTATE",        # 2순위: 브릭 회전
    "REBUILD",       # 3순위: 잘못된 부분 삭제 + 재배치
]

# 물리적 안정화 전략 (floating 처리) - ADD_SUPPORT는 최후의 수단
PHYSICAL_STRATEGIES = [
    "SELECTIVE_REMOVE",  # 1순위: 불필요한 부유 브릭 삭제
    "BRIDGE",            # 2순위: 부유 클러스터 연결
    "ADD_SUPPORT",       # 3순위 (최후의 수단): 지지대 추가
    "ROLLBACK",          # 비상용: 원본 복원
]

# Vision 전략 (하위 호환용 alias)
VISION_STRATEGIES = SHAPE_STRATEGIES

# DEBATE 에이전트 및 가중치
AGENTS = ["FIDELITY", "CREATIVE"]

AGENT_WEIGHTS = {
    "FIDELITY": 0.5,  # 충실도 (원본 유지)
    "CREATIVE": 0.5,  # 창의성 (개선)
}
