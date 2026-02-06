# Evolver Agent Constants

# --- 반복 제어 ---
# 최대 반복 횟수 (무한 루프 방지)
FIXED_ITERATIONS = 3

# --- LLM 설정 ---
LLM_MODEL = "gpt-4o"
LLM_TIMEOUT = 60

# --- 제약 조건 ---
# 원본 브릭 수 대비 최대 제거 허용 비율 (0.2 = 20%)
MAX_REMOVAL_RATIO = 0.2

# --- 대칭 분석 예외 ---
# 비대칭이 자연스러운 모델 타입 목록
SKIP_SYMMETRY_TYPES = [
    "animal", 
    "plant", 
    "organic", 
    "character", 
    "asymmetric",
    "irregular"
]

# --- Reflect Node ---
# 교훈 반영 임계값
LESSONS_THRESHOLD = 0.7
# 요약 타임아웃
SUMMARIZE_TIMEOUT = 30

# --- Debate Node ---
# 에이전트별 의사결정 가중치
AGENT_WEIGHTS = {
    "stability": 0.4,
    "aesthetics": 0.4,
    "efficiency": 0.2
}
