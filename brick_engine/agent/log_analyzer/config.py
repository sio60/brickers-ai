"""
Log Analyzer — 설정 및 상수
============================
도구 매핑, 프로젝트 모듈 화이트리스트, 실행 제한값 등.
"""

from ..agent_tools import (
    ReadFileSnippet,
    CheckDBStatus,
    CheckSystemHealth,
    CheckSQSStatus,
    execute_read_file,
    execute_check_db,
    execute_check_system,
    execute_check_sqs,
)

# ============================================================
# 프로젝트 파일 범위 (Import Crawler 화이트리스트)
# ============================================================
PROJECT_MODULES = {
    'route', 'service', 'brick_engine', 'chat', 'vectordb',
    'physical_verification', 'config', 'db', 'app',
}

EXCLUDE_FILENAMES = {
    'test_', 'debug_', 'check_', 'verify_', 'make_', 'inspect_',
    'ingest_', 'seed', 'smoke', 'output_test',
}

# ============================================================
# Tool Calling 설정
# ============================================================

# LLM bind_tools에 전달할 Pydantic 스키마 목록
TOOL_SCHEMAS = [ReadFileSnippet, CheckDBStatus, CheckSystemHealth, CheckSQSStatus]

# Tool 이름 → 실행 함수 매핑
TOOL_EXECUTOR_MAP = {
    "ReadFileSnippet": execute_read_file,
    "CheckDBStatus": execute_check_db,
    "CheckSystemHealth": execute_check_system,
    "CheckSQSStatus": execute_check_sqs,
}

# ============================================================
# 실행 제한
# ============================================================
MAX_INVESTIGATION_ROUNDS = 5

# Docker 로그 수집 설정
DOCKER_LOG_TAIL_LINES = 2000

# Job ID 탐지 패턴 (V1에서 이식)
JOB_FAILURE_PATTERN = r"요청 실패! \| jobId=([a-f0-9-]+)"
JOB_START_PATTERN = r"요청 시작 \| jobId=([a-f0-9-]+)"

# ============================================================
# 에러 카테고리 분류 (분기 2: route_by_category)
# ============================================================
# 이 에러 타입들이 감지되면 → investigate_infra 경로로 분기
INFRA_ERROR_TYPES = {
    "ConnectionError", "ConnectionRefusedError", "ConnectionResetError",
    "TimeoutError", "ServerSelectionTimeoutError", "NetworkError",
    "OSError", "SocketError", "BrokenPipeError",
    "ClientConnectorError",  # aiohttp
    "ConnectTimeoutError",   # httpx
    "ReadTimeoutError",      # httpx
}

# 에러 메시지에서 인프라 문제를 감지하는 키워드
INFRA_ERROR_KEYWORDS = {
    "connection refused", "connection reset", "timed out", "timeout",
    "server selection", "no host", "dns resolution", "unreachable",
    "broken pipe", "oom", "out of memory", "disk full", "no space",
    "sqs", "queue", "dead letter",
}

# ============================================================
# 리포트 검증 (분기 4: validate_report)
# ============================================================
MAX_REPORT_RETRIES = 2

# 리포트 JSON에 반드시 있어야 하는 필드
REPORT_REQUIRED_FIELDS = {
    "error_identification", "root_cause", "summary",
}

# ============================================================
# 조사 깊이 (분기 3: 라운드 3+ 시 deep dive 프롬프트 적용)
# ============================================================
DEEP_DIVE_THRESHOLD = 3

