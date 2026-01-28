# ============================================
# Stage 1: Base Image (패키지만 설치)
# - 이 레이어는 requirements.txt 변경 시에만 재빌드
# ============================================
FROM python:3.11.9-slim AS base

WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    libgfortran5 \
    pkg-config \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

# ✅ 로컬 wheels 폴더 복사 (pybullet 등 미리 빌드된 패키지)
COPY ./wheels /tmp/wheels

# requirements.txt만 먼저 복사 (레이어 캐싱 최적화)
COPY requirements.txt .

# ✅ 패키지 설치 (로컬 wheels 우선 사용)
RUN pip install --no-cache-dir \
    --find-links=/tmp/wheels \
    -r requirements.txt

# ============================================
# Stage 2: Runtime (소스코드는 볼륨 마운트)
# ============================================
FROM base AS runtime

WORKDIR /app

# 소스코드는 볼륨으로 마운트되므로 여기서는 복사하지 않음
# COPY . . <- 이 줄 제거

EXPOSE 8000

# uvicorn reload 모드로 실행 (코드 변경 시 자동 재시작)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
