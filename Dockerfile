# ============================================
# AI Server Runtime
# 베이스 이미지를 미리 빌드해두면 빠른 시작 가능
# ============================================

# ✅ 미리 빌드된 베이스 이미지 사용 (Docker Hub에 푸시해둔 경우)
# FROM johnbyeon/brickers-ai-base:latest AS runtime

# ⚠️ 베이스 이미지가 없으면 로컬에서 빌드 (느림)
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

# 로컬 wheels 폴더 복사
COPY ./wheels /tmp/wheels

# requirements.txt 복사 및 패키지 설치
COPY requirements.txt .

RUN pip install --no-cache-dir \
    --find-links=/tmp/wheels \
    -r requirements.txt

# ============================================
# Runtime Stage
# ============================================
FROM base AS runtime

WORKDIR /app

EXPOSE 8000

# uvicorn reload 모드로 실행
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
