# ============================================
# AI Server Runtime
# 로컬 wheels에서 패키지 설치 → 빠른 빌드
# ============================================

FROM python:3.11.9-slim

WORKDIR /app

# 런타임 라이브러리만 설치 (빌드 도구 제외)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libatlas-base-dev \
    libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

# 로컬 wheels 복사 (미리 빌드된 패키지)
COPY ./wheels /tmp/wheels

# requirements.txt 복사
COPY requirements.txt .

# wheels에서 설치 (컴파일 불필요 → 빠름)
RUN pip install --no-cache-dir \
    --find-links=/tmp/wheels \
    -r requirements.txt \
    && rm -rf /tmp/wheels

# 소스코드 복사
COPY . .

EXPOSE 8000

# uvicorn 실행 (운영 대응을 위해 reload는 환경에 따라 선택 가능하지만 기본 유지)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
