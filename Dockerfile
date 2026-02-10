# ============================================
# Stage 1: Rust Builder
# ============================================
FROM rust:1.80-slim-bookworm AS rust-builder

WORKDIR /build/rust
# Rust 빌드 도중 maturin이 전역 파이썬을 사용할 수 있어야 함
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv

# 소스 복사 (Cargo.toml, pyproject.toml 포함)
COPY brick_judge/rust /build/rust

# Maturin 설치 및 빌드
RUN pip3 install --break-system-packages maturin
RUN maturin build --release --out /build/wheels

# ============================================
# Stage 2: Python Builder
# ============================================
FROM python:3.11.9-slim AS python-builder

WORKDIR /build

# 빌드 도구 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

# requirements 복사 및 wheel 빌드 (Docker 레이어 캐시로 재빌드 스킵)
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt

# ============================================
# Stage 3: Runtime
# ============================================
FROM python:3.11.9-slim AS runtime

WORKDIR /app

# ✅ stdout 버퍼링 비활성화 (Docker 컨테이너에서 실시간 로그 출력)
ENV PYTHONUNBUFFERED=1

# 런타임 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends \
    libatlas-base-dev \
    libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

# builder 스테이지에서 빌드된 wheel들 복사
COPY --from=python-builder /build/wheels /tmp/wheels
COPY --from=rust-builder /build/wheels /tmp/wheels

COPY requirements.txt .

# 모든 wheel 설치 (pip가 종속성 해결을 위해 /tmp/wheels를 먼저 보게 함)
RUN pip install --no-cache-dir \
    --find-links=/tmp/wheels \
    -r requirements.txt \
    && pip install --no-cache-dir /tmp/wheels/brick_judge_rs*.whl \
    && rm -rf /tmp/wheels

# 소스코드 복사
COPY . .

EXPOSE 8000

# uvicorn 실행
# ✅ 프로덕션에서는 --reload 제거 (subprocess 파이프 버퍼링 방지)
CMD ["python", "-u", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
