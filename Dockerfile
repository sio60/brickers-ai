# syntax=docker/dockerfile:1
# ============================================
# Stage 1: Rust Builder
# ============================================
FROM rust:1.80-slim-bookworm AS rust-builder

WORKDIR /build/rust
# Rust 빌드 도중 maturin이 전역 파이썬을 사용할 수 있어야 함
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv

# 소스 복사 (Cargo.toml, pyproject.toml 포함)
COPY brick_judge/rust /build/rust

# Maturin 설치 및 빌드 (cargo 캐시 활용)
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/rust/target \
    pip3 install --break-system-packages maturin \
    && maturin build --release --out /build/wheels

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

# requirements 복사 및 wheel 빌드 (pip 캐시로 재빌드 속도 향상)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel --wheel-dir /build/wheels -r requirements.txt

# ============================================
# Stage 3: Runtime
# ============================================
FROM python:3.11.9-slim AS runtime

WORKDIR /app

# 런타임 라이브러리, 폰트, LDView 의존성 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    libatlas-base-dev \
    libgfortran5 \
    fonts-nanum \
    xvfb \
    libosmesa6 \
    libgl1-mesa-glx \
    libglu1-mesa \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libfuse2 \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# LDView AppImage → extract (Docker에서 FUSE 불가하므로 풀어서 사용)
RUN wget -q https://github.com/tcobbs/ldview/releases/download/v4.5/LDView-39cc01ab-x86_64.AppImage \
        -O /tmp/LDView.AppImage \
    && chmod +x /tmp/LDView.AppImage \
    && cd /opt && /tmp/LDView.AppImage --appimage-extract \
    && ln -s /opt/squashfs-root/AppRun /usr/local/bin/LDView \
    && rm /tmp/LDView.AppImage

# LDraw parts library
RUN mkdir -p /usr/share/ldraw && \
    wget -q https://library.ldraw.org/library/updates/complete.zip -O /tmp/ldraw.zip && \
    unzip -q /tmp/ldraw.zip -d /usr/share/ && \
    rm /tmp/ldraw.zip

ENV LDRAWDIR=/usr/share/ldraw

# builder 스테이지에서 빌드된 wheel들 복사
COPY --from=python-builder /build/wheels /tmp/wheels
COPY --from=rust-builder /build/wheels /tmp/wheels

COPY requirements.txt .

# 모든 wheel 설치 (pip가 종속성 해결을 위해 /tmp/wheels를 먼저 보게 함)
RUN pip install --no-cache-dir \
    --find-links=/tmp/wheels \
    -r requirements.txt

# brick_judge_rs 별도 설치 (requirements.txt에 없으므로)
RUN pip install --no-cache-dir /tmp/wheels/brick_judge_rs*.whl \
    && rm -rf /tmp/wheels

# 소스코드 복사
COPY . .

EXPOSE 8000

# uvicorn 실행
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
