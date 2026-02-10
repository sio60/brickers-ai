# ============================================
# Brickers Screenshot Server
# LDView + LDraw + Python (AI 라이브러리 없음)
# ============================================

# ============================================
# Stage 1: Python Builder
# ============================================
FROM python:3.11.9-slim AS python-builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt

# ============================================
# Stage 2: Runtime
# ============================================
FROM python:3.11.9-slim AS runtime

WORKDIR /app

# 런타임 라이브러리, LDView 의존성 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
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
    xauth \
    libharfbuzz0b \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# LDView AppImage -> extract (Docker에서 FUSE 불가하므로 풀어서 사용)
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
ENV PYTHONUNBUFFERED=1

# builder 스테이지에서 빌드된 wheel 복사 및 설치
COPY --from=python-builder /build/wheels /tmp/wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --find-links=/tmp/wheels -r requirements.txt \
    && rm -rf /tmp/wheels

# 소스코드 복사
COPY . .

EXPOSE 8002

CMD ["python", "-u", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
