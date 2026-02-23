# ============================================
# Brickers Blueprint PDF Server
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
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt

# ============================================
# Stage 2: Runtime (ubuntu 22.04 — LDView .deb 호환)
# ============================================
FROM ubuntu:22.04 AS runtime

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Python 3.11 + 런타임 라이브러리 + LDView 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    fonts-nanum ca-certificates \
    xvfb xauth \
    libosmesa6 libgl1-mesa-glx libglu1-mesa \
    libx11-6 libxext6 libxrender1 libfuse2 \
    libharfbuzz0b libfontconfig1 \
    wget unzip \
    && rm -rf /var/lib/apt/lists/*

# LDView .deb (AppImage → .deb: Mac ARM Docker 호환)
RUN wget -q https://github.com/tcobbs/ldview/releases/download/v4.5/ldview-qt5-4.5-ubuntu-22.04.amd64.deb \
    -O /tmp/ldview.deb \
    && dpkg --force-all -i /tmp/ldview.deb || true \
    && apt-get update && apt-get install -f -y || true \
    && rm -f /tmp/ldview.deb \
    && rm -rf /var/lib/apt/lists/*

# LDraw parts library — unified with root Dockerfile (much faster than library.ldraw.org)
RUN wget -q --tries=10 --timeout=30 https://github.com/gkjohnson/ldraw-parts-library/archive/refs/heads/master.zip -O /tmp/ldraw-gh.zip && \
    unzip -q /tmp/ldraw-gh.zip -d /tmp/ && \
    mkdir -p /usr/share/ldraw && \
    mv /tmp/ldraw-parts-library-master/complete/ldraw /usr/share/ && \
    rm -rf /tmp/ldraw-gh.zip /tmp/ldraw-parts-library-master

ENV LDRAWDIR=/usr/share/ldraw

# builder 스테이지에서 빌드된 wheel 복사 및 설치
COPY --from=python-builder /build/wheels /tmp/wheels
COPY requirements.txt .
RUN pip3 install --no-cache-dir --find-links=/tmp/wheels -r requirements.txt \
    && rm -rf /tmp/wheels

# 소스코드 복사
COPY . .

EXPOSE 8001

CMD ["python3", "-u", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
