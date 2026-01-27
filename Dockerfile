FROM python:3.11.9-slim

WORKDIR /app

# 1. 시스템 의존성 설치
# (빌드 도구들은 여전히 필요할 수 있으니 유지하는 게 좋습니다)
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    libgfortran5 \
    pkg-config \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

# [핵심 변경 1] 로컬에 미리 빌드해둔 wheels 폴더를 컨테이너 임시 폴더로 복사
COPY ./wheels /tmp/wheels

COPY requirements.txt .

# [핵심 변경 2] --find-links 옵션을 사용해 로컬 파일을 우선 사용하도록 설정
# 이제 pybullet은 컴파일하지 않고 /tmp/wheels에 있는 파일을 바로 설치합니다.
RUN pip install --no-cache-dir \
    --find-links=/tmp/wheels \
    -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]