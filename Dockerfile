FROM python:3.11.9-slim

WORKDIR /app

# 1. 시스템 의존성 설치
# numpy, scipy, pillow 등이 내부적으로 사용하는 C 라이브러리들을 미리 설치해두면 빌드 실패 확률이 줄어들어.
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    libgfortran5 \
    pkg-config \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 2. 패키지 설치 (여기가 핵심!)
# --extra-index-url https://www.piwheels.org/simple 옵션을 추가해서
# 라즈베리파이용으로 미리 빌드된 파일(Wheel)을 우선적으로 찾아 설치하게 함.
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://www.piwheels.org/simple

COPY . .

EXPOSE 8000

# uvicorn 실행
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]