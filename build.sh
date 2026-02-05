#!/bin/bash
# ============================================
# AI Server 빌드 스크립트
# wheels 폴더에 패키지 없으면 빌드, 있으면 스킵
# ============================================

set -e

WHEELS_DIR="./wheels"
REQUIREMENTS="requirements.txt"

# 필수 패키지 목록 (빌드 오래 걸리는 것들)
REQUIRED_PACKAGES=(
    "torch"
    "scipy"
    "numpy"
    "tokenizers"
    "sentencepiece"
)

echo "🔍 Checking wheels directory..."

# wheels 폴더 생성
mkdir -p "$WHEELS_DIR"

# 필수 패키지 체크
MISSING_PACKAGES=()
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if ! ls "$WHEELS_DIR"/${pkg}*.whl 1>/dev/null 2>&1; then
        echo "  ❌ $pkg not found in wheels"
        MISSING_PACKAGES+=("$pkg")
    else
        echo "  ✅ $pkg found"
    fi
done

# 없는 패키지가 있으면 빌드
if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo ""
    echo "📦 Building missing packages: ${MISSING_PACKAGES[*]}"
    echo "⏳ This may take a while..."
    echo ""

    pip wheel ${MISSING_PACKAGES[*]} \
        --wheel-dir "$WHEELS_DIR" \
        --find-links "$WHEELS_DIR" \
        --no-cache-dir

    echo ""
    echo "✅ Wheel build complete!"
else
    echo ""
    echo "✅ All required packages found in wheels"
fi

echo ""
echo "🐳 Building Docker image..."
docker compose build

echo ""
echo "🛑 Stopping existing container..."
docker compose down

echo ""
echo "🚀 Starting container..."
docker compose up -d

echo ""
echo "✅ Done! Container is running."
echo "📋 Logs: docker compose logs -f"
