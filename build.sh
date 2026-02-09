#!/bin/bash
# ============================================
# AI Server 빌드 스크립트
# Docker 멀티스테이지 빌드 + BuildKit 캐시
# ============================================

set -e

# BuildKit 활성화 (캐시 마운트 사용)
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

echo "🐳 Building Docker image (BuildKit cache enabled)..."
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
