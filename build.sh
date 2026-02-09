#!/bin/bash
# ============================================
# AI Server 빌드 스크립트
# Docker 멀티스테이지 빌드
# ============================================

set -e

echo "🐳 Building Docker image..."
docker-compose build

echo ""
echo "🛑 Stopping existing container..."
docker-compose down

echo ""
echo "🚀 Starting container..."
docker-compose up -d

echo ""
echo "✅ Done! Container is running."
echo "📋 Logs: docker-compose logs -f"
