"""
Intelligence Service (MCP-Ready Toolset)
DB(MongoDB)와 Analytics(GA4) 데이터를 통합하여 전략적 분석을 수행하는 핵심 서비스 모듈.
"""
from __future__ import annotations
import logging
from typing import Dict, Any, List
from datetime import datetime
from db import get_db
from service import backend_client

log = logging.getLogger("admin_analyst.intelligence")

class IntelligenceService:
    @staticmethod
    async def get_combined_snapshot(days: int = 7) -> Dict[str, Any]:
        """거시 트렌드(GA4)와 미시 로그(DB)를 결합한 종합 스냅샷 제공."""
        log.info(f"📊 [Intel] 종합 스냅샷 수집 시작 (기간: {days}일)")
        
        # 1. Analytics (거시)
        summary = await backend_client.get_analytics_summary(days)
        tags = await backend_client.get_top_tags(days, limit=10)
        
        # 2. Database (미시)
        db_stats = {}
        try:
            db = get_db()
            one_day_ago = datetime.now().timestamp() - 86400
            jobs = list(db["kids_jobs"].find({"createdAt": {"$gte": datetime.fromtimestamp(one_day_ago)}}))
            
            db_stats = {
                "total_recent_jobs": len(jobs),
                "error_jobs": [j for j in jobs if j.get("status") == "FAILED"],
                "active_heavy_tasks": [j for j in jobs if j.get("stage") == "MODEL"],
            }
        except Exception as e:
            log.error(f"❌ [Intel] DB 조회 실패: {e}")

        return {
            "timestamp": datetime.now().isoformat(),
            "analytics_summary": summary,
            "popular_trends": tags,
            "db_raw_insights": {
                "total_jobs_24h": db_stats.get("total_recent_jobs", 0),
                "failure_count": len(db_stats.get("error_jobs", [])),
                "processing_peak": len(db_stats.get("active_heavy_tasks", [])),
            }
        }

    @staticmethod
    def identify_causal_factors(snapshot: Dict[str, Any]) -> List[str]:
        """데이터 간의 상관관계를 분석하여 인과 요인을 추정 (Heuristic logic)."""
        factors = []
        gen_fail_rate = 0
        
        # 예시: 생성 실패와 특정 태그/시간대의 상관관계 분석
        stats = snapshot.get("db_raw_insights", {})
        if stats.get("total_jobs_24h", 0) > 0:
            gen_fail_rate = stats["failure_count"] / stats["total_jobs_24h"]
            
        if gen_fail_rate > 0.2:
            factors.append("🚨 생성 실패율 급상승 (20% 초과): 인프라 부하 또는 외부 API 장애 의심")
            
        return factors
