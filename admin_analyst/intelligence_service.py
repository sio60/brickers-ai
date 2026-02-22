"""
IntelligenceService (MCP-Ready Toolset)
DB(MongoDB)와 Analytics(GA4) 데이터를 통합하여 전략적 분석을 수행하는 핵심 서비스 모듈.
"""
import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
from db import get_db
from service import backend_client

log = logging.getLogger("admin_analyst.intelligence")

class IntelligenceService:
    @staticmethod
    async def get_combined_snapshot(days: int = 7) -> Dict[str, Any]:
        """거시 트렌드(GA4)와 미시 로그(DB)를 결합한 종합 스냅샷 제공 (Analyst Service용)."""
        log.info(f"📊 [Intel] 종합 스냅샷 수집 시작 (기간: {days}일)")
        
        # 1. Analytics (거시) - 병렬 수집
        summary_task = backend_client.get_analytics_summary(days)
        tags_task = backend_client.get_top_tags(days, limit=10)
        product_intel_task = backend_client.get_product_intelligence(days=14)
        performance_task = backend_client.get_performance_summary(days)
        heavy_users_task = backend_client.get_heavy_users(days, limit=5)
        
        tags, product_intel, summary, performance, heavy_users = await asyncio.gather(
            tags_task, product_intel_task, summary_task, performance_task, heavy_users_task
        )
        
        # 2. Database (미시) - 24시간 스냅샷
        db_raw = await IntelligenceService.get_miner_data(days=1)

        return {
            "timestamp": datetime.now().isoformat(),
            "analytics_summary": summary,
            "popular_trends": tags,
            "performance": performance.get("performance", {}) if performance else {},
            "heavy_users": heavy_users,
            "product_intelligence": product_intel,
            "db_raw_insights": db_raw
        }

    @staticmethod
    async def get_miner_data(days: int = 7) -> Dict[str, Any]:
        """
        [NEW] LangGraph Miner Node 전용 통합 데이터 수집 함수.
        GA4 Full Report와 DB Raw Data를 결합하여 수집합니다.
        """
        log.info(f"⛏️ [Intel] Miner용 정밀 데이터 마이닝 시작 (기간: {days}일)")
        
        # 1. Analytics & Product Intel 병렬 수집
        full_report_ptr = backend_client.get_full_report(days=days)
        prod_intel_ptr = backend_client.get_product_intelligence(days=days * 2) # 퍼널은 더 긴 호흡

        # 2. DB Raw Jobs (24시간 집중 분석)
        db_raw = await asyncio.to_thread(_fetch_db_raw_internal)
        
        full_report, prod_intel = await asyncio.gather(full_report_ptr, prod_intel_ptr, return_exceptions=True)

        return {
            "full_report": full_report if not isinstance(full_report, Exception) else {},
            "product_intelligence": prod_intel if not isinstance(prod_intel, Exception) else {},
            "db_metrics": db_raw,
            "collected_at": datetime.now().isoformat()
        }

    @staticmethod
    def identify_causal_factors(snapshot: Dict[str, Any]) -> List[str]:
        """데이터 간의 상관관계를 분석하여 인과 요인을 추정 (Heuristic logic)."""
        factors = []
        stats = snapshot.get("db_raw_insights", {}).get("total_jobs_24h", 0)
        fails = snapshot.get("db_raw_insights", {}).get("failure_count", 0)
        
        if stats > 0:
            gen_fail_rate = fails / stats
            if gen_fail_rate > 0.2:
                factors.append(f"🚨 생성 실패율 급상승 ({gen_fail_rate*100:.1f}%): 인프라 부하 또는 외부 API 장애 의심")
            
        # 퍼포먼스 체크
        avg_wait = snapshot.get("performance", {}).get("avgWaitTime", 0)
        if avg_wait > 60000: # 1분 초과
            factors.append("⚠️ 전체 평균 대기 시간 60초 초과: 서버 처리 지연 발생 중")

        return factors

def _fetch_db_raw_internal():
    """MongoDB에서 최근 24시간 실시간 지표를 집계합니다. (Thread-safe)"""
    try:
        db = get_db()
        if db is None:
            return {"error": "DB Unavailable"}
            
        one_day_ago = datetime.now() - timedelta(days=1)
        jobs_col = db["kids_jobs"]
        
        recent_jobs = list(jobs_col.find({
            "createdAt": {"$gte": one_day_ago}
        }).sort("createdAt", -1).limit(200))

        result = {
            "total_jobs_24h": len(recent_jobs),
            "failure_count": 0,
            "avg_stability": 0.0,
            "avg_gen_time": 0.0,
            "avg_brick_count": 0,
            "error_dist": {},
            "stage_dist": {},
            "input_type_dist": {"Text Prompt": 0, "Image Upload": 0}
        }

        if not recent_jobs:
            return result

        stabilities = []
        gen_times = []
        brick_counts = []

        for j in recent_jobs:
            # 상태 요약
            stage = j.get("stage", "UNKNOWN")
            result["stage_dist"][stage] = result["stage_dist"].get(stage, 0) + 1
            
            if j.get("status") == "FAILED":
                result["failure_count"] += 1
                err = str(j.get("error", "Unknown Error"))[:50]
                result["error_dist"][err] = result["error_dist"].get(err, 0) + 1
            
            # 메트릭 누적
            if "result" in j and isinstance(j["result"], dict):
                res = j["result"]
                if res.get("stabilityScore"): stabilities.append(res["stabilityScore"])
                if res.get("brickCount"): brick_counts.append(res["brickCount"])

            if j.get("startedAt") and j.get("endedAt"):
                dur = (j["endedAt"] - j["startedAt"]).total_seconds()
                if 0 < dur < 600: gen_times.append(dur)
            
            inp = j.get("inputType", "Text Prompt")
            result["input_type_dist"][inp] = result["input_type_dist"].get(inp, 0) + 1

        # 통계 계산
        if stabilities: result["avg_stability"] = round(sum(stabilities) / len(stabilities), 2)
        if gen_times: result["avg_gen_time"] = round(sum(gen_times) / len(gen_times), 1)
        if brick_counts: result["avg_brick_count"] = int(sum(brick_counts) / len(brick_counts))

        return result
    except Exception as e:
        log.error(f"❌ [_fetch_db_raw_internal] Error: {e}")
        return {"error": str(e)}
