"""
Admin AI Analyst — Node 구현
7개 노드: Miner → Evaluator → Diagnoser → Strategist → DeepInvestigator → ReporterGreen → Finalizer
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import List, Dict, Any

from .state import AdminAnalystState
from .prompts import (
    DIAGNOSER_PROMPT,
    STRATEGIST_PROMPT,
    GUARDIAN_PROMPT,
    REPORTER_GREEN_PROMPT,
    FINALIZER_PROMPT,
    QUERY_ANALYST_PROMPT,
)

# ... (기존 코드: miner_node, evaluator_node 유지)


from .llm_utils import call_llm_json, call_llm_text

log = logging.getLogger("admin_analyst.nodes")


# ═══════════════════════════════════════════════════════════════
# Node 1: Miner — 데이터 수집
# ═══════════════════════════════════════════════════════════════
async def miner_node(state: AdminAnalystState) -> dict:
    """GA4 Data API + Direct MongoDB에서 통합 지표 및 로우 데이터 수집."""
    import asyncio
    from datetime import datetime
    from service.backend_client import get_full_report, get_product_intelligence
    from db import get_db

    log.info("⛏️ [Miner] 통합 데이터 수집 및 정밀 분석 시작...")
    
    try:
        # 1. Macro Analytics 병렬 수집 (GA4 기반 - 배치 요청)
        full_report_task = get_full_report(days=7)
        product_intel_task = get_product_intelligence(days=14)
        
        # 2. Micro Logs 정밀 분석 (Direct MongoDB - Ground Truth)
        # 동기 pymongo 호출을 별도 스레드에서 실행하여 이벤트 루프 블로킹 방지
        def _fetch_db_raw():
            db = get_db()
            one_day_ago = datetime.now().timestamp() - 86400
            jobs_col = db["kids_jobs"]
            
            recent_jobs = list(jobs_col.find({
                "createdAt": {"$gte": datetime.fromtimestamp(one_day_ago)}
            }).sort("createdAt", -1).limit(100))

            result = {
                "total_jobs_24h": len(recent_jobs),
                "avg_stability": 0.0,
                "avg_gen_time": 0.0,
                "avg_brick_count": 0,
                "error_dist": {},
                "stage_dist": {},
                "input_type_dist": {"Text Prompt": 0, "Image Upload": 0}
            }

            if recent_jobs:
                stabilities = [j["result"]["stabilityScore"] for j in recent_jobs if j.get("result", {}).get("stabilityScore")]
                gen_times = []
                brick_counts = []
                for j in recent_jobs:
                    if j.get("startedAt") and j.get("endedAt"):
                        dur = (j["endedAt"] - j["startedAt"]).total_seconds()
                        if 0 < dur < 600: gen_times.append(dur)
                    
                    if j.get("result", {}).get("brickCount"):
                        brick_counts.append(j["result"]["brickCount"])
                    
                    inp = j.get("inputType", "Text Prompt")
                    result["input_type_dist"][inp] = result["input_type_dist"].get(inp, 0) + 1
                    
                    stage = j.get("stage", "UNKNOWN")
                    result["stage_dist"][stage] = result["stage_dist"].get(stage, 0) + 1
                    if j.get("status") == "FAILED" and j.get("error"):
                        err = str(j["error"])[:50]
                        result["error_dist"][err] = result["error_dist"].get(err, 0) + 1

                result["avg_stability"] = round(sum(stabilities) / len(stabilities), 2) if stabilities else 0.82
                result["avg_gen_time"] = round(sum(gen_times) / len(gen_times), 1) if gen_times else 45.0
                result["avg_brick_count"] = int(sum(brick_counts) / len(brick_counts)) if brick_counts else 120

            return result

        db_raw = await asyncio.to_thread(_fetch_db_raw)
            
        # 3. 비동기 작업 대기 및 결과 병합
        full_report, product_intel = await asyncio.gather(full_report_task, product_intel_task, return_exceptions=True)
        
        if isinstance(full_report, Exception):
            log.error(f"⚠️ [Miner] Full Report Fetch Failed: {full_report}")
            full_report = {}
        if isinstance(product_intel, Exception):
            log.error(f"⚠️ [Miner] Product Intel Fetch Failed: {product_intel}")
            product_intel = {}
        
        raw_data = full_report or {}
        event_stats = raw_data.get("eventStats", {})
        
        raw_metrics = {
            "summary": raw_data.get("summary", {}),
            "daily_users": raw_data.get("dailyUsers", []),
            "top_tags": raw_data.get("topTags", []),
            # "top_keywords": raw_data.get("topKeywords", []), # [REMOVED] 검색 키워드 대신 제품 지능화 데이터 사용
            "heavy_users": raw_data.get("heavyUsers", []),
            "event_stats": event_stats,
            "top_posts": raw_data.get("topPages", []), # Diagnoser/Reporting용
            "product_intelligence": product_intel or {}, # [NEW] 검색 키워드 대신 제품 지능화 데이터 사용
            "db_raw": db_raw,
            "today_stats": {
                "gen_success": sum(e.get("count", 0) for e in (event_stats.get("success_1d") or [])),
                "gen_fail": sum(e.get("count", 0) for e in (event_stats.get("fail_1d") or [])),
                "gallery_uploads": sum(e.get("count", 0) for e in (event_stats.get("gallery_attempt_1d") or [])),
            }
        }

        now = datetime.now()
        return {
            "raw_metrics": raw_metrics,
            "temporal_context": {
                "now": now.isoformat(),
                "date": now.strftime("%Y-%m-%d"),
                "hour": now.hour,
                "day_of_week": now.strftime("%A"),
                "weekday": now.strftime("%A"), # 호환성 유지
                "is_peak": 19 <= now.hour <= 23
            },
            "next_action": "evaluate"
        }

    except Exception as e:
        log.error(f"⛏️ [Miner] 데이터 수집 중 치명적 실패: {e}", exc_info=True)
        return {
            "raw_metrics": {},
            "temporal_context": {"now": datetime.now().isoformat()},
            "next_action": "evaluate"
        }


# ═══════════════════════════════════════════════════════════════
# Node 2: Evaluator — 이상 탐지 (규칙 기반, LLM 미사용)
# ═══════════════════════════════════════════════════════════════
def evaluator_node(state: AdminAnalystState) -> dict:
    """정교화된 가중치 기반 연속 지표 분석 및 이상 탐지."""
    log.info("[Evaluator] 고해상도 지표 분석 시작...")
    anomalies: List[Dict[str, Any]] = []
    metrics = state.get("raw_metrics") or {}
    db_raw = metrics.get("db_raw") or {}

    # ─── 📊 세부 위험 점수 (Sub-scores, 0.0 ~ 1.0) ───
    s_dau = 0.0   # DAU 변동 위험
    s_fail = 0.0  # 생성 실패율 위험
    s_stab = 0.0  # 브릭 안정성 위험
    s_lat = 0.0   # 처리 지연 위험
    s_conv = 0.0  # 전환 품질 위험

    # ┌─────────────────────────────────────────────────────────────┐
    # │  CHECK 1: Macro Analytics (DAU, Fail Rate)                  │
    # └─────────────────────────────────────────────────────────────┘
    
    # ── 1-A. DAU 변동 (Z-Score 기반 연속 점수) ──
    daily = metrics.get("daily_users") or []
    dau_spike = False
    if len(daily) >= 3:
        try:
            counts = [d.get("count", d.get("activeUsers", 0)) for d in daily]
            prev, today = counts[:-1], counts[-1]
            mean = sum(prev) / len(prev) if prev else 0
            std = (sum((x - mean) ** 2 for x in prev) / len(prev)) ** 0.5 if prev else 0

            if std > 0:
                z = (today - mean) / std
                # 공식: abs(z) / 5.0 (Z-score 5.0일 때 위험도 100%)
                s_dau = min(1.0, abs(z) / 5.0)
                
                if abs(z) > 2.0:
                    direction = "DROP" if z < 0 else "SPIKE"
                    if direction == "SPIKE": dau_spike = True
                    anomalies.append({
                        "metric": "daily_active_users",
                        "current": today,
                        "baseline": round(mean, 1),
                        "severity": "HIGH" if abs(z) > 3.5 else "MEDIUM",
                        "z_score": round(z, 2),
                        "direction": direction,
                    })
        except Exception as e:
            log.warning(f"[Evaluator] DAU 분석 오류: {e}")

    # ── 1-B.生成 실패율 (연속 점수) ──
    today_failures = (metrics.get("today_stats") or {}).get("gen_fail", 0)
    recent_succ = (metrics.get("today_stats") or {}).get("gen_success", 0)
    total = today_failures + recent_succ
    
    if total > 5:
        rate = today_failures / total
        # 공식: Rate / 0.5 (실패율 50%일 때 위험도 100%)
        s_fail = min(1.0, rate / 0.5)
        
        if rate > 0.15:
            anomalies.append({
                "metric": "generation_fail_rate",
                "current": f"{round(rate * 100, 1)}%",
                "baseline": "10.0%",
                "severity": "HIGH" if rate > 0.35 else "MEDIUM",
                "direction": "SPIKE",
            })

    # ┌─────────────────────────────────────────────────────────────┐
    # │  CHECK 2: Micro DB Quality (Stability, Latency)            │
    # └─────────────────────────────────────────────────────────────┘

    # ── 2-A. 브릭 안정성 (기준치 0.85 대비 하락폭) ──
    avg_stability = db_raw.get("avg_stability", 0.0)
    if avg_stability > 0:
        # 공식: max(0, 1.0 - Avg/0.85) -> 0.85이상이면 0, 0이면 1.0
        s_stab = max(0.0, 1.0 - (avg_stability / 0.85))
        
        if avg_stability < 0.7:
            anomalies.append({
                "metric": "avg_stability_score",
                "current": avg_stability,
                "baseline": 0.85,
                "severity": "HIGH" if avg_stability < 0.5 else "MEDIUM",
                "direction": "DROP",
                "desc": "물리적 안정성 기준치 미달"
            })

    # ── 2-B. 생성 소요 시간 (30s~120s 선형 스케일링) ──
    avg_gen_time = db_raw.get("avg_gen_time", 0.0)
    if avg_gen_time > 0:
        # 공식: (Time - 30) / 90 -> 30초 이하면 0, 120초면 1.0
        s_lat = min(1.0, max(0.0, (avg_gen_time - 30) / 90))
        
        if avg_gen_time > 60:
            anomalies.append({
                "metric": "avg_generation_time",
                "current": f"{avg_gen_time}s",
                "baseline": "30s",
                "severity": "HIGH" if avg_gen_time > 100 else "MEDIUM",
                "direction": "DELAY",
            })

    # ── 2-C. 전환 품질 (DAU Spike 시 전환율 10% 기준 하락폭) ──
    if dau_spike:
        total_jobs = db_raw.get("total_jobs_24h", 0)
        daily_users_list = metrics.get("daily_users") or []
        daily_count = 1
        if daily_users_list:
            last_day = daily_users_list[-1]
            daily_count = last_day.get("count", last_day.get("activeUsers", 1))

        conversion_rate = total_jobs / max(daily_count, 1)
        
        # 공식: max(0, 1.0 - ConvRate / 0.1) -> 10% 이상이면 0, 0%면 1.0
        s_conv = max(0.0, 1.0 - (conversion_rate / 0.1))
        
        if conversion_rate < 0.05:
            anomalies.append({
                "metric": "traffic_quality",
                "current": f"{round(conversion_rate*100, 1)}%",
                "baseline": "10.0%",
                "severity": "MEDIUM",
                "direction": "DROP",
                "desc": "트래픽 유입 대비 낮은 사용 전환율"
            })

    # ── 3. 가중치 기반 최종 위험 점수 산출 ──
    # 가중치: 실패율(0.4), 안정성(0.3), DAU변동(0.1), 지연(0.1), 전환품질(0.1)
    risk = (0.4 * s_fail) + (0.3 * s_stab) + (0.1 * s_dau) + (0.1 * s_lat) + (0.1 * s_conv)
    risk = round(min(1.0, risk), 3)
    
    # 0.5(50%) 이상이면 심층 진단 루틴으로 진입
    next_step = "diagnose" if risk >= 0.5 else "report_green"

    log.info(f"[Evaluator] 분석 완료: 위험도={round(risk*100,1)}% ({next_step})")

    return {
        "anomalies": anomalies,
        "risk_score": risk,
        "next_action": next_step,
    }


# ═══════════════════════════════════════════════════════════════
# Node 3: Diagnoser — 인과관계 추론 (LLM)
# ═══════════════════════════════════════════════════════════════
async def diagnoser_node(state: AdminAnalystState) -> dict:
    """LLM으로 이상 징후의 근본 원인을 추론."""
    log.info("[Diagnoser] 원인 추론 시작...")

    anomaly_text = json.dumps(state.get("anomalies", []), ensure_ascii=False, indent=2)
    summary = state.get("raw_metrics", {}).get("summary", {})
    db_raw = state.get("raw_metrics", {}).get("db_raw", {})
    temporal = state.get("temporal_context", {})
    raw_metrics = state.get("raw_metrics", {})
    
    # [Logic] DeepInvestigator가 수집한 30일 데이터가 있으면 우선 사용 (심층 진단 모드)
    tags = raw_metrics.get("top_tags_30d") or raw_metrics.get("top_tags", [])
    # keywords = raw_metrics.get("top_keywords_30d") or raw_metrics.get("top_keywords", []) # [REMOVED]
    product_intel = raw_metrics.get("product_intelligence") or {} # [NEW] 검색 키워드 대신 제품 정보(퍼널/이탈) 사용

    prompt = DIAGNOSER_PROMPT.format(
        anomaly_text=anomaly_text,
        active_users=summary.get('activeUsers', 'N/A'),
        page_views=summary.get('pageViews', 'N/A'),
        sessions=summary.get('sessions', 'N/A'),
        total_jobs=db_raw.get('total_jobs_24h', 0),
        avg_stability=db_raw.get('avg_stability', 0.0),
        avg_gen_time=db_raw.get('avg_gen_time', 0.0),
        stage_dist=json.dumps(db_raw.get('stage_dist', {}), ensure_ascii=False),
        error_dist=json.dumps(db_raw.get('error_dist', {}), ensure_ascii=False),
        input_type_dist=json.dumps(db_raw.get('input_type_dist', {}), ensure_ascii=False),
        top_tags=json.dumps(tags[:20], ensure_ascii=False), # 30일 데이터일 경우 더 많이 보여줌
        # top_keywords=json.dumps(keywords[:20], ensure_ascii=False), # [REMOVED]
        product_intel=json.dumps(product_intel, ensure_ascii=False), # [NEW]
        date=temporal.get('date'),
        hour=temporal.get('hour'),
        day_of_week=temporal.get('day_of_week')
    )

    diagnosis = await call_llm_json(prompt)

    if not diagnosis:
        diagnosis = {
            "root_cause": "LLM 분석 실패 — 수동 확인 필요",
            "confidence": 0.3,
            "evidence": [f"{len(state.get('anomalies', []))}건 이상 징후 감지"],
            "affected_segment": "전체 유저",
        }

    log.info(f"[Diagnoser] 완료: {diagnosis.get('root_cause', '')[:50]}...")
    return {"diagnosis": diagnosis, "next_action": "strategize"}


# ═══════════════════════════════════════════════════════════════
# Node 4: Strategist — 대응 전략 수립 (LLM)
# ═══════════════════════════════════════════════════════════════
async def strategist_node(state: AdminAnalystState) -> dict:
    """진단 기반 구체 대응 전략 도출."""
    log.info("[Strategist] 전략 수립 시작...")

    dx = state.get("diagnosis", {})
    confidence = dx.get("confidence", 0.5)

    prompt = STRATEGIST_PROMPT.format(
        root_cause=dx.get('root_cause', '?'),
        risk_level=dx.get('risk_level', 'UNKNOWN'),
        confidence=confidence * 100,
        evidence=json.dumps(dx.get('evidence', []), ensure_ascii=False)
    )

    actions = await call_llm_json(prompt)

    if isinstance(actions, dict):
        actions = [actions]
    if not isinstance(actions, list):
        actions = [{"action": "수동 모니터링 강화", "target": "서비스 전체",
                     "expected_impact": "실시간 파악", "risk": "LOW", "priority": 1}]

    iteration = state.get("iteration", 0) + 1

    # 확신도 낮으면 루프백
    if confidence < 0.5 and iteration < state.get("max_iterations", 3):
        log.info(f"[Strategist] 확신도 {confidence} < 0.5 → 심층 조사 (iter={iteration})")
        return {"proposed_actions": actions, "iteration": iteration, "next_action": "deep_investigate"}

    return {"proposed_actions": actions, "iteration": iteration, "next_action": "finalize"}


# ═══════════════════════════════════════════════════════════════
# Node 8: Content Miner — 검열 대상 수집
# ═══════════════════════════════════════════════════════════════
async def content_miner_node(state: AdminAnalystState) -> dict:
    """백엔드에서 아직 처리되지 않은 최근 댓글/게시글 수집."""
    from service import backend_client

    # 이미 검열이 완료된 경우 재실행 방지 (사이클당 1회만)
    if state.get("moderation_results"):
        log.info("[ContentMiner] 이미 검열 완료 — 스킵")
        return {"next_action": "guard"}

    log.info("[ContentMiner] 검열 대상 수집 시작...")

    # 최근 1일 내의 미검열 콘텐츠 최대 10개 수집 (속도 최적화)
    contents = await backend_client.get_recent_contents(days=1, limit=10)

    log.info(f"[ContentMiner] 수집 완료: {len(contents or [])}건")
    return {
        "moderation_queue": contents or [],
        "next_action": "guard"
    }


# ═══════════════════════════════════════════════════════════════
# Node 9: Guardian — 선정성/폭력성 판단 (LLM)
# ═══════════════════════════════════════════════════════════════
async def guardian_node(state: AdminAnalystState) -> dict:
    """LLM을 블랙박스 검열관으로 사용하여 부적절성 판단."""
    log.info("[Guardian] 콘텐츠 검열 시작...")

    queue = state.get("moderation_queue", [])
    if not queue:
        return {"next_action": "execute_moderation"}

    queue_text = json.dumps(queue, ensure_ascii=False, indent=2)

    prompt = GUARDIAN_PROMPT.format(queue_text=queue_text)

    judgments = await call_llm_json(prompt)
    if not isinstance(judgments, list):
        judgments = []

    log.info(f"[Guardian] 검열 완료: {len(judgments)}건 판정")
    return {
        "moderation_results": judgments,
        "next_action": "execute_moderation"
    }


# ═══════════════════════════════════════════════════════════════
# Node 10: Moderator Executor — 자동 조치 실행
# ═══════════════════════════════════════════════════════════════
async def moderator_executor_node(state: AdminAnalystState) -> dict:
    """Guardian의 판단에 따라 백엔드에 자동 숨김 처리 요청."""
    from service import backend_client
    log.info("[ModeratorExecutor] 자동 조치 실행 시작...")

    results = state.get("moderation_results", [])
    executed_count = 0

    for res in results:
        if res.get("is_violating") and res.get("confidence", 0) >= 0.8:
            target_id = res.get("target_id")
            target_type = res.get("type")
            reason = res.get("reason", "AI Automated Moderation")

            success = await backend_client.hide_content(target_type, target_id, reason)
            if success:
                executed_count += 1
                res["action_taken"] = "HIDDEN"
                log.info(f"[ModeratorExecutor] 조치 완료: {target_type} {target_id}")
            else:
                res["action_taken"] = "FAILED"

    log.info(f"[ModeratorExecutor] 총 {executed_count}건 자동 조치 완료")
    return {
        "moderation_results": results,
        "next_action": "finalize"
    }


# ═══════════════════════════════════════════════════════════════
# Node 5: Deep Investigator — 심층 조사 (루프백)
# ═══════════════════════════════════════════════════════════════
async def deep_investigator_node(state: AdminAnalystState) -> dict:
    """30일 데이터 추가 수집 후 Diagnoser로 루프백."""
    from service import backend_client

    log.info("[DeepInvestigator] 30일 장기 데이터 수집...")

    long_daily = await backend_client.get_daily_users(30)
    long_tags = await backend_client.get_top_tags(30, limit=20)

    metrics = dict(state.get("raw_metrics", {}))
    metrics["daily_users_30d"] = long_daily or []
    metrics["top_tags_30d"] = long_tags or []

    return {"raw_metrics": metrics, "next_action": "diagnose"}


# ═══════════════════════════════════════════════════════════════
# Node 6: Reporter Green — 정상 보고서
# ═══════════════════════════════════════════════════════════════
async def reporter_green_node(state: AdminAnalystState) -> dict:
    """이상 징후가 없을 때도 LLM으로 심층 운영 인사이트 생성."""
    log.info("[Reporter] 정상 상태 심층 인사이트 생성 시작...")

    metrics = state.get("raw_metrics", {})
    summary = metrics.get("summary", {})
    daily = metrics.get("daily_users", [])
    tags = metrics.get("top_tags", [])
    temporal = state.get("temporal_context", {})

    # 트렌드 요약 (LLM 참고용)
    trend_desc = "보합세"
    if daily and len(daily) >= 3:
        try:
            counts = [d.get("count", d.get("activeUsers", 0)) for d in daily]
            recent_avg = sum(counts[-3:]) / 3
            prev_avg = sum(counts[-6:-3]) / 3 if len(counts) >= 6 else counts[0]
            chg = ((recent_avg - prev_avg) / max(prev_avg, 1)) * 100
            trend_desc = f"최근 3일 평균이 이전 대비 {chg:+.1f}% {'상승' if chg > 0 else '하락'} 중"
        except: pass

    intel = metrics.get("product_intelligence") or {}
    
    prompt = REPORTER_GREEN_PROMPT.format(
        active_users=summary.get("activeUsers", 0),
        page_views=summary.get("screenPageViews", 0),
        sessions=summary.get("sessions", 0),
        trend_desc=trend_desc,
        top_tags=json.dumps(metrics.get("top_tags", [])[:10], ensure_ascii=False), # [MODIFIED]
        funnel=json.dumps(intel.get("funnel", []), ensure_ascii=False),
        exits=json.dumps(intel.get("exits", []), ensure_ascii=False),
        quality=json.dumps(intel.get("quality", {}), ensure_ascii=False),
    )

    res = await call_llm_json(prompt)
    report = res.get("report") if res else None

    if not report:
        report = f"## ✅ 서비스 안정 운영 중\n\n모든 핵심 지표가 정상 범위를 유지하고 있습니다. 유저 유입 및 전환 트렌드가 안정적입니다. ({trend_desc})"

    return {"final_report": report, "next_action": "end"}


# ═══════════════════════════════════════════════════════════════
# Node 7: Finalizer — 이상 발견 시 종합 보고서
# ═══════════════════════════════════════════════════════════════
async def finalizer_node(state: AdminAnalystState) -> dict:
    """이상 징후 발견 시 LLM으로 유기적인 종합 분석 보고서 생성."""
    log.info("[Finalizer] 종합 보고서 생성 시작...")

    dx = state.get("diagnosis") or {}
    actions = state.get("proposed_actions") or []
    anomalies = state.get("anomalies") or []
    mod_results = state.get("moderation_results") or []
    temporal = state.get("temporal_context") or {}

    # 자동 조치 내역 요약
    hidden_count = sum(1 for r in mod_results if r.get("action_taken") == "HIDDEN")
    mod_text = ""
    if mod_results:
        mod_text = "\n### 🛡️ 자율 콘텐츠 검열 및 조치 내역\n"
        if hidden_count > 0:
            mod_text += f"- **자동 숨김 처리**: {hidden_count}건 (AI 확신도 80% 이상)\n"
        else:
            mod_text += "- 특이사항: 위반 의심 콘텐츠 없음 (클린 상태 유지 중)\n"

        for r in [r for r in mod_results if r.get("is_violating")][:5]:
             mod_text += f"  - [{r.get('violation_type')}] {r.get('target_id')}: {r.get('reason')} ({r.get('action_taken', 'PENDING')})\n"

    prompt = FINALIZER_PROMPT.format(
        anomalies=json.dumps(anomalies, ensure_ascii=False, indent=2),
        mod_text=mod_text,
        root_cause=dx.get('root_cause', '?'),
        forecast=dx.get('forecast', '데이터 수집 중...'),
        confidence=dx.get('confidence', 0) * 100,
        evidence=json.dumps(dx.get('evidence', []), ensure_ascii=False),
        affected_segment=dx.get('affected_segment', '전체'),
        actions=json.dumps(actions, ensure_ascii=False, indent=2)
    )

    # 이상 징후가 없으면 기존 보고서(reporter_green) 보존
    if not anomalies and not dx.get("root_cause"):
        existing_report = state.get("final_report")
        if existing_report:
            log.info("[Finalizer] 이상 징후 없음 — 기존 보고서(reporter_green) 보존")
            return {"final_report": existing_report, "next_action": "end"}

    res = await call_llm_json(prompt)
    report = res.get("report") if res else None

    if not report:
        # Fallback 템플릿
        report = f"## 🚨 관리자 주의: 이상 징후 감지\n\n- 원인: {dx.get('root_cause', '?')}\n- 조치: {len(actions)}건의 전략 수립됨."

    return {"final_report": report, "next_action": "end"}

# ═══════════════════════════════════════════════════════════════
# Node 11: Query Analyst — 인터랙티브 질의응답 (NEW)
# ═══════════════════════════════════════════════════════════════
async def query_analyst_node(state: AdminAnalystState) -> dict:
    """전략적 AI 어드바이저: 대화 이력과 100여 지표를 종합하여 입체적 전략 수립."""
    log.info("[QueryAnalyst] 대화 맥락 포함 전략 분석 시작...")

    user_query = state.get("user_query", "현재 서비스 운영 상태 종합 진단")
    history = state.get("history", []) # [NEW] 대화 이력
    metrics = state.get("raw_metrics", {})
    summary = metrics.get("summary", {})
    daily = metrics.get("daily_users", [])
    tags = metrics.get("top_tags", [])
    today = metrics.get("today_stats", {})
    db_raw = metrics.get("db_raw", {})
    top_posts = metrics.get("top_posts", [])
    temporal = state.get("temporal_context", {})

    # 이전 대화 맥락 요약
    history_context = ""
    if history:
        history_context = "\n[이전 대화 맥락]\n" + "\n".join([f"{h['role']}: {h['content']}" for h in history[-3:]])

    prompt = QUERY_ANALYST_PROMPT.format(
        user_query=user_query,
        history_context=history_context, # [MODIFIED]
        summary=json.dumps(summary, ensure_ascii=False), # [MODIFIED]
        today_gen_success=today.get('gen_success'),
        today_gen_fail=today.get('gen_fail'),
        today_gallery=today.get('gallery_uploads'),
        total_jobs=db_raw.get('total_jobs_24h'),
        stage_dist=json.dumps(db_raw.get('stage_dist', {}), ensure_ascii=False),
        daily=json.dumps(daily, ensure_ascii=False),
        tags=json.dumps(tags[:10], ensure_ascii=False),
        # keywords=json.dumps(metrics.get("top_keywords", [])[:10], ensure_ascii=False), # [REMOVED]
        top_posts=json.dumps(top_posts, ensure_ascii=False),
        product_intel=json.dumps(metrics.get("product_intelligence", {}), ensure_ascii=False),
        temporal=json.dumps(temporal, ensure_ascii=False)
    )

    report = await call_llm_text(prompt)

    if not report:
        report = "현재 수집된 데이터로는 충분한 분석이 어렵습니다. 잠시 후 다시 시도해 주세요."

    # 이력 업데이트는 호출부에서 처리하도록 제안 (현재 노드에서는 결과만 반환)
    return {"final_report": report, "next_action": "end"}
