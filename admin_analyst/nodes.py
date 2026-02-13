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


from .llm_utils import call_llm_json

log = logging.getLogger("admin_analyst.nodes")


# ═══════════════════════════════════════════════════════════════
# Node 1: Miner — 데이터 수집
# ═══════════════════════════════════════════════════════════════
async def miner_node(state: AdminAnalystState) -> dict:
    """GA4 Data API + Direct MongoDB에서 원합 지표 및 로우 데이터 수집."""
    from service import backend_client
    from db import get_db
    import config

    log.info("[Miner] 통합 데이터 수집 시작 (Analytics + DB)...")

    # ┌─────────────────────────────────────────────────────────────┐
    # │  PART 1: Macro Analytics (GA4 & Backend Stats)              │
    # │  - 전체 서비스의 거시적 흐름(트래픽, 유입) 파악                 │
    # └─────────────────────────────────────────────────────────────┘
    import asyncio

    # ┌─────────────────────────────────────────────────────────────┐
    # │  PART 1: Macro Analytics (GA4 & Backend Stats)              │
    # │  - 병렬(Parallel) 처리로 속도 10배 향상: 모든 API 동시 호출      │
    # └─────────────────────────────────────────────────────────────┘
    results = await asyncio.gather(
        backend_client.get_analytics_summary(7),
        backend_client.get_daily_users(14),
        backend_client.get_top_tags(7, limit=15),
        backend_client.get_heavy_users(7, limit=10),
        backend_client.get_top_posts(7, limit=5),
        backend_client.get_event_stats("generate_fail", 7),
        backend_client.get_event_stats("generate_success", 7),
        backend_client.get_event_stats("generate_success", 1),
        backend_client.get_event_stats("generate_fail", 1),
        backend_client.get_event_stats("gallery_register_attempt", 1),
        return_exceptions=True
    )

    # 결과 매핑 (에러 발생 시 None/빈값 처리)
    summary = results[0] if not isinstance(results[0], Exception) else {}
    daily = results[1] if not isinstance(results[1], Exception) else []
    tags = results[2] if not isinstance(results[2], Exception) else []
    users = results[3] if not isinstance(results[3], Exception) else []
    top_posts = results[4] if not isinstance(results[4], Exception) else []
    fail_7d = results[5] if not isinstance(results[5], Exception) else []
    success_7d = results[6] if not isinstance(results[6], Exception) else []
    today_gen_success = results[7] if not isinstance(results[7], Exception) else []
    today_gen_fail = results[8] if not isinstance(results[8], Exception) else []
    today_gallery = results[9] if not isinstance(results[9], Exception) else []

    # ┌─────────────────────────────────────────────────────────────┐
    # │  PART 2: Micro Logs (Direct MongoDB Access)                 │
    # │  - 개별 작업의 구체적 상태, 품질, 에러 등 미시적 데이터 파악      │
    # └─────────────────────────────────────────────────────────────┘
    db_raw = {}
    try:
        db = get_db()
        # 최근 24시간 내 생성된 작업들의 원본 상태 요약
        one_day_ago = datetime.now().timestamp() - 86400
        jobs_col = db["kids_jobs"]
        
        # 성공했거나 실패한 작업 모두 포함하여 분석 (최대 200건 샘플링)
        recent_jobs = list(jobs_col.find({
            "createdAt": {"$gte": datetime.fromtimestamp(one_day_ago)}
        }).limit(200))
        
        db_raw["total_jobs_24h"] = len(recent_jobs)
        db_raw["stage_dist"] = {}
        
        # [NEW] 미시적 품질 지표 계산 (Custom Definitions 대체/보완)
        stability_scores = []
        gen_times = []
        brick_counts = []
        error_dist = {}
        input_type_dist = {}
        
        for j in recent_jobs:
            st = j.get("stage", "UNKNOWN")
            db_raw["stage_dist"][st] = db_raw["stage_dist"].get(st, 0) + 1
            
            # 안정성 점수 (result.stabilityScore)
            if j.get("result") and "stabilityScore" in j["result"]:
                stability_scores.append(j["result"]["stabilityScore"])
                
            # 생성 소요 시간 (endedAt - startedAt)
            if j.get("startedAt") and j.get("endedAt"):
                try:
                    dur = (j["endedAt"] - j["startedAt"]).total_seconds()
                    if 0 < dur < 600: # 10분 이상은 이상치 제외
                        gen_times.append(dur)
                except: pass
                
            # 브릭 개수 (result.brickCount)
            if j.get("result") and "brickCount" in j["result"]:
                brick_counts.append(j["result"]["brickCount"])
            
            # 에러 유형 분포 (실패 원인 분석용)
            if j.get("error"):
                # 에러 메시지나 코드를 단순화해서 카운팅
                err_msg = str(j["error"])[:50] 
                error_dist[err_msg] = error_dist.get(err_msg, 0) + 1
            
            # 입력 방식 선호도 (Text Prompt vs Image Upload)
            inp = j.get("inputType", "unknown")
            input_type_dist[inp] = input_type_dist.get(inp, 0) + 1

        # 평균값 및 분포 산출
        db_raw["avg_stability"] = round(sum(stability_scores) / len(stability_scores), 2) if stability_scores else 0.0
        db_raw["avg_gen_time"] = round(sum(gen_times) / len(gen_times), 1) if gen_times else 0.0
        db_raw["avg_brick_count"] = int(sum(brick_counts) / len(brick_counts)) if brick_counts else 0
        db_raw["error_dist"] = error_dist
        db_raw["input_type_dist"] = input_type_dist
            
        log.info(f"[Miner] DB 데이터 수집 완료: Jobs={len(recent_jobs)} (AvgStability={db_raw['avg_stability']})")
    except Exception as e:
        log.warning(f"[Miner] DB 수집 중 오류 (무시하고 진행): {e}")

    now = datetime.now()
    temporal = {
        "day_of_week": now.strftime("%a"),
        "is_weekend": now.weekday() >= 5,
        "hour": now.hour,
        "is_peak": 19 <= now.hour <= 23,
        "date": now.strftime("%Y-%m-%d"),
    }

    log.info(f"[Miner] 수집 완료: summary={bool(summary)}, db_raw={bool(db_raw)}, today_gen={bool(today_gen_success)}")

    return {
        "raw_metrics": {
            "summary": summary or {},
            "daily_users": daily or [],
            "top_tags": tags or [],
            "heavy_users": users or [],
            "fail_events": fail_7d or [],       # [복구] Evaluator용
            "success_events": success_7d or [], # [복구] Evaluator용
            "db_raw": db_raw,
            "today_stats": {
                "gen_success": sum(e.get("count", 0) for e in (today_gen_success or [])),
                "gen_fail": sum(e.get("count", 0) for e in (today_gen_fail or [])),
                "gallery_uploads": sum(e.get("count", 0) for e in (today_gallery or [])),
            },
            "top_posts": top_posts or [],
        },
        "temporal_context": temporal,
        "moderation_queue": [],
        "moderation_results": [],
        "next_action": "evaluate",
    }


# ═══════════════════════════════════════════════════════════════
# Node 2: Evaluator — 이상 탐지 (규칙 기반, LLM 미사용)
# ═══════════════════════════════════════════════════════════════
def evaluator_node(state: AdminAnalystState) -> dict:
    """Z-Score 및 DB 품질 지표 기반 이상 탐지."""
    log.info("[Evaluator] 이상 탐지 시작...")
    anomalies: List[Dict[str, Any]] = []
    metrics = state.get("raw_metrics", {})
    db_raw = metrics.get("db_raw", {})

    # ┌─────────────────────────────────────────────────────────────┐
    # │  CHECK 1: Macro Analytics Anomalies (DAU, Fail Rare)        │
    # └─────────────────────────────────────────────────────────────┘
    
    # ── 1-A. DAU 급변 감지 ──
    daily = metrics.get("daily_users") or []
    dau_spike = False # 마케팅 감지용 플래그
    if len(daily) >= 3:
        try:
            counts = [d.get("count", d.get("activeUsers", 0)) for d in daily]
            prev, today = counts[:-1], counts[-1]
            mean = sum(prev) / len(prev) if prev else 0
            std = (sum((x - mean) ** 2 for x in prev) / len(prev)) ** 0.5 if prev else 0

            if std > 0:
                z = (today - mean) / std
                if abs(z) > 2.0:
                    severity = "HIGH" if abs(z) > 3.5 else "MEDIUM"
                    direction = "DROP" if z < 0 else "SPIKE"
                    if direction == "SPIKE":
                        dau_spike = True
                        
                    anomalies.append({
                        "metric": "daily_active_users",
                        "current": today,
                        "baseline": round(mean, 1),
                        "severity": severity,
                        "z_score": round(z, 2),
                        "direction": direction,
                    })
        except Exception as e:
            log.warning(f"[Evaluator] DAU 분석 오류: {e}")

    # ── 1-B. 생성 실패율 급증 ──
    fail_ev = metrics.get("fail_events") or []
    succ_ev = metrics.get("success_events") or []
    today_failures = metrics.get("today_stats", {}).get("gen_fail", 0)
    
    if fail_ev and succ_ev:
        try:
            fc = [e.get("count", 0) for e in fail_ev]
            # 오늘 데이터가 API 갱신 전일 수 있으므로 실시간 today_stats 우선 고려
            if today_failures > 0:
                recent_fail = today_failures
            else:
                recent_fail = sum(fc[-1:]) if fc else 0
            
            recent_succ = metrics.get("today_stats", {}).get("gen_success", 0)
            total = recent_fail + recent_succ

            if total > 5:
                rate = recent_fail / total
                prev_rate_avg = 0.1 # 기본값
                
                if rate > 0.2: # 20% 이상 실패 시 체크
                    anomalies.append({
                        "metric": "generation_fail_rate",
                        "current": round(rate * 100, 1),
                        "baseline": "10.0",
                        "severity": "HIGH" if rate > 0.4 else "MEDIUM",
                        "z_score": round(rate / 0.1, 2),
                        "direction": "SPIKE",
                    })
        except Exception as e:
            log.warning(f"[Evaluator] 실패율 분석 오류: {e}")

    # ┌─────────────────────────────────────────────────────────────┐
    # │  CHECK 2: Micro DB Anomalies (Quality, Latency, Marketing)  │
    # └─────────────────────────────────────────────────────────────┘

    # ── 2-A. [NEW] 평균 안정성 점수 하락 (0.7 미만이면 주의) ──
    avg_stability = db_raw.get("avg_stability", 0.0)
    if avg_stability > 0 and avg_stability < 0.7:
        anomalies.append({
            "metric": "avg_stability_score",
            "current": avg_stability,
            "baseline": 0.85,
            "severity": "HIGH" if avg_stability < 0.5 else "MEDIUM",
            "direction": "DROP",
            "desc": "생성된 브릭의 물리적 안정성이 크게 떨어짐"
        })

    # ── 2-B. [NEW] 생성 시간 지연 (평균 60초 초과 시 주의) ──
    avg_gen_time = db_raw.get("avg_gen_time", 0.0)
    if avg_gen_time > 60:
        anomalies.append({
            "metric": "avg_generation_time",
            "current": f"{avg_gen_time}s",
            "baseline": "30s",
            "severity": "HIGH" if avg_gen_time > 120 else "MEDIUM",
            "direction": "DELAY",
            "desc": "AI 엔진 처리 속도 저하 감지"
        })

    # ── 2-C. [NEW] 마케팅 효율/트래픽 품질 감지 ──
    # DAU는 급증했는데(SPIKE), 생성 시도는 늘지 않았다면 허수 유입 가능성
    if dau_spike:
        total_gens = db_raw.get("total_jobs_24h", 0)
        # 평소 100명당 10개 생성한다고 가정 (10%)
        # 트래픽 대비 생성 비율이 너무 낮으면 마케팅 효율 저하로 의심
        daily_count = metrics.get("daily_users", [])[-1].get("activeUsers", 1) if metrics.get("daily_users") else 1
        conversion_rate = total_gens / max(daily_count, 1)
        
        if conversion_rate < 0.05: # 5% 미만이면 체리피커 유입 의심
            anomalies.append({
                "metric": "traffic_quality_drop",
                "current": f"{round(conversion_rate*100, 1)}%",
                "baseline": "10.0%",
                "severity": "MEDIUM",
                "direction": "DROP",
                "desc": "트래픽 급증 대비 실제 사용 전환율 저조 (저품질 유입/마케팅 효율 의심)"
            })


    # ── 3. 종합 위험 점수 (Threshold Tuned) ──
    # HIGH = 0.5 (하나만 있어도 즉시 리포트 전환)
    # MEDIUM = 0.2 (최소 3개는 모여야 리포트 전환)
    # Threshold = 0.5
    risk = sum(0.5 if a["severity"] == "HIGH" else 0.2 for a in anomalies)
    risk = min(1.0, risk)
    
    next_step = "diagnose" if risk >= 0.5 else "report_green"

    log.info(f"[Evaluator] 완료: {len(anomalies)}건 이상, risk={risk} → {next_step}")

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
    tags = state.get("raw_metrics", {}).get("top_tags", [])

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
        top_tags=json.dumps(tags[:10], ensure_ascii=False),
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
    log.info("[ContentMiner] 검열 대상 수집 시작...")

    # 최근 1일 내의 미검열 콘텐츠 최대 50개 수집
    contents = await backend_client.get_recent_contents(days=1, limit=50)

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

    prompt = REPORTER_GREEN_PROMPT.format(
        active_users=summary.get('activeUsers', 'N/A'),
        page_views=summary.get('pageViews', 'N/A'),
        sessions=summary.get('sessions', 'N/A'),
        trend_desc=trend_desc,
        top_tags=', '.join(f"#{t.get('tag', '알수없음')}" for t in tags[:7]),
        day_of_week=temporal.get('day_of_week'),
        hour=temporal.get('hour'),
        is_peak=temporal.get('is_peak')
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

    dx = state.get("diagnosis", {})
    actions = state.get("proposed_actions", [])
    anomalies = state.get("anomalies", [])
    mod_results = state.get("moderation_results", [])
    temporal = state.get("temporal_context", {})

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
        history_context=history_context,
        user_query=user_query,
        summary=json.dumps(summary, ensure_ascii=False),
        today_gen_success=today.get('gen_success'),
        today_gen_fail=today.get('gen_fail'),
        today_gallery=today.get('gallery_uploads'),
        total_jobs=db_raw.get('total_jobs_24h'),
        stage_dist=json.dumps(db_raw.get('stage_dist', {}), ensure_ascii=False),
        daily=json.dumps(daily, ensure_ascii=False),
        tags=json.dumps(tags[:10], ensure_ascii=False),
        top_posts=json.dumps(top_posts, ensure_ascii=False),
        temporal=json.dumps(temporal, ensure_ascii=False)
    )

    res = await call_llm_json(prompt)
    report = res.get("report") if isinstance(res, dict) else str(res)
    
    if not report or report == "None":
        from .llm_utils import call_llm_text
        report = await call_llm_text(prompt)

    # 이력 업데이트는 호출부에서 처리하도록 제안 (현재 노드에서는 결과만 반환)
    return {"final_report": report, "next_action": "end"}
