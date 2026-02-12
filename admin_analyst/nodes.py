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
from .llm_utils import call_llm_json

log = logging.getLogger("admin_analyst.nodes")


# ═══════════════════════════════════════════════════════════════
# Node 1: Miner — 데이터 수집
# ═══════════════════════════════════════════════════════════════
async def miner_node(state: AdminAnalystState) -> dict:
    """GA4 Data API + 백엔드에서 원본 지표 수집."""
    from service import backend_client

    log.info("[Miner] 데이터 수집 시작...")

    summary = await backend_client.get_analytics_summary(7)
    daily = await backend_client.get_daily_users(14)
    tags = await backend_client.get_top_tags(7, limit=15)
    users = await backend_client.get_heavy_users(7, limit=10)
    fail_stats = await backend_client.get_event_stats("generate_fail", 7)
    success_stats = await backend_client.get_event_stats("generate_success", 7)

    now = datetime.now()
    temporal = {
        "day_of_week": now.strftime("%a"),
        "is_weekend": now.weekday() >= 5,
        "hour": now.hour,
        "is_peak": 19 <= now.hour <= 23,
        "date": now.strftime("%Y-%m-%d"),
    }

    log.info(f"[Miner] 수집 완료: summary={bool(summary)}, daily={len(daily or [])}")

    return {
        "raw_metrics": {
            "summary": summary or {},
            "daily_users": daily or [],
            "top_tags": tags or [],
            "heavy_users": users or [],
            "fail_events": fail_stats or [],
            "success_events": success_stats or [],
        },
        "temporal_context": temporal,
        "next_action": "evaluate",
    }


# ═══════════════════════════════════════════════════════════════
# Node 2: Evaluator — 이상 탐지 (규칙 기반, LLM 미사용)
# ═══════════════════════════════════════════════════════════════
def evaluator_node(state: AdminAnalystState) -> dict:
    """Z-Score 기반 통계적 이상 탐지."""
    log.info("[Evaluator] 이상 탐지 시작...")
    anomalies: List[Dict[str, Any]] = []
    metrics = state.get("raw_metrics", {})

    # ── 1. DAU 급변 감지 ──
    daily = metrics.get("daily_users") or []
    if len(daily) >= 3:
        try:
            counts = [d.get("count", d.get("activeUsers", 0)) for d in daily]
            prev, today = counts[:-1], counts[-1]
            mean = sum(prev) / len(prev)
            std = (sum((x - mean) ** 2 for x in prev) / len(prev)) ** 0.5

            if std > 0:
                z = (today - mean) / std
                if abs(z) > 2.0:
                    anomalies.append({
                        "metric": "daily_active_users",
                        "current": today,
                        "baseline": round(mean, 1),
                        "severity": "HIGH" if abs(z) > 3 else "MEDIUM",
                        "z_score": round(z, 2),
                        "direction": "DROP" if z < 0 else "SPIKE",
                    })
        except Exception as e:
            log.warning(f"[Evaluator] DAU 분석 오류: {e}")

    # ── 2. 생성 실패율 급증 ──
    fail_ev = metrics.get("fail_events") or []
    succ_ev = metrics.get("success_events") or []
    if fail_ev and succ_ev:
        try:
            fc = [e.get("count", 0) for e in fail_ev]
            sc = [e.get("count", 0) for e in succ_ev]
            recent_fail = sum(fc[-3:]) if len(fc) >= 3 else sum(fc)
            recent_succ = sum(sc[-3:]) if len(sc) >= 3 else sum(sc)
            total = recent_fail + recent_succ

            if total > 5:
                rate = recent_fail / total
                prev_f = sum(fc[:-3]) if len(fc) > 3 else 0
                prev_s = sum(sc[:-3]) if len(sc) > 3 else 0
                prev_t = prev_f + prev_s
                prev_rate = prev_f / prev_t if prev_t > 0 else 0

                if rate > 0.2 and rate > prev_rate * 1.5:
                    anomalies.append({
                        "metric": "generation_fail_rate",
                        "current": round(rate, 3),
                        "baseline": round(prev_rate, 3),
                        "severity": "HIGH" if rate > 0.4 else "MEDIUM",
                        "z_score": round(rate / max(prev_rate, 0.01), 2),
                        "direction": "SPIKE",
                    })
        except Exception as e:
            log.warning(f"[Evaluator] 실패율 분석 오류: {e}")

    # ── 3. 종합 위험 점수 ──
    risk = min(1.0, sum(0.4 if a["severity"] == "HIGH" else 0.2 for a in anomalies))
    next_step = "diagnose" if risk >= 0.3 else "report_green"

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
    temporal = state.get("temporal_context", {})
    tags = state.get("raw_metrics", {}).get("top_tags", [])

    prompt = f"""당신은 브릭커스(Brickers) 서비스의 운영 분석 전문가입니다.
아래 이상 징후를 분석하고 근본 원인을 추론하세요.

[감지된 이상 징후]
{anomaly_text}

[서비스 지표]
- 활성 유저: {summary.get('activeUsers', 'N/A')}
- 페이지뷰: {summary.get('pageViews', 'N/A')}
- 세션 수: {summary.get('sessions', 'N/A')}

[인기 태그 TOP 5]
{json.dumps(tags[:5], ensure_ascii=False) if tags else '없음'}

[시간 맥락]
- 요일: {temporal.get('day_of_week')}, 시간: {temporal.get('hour')}시, 피크: {temporal.get('is_peak')}

다음 JSON으로만 응답하세요:
{{"root_cause":"근본 원인 한 줄 (한국어)","confidence":0.0~1.0,"evidence":["증거1","증거2"],"affected_segment":"영향 유저군"}}"""

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

    prompt = f"""브릭커스 서비스에서 문제가 감지되었습니다.

- 원인: {dx.get('root_cause', '?')}
- 확신도: {confidence}
- 증거: {json.dumps(dx.get('evidence', []), ensure_ascii=False)}
- 영향: {dx.get('affected_segment', '전체')}

실행 가능한 대응 전략을 최대 3개, 다음 JSON 배열로만 응답하세요:
[{{"action":"조치명","target":"대상","expected_impact":"효과","risk":"LOW|MEDIUM|HIGH","priority":1}}]"""

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

    prompt = f"""당신은 브릭커스(Brickers) 서비스의 데이터 과학자이자 운영 컨설턴트입니다.
현재 서비스의 주요 지표는 통계적으로 안정 범위에 있습니다. 
단순 지표 요약을 넘어, 데이터를 다각도로 해석하여 관리자에게 가치 있는 '심층 인사이트'를 제공하세요.

[수집된 데이터]
- 활성 유저(DAU): {summary.get('activeUsers', 'N/A')}
- 페이지뷰: {summary.get('pageViews', 'N/A')}
- 세션당 활동: {summary.get('sessions', 'N/A')}
- 현 시점 트렌드: {trend_desc}
- 인기 관심사(태그): {', '.join(f"#{t.get('tag', '알수없음')}" for t in tags[:5])}
- 시간대 맥락: {temporal.get('day_of_week')}요일 {temporal.get('hour')}시 (피크타임 여부: {temporal.get('is_peak')})

[보고서 구성 가이드]
1. '지표 해석 (Metrics Interpretation)': 현재 수치가 의미하는 서비스의 건강 상태
2. '유저 페르소나 및 행동 추론': 인기 태그와 시간대를 바탕으로 현재 어떤 유저층이 무엇을 위해 접속하는지 분석
3. '성장 기회 (Growth Opportunity)': 지표를 한 단계 더 끌어올리기 위한 구체적인 실험이나 마케팅 제안
4. 마크다운 형식을 적극 활용하여 가독성 있게 작성하세요.

다음 JSON으로만 응답하세요:
{{"report": "심층 분석 내용 (마크다운 형식)"}}"""

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
    metrics = state.get("raw_metrics", {})
    temporal = state.get("temporal_context", {})

    prompt = f"""당신은 브릭커스(Brickers) 서비스의 위기 대응 본부장입니다.
감지된 이상 징후에 대해 경영진이 즉시 의사결정을 내릴 수 있도록 '심층 분석 및 대응 보고서'를 작성하세요.

[수집된 이상 징후]
{json.dumps(anomalies, ensure_ascii=False, indent=2)}

[진단 결과 (원인)]
- 근본 원인: {dx.get('root_cause', '?')}
- 증거 및 영향: {json.dumps(dx.get('evidence', []), ensure_ascii=False)} / {dx.get('affected_segment', '전체')}

[권장 대응 전략]
{json.dumps(actions, ensure_ascii=False, indent=2)}

[보고서 작성 가이드]
1. 제목은 상황의 심각도를 나타내는 이모지와 함께 작성하세요 (예: � 긴급 대응 보고서)
2. '브리핑': 무엇이 문제이고 얼마나 심각한지 전문가 시각에서 한 문단 요약
3. '인과관계 분석': 왜 이런 일이 발생했는지 데이터와 맥락을 연결하여 설명
4. '우선순위 조치 계획': 제안된 전략들을 실행 순서와 기대 효과 중심으로 재구성
5. 마크다운 형식을 사용하여 가독성 있게 작성하세요 (테이블, 인용구 등 권장).

다음 JSON으로만 응답하세요:
{{"report": "종합 분석 보고서 내용 (마크다운 형식)"}}"""

    res = await call_llm_json(prompt)
    report = res.get("report") if res else None

    if not report:
        # Fallback 템플릿
        report = f"## 🚨 관리자 주의: 이상 징후 감지\n\n- 원인: {dx.get('root_cause', '?')}\n- 조치: {len(actions)}건의 전략 수립됨."

    return {"final_report": report, "next_action": "end"}
