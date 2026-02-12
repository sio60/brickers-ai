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
    """이상 없을 때 간결한 녹색 보고서."""
    log.info("[Reporter] 정상 보고서 생성...")

    s = state.get("raw_metrics", {}).get("summary", {})
    daily = state.get("raw_metrics", {}).get("daily_users", [])
    tags = state.get("raw_metrics", {}).get("top_tags", [])
    t = state.get("temporal_context", {})

    # 트렌드
    trend = ""
    if daily and len(daily) >= 2:
        try:
            c = [d.get("count", d.get("activeUsers", 0)) for d in daily]
            r_avg = sum(c[-3:]) / min(3, len(c))
            p_avg = sum(c[:-3]) / max(1, len(c) - 3) if len(c) > 3 else r_avg
            chg = ((r_avg - p_avg) / max(p_avg, 1)) * 100
            trend = f"최근 3일 {'📈 증가' if chg > 0 else '📉 감소'} ({chg:+.1f}%)"
        except Exception:
            trend = "계산 불가"

    top = ", ".join(f"#{x.get('tag', x.get('name', '?'))}" for x in (tags or [])[:5]) or "없음"

    report = f"""## ✅ 서비스 정상 운영 중

| 항목 | 값 |
|------|-----|
| 활성 유저 | {s.get('activeUsers', 'N/A')}명 |
| 페이지뷰 | {s.get('pageViews', 'N/A')} |
| 세션 수 | {s.get('sessions', 'N/A')} |
| 이상 징후 | 없음 |
| 트렌드 | {trend} |
| 인기 태그 | {top} |
| 분석 시각 | {t.get('date', '')} {t.get('hour', '')}시 |

> 모든 핵심 지표가 정상 범위 입니다. 🎉"""

    return {"final_report": report, "next_action": "end"}


# ═══════════════════════════════════════════════════════════════
# Node 7: Finalizer — 이상 발견 시 종합 보고서
# ═══════════════════════════════════════════════════════════════
async def finalizer_node(state: AdminAnalystState) -> dict:
    """종합 분석 보고서 생성 (이상 징후 + 원인 + 전략)."""
    log.info("[Finalizer] 종합 보고서 생성...")

    dx = state.get("diagnosis", {})
    actions = state.get("proposed_actions", [])
    anomalies = state.get("anomalies", [])

    a_lines = []
    for a in anomalies:
        icon = "🔴" if a.get("direction") == "DROP" else "🔺"
        a_lines.append(
            f"- {icon} **{a.get('metric')}**: 현재 {a.get('current')} vs "
            f"기준 {a.get('baseline')} (심각도: {a.get('severity')}, Z: {a.get('z_score')})"
        )

    act_lines = []
    for i, ac in enumerate(actions, 1):
        ri = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(ac.get("risk", ""), "⚪")
        act_lines.append(
            f"{i}. **{ac.get('action')}** → {ac.get('target', '?')}\n"
            f"   - 기대: {ac.get('expected_impact', '?')} | 리스크: {ri} {ac.get('risk', '?')}"
        )

    ev = "\n".join(f"- {e}" for e in dx.get("evidence", []))

    report = f"""## 🚨 이상 징후 분석 보고서

### 감지된 이상
{chr(10).join(a_lines) or '- 없음'}

### 근본 원인
> {dx.get('root_cause', '분석 불가')}

### 증거
{ev or '- 없음'}

### 영향 범위
{dx.get('affected_segment', '전체 유저')}

### 제안 조치 ({len(actions)}건)
{chr(10).join(act_lines) or '- 없음'}

---
📊 확신도: {dx.get('confidence', 0) * 100:.0f}% | 🔄 반복: {state.get('iteration', 1)}회 | ⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}"""

    return {"final_report": report, "next_action": "end"}
