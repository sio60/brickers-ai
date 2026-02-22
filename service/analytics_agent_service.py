import logging
import os
import asyncio
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from service import backend_client

log = logging.getLogger(__name__)

class AnalyticsAgentService:
    def __init__(self, http_client=None):
        """
        http_client: LLM 호출을 위한 httpx AsyncClient (OpenAI/Gemini 호환 API)
        """
        self.http = http_client
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")

    async def get_analyst_report(self, days: int = 7) -> str:
        """
        1번 기능: 데이터 분석가 에이전트
        IntelligenceService에서 통합 데이터를 가져와 LLM이 해석한 보고서를 반환합니다.
        """
        from admin_analyst.intelligence_service import IntelligenceService
        
        snapshot = await IntelligenceService.get_combined_snapshot(days)
        
        analytics_summary = snapshot.get("analytics_summary")
        product_intel = snapshot.get("product_intelligence")
        performance = snapshot.get("performance", {})
        top_tags = snapshot.get("popular_trends", [])
        heavy_users = snapshot.get("heavy_users", [])
        
        if not analytics_summary:
            return "현재 분석 데이터를 불러올 수 없습니다. 백엔드 연결을 확인해주세요."

        # LLM에게 전달할 컨텍스트 구성
        context = f"""
[Brickers GA4 Analytics Data - Last {days} days]
- Active Users: {analytics_summary.get('activeUsers', 'N/A')}
- Page Views: {analytics_summary.get('pageViews', 'N/A')}
- Sessions: {analytics_summary.get('sessions', 'N/A')}

[Product Intelligence (Conversion & Churn)]
{json.dumps(product_intel, ensure_ascii=False, indent=2)}

[Popular Tags (Top 10)]
{top_tags}

[System Performance]
- Avg Wait Time: {performance.get('avgWaitTime', 0)/1000:.1f}s
- Avg Latency (AI): {performance.get('avgLatency', 0)/1000:.1f}s
- Avg Cost: ${performance.get('avgCost', 0):.4f}
- Avg Brick Count: {performance.get('avgBrickCount', 0):.1f}

[Power Users (Top 5)]
{heavy_users}
"""
        prompt = f"""
You are the 'Brickers Data Analyst Agent'. 
Based on the following GA4 data, provide a very detailed and professional analysis in Korean.
Focus on:
1. Overall performance trend and daily user patterns.
2. Popular brick themes and trends based on 'Popular Brick Tags'.
3. User engagement insights looking at total metrics and 'Key Active Users'.
4. Specific, actionable suggestions for product improvement or marketing based on the data.

Rules:
- Format the report with clear headings and bullet points.
- provide strategic insights for a LEGO-like toy/creation platform.
- Tone: Insightful, professional, yet encouraging. Use Korean.

Data:
{context}
"""
        return await self._call_llm(prompt)

    async def run_anomaly_detection(self) -> Dict[str, Any]:
        """
        3번 기능: 이상 징후 감지
        최근 데이터를 분석하여 급격한 하락이나 이상 현상을 탐지합니다.
        """
        # 최근 7일간의 성공량 확인
        stats = await backend_client.get_event_stats("generate_success", days=7)
        if not stats or len(stats) < 2:
            return {"status": "insufficient_data", "message": "데이터가 부족하여 분석할 수 없습니다."}

        # 오늘 데이터와 평균 비교 (간단한 로직)
        # stats: [{"date": "20260211", "count": 10}, ...]
        counts = [s['count'] for s in stats]
        today_count = counts[-1]
        prev_avg = sum(counts[:-1]) / len(counts[:-1]) if len(counts) > 1 else today_count

        threshold = 0.5 # 50% 이하로 떨어지면 경고
        is_anomaly = today_count < (prev_avg * threshold) and prev_avg > 5

        result = {
            "status": "anomaly" if is_anomaly else "normal",
            "today": today_count,
            "previous_average": round(prev_avg, 2),
            "drop_rate": round((1 - today_count/prev_avg) * 100, 1) if prev_avg > 0 else 0
        }

        if is_anomaly:
            result["message"] = f"🚨 경고: 브릭 생성 성공률이 평소 대비 {result['drop_rate']}% 하락했습니다. 서버 확인이 필요합니다."
        else:
            result["message"] = "✅ 서비스 상태가 정상입니다."
        
        return result

    async def _call_llm(self, prompt: str) -> str:
        if not self.http:
            return "LLM 클라이언트가 설정되지 않았습니다."
        
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        }
        try:
            resp = await self.http.post("chat/completions", json=body)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            log.error(f"LLM call failed in AnalyticsAgent: {e}")
            return f"보고서 생성 중 오류가 발생했습니다: {str(e)}"

    async def get_personalized_quality_advice(self, user_id: str) -> str:
        """
        AI 에이전트 기반 데이터 품질 향상 제안 기능.
        유저의 과거 행동 패턴을 분석하여 고품질 결과물을 위한 맞춤형 팁을 제공합니다.
        """
        activity = await backend_client.get_user_activity(user_id, days=30)
        
        if not activity:
            return "유저의 활동 데이터가 부족하여 맞춤 제안을 생성할 수 없습니다."

        # 활동 내역 요약 (예: 어떤 이벤트를 많이 했는지)
        context = f"User {user_id} Activity (Last 30 days):\n{activity}"
        
        prompt = f"""
You are the 'Brickers Quality Pro Agent'. 
Analyze the following user activity data and provide a personalized advice to improve their 'Bricker Creation' quality.
Focus on helping them choose better prompts, themes, or settings based on their past successful/failed patterns.

Rules:
- If they have many 'generate_success' but low 'download', suggest how to make the results more 'printable' or 'useful'.
- If they have many 'generate_fail', analyze potential issues and give prompt engineering tips.
- Tone: Friendly, expert-like, and proactive. Use Korean.

User Data:
{context}
"""
        return await self._call_llm(prompt)
