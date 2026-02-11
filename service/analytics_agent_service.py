import logging
import os
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
        백엔드 API에서 데이터를 가져와 LLM이 해석한 보고서를 반환합니다.
        """
        summary = await backend_client.get_analytics_summary(days)
        daily_users = await backend_client.get_daily_users(days)
        
        if not summary:
            return "현재 분석 데이터를 불러올 수 없습니다. 백엔드 연결을 확인해주세요."

        # LLM에게 전달할 컨텍스트 구성
        context = f"""
[Brickers GA4 Analytics Data - Last {days} days]
- Total Active Users: {summary.get('activeUsers')}
- Total Page Views: {summary.get('pageViews')}
- Total Sessions: {summary.get('sessions')}

[Daily Users Trend]
{daily_users}
"""
        prompt = f"""
You are the 'Brickers Data Analyst Agent'. 
Based on the following GA4 data, provide a brief, professional, and friendly analysis in Korean.
Focus on:
1. Overall performance trend.
2. Any notable insights (growth, user engagement).
3. Suggestions for improvement.

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
