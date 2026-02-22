# route/admin_analytics.py
"""Admin AI Analytics Endpoints"""
from fastapi import APIRouter, HTTPException, Query, Body, Request
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging

from service.analytics_agent_service import AnalyticsAgentService
from admin_analyst import analyst_graph

logger = logging.getLogger("api.admin.analytics")
router = APIRouter()

# --- Models ---
class AnalyticsReportResponse(BaseModel):
    report: str
    days: int

class AnomalyResponse(BaseModel):
    status: str
    message: str
    today: int
    previous_average: float
    drop_rate: float

@router.get("/ai-report", response_model=AnalyticsReportResponse)
async def get_ai_analytics_report(request: Request, days: int = Query(7, ge=1, le=30)):
    agent: AnalyticsAgentService = request.app.state.analytics_agent
    if not agent:
        raise HTTPException(status_code=500, detail="Analytics Agent not initialized")
    
    report = await agent.get_analyst_report(days)
    return {"report": report, "days": days}

@router.get("/check-anomaly", response_model=AnomalyResponse)
async def check_analytics_anomaly(request: Request):
    agent: AnalyticsAgentService = request.app.state.analytics_agent
    if not agent:
        raise HTTPException(status_code=500, detail="Analytics Agent not initialized")
    
    result = await agent.run_anomaly_detection()
    return result

# Global state for Admin Analysis to prevent parallel execution
_analysis_lock = False

@router.post("/deep-analyze")
async def deep_analyze():
    global _analysis_lock
    if _analysis_lock:
        logger.warning("⚠️ [Admin] Deep Analysis is already running, skipping...")
        return {"status": "busy", "message": "Analysis is already in progress"}

    logger.info("🧠 [Admin] LangGraph Deep Analysis 시작...")
    _analysis_lock = True
    try:
        initial_state = {
            "raw_metrics": {},
            "temporal_context": {},
            "anomalies": [],
            "risk_score": 0.0,
            "diagnosis": None,
            "proposed_actions": [],
            "iteration": 0,
            "max_iterations": 1,
            "next_action": "mine",
            "final_report": None,
            "moderation_queue": [],
            "moderation_results": [],
            "history": [],
        }

        result = await analyst_graph.ainvoke(initial_state)

        logger.info(f"✅ [Admin] Deep Analysis 완료 (risk={result.get('risk_score', 0)})")
        
        final_report = result.get("final_report")
        if not final_report:
            final_report = "보고서 생성에 실패했습니다."

        return {
            "status": "success",
            "report": final_report,
            "risk_score": result.get("risk_score", 0),
            "anomalies": result.get("anomalies", []),
            "diagnosis": result.get("diagnosis") or {},
            "proposed_actions": result.get("proposed_actions", []),
            "moderation_results": result.get("moderation_results", []),
            "iteration": result.get("iteration", 0),
        }
    except Exception as e:
        logger.error(f"❌ [Admin] Deep Analysis 실패: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis Failed: {str(e)}")
    finally:
        _analysis_lock = False

@router.post("/query")
async def query_analytics(body: Dict[str, Any] = Body(...)):
    query = body.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    logger.info(f"💬 [Admin] 인터랙티브 쿼리 요청: {query[:50]}...")
    
    try:
        from admin_analyst.nodes import miner_node, query_analyst_node
        
        state = await miner_node({"user_query": query})
        state["user_query"] = query
        result = await query_analyst_node(state)
        
        return {
            "status": "success",
            "answer": result.get("final_report", "답변 생성 실패"),
        }
    except Exception as e:
        logger.error(f"❌ [Admin] Query Analysis 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))
