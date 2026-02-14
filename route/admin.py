# route/admin.py
"""Admin API for System Monitoring & Logs"""
from __future__ import annotations

import os
import json
import docker
import logging # Added import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, Body, Request
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from db import get_db


# Logger configuration
logger = logging.getLogger("api.admin") # Added logger setup

try:
    from brick_engine.agent.log_analyzer.graph import app as log_agent_app
except ImportError:
    # Local dev path fallback if needed
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from brick_engine.agent.log_analyzer.graph import app as log_agent_app

from service.analytics_agent_service import AnalyticsAgentService
from admin_analyst import analyst_graph

# Create router (Prefix를 쓰지 않고 데코레이터에서 직접 명시)
router = APIRouter(tags=["admin"])

print("[Admin] Initializing admin routes...", flush=True)

@router.get("/ai-admin/ping")
def ping_admin():
    print("[Admin] Ping received!", flush=True)
    return {"status": "ai_admin_ok", "timestamp": datetime.now().isoformat()}

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


@router.get("/ai-admin/analytics/ai-report")
async def get_ai_analytics_report(request: Request, days: int = Query(7, ge=1, le=30)):
    agent: AnalyticsAgentService = request.app.state.analytics_agent
    if not agent:
        raise HTTPException(status_code=500, detail="Analytics Agent not initialized")
    
    report = await agent.get_analyst_report(days)
    return {"report": report, "days": days}

@router.get("/ai-admin/analytics/check-anomaly", response_model=AnomalyResponse)
async def check_analytics_anomaly(request: Request):
    """
    [NEW] 이상 징후 감지 실행
    """
    agent: AnalyticsAgentService = request.app.state.analytics_agent
    if not agent:
        raise HTTPException(status_code=500, detail="Analytics Agent not initialized")
    
    result = await agent.run_anomaly_detection()
    return result


@router.post("/ai-admin/analytics/deep-analyze")
async def deep_analyze():
    """
    [NEW] LangGraph 기반 심층 분석 실행.
    데이터 수집 → 이상 탐지 → 인과 추론 → 전략 수립 파이프라인.
    """
    logger.info("🧠 [Admin] LangGraph Deep Analysis 시작...")
    try:
        initial_state = {
            "raw_metrics": {},
            "temporal_context": {},
            "anomalies": [],
            "risk_score": 0.0,
            "diagnosis": None,
            "proposed_actions": [],
            "iteration": 0,
            "max_iterations": 3,
            "next_action": "mine",
            "final_report": None,
        }

        result = await analyst_graph.ainvoke(initial_state)

        logger.info(f"🔍 [Admin] Deep Analysis Result Type: {type(result)}")
        logger.info(f"🔍 [Admin] Deep Analysis Result Content: {str(result)[:500]}")

        # If result is a list, try to get the last element (state)
        if isinstance(result, list):
             logger.warning("⚠️ [Admin] Result is a list, using the last element as state.")
             if result:
                 result = result[-1]
             else:
                 result = {}

        logger.info(f"✅ [Admin] Deep Analysis 완료 (risk={result.get('risk_score', 0)})")
        return {
            "status": "success",
            "report": result.get("final_report", "보고서 생성 실패"),
            "risk_score": result.get("risk_score", 0),
            "anomalies": result.get("anomalies", []),
            "diagnosis": result.get("diagnosis"),
            "proposed_actions": result.get("proposed_actions", []),
            "moderation_results": result.get("moderation_results", []), # ✅ [NEW]
            "iteration": result.get("iteration", 0),
        }
    except Exception as e:
        logger.error(f"❌ [Admin] Deep Analysis 실패: {e}")
        raise HTTPException(status_code=500, detail=f"Deep Analysis Failed: {str(e)}")
 
 
@router.post("/ai-admin/analytics/query")
async def query_analytics(body: Dict[str, Any] = Body(...)):
    """
    [NEW] 관리자의 자연어 질문에 따른 커스텀 분석 수행.
    """
    query = body.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    logger.info(f"💬 [Admin] 인터랙티브 쿼리 요청: {query[:50]}...")
    
    try:
        from admin_analyst.nodes import miner_node, query_analyst_node
        
        # 1. 데이터 수집
        state = await miner_node({"user_query": query})
        
        # 2. 질문 기반 분석
        state["user_query"] = query
        result = await query_analyst_node(state)
        
        return {
            "status": "success",
            "answer": result.get("final_report", "답변 생성 실패"),
        }
    except Exception as e:
        logger.error(f"❌ [Admin] Query Analysis 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ai-admin/moderation/restore")
async def restore_moderated_content(body: Dict[str, Any] = Body(...)):
    """
    [NEW] AI가 숨긴 콘텐츠를 관리자가 수동으로 복구.
    """
    from service import backend_client
    target_type = body.get("type")
    target_id = body.get("targetId")

    if not target_type or not target_id:
        raise HTTPException(status_code=400, detail="Missing type or targetId")

    logger.info(f"♻️ [Admin] 콘텐츠 복구 시도: {target_type} {target_id}")
    
    # backend_client에 restore_content 함수를 미리 만들어두었다고 가정 (또는 추가 예정)
    success = await backend_client.restore_content(target_type, target_id)
    
    if success:
        return {"status": "success", "message": "Content restored"}
    else:
        raise HTTPException(status_code=500, detail="Failed to restore content")

class LogResponse(BaseModel):
    container: str
    logs: str

class AnalysisRequest(BaseModel):
    container_name: str = "brickers-ai-container"
    job_id: Optional[str] = None

class AnalysisResponse(BaseModel):
    container: str
    is_error: bool
    plain_summary: str           # [NEW] 관리자용 한글 요약
    user_impact_level: str       # [NEW] critical | high | low
    suggested_actions: List[str] # [NEW] 권장 조치 목록
    business_insight: Optional[str] = None
    job_id: Optional[str] = None

class ArchivedLogResponse(BaseModel):
    job_id: str
    logs: str
    timestamp: str
    container: str


class ArchiveLogRequest(BaseModel):
    job_id: str
    logs: str
    container_name: str = "brickers-ai-container"
    status: str = "FAILED"  # [추가] 상태 수신 (기본값: FAILED)
    client_timestamp: Optional[str] = None  # [NEW] Race Condition 방지용 timestamp


class SystemLogRequest(BaseModel):
    logs: List[str] # [CHANGE] str -> List[str] for $push
    container_name: str = "brickers-ai-container"
    timestamp: str
    session_id: str # [NEW]

# ... (middle parts omitted, keeping existing code) ...

@router.post("/ai-admin/archive/system")
async def archive_system_log(request: SystemLogRequest):
    """
    [NEW] Archive system-level logs to 'system_logs' collection.
    Grouping: (session_id, date)
    Action: $push to 'logs' array
    """
    try:
        db = get_db()
        if db is None:
            raise HTTPException(status_code=503, detail="Database connection unavailable")
        
        collection = db["system_logs"]
        
        # Extract Date (YYYY-MM-DD) from timestamp or current time
        try:
            dt = datetime.fromisoformat(request.timestamp.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y-%m-%d")
        except:
            date_str = datetime.utcnow().strftime("%Y-%m-%d")

        # Query Key
        filter_query = {
            "session_id": request.session_id,
            "date": date_str
        }
        
        # Update Operation ($push with $each)
        update_op = {
            "$push": {
                "logs": {"$each": request.logs}
            },
            "$setOnInsert": {
                "container": request.container_name,
                "created_at": datetime.utcnow(),
                "session_id": request.session_id,
                "date": date_str
            },
            "$set": {
                "last_updated": datetime.utcnow()
            }
        }
        
        collection.update_one(filter_query, update_op, upsert=True)
        return {"status": "success"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/logs/{container_name}", response_model=LogResponse)
def get_container_logs(
    container_name: str, 
    tail: int = Query(100, ge=1, le=2000),
    since_seconds: Optional[int] = Query(None, ge=1)
):
    """
    Fetch logs from a running Docker container (Read-Only).
    """
    logger.info(f"🌐 [API: GET /logs/{container_name}] Requesting last {tail} lines.")
    try:
        # Connect to Docker Socket
        client = docker.from_env()
        
        # Get container
        container = client.containers.get(container_name)
        logger.info(f"📦 [admin.py] Connected to container: {container.id[:12]}")
        
        # Fetch logs
        log_bytes = container.logs(tail=tail, since=since_seconds)
        log_str = log_bytes.decode("utf-8", errors="replace")
        
        logger.info(f"✅ [admin.py] Fetched {len(log_str)} bytes of logs.")
        return {
            "container": container_name,
            "logs": log_str
        }

    except docker.errors.NotFound:
        logger.error(f"❌ [admin.py] Container '{container_name}' not found.")
        raise HTTPException(status_code=404, detail=f"Container '{container_name}' not found")
    except Exception as e:
        logger.error(f"❌ [admin.py] Error fetching logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Error: {str(e)}")


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_logs(request: AnalysisRequest = Body(...)):
    """
    Trigger AI Log Analysis Agent.
    """
    logger.info(f"🌐 [API: POST /analyze] Container: {request.container_name}")
    try:
        # 1. Initialize State
        initial_state = {
            "container_name": request.container_name,
            "logs": "",
            "analysis_result": None,
            "messages": [],
            "iteration": 0,
            "investigation_notes": []
        }
        
        # 2. Run Agent
        logger.info(f"🧠 [admin.py] Invoking LangGraph Agent for {request.container_name}...")
        result_state = await log_agent_app.ainvoke(initial_state)
        logger.info(f"✅ [admin.py] Agent execution finished after {result_state.get('iteration')} iterations.")
        
        # 3. Parse Result
        raw_result_str = result_state.get("analysis_result")
        if not raw_result_str:
            logger.error("❌ [admin.py] Agent returned EMPTY analysis_result.")
            raise HTTPException(status_code=500, detail="Agent returned no analysis result")
            
        try:
            result_json = json.loads(raw_result_str)
            analysis_data = result_json.get("analysis", result_json)
            logger.info(f"📝 [admin.py] Final Analysis Summary: {analysis_data.get('summary', 'No summary')[:100]}...")

        except json.JSONDecodeError:
             logger.warning(f"⚠️ [admin.py] AI Response is not valid JSON: {raw_result_str[:200]}")
             return {
                "container": request.container_name,
                "is_error": True,
                "summary": "AI 응답 파싱 실패",
                "root_cause": raw_result_str,
                "suggestion": "Log Agent 로직을 점검하세요."
             }

        # --- [NEW] Auto-Archive Log to DB ---
        job_id = result_state.get("job_id")
        logs_content = result_state.get("logs", "")
        
        if job_id and logs_content:
            try:
                db = get_db()
                if db is not None:
                    collection = db["failed_job_logs"]
                    doc = {
                        "jobId": job_id,
                        "logs": logs_content,
                        "timestamp": datetime.utcnow().isoformat(),
                        "container": request.container_name,
                        "status": "FAILED",
                        "bia_insight": {
                            "summary": result_state.get("plain_summary"),
                            "impact": result_state.get("user_impact_level"),
                            "actions": result_state.get("suggested_actions"),
                            "business": result_state.get("business_insight")
                        }
                    }
                    collection.replace_one({"jobId": job_id}, doc, upsert=True)
                    logger.info(f"💾 [admin.py] Automatically archived logs for Job ID: {job_id}")
                else:
                    logger.warning("⚠️ [admin.py] DB connection unavailable, skipping auto-archive.")
            except Exception as db_err:
                logger.error(f"❌ [admin.py] Failed to auto-archive logs: {db_err}")
        # ------------------------------------

        # 4. Return Structured Response
        return {
            "container": request.container_name,
            "is_error": True,
            "plain_summary": result_state.get("plain_summary") or "분석 완료",
            "user_impact_level": result_state.get("user_impact_level") or "low",
            "suggested_actions": result_state.get("suggested_actions") or [],
            "business_insight": result_state.get("business_insight"),
            "job_id": job_id
        }

    except Exception as e:
        logger.error(f"❌ [admin.py] Analysis Failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis Failed: {str(e)}")


@router.post("/ai-admin/archive")
async def archive_log(request: ArchiveLogRequest):
    """
    Archive job logs to MongoDB.
    FAILED 상태인 경우 AI 에이전트 분석도 함께 실행하여 결과를 저장합니다.
    """
    logger.info(f"🌐 [API: POST /archive] Archiving logs for Job ID: {request.job_id} ({request.status})")
    try:
        db = get_db()
        if db is None:
            raise HTTPException(status_code=503, detail="Database connection unavailable")
            
        collection = db["failed_job_logs"]
        
        # [NEW] Race Condition Check & Safe Update
        # 1. 문서가 존재하는지 확인
        existing_doc = collection.find_one({"jobId": request.job_id})
        
        should_update = True
        if existing_doc and request.client_timestamp:
            # 2. 기존 문서에 client_timestamp가 있고, 요청된 timestamp가 더 오래된 경우 업데이트 Skip
            # [CHANGE] 단, 로그 길이가 더 길어졌다면 (내용이 추가되었다면) 타임스탬프가 조금 뒤집혀도 업데이트 허용
            last_ts = existing_doc.get("client_timestamp")
            last_logs = existing_doc.get("logs", "")
            
            is_newer_ts = (last_ts is None) or (request.client_timestamp >= last_ts)
            is_more_logs = len(request.logs) > len(last_logs)

            if not is_newer_ts and not is_more_logs:
                logger.warning(f"⚠️ [admin.py] Skipping archival: Old request & No new content ({last_ts} > {request.client_timestamp})")
                should_update = False
        
        if should_update:
            # 3. Safe Update using $set (Preserves other fields like ai_analysis)
            update_fields = {
                "logs": request.logs,
                "timestamp": datetime.utcnow().isoformat(),
                "container": request.container_name,
                "status": request.status
            }
            if request.client_timestamp:
                update_fields["client_timestamp"] = request.client_timestamp
                
            collection.update_one(
                {"jobId": request.job_id},
                {"$set": update_fields},
                upsert=True
            )
            logger.info(f"✅ [admin.py] Archived logs for {request.job_id} (Safe Update)")
        else:
            logger.info(f"⏭️ [admin.py] Skipped logs for {request.job_id} (Outdated)")

        # --- AI 분석 자동 실행 로직 제거 (Decoupled) ---
        # 이제 BIA 인사이트는 관리자 페이지에서 '분석' 버튼을 누를 때만 (on-demand) 실행됩니다.
        # 이를 통해 백그라운드 작업의 오버헤드를 줄이고 AI를 독립적인 도구로 분리합니다.
        
        # [Restored as comments per user request]
        # ai_analysis = None
        # if request.status == "FAILED" and request.logs:
        #     try:
        #         logger.info(f"🧠 [admin.py] FAILED 로그 감지 → AI 분석 시작 (Job: {request.job_id})")
        #         analysis_state = {
        #             "container_name": request.container_name,
        #             "logs": request.logs,
        #             "analysis_result": None,
        #             "messages": [],
        #             "iteration": 0,
        #             "job_id": request.job_id,
        #             "investigation_notes": []
        #         }
        #         result_state = await log_agent_app.ainvoke(analysis_state)
        #         raw_result = result_state.get("analysis_result", "")
        #         if raw_result:
        #             result_json = json.loads(raw_result)
        #             ai_analysis = result_json.get("analysis", result_json)
        #             collection.update_one(
        #                 {"jobId": request.job_id},
        #                 {"$set": {
        #                     "bia_insight": {
        #                         "summary": result_state.get("plain_summary"),
        #                         "impact": result_state.get("user_impact_level"),
        #                         "actions": result_state.get("suggested_actions"),
        #                         "business": result_state.get("business_insight")
        #                     },
        #                     "ai_analyzed_at": datetime.utcnow().isoformat()
        #                 }}
        #             )
        #             logger.info(f"✅ [admin.py] AI 분석 완료 & 저장 (Job: {request.job_id})")
        #     except Exception as ai_err:
        #         logger.error(f"⚠️ [admin.py] AI 분석 실패 (로그 저장은 정상): {ai_err}")

        return {
            "status": "success",
            "jobId": request.job_id
        }
        
    except Exception as e:
        logger.error(f"❌ [admin.py] Archive Failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Archive Failed: {str(e)}")



@router.get("/ai-admin/archived/{job_id}", response_model=ArchivedLogResponse)
async def get_archived_log(job_id: str):
    """
    Fetch archived logs from MongoDB.
    """
    logger.info(f"🌐 [API: GET /archived/{job_id}] Fetching archived logs.")
    try:
        db = get_db()
        if db is None:
            raise HTTPException(status_code=503, detail="Database connection unavailable")
            
        collection = db["failed_job_logs"]
        doc = collection.find_one({"jobId": job_id})
        
        if not doc:
            raise HTTPException(status_code=404, detail=f"Logs for Job ID {job_id} not found")
            
        return {
            "job_id": doc["jobId"],
            "logs": doc["logs"],
            "timestamp": doc["timestamp"],
            "container": doc.get("container", "unknown")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [admin.py] Fetch Archived Failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Fetch Archived Failed: {str(e)}")
