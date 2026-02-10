# route/admin.py
"""Admin API for System Monitoring & Logs"""
from __future__ import annotations

import os
import json
import docker
import logging # Added import logging
from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# Logger configuration
logger = logging.getLogger("api.admin") # Added logger setup

# Create router
router = APIRouter(prefix="/api/admin", tags=["admin"])

# --- Models ---
class LogResponse(BaseModel):
    container: str
    logs: str

class AnalysisRequest(BaseModel):
    container_name: str = "brickers-ai-container"
    job_id: Optional[str] = None

class AnalysisResponse(BaseModel):
    container: str
    is_error: bool
    summary: str
    root_cause: Optional[str] = None
    suggestion: Optional[str] = None
    job_id: Optional[str] = None # 분석된 Job ID 반환

class ArchiveLogRequest(BaseModel):
    job_id: str
    logs: str
    container_name: str = "brickers-ai-container"

class ArchivedLogResponse(BaseModel):
    job_id: str
    logs: str
    timestamp: str
    container: str

# --- Agent & DB Import ---
from brick_engine.agent.log_analyzer import app as log_agent_app
from brick_engine.agent.yang_db import get_db
from datetime import datetime

# --- Endpoints ---

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
            "error_count": 0,
            "messages": [],
            "iteration": 0
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

        # 4. Return Structured Response
        return {
            "container": request.container_name,
            "is_error": analysis_data.get("error_found", False),
            "summary": analysis_data.get("summary", "분석 완료"),
            "root_cause": analysis_data.get("root_cause"),
            "suggestion": analysis_data.get("suggestion"),
            "job_id": result_state.get("job_id")
        }

    except Exception as e:
        logger.error(f"❌ [admin.py] Analysis Failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis Failed: {str(e)}")


@router.post("/archive")
async def archive_log(request: ArchiveLogRequest):
    """
    Archive failed job logs to MongoDB.
    """
    logger.info(f"🌐 [API: POST /archive] Archiving logs for Job ID: {request.job_id}")
    try:
        db = get_db()
        if db is None:
            raise HTTPException(status_code=503, detail="Database connection unavailable")
            
        collection = db["failed_job_logs"]
        doc = {
            "jobId": request.job_id,
            "logs": request.logs,
            "timestamp": datetime.utcnow().isoformat(),
            "container": request.container_name,
            "status": "FAILED"
        }
        collection.replace_one({"jobId": request.job_id}, doc, upsert=True)
        logger.info(f"✅ [admin.py] Archived logs for {request.job_id}")
        return {"status": "success", "jobId": request.job_id}
        
    except Exception as e:
        logger.error(f"❌ [admin.py] Archive Failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Archive Failed: {str(e)}")


@router.get("/archived/{job_id}", response_model=ArchivedLogResponse)
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
