# route/admin_logs.py
"""Admin Logging & AI Analysis Endpoints"""
from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import json
import docker
from datetime import datetime
from db import get_db

# Try log agent import
try:
    from brick_engine.agent.log_analyzer.graph import app as log_agent_app
except ImportError:
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from brick_engine.agent.log_analyzer.graph import app as log_agent_app

logger = logging.getLogger("api.admin.logs")
router = APIRouter()

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
    plain_summary: str
    user_impact_level: str
    suggested_actions: List[str]
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
    status: str = "FAILED"
    client_timestamp: Optional[str] = None

class SystemLogRequest(BaseModel):
    logs: List[str]
    container_name: str = "brickers-ai-container"
    timestamp: str
    session_id: str

# --- Endpoints ---

@router.post("/ai-admin/archive/system")
async def archive_system_log(request: SystemLogRequest):
    try:
        db = get_db()
        if db is None:
            raise HTTPException(status_code=503, detail="Database connection unavailable")
        
        collection = db["system_logs"]
        try:
            dt = datetime.fromisoformat(request.timestamp.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y-%m-%d")
        except:
            date_str = datetime.utcnow().strftime("%Y-%m-%d")

        filter_query = {"session_id": request.session_id, "date": date_str}
        update_op = {
            "$push": {"logs": {"$each": request.logs}},
            "$setOnInsert": {
                "container": request.container_name,
                "created_at": datetime.utcnow(),
                "session_id": request.session_id,
                "date": date_str
            },
            "$set": {"last_updated": datetime.utcnow()}
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
    try:
        client = docker.from_env()
        container = client.containers.get(container_name)
        log_bytes = container.logs(tail=tail, since=since_seconds)
        log_str = log_bytes.decode("utf-8", errors="replace")
        return {"container": container_name, "logs": log_str}
    except Exception as e:
        logger.error(f"❌ Error fetching logs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_logs(request: AnalysisRequest = Body(...)):
    try:
        initial_state = {
            "container_name": request.container_name,
            "logs": "",
            "analysis_result": None,
            "messages": [],
            "iteration": 0,
            "investigation_notes": []
        }
        result_state = await log_agent_app.ainvoke(initial_state)
        
        raw_result_str = result_state.get("analysis_result")
        if not raw_result_str:
            raise HTTPException(status_code=500, detail="Agent returned no analysis result")
            
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
            except Exception as db_err:
                logger.error(f"❌ Failed to auto-archive logs: {db_err}")

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
        logger.error(f"❌ Analysis Failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ai-admin/archive")
async def archive_log(request: ArchiveLogRequest):
    try:
        db = get_db()
        if db is None:
            raise HTTPException(status_code=503, detail="Database connection unavailable")
        collection = db["failed_job_logs"]
        
        # Safe Update logic
        existing_doc = collection.find_one({"jobId": request.job_id})
        should_update = True
        if existing_doc and request.client_timestamp:
            last_ts = existing_doc.get("client_timestamp")
            last_logs = existing_doc.get("logs", "")
            is_newer_ts = (last_ts is None) or (request.client_timestamp >= last_ts)
            is_more_logs = len(request.logs) > len(last_logs)
            if not is_newer_ts and not is_more_logs:
                should_update = False
        
        if should_update:
            update_fields = {
                "logs": request.logs,
                "timestamp": datetime.utcnow().isoformat(),
                "container": request.container_name,
                "status": request.status
            }
            if request.client_timestamp:
                update_fields["client_timestamp"] = request.client_timestamp
            collection.update_one({"jobId": request.job_id}, {"$set": update_fields}, upsert=True)
            
        return {"status": "success", "jobId": request.job_id}
    except Exception as e:
        logger.error(f"❌ Archive Failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ai-admin/archived/{job_id}", response_model=ArchivedLogResponse)
async def get_archived_log(job_id: str):
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
    except Exception as e:
        logger.error(f"❌ Fetch Archived Failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
