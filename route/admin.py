# route/admin.py
"""Admin API for System Monitoring & Logs"""
from __future__ import annotations

import os
import json
import docker
import logging # Added import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from db import get_db


# Logger configuration
logger = logging.getLogger("api.admin") # Added logger setup

try:
    from brick_engine.agent.log_analyzer.agent import app as log_agent_app
except ImportError:
    # Local dev path fallback if needed
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from brick_engine.agent.log_analyzer.agent import app as log_agent_app

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
                        "status": "FAILED", # Analysis run implies something to check, usually failure
                        "analysis_summary": analysis_data.get("summary"),
                        "root_cause": analysis_data.get("root_cause")
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
            "is_error": analysis_data.get("error_found", False),
            "summary": analysis_data.get("summary", "분석 완료"),
            "root_cause": analysis_data.get("root_cause"),
            "suggestion": analysis_data.get("suggestion"),
            "job_id": job_id
        }

    except Exception as e:
        logger.error(f"❌ [admin.py] Analysis Failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis Failed: {str(e)}")


@router.post("/archive")
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
        doc = {
            "jobId": request.job_id,
            "logs": request.logs,
            "timestamp": datetime.utcnow().isoformat(),
            "container": request.container_name,
            "status": request.status
        }
        collection.replace_one({"jobId": request.job_id}, doc, upsert=True)
        logger.info(f"✅ [admin.py] Archived logs for {request.job_id}")

        # --- FAILED일 때 AI 분석 자동 실행 ---
        ai_analysis = None
        if request.status == "FAILED" and request.logs:
            try:
                logger.info(f"🧠 [admin.py] FAILED 로그 감지 → AI 분석 시작 (Job: {request.job_id})")
                analysis_state = {
                    "container_name": request.container_name,
                    "logs": request.logs,       # 로그를 직접 전달 (DB 재조회 불필요)
                    "analysis_result": None,
                    "error_count": 0,
                    "messages": [],
                    "iteration": 0,
                    "job_id": request.job_id
                }
                result_state = await log_agent_app.ainvoke(analysis_state)

                raw_result = result_state.get("analysis_result", "")
                if raw_result:
                    result_json = json.loads(raw_result)
                    ai_analysis = result_json.get("analysis", result_json)

                    # 같은 문서에 AI 분석 결과 추가
                    collection.update_one(
                        {"jobId": request.job_id},
                        {"$set": {
                            "ai_analysis": ai_analysis,
                            "ai_analyzed_at": datetime.utcnow().isoformat()
                        }}
                    )
                    logger.info(f"✅ [admin.py] AI 분석 완료 & 저장 (Job: {request.job_id})")
                    logger.info(f"📝 요약: {ai_analysis.get('summary', 'N/A')[:100]}")

            except Exception as ai_err:
                logger.error(f"⚠️ [admin.py] AI 분석 실패 (로그 저장은 정상): {ai_err}")
                # AI 분석 실패해도 로그 저장은 이미 성공했으므로 에러 발생시키지 않음

        return {
            "status": "success",
            "jobId": request.job_id,
            "ai_analysis": ai_analysis  # 분석 결과가 있으면 함께 반환
        }
        
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
