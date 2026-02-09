# route/admin.py
"""Admin API for System Monitoring & Logs"""
from __future__ import annotations

import os
import docker
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List

# Create router
router = APIRouter(prefix="/api/admin", tags=["admin"])

class LogResponse(BaseModel):
    container: str
    logs: str

@router.get("/logs/{container_name}", response_model=LogResponse)
def get_container_logs(
    container_name: str, 
    tail: int = Query(100, ge=1, le=2000),
    since_seconds: Optional[int] = Query(None, ge=1)
):
    """
    Fetch logs from a running Docker container (Read-Only).
    - tail: Number of lines to return (default 100)
    - since_seconds: Get logs from N seconds ago
    """
    try:
        # Connect to Docker Socket (mounted at /var/run/docker.sock)
        client = docker.from_env()
        
        # Get container
        container = client.containers.get(container_name)
        
        # Fetch logs
        # timestamps=True makes parsing harder for simple views, let's keep it simple for now
        log_bytes = container.logs(
            tail=tail, 
            since=since_seconds
        )
        
        # Decode
        log_str = log_bytes.decode("utf-8", errors="replace")
        
        return {
            "container": container_name,
            "logs": log_str
        }

    except docker.errors.NotFound:
        raise HTTPException(status_code=404, detail=f"Container '{container_name}' not found")
    except docker.errors.APIError as e:
        raise HTTPException(status_code=500, detail=f"Docker API Error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Error: {str(e)}")
