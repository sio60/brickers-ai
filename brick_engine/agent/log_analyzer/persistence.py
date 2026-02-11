import logging
import httpx
import config
from typing import Optional
import os

logger = logging.getLogger("agent.log_analyzer.persistence")

# API Base URL (관리자 로그 관련 API 엔드포인트)
ADMIN_API_BASE = f"{config.API_PUBLIC_BASE_URL}/api/admin"





async def archive_job_logs(job_id: str, logs: list[str], status: str = "FAILED", container_name: str = "brickers-ai-container"):
    """
    Apply in-memory log persistence (No Docker API).
    logs: list[str] - accumulated logs from kids_render.py buffer.
    status: RUNNING, SUCCESS, FAILED
    """
    if container_name == "brickers-ai-container" and "HOSTNAME" in os.environ:
         # Docker container ID usually in HOSTNAME
        container_name = os.environ["HOSTNAME"]

    # [NEW] Client Timestamp (for race condition handling)
    # Use localized timestamp if possible, or UTC
    from datetime import datetime, timezone
    client_timestamp = datetime.now(timezone.utc).isoformat()

    logger.info(f"📦 [로그 아카이브] Job ID [{job_id}] ({status}) 로그 백업 시작 ({len(logs)} lines)...")
    
    full_log_text = "\n".join(logs) if logs else f"[{status}] No logs recorded."
    
    try:
        # 백엔드 API 호출 (직접 DB 저장 대신 정석적인 방식 채택)
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ADMIN_API_BASE}/archive",
                json={
                    "job_id": job_id,
                    "logs": full_log_text,
                    "container_name": container_name,
                    "status": status,
                    "client_timestamp": client_timestamp  # [NEW]
                },
                timeout=10.0
            )
            
            if response.status_code == 200:
                logger.info(f"✅ [로그 아카이브] Job ID [{job_id}] ({status}) 백엔드 전송 완료")
                return True
            else:
                logger.error(f"❌ [로그 아카이브] 백엔드 전송 실패: {response.status_code} {response.text}")
                return False
            
    except Exception as e:
        logger.error(f"❌ [로그 아카이브] 에러 발생: {str(e)}")
        return False

# 호환성 유지를 위한 앨리어스
archive_failed_job_logs = archive_job_logs

async def get_archived_logs(job_id: str) -> Optional[str]:
    """백엔드 API를 통해 아카이빙된 로그를 가져옵니다."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{ADMIN_API_BASE}/archived/{job_id}", timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                return data.get("logs")
            else:
                logger.warning(f"⚠️ [로그 아카이브] 로그 조회 실패 ({job_id}): {response.status_code}")
                return None

    except Exception as e:
        logger.error(f"❌ [로그 아카이브] 조회 중 에러 발생: {str(e)}")
        return None


async def archive_system_logs(logs: list[str], session_id: str = "unknown", container_name: str = "brickers-ai-container"):
    """
    [NEW] 시스템 로그(Job ID 없는 전역 로그)를 아카이빙 (Fire-and-forget)
    """
    if not logs:
        return
        
    full_log_text = "\n".join(logs) # 그냥 문자열로 보내지만, 받는 쪽(admin)에서 리스트로 잘라서 $push 할 예정
    from datetime import datetime
    timestamp = datetime.utcnow().isoformat()
    
    try:
        async with httpx.AsyncClient() as client:
            # /archive/system 엔드포인트 호출
            response = await client.post(
                f"{ADMIN_API_BASE}/archive/system",
                json={
                    "logs": logs, # [Fix] 리스트 자체를 보냄 ($push 위해)
                    "container_name": container_name,
                    "timestamp": timestamp,
                    "session_id": session_id # [NEW]
                },
                timeout=5.0
            ) 
            # 성공 여부 체크만 하고 에러 발생 시 무시 (무한 루프 방지)
            if response.status_code != 200:
                pass
    except Exception:
        # 시스템 로그 전송 실패 자체가 또 로그를 남기면 무한 루프 위험이 있으므로 조용히 넘어감
        pass
