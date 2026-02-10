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
                    "status": status  # [수정] status 전송 활성화
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
