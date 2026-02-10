import docker
import logging
import httpx
import config
from typing import Optional
from .state import LogAnalysisState

logger = logging.getLogger("agent.log_agent.persistence")

# API Base URL (관리자 로그 관련 API 엔드포인트)
ADMIN_API_BASE = f"{config.API_PUBLIC_BASE_URL}/api/admin"

async def archive_failed_job_logs(job_id: str, container_name: str = "brickers-ai-container"):
    """
    실패한 Job의 로그를 Docker에서 추출하여 백엔드 API를 통해 아카이빙합니다.
    """
    logger.info(f"📦 [로그 아카이브] Job ID [{job_id}] 로그 백업 시작...")
    
    try:
        client = docker.from_env()
        container = client.containers.get(container_name)
        # 최근 5000줄 정도를 훑어서 해당 Job ID가 포함된 맥락을 추출
        raw_logs = container.logs(tail=5000).decode("utf-8", errors="replace")
        job_logs = [line for line in raw_logs.splitlines() if job_id in line]
        
        if not job_logs:
            logger.warning(f"⚠️ [로그 아카이브] Job ID [{job_id}]를 로그에서 찾지 못했습니다.")
            return False
            
        full_log_text = "\n".join(job_logs)
        
        # 백엔드 API 호출 (직접 DB 저장 대신 정석적인 방식 채택)
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ADMIN_API_BASE}/archive",
                json={
                    "job_id": job_id,
                    "logs": full_log_text,
                    "container_name": container_name
                },
                timeout=10.0
            )
            
            if response.status_code == 200:
                logger.info(f"✅ [로그 아카이브] Job ID [{job_id}] 백엔드 전송 완료")
                return True
            else:
                logger.error(f"❌ [로그 아카이브] 백엔드 전송 실패: {response.status_code} {response.text}")
                return False
            
    except Exception as e:
        logger.error(f"❌ [로그 아카이브] 에러 발생: {str(e)}")
        return False

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
