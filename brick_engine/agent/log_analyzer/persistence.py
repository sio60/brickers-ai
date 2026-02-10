import docker
import logging
import httpx
import config
from typing import Optional
import datetime
from .state import LogAnalysisState

logger = logging.getLogger("agent.log_analyzer.persistence")

# API Base URL (관리자 로그 관련 API 엔드포인트)
ADMIN_API_BASE = f"{config.API_PUBLIC_BASE_URL}/api/admin"

async def archive_job_logs(job_id: str, status: str = "FAILED", container_name: str = "brickers-ai-container", start_time: Optional[datetime.datetime] = None):
    """
    Job의 로그를 Docker에서 추출하여 백엔드 API를 통해 아카이빙합니다.
    status: RUNNING, SUCCESS, FAILED
    start_time: Job 시작 시간 (제공 시 해당 시간 이후 모든 로그 수집)
    """
    logger.info(f"📦 [로그 아카이브] Job ID [{job_id}] ({status}) 로그 백업 시작...")
    
    try:
        client = docker.from_env()
        container = client.containers.get(container_name)
        
        raw_logs = ""
        if start_time:
            # 타임스탬프 기반 수집 (Job ID 필터링 없이 해당 시간 대 모든 로그)
            # Docker requires timestamp in int/float or string, datetime works too
            raw_logs = container.logs(since=start_time).decode("utf-8", errors="replace")
            full_log_text = raw_logs
            logger.info(f"🕒 [로그 아카이브] Timestamp 기반 수집: {start_time} 이후 {len(raw_logs)} bytes")
        else:
            # 기존 방식: 최근 5000줄에서 ID 필터링
            raw_logs = container.logs(tail=5000).decode("utf-8", errors="replace")
            job_logs = [line for line in raw_logs.splitlines() if job_id in line]
            full_log_text = "\n".join(job_logs) if job_logs else ""

        if not full_log_text and status != "RUNNING":
             # RUNNING 초기에는 아직 로그가 안 찍혔을 수도 있으므로 경고 스킵
            logger.warning(f"⚠️ [로그 아카이브] Job ID [{job_id}] 관련 로그를 찾지 못했습니다.")
            full_log_text = f"[{status}] Log not found for Job {job_id}"
        elif not full_log_text and status == "RUNNING":
             full_log_text = f"[{status}] Job Started"
        
        # 백엔드 API 호출 (직접 DB 저장 대신 정석적인 방식 채택)
        # TODO: admin.py의 /archive 엔드포인트도 status를 받을 수 있게 확장하면 좋음
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ADMIN_API_BASE}/archive",
                json={
                    "job_id": job_id,
                    "logs": full_log_text,
                    "container_name": container_name,
                    # "status": status  <-- admin.py API 확장이 필요할 수 있음
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
