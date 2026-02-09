import docker
import logging
from datetime import datetime
from .state import LogAnalysisState

# DB Connection (Lazy import to avoid issues)
def get_db_collection():
    try:
        from ..yang_db import get_db
        db = get_db()
        return db["failed_job_logs"] if db is not None else None
    except:
        return None

logger = logging.getLogger("agent.log_analyzer.persistence")

async def archive_failed_job_logs(job_id: str, container_name: str = "brickers-ai-container"):
    """
    실패한 Job의 로그를 Docker에서 추출하여 MongoDB에 아카이빙합니다.
    (Full Context 보존 전략)
    """
    logger.info(f"📦 [로그 아카이브] Job ID [{job_id}] 로그 백업 시작...")
    
    try:
        client = docker.from_env()
        container = client.containers.get(container_name)
        # 넉넉하게 최근 5000줄 정도를 훑어서 해당 Job ID가 포함된 맥락을 다 긁어옴
        raw_logs = container.logs(tail=5000).decode("utf-8", errors="replace")
        
        job_logs = [line for line in raw_logs.splitlines() if job_id in line]
        
        if not job_logs:
            logger.warning(f"⚠️ [로그 아카이브] Job ID [{job_id}]를 로그에서 찾지 못했습니다.")
            return False
            
        full_log_text = "\n".join(job_logs)
        
        collection = get_db_collection()
        if collection is not None:
            doc = {
                "jobId": job_id,
                "logs": full_log_text,
                "timestamp": datetime.utcnow().isoformat(),
                "container": container_name,
                "status": "FAILED"
            }
            collection.replace_one({"jobId": job_id}, doc, upsert=True)
            logger.info(f"✅ [로그 아카이브] Job ID [{job_id}] DB 저장 완료 ({len(job_logs)}줄)")
            return True
        else:
            logger.error("❌ [로그 아카이브] DB 연결 실패.")
            return False
            
    except Exception as e:
        logger.error(f"❌ [로그 아카이브] 에러 발생: {str(e)}")
        return False

async def get_archived_logs(job_id: str) -> Optional[str]:
    """DB에서 아카이빙된 로그를 가져옵니다."""
    collection = get_db_collection()
    if collection is not None:
        doc = collection.find_one({"jobId": job_id})
        return doc.get("logs") if doc else None
    return None
