# vectordb/maintenance.py
import time
import logging
from datetime import datetime, timezone
from threading import Thread

import config
from vectordb.loader import ingest_parts, ingest_models, ensure_indexes
from vectordb.processor import update_all_bboxes, update_all_embeddings

logger = logging.getLogger("VectorDB.Maintenance")

_sync_running = False

def run_full_sync(background=False):
    # 다운로드 기믹 일단 삭제
    """DB 인제스트, BBox/임베딩 연산까지 전체를 동기화합니다. (다운로드는 제외)"""
    global _sync_running
    if _sync_running:
        logger.warning("Sync is already running. Skipping...")
        return False

    def _internal_sync():
        global _sync_running
        _sync_running = True
        try:
            # Docker에서 이미 파일을 받아두므로 바로 인제스트 시작
            ensure_indexes()
            ingest_parts()
            ingest_models()
            update_all_bboxes(only_missing=True)
            update_all_embeddings(only_missing=True)
            logger.info("Full sync (ingestion & processing) completed successfully.")
        finally:
            _sync_running = False

    if background:
        Thread(target=_internal_sync, daemon=True).start()
        logger.info("Full sync started in background thread.")
        return True
    else:
        _internal_sync()
        return True

def start_scheduler():
    """매월 1일 자정에 전체 동기화가 실행되도록 스케줄러를 가동합니다."""
    """월간 스케줄러 시작 (백그라운드 스레드에서 작동)"""
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        sched = BackgroundScheduler(timezone="UTC")
        # 스케줄러 자체는 이미 별도 스레드에서 동작하므로 background=False로 호출
        sched.add_job(run_full_sync, 'cron', day=1, hour=0, minute=0, kwargs={"background": False})
        sched.start()
        logger.info("APScheduler started: Monthly sync set for day 1.")
    except ImportError:
        logger.warning("APScheduler not found. Falling back to simple loop.")
        def loop():
            while True:
                now = datetime.now(timezone.utc)
                if now.day == 1 and now.hour == 0 and now.minute == 0:
                    run_full_sync(); time.sleep(120)
                time.sleep(30)
        Thread(target=loop, daemon=True).start()
