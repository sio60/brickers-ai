# vectordb/maintenance.py
import time
import requests
import zipfile
import logging
from datetime import datetime, timezone
from threading import Thread

import config
from vectordb.loader import ingest_parts, ingest_models, ensure_indexes
from vectordb.processor import update_all_bboxes, update_all_embeddings

logger = logging.getLogger("VectorDB.Maintenance")

def download_and_extract_ldraw_zip():
    """최신 LDraw 라이브러리를 공식 사이트에서 내려받아 압축을 풉니다."""
    url = "https://library.ldraw.org/library/updates/complete.zip"
    base_dir = config.LDRAW_BASE_DIR
    temp_zip = base_dir.parent / "ldraw_complete.zip"
    
    logger.info(f"Downloading from {url}...")
    try:
        r = requests.get(url, timeout=300, stream=True)
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        
        from tqdm import tqdm
        with open(temp_zip, "wb") as f, tqdm(
            desc="Downloading LDraw Library",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
            
        logger.info(f"Extracting to {base_dir}...")
        with zipfile.ZipFile(temp_zip, 'r') as z: z.extractall(base_dir.parent)
        return True
    except Exception as e:
        logger.error(f"Update failed: {e}"); return False
    finally:
        if temp_zip.exists(): temp_zip.unlink()

_sync_running = False

def run_full_sync(background=False):
    """다운로드부터 DB 인제스트, BBox/임베딩 연산까지 전체를 동기화합니다."""
    global _sync_running
    if _sync_running:
        logger.warning("Sync is already running. Skipping...")
        return False

    def _internal_sync():
        global _sync_running
        _sync_running = True
        try:
            if download_and_extract_ldraw_zip():
                ensure_indexes()
                ingest_parts()
                ingest_models()
                update_all_bboxes(only_missing=True)
                update_all_embeddings(only_missing=True)
                logger.info("Full sync completed successfully.")
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
