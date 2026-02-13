# ============================================================================
# BackgroundSaver: 비동기 DB 저장 워커
# ============================================================================

import logging
import threading
from queue import Queue
from typing import Optional

logger = logging.getLogger("CoScientistMemory")


class BackgroundSaver:
    """비동기 DB 저장을 위한 백그라운드 워커"""

    def __init__(self):
        self.queue: Queue = Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self.running = False

    def start(self):
        if self.running:
            return
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        logger.info("Background saver started")

    def stop(self):
        self.running = False
        if self.worker_thread:
            self.queue.put(None)  # Poison pill
            self.worker_thread.join(timeout=5)

    def enqueue(self, task: callable):
        """저장 작업을 큐에 추가"""
        if not self.running:
            self.start()
        self.queue.put(task)

    def _worker(self):
        while self.running:
            try:
                task = self.queue.get(timeout=1)
                if task is None:
                    break
                task()  # Execute the save task
            except Exception as e:
                if "Empty" not in str(type(e).__name__):
                    logger.error(f"Background save failed: {e}")


# Global background saver instance
bg_saver = BackgroundSaver()
