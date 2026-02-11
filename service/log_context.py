import sys
import asyncio
from contextvars import ContextVar
from typing import Optional, List, TextIO
from brick_engine.agent.log_analyzer.persistence import archive_system_logs

# 1. ContextVar 정의 (각 요청/Job별 로그 버퍼를 저장)
job_log_buffer_var: ContextVar[Optional[List[str]]] = ContextVar("job_log_buffer", default=None)

import uuid

# 2. Global Log Hook Class
class GlobalLogCapture:
    """
    시스템 전체의 stdout/stderr를 가로채는 싱글톤 클래스.
    """
    _instance = None
    _system_log_buffer: List[str] = []
    _flush_task = None
    session_id: str = ""

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalLogCapture, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.session_id = str(uuid.uuid4()) # [NEW] Server Session ID
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        sys.stdout = self._Tee(self.original_stdout)
        sys.stderr = self._Tee(self.original_stderr)
        
        self._initialized = True
        
        print(f"[GlobalLogCapture] System stdout/stderr hooked globally. SessionID={self.session_id}", flush=True)

    def start_flusher(self):
        """앱 시작 시 호출 (Event Loop 필요)"""
        if self._flush_task is None:
            self._flush_task = asyncio.create_task(self._flush_loop())
            print("[GlobalLogCapture] Background log flusher started.", flush=True)

    async def _flush_loop(self):
        """Smart Batching Flush Loop"""
        while True:
            try:
                # 1. 버퍼가 비었으면 긴 대기 (CPU 절약)
                if not self._system_log_buffer:
                    await asyncio.sleep(1.0) # 1초마다 체크 (가벼움)
                    continue

                # 2. 버퍼에 내용이 있으면:
                #    - 양이 많으면(100줄 이상) -> 즉시 전송
                #    - 양이 적으면 -> 좀 더 기다리며 모으기 (최대 5분=300초)
                if len(self._system_log_buffer) < 100:
                    await asyncio.sleep(300.0)

                # 3. 전송 (Flush)
                if self._system_log_buffer:
                    # 복사본 뜨고 비우기
                    chunk = self._system_log_buffer[:]
                    self._system_log_buffer.clear() # 비우기
                    
                    # 전송 (세션 ID 포함)
                    await archive_system_logs(chunk, session_id=self.session_id)
                    
            except Exception as e:
                # 플러시 루프 죽지 않게 방어
                await asyncio.sleep(5.0) # 에러 시 잠시 대기

    class _Tee(TextIO):
        def __init__(self, original: TextIO):
            self.original = original

        def write(self, message: str):
            # 1. Docker/Console
            self.original.write(message)
            self.original.flush()

            # 2. Job buffering or System buffering
            if message.strip():
                # A. Job Context 확인
                buffer = job_log_buffer_var.get()
                if buffer is not None:
                    # 작업 중인 로그 -> Job Buffer로
                    buffer.append(message.rstrip())
                else:
                    # B. 작업 없는 로그 -> System Buffer로 (전역 공유 리스트)
                    GlobalLogCapture._system_log_buffer.append(message.rstrip())


        def flush(self):
            self.original.flush()

        # TextIO compatibility
        def isatty(self): return self.original.isatty()
        def fileno(self): return self.original.fileno()
        def close(self): pass 
        def closed(self): return False
        def encoding(self): return self.original.encoding

# 3. Job Context Manager
class JobLogContext:
    def __init__(self, buffer: List[str]):
        self.buffer = buffer
        self.token = None

    def __enter__(self):
        self.token = job_log_buffer_var.set(self.buffer)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token:
            job_log_buffer_var.reset(self.token)
