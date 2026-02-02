import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional


@dataclass
class ConversationState:
    summary: str
    messages: Deque[dict]  # [{"role":"user/assistant","content":"..."}]
    updated_at: float


class InMemoryConversationStore:
    """
    - 대화 히스토리를 RAM에 저장
    - TTL 지나면 자동 폐기
    - messages가 너무 길어지면 “요약”으로 압축할 수 있게 훅 제공
    """

    def __init__(self, max_messages: int = 20, ttl_seconds: int = 3600):
        self.max_messages = (
            max_messages  # ✅ 최근 메시지 N개 유지 (user/assistant 포함)
        )
        self.ttl_seconds = ttl_seconds
        self._data: Dict[str, ConversationState] = {}
        self._lock = asyncio.Lock()

    async def get(self, conversation_id: str) -> ConversationState:
        async with self._lock:
            self._gc_locked()
            state = self._data.get(conversation_id)
            if state is None:
                state = ConversationState(
                    summary="", messages=deque(), updated_at=time.time()
                )
                self._data[conversation_id] = state
            return state

    async def touch(self, conversation_id: str) -> None:
        async with self._lock:
            state = self._data.get(conversation_id)
            if state:
                state.updated_at = time.time()

    async def append_turn(
        self, conversation_id: str, user_text: str, assistant_text: str
    ) -> None:
        async with self._lock:
            self._gc_locked()
            state = self._data.get(conversation_id)
            if state is None:
                state = ConversationState(
                    summary="", messages=deque(), updated_at=time.time()
                )
                self._data[conversation_id] = state

            state.messages.append({"role": "user", "content": user_text})
            state.messages.append({"role": "assistant", "content": assistant_text})
            state.updated_at = time.time()

            # “요약”을 쓰기 전이라도, 최소한 최근 max_messages는 유지
            while len(state.messages) > self.max_messages:
                state.messages.popleft()

    async def set_summary_and_keep_recent(
        self,
        conversation_id: str,
        new_summary: str,
        keep_recent: List[dict],
    ) -> None:
        async with self._lock:
            state = self._data.get(conversation_id)
            if state is None:
                state = ConversationState(
                    summary="", messages=deque(), updated_at=time.time()
                )
                self._data[conversation_id] = state

            state.summary = new_summary
            state.messages = deque(keep_recent)
            state.updated_at = time.time()

    async def snapshot(self, conversation_id: str) -> tuple[str, List[dict]]:
        async with self._lock:
            self._gc_locked()
            state = self._data.get(conversation_id)
            if state is None:
                return "", []
            return state.summary, list(state.messages)

    def _gc_locked(self) -> None:
        now = time.time()
        expired = [
            cid
            for cid, st in self._data.items()
            if now - st.updated_at > self.ttl_seconds
        ]
        for cid in expired:
            del self._data[cid]
