from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    message: str
    language: Optional[str] = "ko"
    conversation_id: Optional[str] = None  # ✅ 없으면 서버가 생성


class ChatResponse(BaseModel):
    content: str
    conversation_id: str  # ✅ 다음 요청에 다시 보내면 “기억”됨
