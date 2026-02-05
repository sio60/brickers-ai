from fastapi import APIRouter, Request
from .schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/query", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    svc = request.app.state.chat_service  # ✅ startup에서 주입
    content, cid = await svc.process_chat(
        user_message=req.message,
        language=req.language,
        conversation_id=req.conversation_id,
    )
    return ChatResponse(content=content, conversation_id=cid)
