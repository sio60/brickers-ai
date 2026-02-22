# route/admin_moderation.py
"""Admin Moderation Endpoints"""
from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any
import logging
from service import backend_client

logger = logging.getLogger("api.admin.moderation")
router = APIRouter()

@router.post("/moderation/restore")
async def restore_moderated_content(body: Dict[str, Any] = Body(...)):
    """
    [NEW] AI가 숨긴 콘텐츠를 관리자가 수동으로 복구.
    """
    target_type = body.get("type")
    target_id = body.get("targetId")

    if not target_type or not target_id:
        raise HTTPException(status_code=400, detail="Missing type or targetId")

    logger.info(f"♻️ [Admin] 콘텐츠 복구 시도: {target_type} {target_id}")
    
    # backend_client에 restore_content 함수를 미리 만들어두었다고 가정 (또는 추가 예정)
    success = await backend_client.restore_content(target_type, target_id)
    
    if success:
        return {"status": "success", "message": "Content restored"}
    else:
        raise HTTPException(status_code=500, detail="Failed to restore content")
