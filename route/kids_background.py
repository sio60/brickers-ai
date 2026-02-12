
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
from service.background_composer import generate_background_async, composite_images_async
from service.s3_client import upload_bytes_to_s3
import uuid

router = APIRouter(prefix="/api/v1/kids", tags=["kids-background"])

@router.post("/bg-composite")
async def composite_background(
    file: UploadFile = File(...),
    subject: str = Form(...),
):
    """
    1. 전경(투명 PNG) 업로드
    2. 주제(subject)로 배경 생성 (Gemini)
    3. 합성 (Pillow)
    4. S3 업로드 후 URL 반환
    """
    try:
        # 1. 파일 읽기
        fg_bytes = await file.read()
        
        # 2. 배경 생성
        bg_bytes = await generate_background_async(subject)
        
        # 3. 합성
        final_bytes = await composite_images_async(fg_bytes, bg_bytes) # 여기 수정 (async wrapper)
        
        # 4. S3 업로드
        filename = f"composition_{uuid.uuid4().hex}.png"
        s3_url = upload_bytes_to_s3(final_bytes, filename, content_type="image/png")
        
        return {"ok": True, "url": s3_url}
        
    except Exception as e:
        print(f"[BG Composite] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
