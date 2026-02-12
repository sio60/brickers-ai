
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
import uuid
import httpx

from service.background_composer import generate_background_async, composite_images_async
from service.s3_client import upload_bytes_to_s3
from service.render_client import render_ldr_steps

router = APIRouter(prefix="/api/v1/kids", tags=["kids-background"])


@router.post("/bg-composite")
async def composite_background(
    file: Optional[UploadFile] = File(None),
    subject: str = Form(...),
    ldrUrl: Optional[str] = Form(None),
):
    """
    1. LDR URL이 오면 LDView 렌더로 정면 45도 스냅샷 생성
    2. 없으면 업로드된 PNG를 전경으로 사용
    3. Gemini로 배경 생성 후 합성
    4. S3 업로드하여 URL 반환
    """
    try:
        fg_bytes: Optional[bytes] = None

        # 1) LDR 렌더 우선
        if ldrUrl:
            try:
                async with httpx.AsyncClient(timeout=20.0) as client:
                    resp = await client.get(ldrUrl)
                    resp.raise_for_status()
                    ldr_text = resp.text

                # 약 45도(정면+상단) 한 장만 필요
                views = [[200, -150, 200]]
                step_images = await render_ldr_steps(ldr_text, width=1024, height=768, views=views)
                if step_images and step_images[-1] and step_images[-1][0]:
                    fg_bytes = step_images[-1][0]
                else:
                    print("[BG Composite] LDR render returned empty image, fallback to upload")
            except Exception as e:
                print(f"[BG Composite] LDR render failed, fallback to uploaded file: {e}")

        # 2) fallback: 업로드된 PNG 사용
        if fg_bytes is None:
            if file is None:
                raise HTTPException(status_code=400, detail="file or ldrUrl is required")
            fg_bytes = await file.read()

        # 3) 배경 생성
        bg_bytes = await generate_background_async(subject)

        # 4) 합성
        final_bytes = await composite_images_async(fg_bytes, bg_bytes)

        # 5) S3 업로드
        filename = f"composition_{uuid.uuid4().hex}.png"
        s3_url = upload_bytes_to_s3(final_bytes, filename, content_type="image/png")

        return {"ok": True, "url": s3_url}

    except HTTPException as he:
        raise he
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            print(f"[BG Composite] Rate Limit: {e}")
            raise HTTPException(status_code=429, detail="AI Server Busy (Rate Limit Exceeded). Please try again later.")

        print(f"[BG Composite] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
