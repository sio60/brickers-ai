# routes/instructions_pdf.py
from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Tuple
from io import BytesIO
import re
import httpx
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

router = APIRouter(prefix="/api/instructions", tags=["instructions"])

# ---------- Pydantic models ----------
class StepItem(BaseModel):
    index: int = Field(..., ge=1)
    imageUrl: str


class InstructionPdfPayload(BaseModel):
    modelName: str = "instructions"
    coverImageUrl: str
    steps: List[StepItem]


# ---------- helpers ----------
_HTTP_RE = re.compile(r"^https?://", re.IGNORECASE)

async def _fetch_image_bytes(url: str) -> Tuple[bytes, str]:
    """
    Returns: (raw_bytes, content_type)
    Only supports http(s) URL in "B 방식".
    """
    s = (url or "").strip()
    if not _HTTP_RE.match(s):
        raise HTTPException(status_code=400, detail="imageUrl must be http(s) URL")

    headers = {
        "User-Agent": "Mozilla/5.0 (Brickers PDF Generator; +http://localhost)",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Referer": "http://localhost:5173",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True, headers=headers) as client:
            r = await client.get(s)

        ct = (r.headers.get("content-type") or "").lower()

        if r.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"Image fetch failed: {r.status_code} {r.reason_phrase} (content-type={ct})",
            )

        if "image/" not in ct:
            # 이미지 아닌 HTML 에러 페이지 등을 방지
            hint = ""
            try:
                hint = (r.text or "")[:200]
            except Exception:
                hint = ""
            raise HTTPException(
                status_code=415,
                detail=f"Non-image content-type={ct}. body_hint={hint}",
            )

        return r.content, ct

    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Image request error: {type(e).__name__}: {e}")


def _normalize_to_png_bytes(img_bytes: bytes) -> bytes:
    """
    어떤 포맷이든 Pillow로 열어서 PNG로 통일
    """
    try:
        im = Image.open(BytesIO(img_bytes)).convert("RGBA")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image bytes: {e}")

    out = BytesIO()
    im.save(out, format="PNG")
    return out.getvalue()


def _draw_full_page(c: canvas.Canvas, img_reader: ImageReader, margin: int = 36):
    page_w, page_h = A4
    iw, ih = img_reader.getSize()

    max_w = page_w - margin * 2
    max_h = page_h - margin * 2

    scale = min(max_w / iw, max_h / ih)
    dw, dh = iw * scale, ih * scale

    x = (page_w - dw) / 2
    y = (page_h - dh) / 2

    c.drawImage(
        img_reader,
        x, y,
        width=dw, height=dh,
        preserveAspectRatio=True,
        mask="auto",
    )


# ---------- endpoints ----------
@router.post("/debug/fetch")
async def debug_fetch_image(payload: dict = Body(...)):
    target = payload.get("url") or payload.get("imageUrl")
    if not target:
        raise HTTPException(status_code=400, detail="body must include 'url' or 'imageUrl'")

    raw, ct = await _fetch_image_bytes(target)

    # Pillow로 실제 열리는지 확인
    im = Image.open(BytesIO(raw))
    w, h = im.size

    return {
        "ok": True,
        "bytes": len(raw),
        "contentType": ct,
        "format": im.format,
        "mode": im.mode,
        "width": w,
        "height": h,
    }


@router.post("/pdf")
async def create_instructions_pdf(req: InstructionPdfPayload):
    if not req.steps:
        raise HTTPException(status_code=400, detail="steps is empty")

    pdf_buf = BytesIO()
    c = canvas.Canvas(pdf_buf, pagesize=A4)

    # cover
    cover_bytes, _ = await _fetch_image_bytes(req.coverImageUrl)
    cover_png = _normalize_to_png_bytes(cover_bytes)
    cover_reader = ImageReader(BytesIO(cover_png))
    _draw_full_page(c, cover_reader)
    c.showPage()

    # steps
    for step in sorted(req.steps, key=lambda s: s.index):
        step_bytes, _ = await _fetch_image_bytes(step.imageUrl)
        step_png = _normalize_to_png_bytes(step_bytes)
        step_reader = ImageReader(BytesIO(step_png))
        _draw_full_page(c, step_reader)
        c.showPage()

    c.save()
    pdf_buf.seek(0)

    # 파일명 간단 안전화
    safe = re.sub(r'[\\/:*?"<>|]+', "_", req.modelName or "instructions")
    filename = f"{safe}.pdf"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}

    return StreamingResponse(pdf_buf, media_type="application/pdf", headers=headers)
