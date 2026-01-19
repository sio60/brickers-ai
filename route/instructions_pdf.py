# routes/instructions_pdf.py
from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from io import BytesIO
import base64
import re
import httpx
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader


router = APIRouter(prefix="/api/instructions", tags=["instructions"])


# ---------- Pydantic models ----------
class StepItem(BaseModel):
    index: int
    dataUrl: str


class InstructionPdfPayload(BaseModel):
    modelName: str = "instructions"
    coverDataUrl: str
    steps: List[StepItem]


# ---------- helpers ----------
_DATA_URL_RE = re.compile(r"^data:(?P<mime>image\/[a-zA-Z0-9\+\-\.]+);base64,(?P<b64>.+)$")


async def _load_image_bytes(data_or_url: str) -> tuple[bytes, str]:
    """
    Returns: (raw_bytes, mime_or_content_type)
    Supports:
      - data:image/...;base64,...
      - http(s) URL (CDN)
    """
    s = (data_or_url or "").strip()

    m = _DATA_URL_RE.match(s)
    if m:
        mime = m.group("mime")
        try:
            return base64.b64decode(m.group("b64")), mime
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 dataUrl: {e}")

    if s.startswith("http://") or s.startswith("https://"):
        headers = {
            "User-Agent": "Mozilla/5.0 (BrickersAI PDF Generator; +http://localhost)",
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Referer": "http://localhost:5173",
        }
        try:
            async with httpx.AsyncClient(timeout=20.0, follow_redirects=True, headers=headers) as client:
                r = await client.get(s)

            ct = (r.headers.get("content-type") or "").lower()

            # CDN에서 403/404/429 등
            if r.status_code >= 400:
                raise HTTPException(
                    status_code=502,
                    detail=f"CDN fetch failed: {r.status_code} {r.reason_phrase} (content-type={ct})",
                )

            # 이미지 아닌데 200 주는 경우(HTML 에러 페이지 등)
            if "image/" not in ct:
                # 바디 앞부분 조금만 힌트로 넣기(너무 길면 안됨)
                hint = r.text[:200] if r.text else ""
                raise HTTPException(
                    status_code=415,
                    detail=f"CDN returned non-image content-type={ct}. body_hint={hint}",
                )

            return r.content, ct

        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"CDN request error: {type(e).__name__}: {e}")

    raise HTTPException(status_code=400, detail="dataUrl must be data:image;base64,... or http(s) URL")


def _normalize_to_png_bytes(img_bytes: bytes) -> bytes:
    try:
        im = Image.open(BytesIO(img_bytes)).convert("RGBA")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Downloaded bytes are not a valid image: {e}")

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
    c.drawImage(img_reader, x, y, width=dw, height=dh, preserveAspectRatio=True, mask="auto")


# ---------- endpoints ----------
@router.post("/debug/fetch")
async def debug_fetch_image(payload: dict = Body(...)):
    target = payload.get("url") or payload.get("dataUrl")
    if not target:
        raise HTTPException(status_code=400, detail="body must include 'url' or 'dataUrl'")

    raw, ct = await _load_image_bytes(target)

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
    pdf_buf = BytesIO()
    c = canvas.Canvas(pdf_buf, pagesize=A4)

    # cover
    cover_bytes, _ = await _load_image_bytes(req.coverDataUrl)
    cover_png = _normalize_to_png_bytes(cover_bytes)
    cover_reader = ImageReader(BytesIO(cover_png))
    _draw_full_page(c, cover_reader)
    c.showPage()

    # steps
    for step in sorted(req.steps, key=lambda s: s.index):
        step_bytes, _ = await _load_image_bytes(step.dataUrl)
        step_png = _normalize_to_png_bytes(step_bytes)
        step_reader = ImageReader(BytesIO(step_png))
        _draw_full_page(c, step_reader)
        c.showPage()

    c.save()
    pdf_buf.seek(0)

    filename = f"{req.modelName or 'instructions'}.pdf"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}

    return StreamingResponse(pdf_buf, media_type="application/pdf", headers=headers)
