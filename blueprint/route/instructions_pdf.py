# blueprint/route/instructions_pdf.py
"""조립설명서 PDF API 엔드포인트"""
from __future__ import annotations

import os
import re
import base64
import uuid
import httpx
from io import BytesIO
from datetime import datetime
from typing import List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from PIL import Image

from service.s3_client import USE_S3, S3_BUCKET, upload_bytes_to_s3
from service.ldr_parser import parse_ldr_step_boms
from service.pdf_generator import generate_pdf_with_images_and_bom


router = APIRouter(prefix="/api/instructions", tags=["instructions"])

PDF_S3_PREFIX = os.environ.get("S3_PREFIX_PDF", "uploads/pdf").strip().strip("/")


# ─── Pydantic Models ─────────────────────────────────────
class StepImageItem(BaseModel):
    stepIndex: int = Field(..., ge=1)
    images: List[str]

class PdfWithBomRequest(BaseModel):
    modelName: str = "Brickers Model"
    ldrUrl: str
    steps: List[StepImageItem]
    coverImage: Optional[str] = None

class PdfWithBomResponse(BaseModel):
    ok: bool
    pdfUrl: str
    message: Optional[str] = None


# ─── Helpers ──────────────────────────────────────────────
def decode_base64_image(data_url: str) -> bytes:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    return base64.b64decode(data_url)


async def fetch_ldr_text(url: str) -> str:
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.text


_HTTP_RE = re.compile(r"^https?://", re.IGNORECASE)


async def _fetch_image_bytes(url: str) -> Tuple[bytes, str]:
    s = (url or "").strip()
    if not _HTTP_RE.match(s):
        raise HTTPException(status_code=400, detail="imageUrl must be http(s) URL")

    headers = {
        "User-Agent": "Mozilla/5.0 (Brickers PDF Generator)",
        "Accept": "image/*,*/*;q=0.8",
    }
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True, headers=headers) as client:
        r = await client.get(s)
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"Image fetch failed: {r.status_code}")
        ct = (r.headers.get("content-type") or "").lower()
        return r.content, ct


# ─── API: PDF with BOM ───────────────────────────────────
@router.post("/pdf-with-bom", response_model=PdfWithBomResponse)
async def create_pdf_with_bom(req: PdfWithBomRequest):
    """프론트엔드에서 캡처한 Step별 이미지와 LDR BOM을 합쳐 PDF 생성"""
    try:
        print(f"[PDF] Fetching LDR from: {req.ldrUrl}")
        ldr_text = await fetch_ldr_text(req.ldrUrl)
        step_boms = parse_ldr_step_boms(ldr_text)
        print(f"[PDF] Parsed {len(step_boms)} steps from LDR")

        step_images: List[List[bytes]] = []
        for step_item in sorted(req.steps, key=lambda s: s.stepIndex):
            images_bytes = []
            for img_b64 in step_item.images:
                try:
                    images_bytes.append(decode_base64_image(img_b64))
                except Exception as e:
                    print(f"[PDF] Image decode error at step {step_item.stepIndex}: {e}")
                    images_bytes.append(b"")
            step_images.append(images_bytes)

        print(f"[PDF] Decoded {len(step_images)} step image sets")

        cover_bytes = None
        if req.coverImage:
            try:
                cover_bytes = decode_base64_image(req.coverImage)
            except Exception as e:
                print(f"[PDF] Cover image decode error: {e}")

        print(f"[PDF] Generating PDF for: {req.modelName}")
        pdf_bytes = generate_pdf_with_images_and_bom(
            model_name=req.modelName,
            step_images=step_images,
            step_boms=step_boms,
            cover_image=cover_bytes,
        )
        print(f"[PDF] Generated PDF: {len(pdf_bytes)} bytes")

        if USE_S3 and S3_BUCKET:
            now = datetime.now()
            safe_name = re.sub(r'[\\/:*?"<>|]+', "_", req.modelName or "instructions")
            s3_key = f"{PDF_S3_PREFIX}/{now.year:04d}/{now.month:02d}/{uuid.uuid4().hex[:8]}_{safe_name}.pdf"
            pdf_url = upload_bytes_to_s3(pdf_bytes, s3_key, "application/pdf")
            print(f"[PDF] Uploaded to S3: {pdf_url}")
            return PdfWithBomResponse(ok=True, pdfUrl=pdf_url)
        else:
            pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
            return PdfWithBomResponse(
                ok=True,
                pdfUrl=f"data:application/pdf;base64,{pdf_b64}",
                message="S3 disabled, returning base64",
            )

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"LDR fetch failed: {e.response.status_code}")
    except Exception as e:
        print(f"[PDF] Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ─── API: Legacy PDF (URL 기반) ──────────────────────────
class StepItem(BaseModel):
    index: int = Field(..., ge=1)
    imageUrl: str

class InstructionPdfPayload(BaseModel):
    modelName: str = "instructions"
    coverImageUrl: str
    steps: List[StepItem]


@router.post("/pdf")
async def create_instructions_pdf_legacy(req: InstructionPdfPayload):
    """기존 API (URL 기반 이미지) — 하위 호환"""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader

    if not req.steps:
        raise HTTPException(status_code=400, detail="steps is empty")

    pdf_buf = BytesIO()
    c = canvas.Canvas(pdf_buf, pagesize=A4)
    page_w, page_h = A4
    margin = 36

    def draw_full_page(img_reader):
        iw, ih = img_reader.getSize()
        max_w, max_h = page_w - margin * 2, page_h - margin * 2
        scale = min(max_w / iw, max_h / ih)
        dw, dh = iw * scale, ih * scale
        x, y = (page_w - dw) / 2, (page_h - dh) / 2
        c.drawImage(img_reader, x, y, width=dw, height=dh, preserveAspectRatio=True, mask="auto")

    cover_bytes, _ = await _fetch_image_bytes(req.coverImageUrl)
    img = Image.open(BytesIO(cover_bytes)).convert("RGBA")
    buf = BytesIO()
    img.save(buf, format="PNG")
    draw_full_page(ImageReader(BytesIO(buf.getvalue())))
    c.showPage()

    for step in sorted(req.steps, key=lambda s: s.index):
        step_bytes, _ = await _fetch_image_bytes(step.imageUrl)
        img = Image.open(BytesIO(step_bytes)).convert("RGBA")
        buf = BytesIO()
        img.save(buf, format="PNG")
        draw_full_page(ImageReader(BytesIO(buf.getvalue())))
        c.showPage()

    c.save()
    pdf_buf.seek(0)

    safe = re.sub(r'[\\/:*?"<>|]+', "_", req.modelName or "instructions")
    filename = f"{safe}.pdf"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(pdf_buf, media_type="application/pdf", headers=headers)
