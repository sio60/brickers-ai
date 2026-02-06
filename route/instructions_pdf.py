# routes/instructions_pdf.py
"""
조립설명서 PDF 생성 API

- 프론트엔드에서 캡처한 Step별 이미지 (Base64) 수신
- LDR 파일에서 Step별 BOM 파싱
- PDF 생성 (fpdf2) 후 S3 업로드
"""

from __future__ import annotations

import os
import re
import base64
import uuid
import httpx
from io import BytesIO
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
from collections import Counter

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from PIL import Image

from fpdf import FPDF


router = APIRouter(prefix="/api/instructions", tags=["instructions"])


# =============================================
# S3 Integration (service.s3_client에서 공유)
# =============================================
from service.s3_client import USE_S3, S3_BUCKET, upload_bytes_to_s3

# PDF용 S3 prefix (kids_render와 다름)
PDF_S3_PREFIX = os.environ.get("S3_PREFIX_PDF", "uploads/pdf").strip().strip("/")


# =============================================
# LDR 파싱 (프론트엔드 로직과 동일)
# =============================================
@dataclass
class PartInfo:
    id: str
    color: int
    count: int

@dataclass
class StepBOM:
    stepIndex: int
    parts: List[PartInfo]

def normalize_part(token: str) -> str:
    """파일명 정규화: 경로 제거, 확장자 제거, 소문자로"""
    t = token.strip().replace("\\", "/")
    name = t.split("/")[-1].lower()
    return name.replace(".dat", "").replace(".ldr", "")

def parse_ldr_step_boms(ldr_text: str) -> List[StepBOM]:
    """
    프론트엔드 parseBOM()와 동일한 형식으로 LDR 파싱
    """
    lines = ldr_text.replace("\r\n", "\n").split("\n")
    steps: List[StepBOM] = []
    current_parts: Dict[str, PartInfo] = {}
    step_index = 1

    def flush_step():
        nonlocal step_index, current_parts
        steps.append(StepBOM(
            stepIndex=step_index,
            parts=list(current_parts.values())
        ))
        current_parts = {}
        step_index += 1

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # Step/RotStep 구분 (프론트와 동일)
        if re.match(r"^0\s+(STEP|ROTSTEP)\b", line, re.IGNORECASE):
            flush_step()
            continue

        # Type-1 라인 파싱
        if line.startswith("1 "):
            tokens = line.split()
            if len(tokens) >= 15:
                try:
                    color = int(tokens[1])
                except ValueError:
                    continue
                
                part_id = normalize_part(tokens[-1])
                key = f"{part_id}_{color}"
                
                if key in current_parts:
                    current_parts[key].count += 1
                else:
                    current_parts[key] = PartInfo(id=part_id, color=color, count=1)

    # 마지막 Step 처리
    flush_step()
    
    return steps


# =============================================
# LDraw 색상 코드 → 이름 매핑
# =============================================
LDRAW_COLOR_NAMES = {
    0: "Black", 1: "Blue", 2: "Green", 3: "Dark Turquoise",
    4: "Red", 5: "Dark Pink", 6: "Brown", 7: "Light Gray",
    14: "Yellow", 15: "White", 19: "Tan", 25: "Orange",
    70: "Reddish Brown", 71: "Light Bluish Gray", 72: "Dark Bluish Gray"
}

def get_color_name(code: int) -> str:
    return LDRAW_COLOR_NAMES.get(code, f"Color_{code}")


# =============================================
# PDF 생성 (fpdf2)
# =============================================
class InstructionsPDF(FPDF):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.set_auto_page_break(auto=True, margin=15)
        
        # 한글 폰트 (Windows)
        font_path = "C:/Windows/Fonts/malgun.ttf"
        font_bold_path = "C:/Windows/Fonts/malgunbd.ttf"
        if os.path.exists(font_path):
            self.add_font("Malgun", "", font_path)
            if os.path.exists(font_bold_path):
                self.add_font("Malgun", "B", font_bold_path)
            self.korean_font = "Malgun"
        else:
            self.korean_font = "Helvetica"
    
    def header(self):
        self.set_font(self.korean_font, "B", 16)
        self.cell(0, 10, self.model_name, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
    
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Brickers | {datetime.now().strftime('%Y-%m-%d')} | p.{self.page_no()}", align="C")


def generate_pdf_with_images_and_bom(
    model_name: str,
    step_images: List[List[bytes]],  # step_images[step_idx][view_idx] = bytes
    step_boms: List[StepBOM],
    cover_image: Optional[bytes] = None
) -> bytes:
    """
    Step별 이미지 (3개 뷰) + BOM 테이블을 포함한 PDF 생성
    """
    pdf = InstructionsPDF(model_name=model_name)
    
    # 표지 페이지
    pdf.add_page()
    pdf.set_font(pdf.korean_font, "B", 24)
    pdf.ln(30)
    pdf.cell(0, 15, "Assembly Instructions", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    pdf.set_font(pdf.korean_font, "", 16)
    pdf.cell(0, 10, model_name, align="C", new_x="LMARGIN", new_y="NEXT")
    
    if cover_image:
        try:
            img = Image.open(BytesIO(cover_image))
            # 임시 파일로 저장 (fpdf는 파일 경로 또는 BytesIO 지원)
            img_io = BytesIO()
            img.save(img_io, format="PNG")
            img_io.seek(0)
            pdf.ln(10)
            page_w = pdf.w - pdf.l_margin - pdf.r_margin
            pdf.image(img_io, x=pdf.l_margin + 20, w=page_w - 40)
        except Exception as e:
            print(f"[PDF] Cover image error: {e}")
    
    # 각 Step 페이지
    for step_idx, images in enumerate(step_images):
        step_no = step_idx + 1
        bom = step_boms[step_idx] if step_idx < len(step_boms) else StepBOM(stepIndex=step_no, parts=[])
        
        pdf.add_page()
        
        # Step 헤더
        pdf.set_font(pdf.korean_font, "B", 18)
        pdf.cell(0, 12, f"Step {step_no}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)
        
        # 메인 이미지 (View 1)
        if len(images) > 0 and images[0]:
            try:
                img_io = BytesIO(images[0])
                page_w = pdf.w - pdf.l_margin - pdf.r_margin
                pdf.image(img_io, x=pdf.l_margin, w=page_w, h=90)
            except Exception as e:
                print(f"[PDF] Step {step_no} main image error: {e}")
        
        pdf.ln(95)
        
        # 서브 이미지 (View 2, 3) - 나란히 배치
        if len(images) > 1:
            sub_w = (pdf.w - pdf.l_margin - pdf.r_margin - 5) / 2
            start_y = pdf.get_y()
            
            for i, sub_img in enumerate(images[1:3]):
                if sub_img:
                    try:
                        img_io = BytesIO(sub_img)
                        x_pos = pdf.l_margin + (i * (sub_w + 5))
                        pdf.image(img_io, x=x_pos, y=start_y, w=sub_w, h=55)
                    except Exception as e:
                        print(f"[PDF] Step {step_no} sub image {i+2} error: {e}")
            
            pdf.ln(60)
        
        # BOM 테이블
        if bom.parts:
            pdf.set_font(pdf.korean_font, "B", 12)
            pdf.cell(0, 8, f"Parts Needed ({len(bom.parts)} types)", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
            
            # 테이블 헤더
            pdf.set_font(pdf.korean_font, "B", 10)
            pdf.set_fill_color(60, 60, 60)
            pdf.set_text_color(255, 255, 255)
            
            col_qty = 25
            col_part = 80
            col_color = 60
            
            pdf.cell(col_qty, 7, "Qty", border=1, fill=True, align="C")
            pdf.cell(col_part, 7, "Part ID", border=1, fill=True)
            pdf.cell(col_color, 7, "Color", border=1, fill=True, align="C", new_x="LMARGIN", new_y="NEXT")
            
            # 테이블 행
            pdf.set_text_color(0, 0, 0)
            pdf.set_font(pdf.korean_font, "", 10)
            
            for i, part in enumerate(bom.parts[:15]):  # 최대 15개까지만 표시
                pdf.set_fill_color(255, 255, 255 if i % 2 == 0 else 245)
                pdf.cell(col_qty, 6, str(part.count), border=1, fill=True, align="C")
                pdf.cell(col_part, 6, part.id[:40], border=1, fill=True)
                pdf.cell(col_color, 6, get_color_name(part.color), border=1, fill=True, align="C", new_x="LMARGIN", new_y="NEXT")
            
            if len(bom.parts) > 15:
                pdf.set_font(pdf.korean_font, "I", 9)
                pdf.cell(0, 6, f"... and {len(bom.parts) - 15} more parts", new_x="LMARGIN", new_y="NEXT")
    
    # 전체 BOM 요약 페이지
    pdf.add_page()
    pdf.set_font(pdf.korean_font, "B", 16)
    pdf.cell(0, 12, "Full Parts List", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    # 전체 파츠 집계
    all_parts: Dict[str, PartInfo] = {}
    for bom in step_boms:
        for part in bom.parts:
            key = f"{part.id}_{part.color}"
            if key in all_parts:
                all_parts[key].count += part.count
            else:
                all_parts[key] = PartInfo(id=part.id, color=part.color, count=part.count)
    
    total_count = sum(p.count for p in all_parts.values())
    
    pdf.set_font(pdf.korean_font, "", 12)
    pdf.cell(0, 8, f"Total: {total_count} parts ({len(all_parts)} unique)", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    # 전체 BOM 테이블
    pdf.set_font(pdf.korean_font, "B", 10)
    pdf.set_fill_color(231, 76, 60)
    pdf.set_text_color(255, 255, 255)
    
    col_qty = 25
    col_part = 80
    col_color = 60
    
    pdf.cell(col_qty, 7, "Qty", border=1, fill=True, align="C")
    pdf.cell(col_part, 7, "Part ID", border=1, fill=True)
    pdf.cell(col_color, 7, "Color", border=1, fill=True, align="C", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font(pdf.korean_font, "", 10)
    
    sorted_parts = sorted(all_parts.values(), key=lambda p: -p.count)
    for i, part in enumerate(sorted_parts[:50]):  # 최대 50개
        pdf.set_fill_color(255, 255, 255 if i % 2 == 0 else 245)
        pdf.cell(col_qty, 6, str(part.count), border=1, fill=True, align="C")
        pdf.cell(col_part, 6, part.id[:40], border=1, fill=True)
        pdf.cell(col_color, 6, get_color_name(part.color), border=1, fill=True, align="C", new_x="LMARGIN", new_y="NEXT")
    
    # PDF 바이트로 반환
    return bytes(pdf.output())


# =============================================
# Pydantic Models
# =============================================
class StepImageItem(BaseModel):
    stepIndex: int = Field(..., ge=1)
    images: List[str]  # Base64 이미지 배열 (3개 뷰)

class PdfWithBomRequest(BaseModel):
    modelName: str = "Brickers Model"
    ldrUrl: str  # LDR 파일 URL
    steps: List[StepImageItem]
    coverImage: Optional[str] = None  # Base64 커버 이미지

class PdfWithBomResponse(BaseModel):
    ok: bool
    pdfUrl: str
    message: Optional[str] = None


# =============================================
# Helper Functions
# =============================================
def decode_base64_image(data_url: str) -> bytes:
    """Base64 데이터 URL에서 이미지 bytes 추출"""
    if "," in data_url:
        # data:image/png;base64,xxxxx 형식
        data_url = data_url.split(",", 1)[1]
    return base64.b64decode(data_url)

async def fetch_ldr_text(url: str) -> str:
    """LDR 파일 다운로드"""
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.text


# =============================================
# API Endpoints
# =============================================
@router.post("/pdf-with-bom", response_model=PdfWithBomResponse)
async def create_pdf_with_bom(req: PdfWithBomRequest):
    """
    프론트엔드에서 캡처한 Step별 이미지와 LDR BOM을 합쳐 PDF 생성
    """
    try:
        # 1. LDR 파일 다운로드 및 BOM 파싱
        print(f"[PDF] Fetching LDR from: {req.ldrUrl}")
        ldr_text = await fetch_ldr_text(req.ldrUrl)
        step_boms = parse_ldr_step_boms(ldr_text)
        print(f"[PDF] Parsed {len(step_boms)} steps from LDR")
        
        # 2. Base64 이미지 디코딩
        step_images: List[List[bytes]] = []
        for step_item in sorted(req.steps, key=lambda s: s.stepIndex):
            images_bytes = []
            for img_b64 in step_item.images:
                try:
                    img_bytes = decode_base64_image(img_b64)
                    images_bytes.append(img_bytes)
                except Exception as e:
                    print(f"[PDF] Image decode error at step {step_item.stepIndex}: {e}")
                    images_bytes.append(b"")
            step_images.append(images_bytes)
        
        print(f"[PDF] Decoded {len(step_images)} step image sets")
        
        # 3. 커버 이미지 디코딩
        cover_bytes = None
        if req.coverImage:
            try:
                cover_bytes = decode_base64_image(req.coverImage)
            except Exception as e:
                print(f"[PDF] Cover image decode error: {e}")
        
        # 4. PDF 생성
        print(f"[PDF] Generating PDF for: {req.modelName}")
        pdf_bytes = generate_pdf_with_images_and_bom(
            model_name=req.modelName,
            step_images=step_images,
            step_boms=step_boms,
            cover_image=cover_bytes
        )
        print(f"[PDF] Generated PDF: {len(pdf_bytes)} bytes")
        
        # 5. S3 업로드
        if USE_S3 and S3_BUCKET:
            now = datetime.now()
            safe_name = re.sub(r'[\\/:*?"<>|]+', "_", req.modelName or "instructions")
            s3_key = f"{PDF_S3_PREFIX}/{now.year:04d}/{now.month:02d}/{uuid.uuid4().hex[:8]}_{safe_name}.pdf"

            pdf_url = upload_bytes_to_s3(pdf_bytes, s3_key, "application/pdf")
            print(f"[PDF] Uploaded to S3: {pdf_url}")
            
            return PdfWithBomResponse(ok=True, pdfUrl=pdf_url)
        else:
            # S3 미사용 시 Base64로 반환 (또는 로컬 저장)
            pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
            return PdfWithBomResponse(
                ok=True,
                pdfUrl=f"data:application/pdf;base64,{pdf_b64}",
                message="S3 disabled, returning base64"
            )
    
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"LDR fetch failed: {e.response.status_code}")
    except Exception as e:
        print(f"[PDF] Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =============================================
# 기존 API (하위 호환)
# =============================================
class StepItem(BaseModel):
    index: int = Field(..., ge=1)
    imageUrl: str

class InstructionPdfPayload(BaseModel):
    modelName: str = "instructions"
    coverImageUrl: str
    steps: List[StepItem]

_HTTP_RE = re.compile(r"^https?://", re.IGNORECASE)

async def _fetch_image_bytes(url: str) -> Tuple[bytes, str]:
    """URL에서 이미지 다운로드"""
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


@router.post("/pdf")
async def create_instructions_pdf_legacy(req: InstructionPdfPayload):
    """기존 API (URL 기반 이미지) - 하위 호환용"""
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

    # cover
    cover_bytes, _ = await _fetch_image_bytes(req.coverImageUrl)
    img = Image.open(BytesIO(cover_bytes)).convert("RGBA")
    buf = BytesIO(); img.save(buf, format="PNG")
    draw_full_page(ImageReader(BytesIO(buf.getvalue())))
    c.showPage()

    # steps
    for step in sorted(req.steps, key=lambda s: s.index):
        step_bytes, _ = await _fetch_image_bytes(step.imageUrl)
        img = Image.open(BytesIO(step_bytes)).convert("RGBA")
        buf = BytesIO(); img.save(buf, format="PNG")
        draw_full_page(ImageReader(BytesIO(buf.getvalue())))
        c.showPage()

    c.save()
    pdf_buf.seek(0)

    safe = re.sub(r'[\\/:*?"<>|]+', "_", req.modelName or "instructions")
    filename = f"{safe}.pdf"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}

    return StreamingResponse(pdf_buf, media_type="application/pdf", headers=headers)

