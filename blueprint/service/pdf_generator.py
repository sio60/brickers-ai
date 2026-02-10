# blueprint/service/pdf_generator.py
"""조립설명서 PDF 생성 (fpdf2)"""
from __future__ import annotations

import os
from io import BytesIO
from datetime import datetime
from typing import Dict, List, Optional

from fpdf import FPDF
from PIL import Image

from service.ldr_parser import PartInfo, StepBOM
from service.parts_catalog import get_color_name, get_part_size


class InstructionsPDF(FPDF):
    """한글 폰트 + 커스텀 헤더/푸터"""

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.set_auto_page_break(auto=True, margin=15)

        # Windows
        font_path = "C:/Windows/Fonts/malgun.ttf"
        font_bold_path = "C:/Windows/Fonts/malgunbd.ttf"
        # Linux (Docker) — NanumGothic
        linux_font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        linux_font_bold_path = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"

        if os.path.exists(linux_font_path):
            self.add_font("Malgun", "", linux_font_path, uni=True)
            if os.path.exists(linux_font_bold_path):
                self.add_font("Malgun", "B", linux_font_bold_path, uni=True)
            self.korean_font = "Malgun"
        elif os.path.exists(font_path):
            self.add_font("Malgun", "", font_path, uni=True)
            if os.path.exists(font_bold_path):
                self.add_font("Malgun", "B", font_bold_path, uni=True)
            self.korean_font = "Malgun"
        else:
            print("[PDF] Warning: No Korean font found. Using Helvetica.")
            self.korean_font = "Helvetica"

    def header(self):
        self.set_font(self.korean_font, "B", 16)
        self.cell(0, 10, self.model_name, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Brickers | {datetime.now().strftime('%Y-%m-%d')} | p.{self.page_no()}", align="C")


# ─── BOM 테이블 공통 설정 ──────────────────────────────────
_COL_IMG = 18
_COL_SIZE = 55
_COL_COLOR = 45
_COL_QTY = 20
_ROW_H = 14


def _draw_bom_header(pdf: InstructionsPDF, header_bg: tuple = (60, 60, 60)):
    """BOM 테이블 헤더 행 출력"""
    pdf.set_font(pdf.korean_font, "B", 9)
    pdf.set_fill_color(*header_bg)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(_COL_IMG, 7, "Part", border=1, fill=True, align="C")
    pdf.cell(_COL_SIZE, 7, "Size", border=1, fill=True, align="C")
    pdf.cell(_COL_COLOR, 7, "Color", border=1, fill=True, align="C")
    pdf.cell(_COL_QTY, 7, "Qty", border=1, fill=True, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.set_font(pdf.korean_font, "", 9)


def _draw_bom_row(
    pdf: InstructionsPDF,
    part: PartInfo,
    row_idx: int,
    thumbs: Dict[str, bytes],
):
    """BOM 테이블 단일 행 (파츠 이미지 + 사이즈 + 색상 + 수량)"""
    y_before = pdf.get_y()
    x_start = pdf.l_margin
    bg = 255 if row_idx % 2 == 0 else 245
    pdf.set_fill_color(bg, bg, bg)

    # 파츠 썸네일
    thumb_key = f"{part.id}_{part.color}"
    thumb_data = thumbs.get(thumb_key, b"")
    if thumb_data:
        try:
            img_io = BytesIO(thumb_data)
            pdf.image(img_io, x=x_start + 1, y=y_before + 1, w=_COL_IMG - 2, h=_ROW_H - 2)
        except Exception:
            pass

    pdf.cell(_COL_IMG, _ROW_H, "", border=1, fill=True)
    pdf.cell(_COL_SIZE, _ROW_H, get_part_size(part.id), border=1, fill=True, align="C")
    pdf.cell(_COL_COLOR, _ROW_H, get_color_name(part.color), border=1, fill=True, align="C")
    pdf.set_font(pdf.korean_font, "B", 11)
    pdf.cell(_COL_QTY, _ROW_H, str(part.count), border=1, fill=True, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(pdf.korean_font, "", 9)


# ─── 메인 PDF 생성 함수 ──────────────────────────────────
def generate_pdf_with_images_and_bom(
    model_name: str,
    step_images: List[List[bytes]],
    step_boms: List[StepBOM],
    cover_image: Optional[bytes] = None,
    part_thumbnails: Optional[Dict[str, bytes]] = None,
) -> bytes:
    """Step별 이미지 (3개 뷰) + BOM 테이블을 포함한 PDF 생성"""
    pdf = InstructionsPDF(model_name=model_name)
    thumbs = part_thumbnails or {}

    # ── 표지 페이지 ────────────────────────────────────
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
            img_io = BytesIO()
            img.save(img_io, format="PNG")
            img_io.seek(0)
            pdf.ln(10)
            page_w = pdf.w - pdf.l_margin - pdf.r_margin
            pdf.image(img_io, x=pdf.l_margin + 20, w=page_w - 40)
        except Exception as e:
            print(f"[PDF] Cover image error: {e}")

    # ── 각 Step 페이지 ────────────────────────────────
    for step_idx, images in enumerate(step_images):
        step_no = step_idx + 1
        bom = step_boms[step_idx] if step_idx < len(step_boms) else StepBOM(stepIndex=step_no, parts=[])

        pdf.add_page()

        # Step 헤더
        pdf.set_font(pdf.korean_font, "B", 18)
        pdf.cell(0, 12, f"Step {step_no}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

        # 메인 이미지 (View 1 — 쿼터뷰)
        if len(images) > 0 and images[0]:
            try:
                img_io = BytesIO(images[0])
                page_w = pdf.w - pdf.l_margin - pdf.r_margin
                pdf.image(img_io, x=pdf.l_margin, w=page_w, h=90)
            except Exception as e:
                print(f"[PDF] Step {step_no} main image error: {e}")

        pdf.ln(95)

        # 서브 이미지 (View 2 탑뷰, View 3 바닥뷰) — 나란히
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
                        print(f"[PDF] Step {step_no} sub image {i + 2} error: {e}")

            pdf.ln(60)

        # Step BOM 테이블
        if bom.parts:
            pdf.set_font(pdf.korean_font, "B", 12)
            pdf.cell(0, 8, f"Parts Needed ({len(bom.parts)} types)", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

            _draw_bom_header(pdf, header_bg=(60, 60, 60))

            for i, part in enumerate(bom.parts[:15]):
                _draw_bom_row(pdf, part, i, thumbs)

            if len(bom.parts) > 15:
                pdf.set_font(pdf.korean_font, "", 8)
                pdf.cell(0, 6, f"... and {len(bom.parts) - 15} more parts", new_x="LMARGIN", new_y="NEXT")

    # ── 전체 BOM 요약 페이지 ────────────────────────────
    pdf.add_page()
    pdf.set_font(pdf.korean_font, "B", 16)
    pdf.cell(0, 12, "Full Parts List", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

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

    _draw_bom_header(pdf, header_bg=(231, 76, 60))

    sorted_parts = sorted(all_parts.values(), key=lambda p: -p.count)
    for i, part in enumerate(sorted_parts[:50]):
        # 페이지 넘김 체크
        if pdf.get_y() + _ROW_H > pdf.h - 20:
            pdf.add_page()
            pdf.set_font(pdf.korean_font, "B", 14)
            pdf.cell(0, 10, "Full Parts List (cont.)", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)
            _draw_bom_header(pdf, header_bg=(231, 76, 60))

        _draw_bom_row(pdf, part, i, thumbs)

    return bytes(pdf.output())
