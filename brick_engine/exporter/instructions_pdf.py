"""
조립설명서 PDF 생성기 (STEP 기반)

- MongoDB boms 문서 (jobId)에서 steps/items 읽어 PDF 생성
- step마다 "이번 step에 추가되는 부품 리스트"를 페이지로 출력
- 마지막에 전체 BOM 요약 페이지(선택)

fpdf2 사용
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fpdf import FPDF


# ---------------------------
# PDF Base
# ---------------------------
class InstructionsPDF(FPDF):
    def __init__(self, model_name: str, job_id: str, mode: str = "pro"):
        super().__init__()
        self.model_name = model_name
        self.job_id = job_id
        self.mode = mode
        self.set_auto_page_break(auto=True, margin=15)

        # 한글 폰트 (Windows 맑은고딕)
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
        self.set_font(self.korean_font, "B", 16 if self.mode == "pro" else 20)
        self.cell(0, 10, self.model_name, align="C", new_x="LMARGIN", new_y="NEXT")

        self.set_font(self.korean_font, "", 10)
        self.cell(
            0,
            6,
            f"조립 설명서 (STEP) | jobId={self.job_id}",
            align="C",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(
            0,
            10,
            f"Brick CoScientist | {datetime.now().strftime('%Y-%m-%d')} | p.{self.page_no()}",
            align="C",
        )


# ---------------------------
# Helpers
# ---------------------------
def _part_label(item: Dict[str, Any]) -> str:
    # 의미매핑: name 우선. 없으면 canonicalFile/partFile
    return (
        item.get("name")
        or item.get("canonicalFile")
        or item.get("partFile")
        or "Unknown"
    )


def _color_label(item: Dict[str, Any]) -> str:
    # 구조: import_to_mongo에서 colorName 넣어둠
    return item.get("colorName") or f"Color_{item.get('color')}"


def _shorten(s: str, n: int = 60) -> str:
    s = str(s)
    return s if len(s) <= n else (s[: n - 3] + "...")


def _print_kv(pdf: InstructionsPDF, k: str, v: str, size: int = 11):
    # 왼쪽 마진으로 강제 복귀 (가로폭 부족 방지)
    pdf.set_x(pdf.l_margin)

    label_w = 35

    pdf.set_font(pdf.korean_font, "B", size)
    pdf.cell(label_w, 7, str(k), border=0)

    pdf.set_font(pdf.korean_font, "", size)

    # ✅ 현재 X가 label_w 만큼 이동된 상태에서, 남은 폭을 직접 계산해서 multi_cell에 넣기
    remaining_w = pdf.w - pdf.r_margin - pdf.get_x()
    if remaining_w < 10:
        # 혹시라도 너무 좁으면 줄바꿈하고 다시 출력
        pdf.ln(7)
        pdf.set_x(pdf.l_margin)
        remaining_w = pdf.w - pdf.r_margin - pdf.get_x()

    pdf.multi_cell(remaining_w, 7, str(v))

def _try_add_step_image(
    pdf: InstructionsPDF,
    step_images_dir: str,
    model_stem: str,
    step_no: int,
) -> bool:
    """
    step_no: 1-based
    이미지 파일 규칙: {model_stem}_step_{step_no:02d}.png
    """
    if not step_images_dir:
        return False

    img_path = Path(step_images_dir) / f"{model_stem}_step_{step_no:02d}.png"
    if not img_path.exists():
        return False

    # 페이지 가용 폭
    usable_w = pdf.w - pdf.l_margin - pdf.r_margin

    # 이미지 삽입 (현재 커서 위치에)
    pdf.set_x(pdf.l_margin)
    pdf.image(str(img_path), x=pdf.l_margin, w=usable_w)

    pdf.ln(5)
    return True

# ---------------------------
# Pages
# ---------------------------
def add_cover(pdf: InstructionsPDF, boms_doc: Dict[str, Any]):
    model_name = pdf.model_name
    source_path = boms_doc.get("source", {}).get("path", "")
    step_count = boms_doc.get("stepCount", 0)
    total_qty = boms_doc.get("totalQty", 0)
    unique_items = boms_doc.get("uniqueItems", 0)

    pdf.add_page()
    pdf.ln(2)

    pdf.set_font(pdf.korean_font, "B", 14 if pdf.mode == "pro" else 18)
    pdf.cell(0, 10, "요약", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    _print_kv(pdf, "Model", str(model_name))
    _print_kv(pdf, "Steps", str(step_count))
    _print_kv(pdf, "Total", f"{total_qty} parts")
    _print_kv(pdf, "Unique", f"{unique_items} items")
    _print_kv(pdf, "Source", str(source_path))

    pdf.ln(4)
    pdf.set_font(pdf.korean_font, "", 11 if pdf.mode == "pro" else 14)
    pdf.multi_cell(
        0,
        7,
        "이 문서는 LDR 파일의 '0 STEP' 구분을 기준으로 단계별로 추가해야 할 부품을 안내합니다.\n"
        "STEP 1부터 순서대로 조립하세요.",
    )


def add_step_page(
    pdf: InstructionsPDF,
    step_doc: Dict[str, Any],
    step_images_dir: str | None = None,
    model_stem: str = "model",
):
    step_no = int(step_doc.get("step") or 0)
    items: List[Dict[str, Any]] = step_doc.get("items") or []

    pdf.add_page()

    pdf.set_font(pdf.korean_font, "B", 14 if pdf.mode == "pro" else 18)
    pdf.cell(0, 10, f"STEP {step_no}", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font(pdf.korean_font, "", 11 if pdf.mode == "pro" else 14)
    pdf.cell(
        0,
        7,
        f"이번 단계 추가 부품: {step_doc.get('totalQty', 0)}개 (종류 {step_doc.get('uniqueItems', 0)}종)",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(2)

    # ✅ STEP 이미지 넣기
    if step_images_dir:
        ok = _try_add_step_image(pdf, step_images_dir, model_stem, step_no)
        if not ok:
            pdf.set_font(pdf.korean_font, "", 10)
            pdf.cell(
                0,
                6,
                f"(이미지 없음) {model_stem}_step_{step_no:02d}.png",
                new_x="LMARGIN",
                new_y="NEXT",
            )

    # Table header
    pdf.set_font(pdf.korean_font, "B", 10 if pdf.mode == "pro" else 13)
    pdf.set_fill_color(60, 60, 60)
    pdf.set_text_color(255, 255, 255)

    col_part = 115 if pdf.mode == "pro" else 95
    col_color = 45
    col_qty = 25 if pdf.mode == "pro" else 30

    pdf.cell(col_part, 8, "Part", border=1, fill=True)
    pdf.cell(col_color, 8, "Color", border=1, fill=True, align="C")
    pdf.cell(col_qty, 8, "Qty", border=1, fill=True, align="C", new_x="LMARGIN", new_y="NEXT")

    # rows
    pdf.set_text_color(0, 0, 0)
    pdf.set_font(pdf.korean_font, "", 10 if pdf.mode == "pro" else 13)

    for i, it in enumerate(items):
        pdf.set_fill_color(255, 255, 255 if i % 2 == 0 else 248)

        part = _shorten(_part_label(it), 70 if pdf.mode == "pro" else 55)
        color = _shorten(_color_label(it), 25)
        qty = it.get("qty", 0)

        pdf.cell(col_part, 7, str(part), border=1, fill=True)
        pdf.cell(col_color, 7, str(color), border=1, fill=True, align="C")
        pdf.cell(col_qty, 7, str(qty), border=1, fill=True, align="C", new_x="LMARGIN", new_y="NEXT")


def add_overall_bom_page(pdf: InstructionsPDF, boms_doc: Dict[str, Any]):
    items: List[Dict[str, Any]] = boms_doc.get("items") or []
    if not items:
        return

    pdf.add_page()
    pdf.set_font(pdf.korean_font, "B", 14 if pdf.mode == "pro" else 18)
    pdf.cell(0, 10, "전체 부품 목록 (Overall BOM)", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font(pdf.korean_font, "B", 10 if pdf.mode == "pro" else 13)
    pdf.set_fill_color(231, 76, 60)
    pdf.set_text_color(255, 255, 255)

    col_part = 115 if pdf.mode == "pro" else 95
    col_color = 45
    col_qty = 25 if pdf.mode == "pro" else 30

    pdf.cell(col_part, 8, "Part", border=1, fill=True)
    pdf.cell(col_color, 8, "Color", border=1, fill=True, align="C")
    pdf.cell(col_qty, 8, "Qty", border=1, fill=True, align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_text_color(0, 0, 0)
    pdf.set_font(pdf.korean_font, "", 10 if pdf.mode == "pro" else 13)

    for i, it in enumerate(items):
        pdf.set_fill_color(255, 255, 255 if i % 2 == 0 else 248)

        part = _shorten(_part_label(it), 70 if pdf.mode == "pro" else 55)
        color = _shorten(_color_label(it), 25)
        qty = it.get("qty", 0)

        pdf.cell(col_part, 7, str(part), border=1, fill=True)
        pdf.cell(col_color, 7, str(color), border=1, fill=True, align="C")
        pdf.cell(col_qty, 7, str(qty), border=1, fill=True, align="C", new_x="LMARGIN", new_y="NEXT")


# ---------------------------
# Public API
# ---------------------------
def generate_instructions_pdf_from_boms_doc(
    boms_doc: Dict[str, Any],
    output_path: str,
    mode: str = "pro",
    include_overall_bom: bool = True,
    step_images_dir: str | None = None,
) -> str:
    job_id = boms_doc.get("jobId", "unknown_job")
    source_path = (boms_doc.get("source") or {}).get("path") or ""
    model_stem = Path(source_path).stem if source_path else "model"

    model_name = (
        boms_doc.get("modelName")
        or (Path(source_path).stem if source_path else "Model")
    )

    pdf = InstructionsPDF(model_name=model_name, job_id=job_id, mode=mode)

    add_cover(pdf, boms_doc)

    steps = boms_doc.get("steps") or []
    for s in steps:
        if s.get("items"):
            add_step_page(pdf, s, step_images_dir=step_images_dir, model_stem=model_stem)

    if include_overall_bom:
        add_overall_bom_page(pdf, boms_doc)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pdf.output(output_path)
    return output_path

