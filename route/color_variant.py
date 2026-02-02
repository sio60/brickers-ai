# brickers-ai/route/color_variant.py
"""
Color Variant API - 색상 테마 변경 엔드포인트
"""

import os
import base64
import tempfile
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import httpx

# 프로젝트 루트 찾기
def _find_project_root(start: Path) -> Path:
    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent
    for p in [cur] + list(cur.parents):
        if (p / "requirements.txt").exists():
            return p
    return cur

PROJECT_ROOT = _find_project_root(Path(__file__))

# brick-engine 경로 추가
import sys
BRICK_ENGINE_PATH = PROJECT_ROOT / "brick-engine"
EXPORTER_PATH = BRICK_ENGINE_PATH / "exporter"

for p in [str(BRICK_ENGINE_PATH), str(EXPORTER_PATH)]:
    if p not in sys.path:
        sys.path.insert(0, p)

router = APIRouter(prefix="/api/v1", tags=["color"])


# ========================================
# Request/Response 모델
# ========================================

class ColorVariantRequest(BaseModel):
    ldr_url: str  # S3 URL 또는 로컬 URL
    theme: str    # "sunset", "ocean" 등 또는 자유 프롬프트


class ThemeInfo(BaseModel):
    name: str
    description: str


class ColorVariantResponse(BaseModel):
    ok: bool
    message: str
    theme_applied: str
    original_colors: int
    changed_bricks: int
    ldr_data: Optional[str] = None  # base64 encoded LDR


class ThemesResponse(BaseModel):
    themes: List[ThemeInfo]


# ========================================
# 엔드포인트
# ========================================

@router.get("/color-variant/themes", response_model=ThemesResponse)
async def get_themes():
    """사용 가능한 색상 테마 목록"""
    from color_variant import COLOR_THEMES

    themes = [
        ThemeInfo(name=name, description=theme["description"])
        for name, theme in COLOR_THEMES.items()
    ]
    return ThemesResponse(themes=themes)


@router.post("/color-variant", response_model=ColorVariantResponse)
async def apply_color_variant(req: ColorVariantRequest):
    """
    LDR 파일에 색상 테마 적용

    - ldr_url: LDR 파일 URL (S3 또는 로컬)
    - theme: 테마 이름 ("sunset", "ocean" 등) 또는 자유 프롬프트 ("사이버펑크 느낌으로")
    """
    try:
        # 1. LDR 파일 다운로드
        print(f"[ColorVariant] Downloading LDR from: {req.ldr_url}")

        async with httpx.AsyncClient() as client:
            response = await client.get(req.ldr_url, timeout=30.0)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"LDR 다운로드 실패: {response.status_code}"
                )
            ldr_text = response.text

        print(f"[ColorVariant] Downloaded {len(ldr_text)} bytes")

        # 2. 임시 파일로 저장
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.ldr', delete=False, encoding='utf-8'
        ) as f:
            f.write(ldr_text)
            temp_input = f.name

        try:
            # 3. color_variant 실행
            from color_variant import (
                load_parts_db,
                analyze_model_colors,
                get_color_mapping_from_theme,
                get_color_mapping_from_llm,
                apply_color_mapping,
                COLOR_THEMES
            )
            from ldr_converter import ldr_to_brick_model, model_to_ldr

            # 파츠 DB 로드
            parts_db = load_parts_db()

            # 모델 로드
            model = ldr_to_brick_model(temp_input)
            print(f"[ColorVariant] Model loaded: {len(model.bricks)} bricks")

            # 색상 분석
            analysis = analyze_model_colors(model)
            original_colors = analysis["unique_colors"]
            original_codes = [c["code"] for c in analysis["colors"]]

            print(f"[ColorVariant] Original colors: {original_colors}")

            # 색상 매핑 생성
            theme_name = req.theme.lower().strip()

            if theme_name in COLOR_THEMES:
                # 프리셋 테마
                mapping = get_color_mapping_from_theme(original_codes, theme_name)
                applied_theme = theme_name
            else:
                # LLM 자유 프롬프트
                mapping = get_color_mapping_from_llm(analysis, req.theme)
                applied_theme = f"custom: {req.theme}"

            if not mapping:
                raise HTTPException(status_code=500, detail="색상 매핑 생성 실패")

            # 색상 적용
            changed = apply_color_mapping(model, mapping)
            print(f"[ColorVariant] Changed {changed} bricks")

            # 4. LDR 출력
            ldr_output = model_to_ldr(model, parts_db, skip_validation=True, skip_physics=True)
            ldr_bytes = ldr_output.encode('utf-8')
            ldr_data = base64.b64encode(ldr_bytes).decode('utf-8')

            return ColorVariantResponse(
                ok=True,
                message="색상 테마 적용 완료",
                theme_applied=applied_theme,
                original_colors=original_colors,
                changed_bricks=changed,
                ldr_data=ldr_data
            )

        finally:
            # 임시 파일 삭제
            os.unlink(temp_input)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
