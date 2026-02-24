"""LDR → 이미지 렌더링 (로컬 LDView 우선, 없으면 screenshot-server HTTP 폴백)

방향 규칙 (LDraw 좌표계 기준):
- Z 음수 = 앞 (FRONT)
- Z 양수 = 뒤 (BACK)
- X 음수 = 왼쪽 (LEFT)
- X 양수 = 오른쪽 (RIGHT)
"""
import logging
import os
import subprocess
import shutil
import tempfile
import base64
import io
from pathlib import Path
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available, direction labels disabled")

# 로컬 LDView 경로: 환경변수 우선, 없으면 시스템 PATH에서 탐색
LDVIEW_BIN = os.environ.get("LDVIEW_BIN", "LDView")
_ldview_resolved = shutil.which(LDVIEW_BIN)
LDVIEW_LOCAL_AVAILABLE = _ldview_resolved is not None
if LDVIEW_LOCAL_AVAILABLE:
    LDVIEW_BIN = _ldview_resolved

# screenshot-server HTTP 폴백
SCREENSHOT_SERVER_URL = os.environ.get(
    "SCREENSHOT_SERVER_URL", "http://screenshot-api:8002"
)

# 카메라 각도 프리셋 (latitude, longitude, 방향 라벨)
CAMERA_ANGLES = {
    "FRONT":  (20, 0, "FRONT (Z-)"),
    "BACK":   (20, 180, "BACK (Z+)"),
    "LEFT":   (20, -90, "LEFT (X-)"),
    "RIGHT":  (20, 90, "RIGHT (X+)"),
    "BOTTOM": (-90, 0, "BOTTOM"),
}


# ============================================================
# 로컬 LDView 렌더링 (Windows/로컬 개발용)
# ============================================================

def render_ldr_to_image(
    ldr_path: str,
    output_path: Optional[str] = None,
    width: int = 512,
    height: int = 512,
    latitude: int = 30,
    longitude: int = 45
) -> str:
    if not LDVIEW_LOCAL_AVAILABLE:
        raise FileNotFoundError(f"LDView not found: {LDVIEW_BIN}")

    if not Path(ldr_path).exists():
        raise FileNotFoundError(f"LDR file not found: {ldr_path}")

    if output_path is None:
        output_path = tempfile.mktemp(suffix=".png")

    cmd = [
        LDVIEW_BIN,
        str(ldr_path),
        f"-SaveSnapshot={output_path}",
        f"-SaveWidth={width}",
        f"-SaveHeight={height}",
        f"-DefaultLatitude={latitude}",
        f"-DefaultLongitude={longitude}",
        "-SaveZoomToFit=1",
    ]

    subprocess.run(cmd, check=True, capture_output=True)

    if not Path(output_path).exists():
        raise RuntimeError("Rendering failed - no output image")

    return output_path


def _add_direction_label(image_path: str, label: str) -> bytes:
    """이미지에 방향 라벨 텍스트 추가"""
    if not PIL_AVAILABLE:
        with open(image_path, "rb") as f:
            return f.read()

    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except Exception:
        font = ImageFont.load_default()

    text_bbox = draw.textbbox((10, 10), label, font=font)
    padding = 5
    draw.rectangle(
        [text_bbox[0] - padding, text_bbox[1] - padding,
         text_bbox[2] + padding, text_bbox[3] + padding],
        fill=(0, 0, 0, 180)
    )

    draw.text((10, 10), label, fill=(255, 255, 255), font=font)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def render_ldr_to_base64(
    ldr_path: str,
    width: int = 512,
    height: int = 512,
    latitude: int = 30,
    longitude: int = 45,
    label: str = None
) -> str:
    """LDR 파일을 렌더링하고 base64로 반환 (LLM 전송용)"""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        output_path = f.name

    render_ldr_to_image(ldr_path, output_path, width, height, latitude, longitude)

    if label:
        image_data = _add_direction_label(output_path, label)
        image_b64 = base64.b64encode(image_data).decode("utf-8")
    else:
        with open(output_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

    Path(output_path).unlink(missing_ok=True)

    return image_b64


def render_ldr_multi_angle(
    ldr_path: str,
    angles: List[str] = None,
    width: int = 512,
    height: int = 512
) -> Dict[str, str]:
    """LDR 파일을 지정된 방향에서 렌더링하고 base64로 반환"""
    if angles is None:
        angles = list(CAMERA_ANGLES.keys())

    results = {}
    for angle in angles:
        if angle not in CAMERA_ANGLES:
            logger.warning("Unknown angle '%s', skipping", angle)
            continue

        lat, lon, label = CAMERA_ANGLES[angle]
        try:
            b64 = render_ldr_to_base64(ldr_path, width, height, lat, lon, label)
            results[angle] = b64
        except Exception as e:
            logger.error("Error rendering %s: %s", angle, e)
            results[angle] = None

    return results


# ============================================================
# screenshot-server HTTP 폴백 (프로덕션 Docker용)
# ============================================================

def _render_via_http(ldr_text: str, angles: List[str],
                     width: int = 512, height: int = 512) -> Dict[str, str]:
    """screenshot-server의 /render/multi-angle API를 HTTP로 호출"""
    import httpx

    url = f"{SCREENSHOT_SERVER_URL}/render/multi-angle"
    logger.info("[Render] HTTP fallback → %s", url)

    resp = httpx.post(
        url,
        json={
            "ldr_text": ldr_text,
            "angles": angles,
            "width": width,
            "height": height,
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    data = resp.json()

    return data.get("images", {})


# ============================================================
# 메인 엔트리포인트 (observe.py에서 호출)
# ============================================================

def render_model_multi_angle(model, parts_db, angles: List[str] = None,
                             width: int = 512, height: int = 512) -> Dict[str, str]:
    """
    BrickModel을 지정된 방향에서 렌더링.

    전략:
      1. 로컬 LDView 있으면 → 직접 렌더링
      2. 없으면 → screenshot-server HTTP 호출
    """
    import sys
    exporter_dir = Path(__file__).parent.parent
    if str(exporter_dir) not in sys.path:
        sys.path.insert(0, str(exporter_dir))

    from ldr_converter import model_to_ldr

    ldr_content = model_to_ldr(model, parts_db, skip_validation=True, skip_physics=True)

    if angles is None:
        angles = list(CAMERA_ANGLES.keys())

    # 전략 1: 로컬 LDView
    if LDVIEW_LOCAL_AVAILABLE:
        logger.info("[Render] Local LDView: %s", LDVIEW_BIN)
        with tempfile.NamedTemporaryFile(suffix=".ldr", delete=False, mode='w', encoding='utf-8') as f:
            f.write(ldr_content)
            ldr_path = f.name

        try:
            return render_ldr_multi_angle(ldr_path, angles, width, height)
        finally:
            Path(ldr_path).unlink(missing_ok=True)

    # 전략 2: screenshot-server HTTP 폴백
    logger.info("[Render] LDView not found locally, using HTTP fallback")
    try:
        return _render_via_http(ldr_content, angles, width, height)
    except Exception as e:
        logger.error("[Render] HTTP fallback failed: %s", e)
        return {}


if __name__ == "__main__":
    test_ldr = r"C:\Users\301\Desktop\Brickers 관련 문서\테스트 LDR\냥이.ldr"

    print("=== 5방향 렌더링 테스트 ===")
    multi = render_ldr_multi_angle(test_ldr)
    for angle, b64 in multi.items():
        if b64:
            print(f"  {angle}: {len(b64)} bytes")
        else:
            print(f"  {angle}: FAILED")
