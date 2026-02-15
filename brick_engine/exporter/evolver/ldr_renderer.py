"""LDR → 이미지 렌더링 (LDView CLI 사용)

방향 규칙 (LDraw 좌표계 기준):
- Z 음수 = 앞 (FRONT)
- Z 양수 = 뒤 (BACK)
- X 음수 = 왼쪽 (LEFT)
- X 양수 = 오른쪽 (RIGHT)
"""
import subprocess
import tempfile
import base64
import io
from pathlib import Path
from typing import Optional, List, Dict

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[WARNING] PIL not available, direction labels disabled")

LDVIEW_PATH = r"C:\Program Files\LDView\LDView64.exe"

# 카메라 각도 프리셋 (latitude, longitude, 방향 라벨)
# LDraw: Z- = 앞, Z+ = 뒤, X- = 왼, X+ = 오
CAMERA_ANGLES = {
    "FRONT": (20, 0, "FRONT (Z-)"),           # 앞에서 봄
    "BACK": (20, 180, "BACK (Z+)"),           # 뒤에서 봄
    "RIGHT": (20, 90, "RIGHT (X+)"),          # 오른쪽에서 봄
    "LEFT": (20, -90, "LEFT (X-)"),           # 왼쪽에서 봄
    "TOP": (90, 0, "TOP"),                    # 위에서 봄
    "BOTTOM": (-90, 0, "BOTTOM"),             # 아래에서 봄
    "FRONT_RIGHT": (30, 45, "FRONT-RIGHT"),   # 앞-오른쪽 대각선
}


def render_ldr_to_image(
    ldr_path: str,
    output_path: Optional[str] = None,
    width: int = 512,
    height: int = 512,
    latitude: int = 30,
    longitude: int = 45
) -> str:
    """
    LDR 파일을 이미지로 렌더링

    Args:
        ldr_path: LDR 파일 경로
        output_path: 출력 이미지 경로 (없으면 임시 파일)
        width: 이미지 너비
        height: 이미지 높이
        latitude: 카메라 위도 (위아래 각도)
        longitude: 카메라 경도 (좌우 회전)

    Returns:
        렌더링된 이미지 경로
    """
    if not Path(LDVIEW_PATH).exists():
        raise FileNotFoundError(f"LDView not found: {LDVIEW_PATH}")

    if not Path(ldr_path).exists():
        raise FileNotFoundError(f"LDR file not found: {ldr_path}")

    if output_path is None:
        output_path = tempfile.mktemp(suffix=".png")

    cmd = [
        LDVIEW_PATH,
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

    # 폰트 (시스템 기본 폰트 사용)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    # 텍스트 배경 (검정 반투명)
    text_bbox = draw.textbbox((10, 10), label, font=font)
    padding = 5
    draw.rectangle(
        [text_bbox[0] - padding, text_bbox[1] - padding,
         text_bbox[2] + padding, text_bbox[3] + padding],
        fill=(0, 0, 0, 180)
    )

    # 텍스트 (흰색)
    draw.text((10, 10), label, fill=(255, 255, 255), font=font)

    # bytes로 변환
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
    """
    LDR 파일을 렌더링하고 base64로 반환 (LLM 전송용)

    Args:
        label: 이미지에 표시할 방향 라벨 (예: "FRONT (Z-)")
    """
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
    """
    LDR 파일을 7방향에서 렌더링하고 base64로 반환
    각 이미지에 방향 라벨 표시됨

    Args:
        ldr_path: LDR 파일 경로
        angles: 렌더링할 각도 목록 (기본: 7방향 전체)
        width: 이미지 너비
        height: 이미지 높이

    Returns:
        {angle_name: base64_image, ...}
        예: {"FRONT": "...", "BACK": "...", ...}
    """
    if angles is None:
        angles = list(CAMERA_ANGLES.keys())

    results = {}
    for angle in angles:
        if angle not in CAMERA_ANGLES:
            print(f"Warning: Unknown angle '{angle}', skipping")
            continue

        lat, lon, label = CAMERA_ANGLES[angle]
        try:
            b64 = render_ldr_to_base64(ldr_path, width, height, lat, lon, label)
            results[angle] = b64
        except Exception as e:
            print(f"Error rendering {angle}: {e}")
            results[angle] = None

    return results


def render_model_multi_angle(model, parts_db, angles: List[str] = None,
                             width: int = 512, height: int = 512) -> Dict[str, str]:
    """
    BrickModel을 7방향에서 렌더링

    Args:
        model: BrickModel 객체
        parts_db: 파츠 DB
        angles: 렌더링할 각도 목록
        width: 이미지 너비
        height: 이미지 높이

    Returns:
        {angle_name: base64_image, ...}
    """
    import sys
    exporter_dir = Path(__file__).parent.parent
    if str(exporter_dir) not in sys.path:
        sys.path.insert(0, str(exporter_dir))

    from ldr_converter import model_to_ldr

    with tempfile.NamedTemporaryFile(suffix=".ldr", delete=False, mode='w', encoding='utf-8') as f:
        ldr_content = model_to_ldr(model, parts_db, skip_validation=True, skip_physics=True)
        f.write(ldr_content)
        ldr_path = f.name

    try:
        results = render_ldr_multi_angle(ldr_path, angles, width, height)
    finally:
        Path(ldr_path).unlink(missing_ok=True)

    return results


if __name__ == "__main__":
    test_ldr = r"C:\Users\301\Desktop\Brickers 관련 문서\테스트 LDR\냥이.ldr"

    print("=== 7방향 렌더링 테스트 ===")
    multi = render_ldr_multi_angle(test_ldr)
    for angle, b64 in multi.items():
        if b64:
            print(f"  {angle}: {len(b64)} bytes")
        else:
            print(f"  {angle}: FAILED")
