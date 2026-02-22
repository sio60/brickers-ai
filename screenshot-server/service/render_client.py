# screenshot-server/service/render_client.py
"""LDView를 사용하여 LDR 완성 모델의 6면 스크린샷 렌더링"""
from __future__ import annotations

import os
import math
import shutil
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Dict

import anyio
from PIL import Image, ImageOps

LDVIEW_BIN = os.environ.get("LDVIEW_BIN", "LDView")
LDRAWDIR = os.environ.get("LDRAWDIR", "/usr/share/ldraw")

RENDER_ENABLED = shutil.which(LDVIEW_BIN) is not None

# 6면 카메라 뷰 (latitude, longitude)
# 약 30도 위에서 내려다보는 입체감 있는 앵글 + 45도 회전
VIEWS: Dict[str, Dict[str, float]] = {
    "front":  {"lat": 30,  "lon": 45},
    "back":   {"lat": 30,  "lon": 225},
    "left":   {"lat": 30,  "lon": 135},
    "right":  {"lat": 30,  "lon": -45},
    "top":    {"lat": 80,  "lon": 45},
    "bottom": {"lat": -30, "lon": 45},
}


def _render_snapshot(
    ldr_path: str,
    output_path: str,
    width: int,
    height: int,
    lat: float,
    lon: float,
) -> bool:
    """LDView CLI를 xvfb-run으로 실행하여 스냅샷 캡처."""
    cmd = [
        "xvfb-run", "-a",
        LDVIEW_BIN,
        ldr_path,
        f"-SaveSnapshot={output_path}",
        f"-SaveWidth={width}",
        f"-SaveHeight={height}",
        "-SaveZoomToFit=1",
        "-SaveAlpha=0",
        "-BackgroundColor3=0xffffff",
        "-UseOrthographicProjection=1",
        "-EdgeLines=1",
        "-LineSmoothing=1",
        "-ShowHighlightLines=1",
        "-ConditionalHighlights=1",
        f"-DefaultLatitude={lat:.1f}",
        f"-DefaultLongitude={lon:.1f}",
        f"-LDrawDir={LDRAWDIR}",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"[LDView] exit={result.returncode} | {Path(ldr_path).name}")
            if result.stderr:
                print(f"[LDView] stderr: {result.stderr[:500]}")
        exists = Path(output_path).exists() and Path(output_path).stat().st_size > 0
        return exists
    except subprocess.TimeoutExpired:
        print(f"[LDView] Timeout rendering {Path(ldr_path).name}")
        return False
    except FileNotFoundError:
        print(f"[LDView] Binary not found: {LDVIEW_BIN}")
        return False


def _render_6_views_sync(
    ldr_text: str,
    width: int = 1024,
    height: int = 1024,
) -> Dict[str, bytes]:
    """
    (sync) LDR 텍스트 → 6면 렌더링 → {"front": PNG bytes, ...}
    """
    tmpdir = tempfile.mkdtemp(prefix="brickers_screenshot_")
    screenshots: Dict[str, bytes] = {}

    try:
        ldr_file = os.path.join(tmpdir, "model.ldr")
        with open(ldr_file, "w", encoding="utf-8") as f:
            f.write(ldr_text)

        total = len(VIEWS)
        ok_count = 0

        for view_name, angles in VIEWS.items():
            png_file = os.path.join(tmpdir, f"{view_name}.png")

            ok = _render_snapshot(
                ldr_path=ldr_file,
                output_path=png_file,
                width=width,
                height=height,
                lat=angles["lat"],
                lon=angles["lon"],
            )

            if ok:
                img = Image.open(png_file)
                pad = int(max(img.size) * 0.05)
                padded = ImageOps.expand(img, border=pad, fill=(255, 255, 255))
                buf = BytesIO()
                padded.save(buf, format="PNG")
                screenshots[view_name] = buf.getvalue()
                ok_count += 1
                print(f"  [Screenshot] {view_name}: OK ({len(screenshots[view_name])} bytes)")
            else:
                screenshots[view_name] = b""
                print(f"  [Screenshot] {view_name}: FAILED")

        print(f"[Screenshot] Done: {ok_count}/{total} views rendered")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return screenshots


async def render_6_views(
    ldr_text: str,
    width: int = 1024,
    height: int = 1024,
) -> Dict[str, bytes]:
    """
    LDR 텍스트를 6면으로 렌더링 (async wrapper).
    Returns: {"front": PNG bytes, "back": PNG bytes, ...}
    """
    if not RENDER_ENABLED:
        raise RuntimeError(
            f"LDView binary not found ({LDVIEW_BIN}). "
            "Screenshot generation requires LDView installed."
        )

    return await anyio.to_thread.run_sync(
        lambda: _render_6_views_sync(ldr_text, width, height)
    )


def _render_custom_angles_sync(
    ldr_text: str,
    angles: Dict[str, Dict[str, float]],
    width: int = 512,
    height: int = 512,
) -> Dict[str, bytes]:
    """
    (sync) LDR 텍스트를 커스텀 앵글로 렌더링 → {"FRONT": PNG bytes, ...}
    패딩 없이 raw PNG 반환 (비전 분석용).
    """
    tmpdir = tempfile.mkdtemp(prefix="brickers_render_")
    results: Dict[str, bytes] = {}

    try:
        ldr_file = os.path.join(tmpdir, "model.ldr")
        with open(ldr_file, "w", encoding="utf-8") as f:
            f.write(ldr_text)

        ok_count = 0
        for name, angle in angles.items():
            png_file = os.path.join(tmpdir, f"{name}.png")
            ok = _render_snapshot(
                ldr_path=ldr_file,
                output_path=png_file,
                width=width,
                height=height,
                lat=angle["lat"],
                lon=angle["lon"],
            )
            if ok:
                with open(png_file, "rb") as pf:
                    results[name] = pf.read()
                ok_count += 1
                print(f"  [Render] {name}: OK")
            else:
                results[name] = b""
                print(f"  [Render] {name}: FAILED")

        print(f"[Render] Done: {ok_count}/{len(angles)} angles rendered")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return results


async def render_custom_angles(
    ldr_text: str,
    angles: Dict[str, Dict[str, float]],
    width: int = 512,
    height: int = 512,
) -> Dict[str, bytes]:
    """
    LDR 텍스트를 커스텀 앵글로 렌더링 (async wrapper).
    Returns: {"FRONT": PNG bytes, ...}
    """
    if not RENDER_ENABLED:
        raise RuntimeError(
            f"LDView binary not found ({LDVIEW_BIN}). "
            "Rendering requires LDView installed."
        )

    return await anyio.to_thread.run_sync(
        lambda: _render_custom_angles_sync(ldr_text, angles, width, height)
    )
