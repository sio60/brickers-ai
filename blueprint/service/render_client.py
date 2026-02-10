# blueprint/service/render_client.py
"""LDView를 로컬에서 직접 실행하여 LDR 스텝별 렌더링"""
from __future__ import annotations

import os
import re
import math
import base64
import shutil
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anyio
from PIL import Image, ImageOps

LDVIEW_BIN = os.environ.get("LDVIEW_BIN", "LDView")
LDRAWDIR = os.environ.get("LDRAWDIR", "/usr/share/ldraw")

# LDView 사용 가능 여부 (컨테이너 빌드 시 설치됨)
RENDER_ENABLED = shutil.which(LDVIEW_BIN) is not None


def _split_ldr_steps(ldr_text: str) -> List[str]:
    """
    LDR 텍스트를 0 STEP 기준으로 분할하여 누적(cumulative) LDR 리스트 반환.
    step_ldrs[i] = 1번째 ~ (i+1)번째 스텝까지 누적된 LDR 내용
    """
    lines = ldr_text.replace("\r\n", "\n").split("\n")

    header_lines: list[str] = []
    body_lines: list[str] = []
    in_body = False
    for raw in lines:
        line = raw.strip()
        if not in_body:
            if re.match(r"^0\s+(STEP|ROTSTEP)\b", line, re.IGNORECASE):
                in_body = True
                body_lines.append(raw)
            elif line.startswith("1 "):
                in_body = True
                body_lines.append(raw)
            else:
                header_lines.append(raw)
        else:
            body_lines.append(raw)

    steps_parts: list[list[str]] = []
    current: list[str] = []
    for raw in body_lines:
        line = raw.strip()
        if re.match(r"^0\s+(STEP|ROTSTEP)\b", line, re.IGNORECASE):
            if current:
                steps_parts.append(current)
                current = []
        else:
            if line:
                current.append(raw)
    if current:
        steps_parts.append(current)

    if not steps_parts:
        return [ldr_text]

    header_block = "\n".join(header_lines)
    cumulative: list[str] = []
    accumulated: list[str] = []
    for part in steps_parts:
        accumulated.extend(part)
        full = header_block + "\n" + "\n".join(accumulated) + "\n"
        cumulative.append(full)

    return cumulative


def _render_snapshot(
    ldr_path: str,
    output_path: str,
    width: int,
    height: int,
    camera: Tuple[float, float, float],
) -> bool:
    """LDView CLI를 xvfb-run으로 실행하여 스냅샷 캡처."""
    x, y, z = camera
    r = math.sqrt(x * x + y * y + z * z) or 1.0
    lat = math.degrees(math.asin(-y / r))
    lon = math.degrees(math.atan2(x, z))

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
            if result.stdout:
                print(f"[LDView] stdout: {result.stdout[:300]}")
        exists = Path(output_path).exists() and Path(output_path).stat().st_size > 0
        if not exists and result.returncode == 0:
            print(f"[LDView] No output file despite exit=0 | {Path(ldr_path).name}")
        return exists
    except subprocess.TimeoutExpired:
        print(f"[LDView] Timeout rendering {Path(ldr_path).name}")
        return False
    except FileNotFoundError:
        print(f"[LDView] Binary not found: {LDVIEW_BIN}")
        return False


def _render_all_steps_sync(
    ldr_content: str,
    width: int = 1024,
    height: int = 768,
    views: List[List[float]] | None = None,
) -> List[List[bytes]]:
    """
    (sync) LDR -> 스텝 분할 -> 각 스텝 x 뷰 렌더링 -> PNG bytes 리스트 반환.
    """
    if views is None:
        views = [
            [250, -350, 250],    # 45° quarter view (main view)
            [1, -600, 1],        # Top-down view
            [400, -50, 400],     # Ground level view
        ]

    step_ldrs = _split_ldr_steps(ldr_content)
    total = len(step_ldrs)
    print(f"[Renderer] {total} steps, {len(views)} views each")

    tmpdir = tempfile.mkdtemp(prefix="brickers_render_")
    step_images: List[List[bytes]] = []

    try:
        for step_idx, ldr_text in enumerate(step_ldrs):
            ldr_file = os.path.join(tmpdir, f"step_{step_idx}.ldr")
            with open(ldr_file, "w", encoding="utf-8") as f:
                f.write(ldr_text)

            view_bytes: List[bytes] = []
            for view_idx, cam in enumerate(views):
                png_file = os.path.join(tmpdir, f"step_{step_idx}_view_{view_idx}.png")
                camera = (float(cam[0]), float(cam[1]), float(cam[2]))

                ok = _render_snapshot(
                    ldr_path=ldr_file,
                    output_path=png_file,
                    width=width,
                    height=height,
                    camera=camera,
                )

                if ok:
                    # Add 10% white padding for breathing room
                    img = Image.open(png_file)
                    pad_x = int(img.width * 0.10)
                    pad_y = int(img.height * 0.10)
                    padded = ImageOps.expand(
                        img, border=(pad_x, pad_y, pad_x, pad_y),
                        fill=(255, 255, 255),
                    )
                    buf = BytesIO()
                    padded.save(buf, format="PNG")
                    view_bytes.append(buf.getvalue())
                    print(f"  [Step {step_idx+1}/{total}] View {view_idx+1}: OK")
                else:
                    view_bytes.append(b"")
                    print(f"  [Step {step_idx+1}/{total}] View {view_idx+1}: FAILED")

            step_images.append(view_bytes)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return step_images


def _make_single_part_ldr(part_id: str, color: int) -> str:
    """개별 파츠를 위한 완전한 LDR 파일 생성"""
    # part_id가 이미 소문자 + 확장자 제거된 상태
    # LDView가 인식하려면 최소한의 LDraw MPD 헤더 필요
    return (
        f"0 FILE {part_id}_thumb.ldr\n"
        f"0 {part_id}_thumb.ldr\n"
        f"0 Name: {part_id}_thumb.ldr\n"
        "0 Author: Brickers\n"
        f"1 {color} 0 0 0 1 0 0 0 1 0 0 0 1 {part_id}.dat\n"
        "0 STEP\n"
        "0 NOFILE\n"
    )


def _render_part_thumbnails_sync(
    parts: List[Tuple[str, int]],
    width: int = 256,
    height: int = 256,
) -> Dict[str, bytes]:
    """
    개별 파츠 썸네일 렌더링.
    Returns: {"partId_color": PNG bytes, ...}
    """
    thumbnails: Dict[str, bytes] = {}
    if not parts:
        return thumbnails

    tmpdir = tempfile.mkdtemp(prefix="brickers_thumbs_")
    camera = (250, -350, 250)  # 45° quarter view
    ok_count = 0
    fail_count = 0

    try:
        for part_id, color in parts:
            key = f"{part_id}_{color}"
            if key in thumbnails:
                continue

            ldr_content = _make_single_part_ldr(part_id, color)
            ldr_file = os.path.join(tmpdir, f"{key}.ldr")
            png_file = os.path.join(tmpdir, f"{key}.png")

            with open(ldr_file, "w", encoding="utf-8") as f:
                f.write(ldr_content)

            ok = _render_snapshot(ldr_file, png_file, width, height, camera)
            if ok:
                img = Image.open(png_file)
                pad = int(max(img.size) * 0.05)
                padded = ImageOps.expand(
                    img, border=pad, fill=(255, 255, 255),
                )
                buf = BytesIO()
                padded.save(buf, format="PNG")
                thumbnails[key] = buf.getvalue()
                ok_count += 1
            else:
                thumbnails[key] = b""
                fail_count += 1
                print(f"  [Thumb] FAILED: {key} (part={part_id}.dat, color={color})")

        print(f"[Thumb] Done: {ok_count} OK, {fail_count} FAILED out of {len(parts)} parts")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return thumbnails


async def render_part_thumbnails(
    parts: List[Tuple[str, int]],
    width: int = 256,
    height: int = 256,
) -> Dict[str, bytes]:
    """파츠 썸네일 렌더링 (async wrapper)"""
    if not RENDER_ENABLED:
        return {}
    return await anyio.to_thread.run_sync(
        lambda: _render_part_thumbnails_sync(parts, width, height)
    )


async def render_ldr_steps(
    ldr_content: str,
    width: int = 1024,
    height: int = 768,
    views: Optional[List[List[float]]] = None,
) -> List[List[bytes]]:
    """
    LDR 텍스트를 스텝별로 렌더링하여 이미지 bytes 리스트 반환. (async wrapper)

    Returns:
        step_images[step_idx][view_idx] = PNG bytes
    Raises:
        RuntimeError: LDView 미설치 또는 렌더링 실패
    """
    if not RENDER_ENABLED:
        raise RuntimeError(
            f"LDView binary not found ({LDVIEW_BIN}). "
            "PDF generation requires LDView installed in the container."
        )

    return await anyio.to_thread.run_sync(
        lambda: _render_all_steps_sync(ldr_content, width, height, views)
    )
