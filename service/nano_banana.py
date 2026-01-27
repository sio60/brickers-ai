import os
import base64
from typing import Union

import anyio
from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

PROMPT = """
Create ONE SINGLE IMAGE that is a 2x2 grid collage (four panels inside one image).
Do NOT output separate images. Do NOT output a single-view image.

Layout (must be visible as 4 distinct panels with clear spacing/borders):
- Top-left: front view
- Top-right: left 3/4 view
- Bottom-left: right 3/4 view
- Bottom-right: back view

Rules:
- The result must be ONE composite image containing all four panels.
- Clean white studio background in every panel.
- Soft shadow in every panel.
- Keep the exact same LEGO model/subject consistent across panels.
- Same lighting and scale across panels.
- Add thin dividers or margins between panels so the 2x2 grid is obvious.
"""

def _decode_if_base64_image(data: Union[bytes, str]) -> bytes:
    if data is None:
        return b""

    if isinstance(data, str):
        s = data.strip()
    else:
        try:
            s = data.decode("utf-8", errors="strict").strip()
        except Exception:
            return data

    if s.startswith("data:image"):
        try:
            s = s.split(",", 1)[1].strip()
        except Exception:
            pass

    looks_like_base64 = (
        s.startswith("iVBOR") or
        s.startswith("/9j/") or
        all(c.isalnum() or c in "+/=\n\r" for c in s[:200])
    )

    if looks_like_base64:
        try:
            return base64.b64decode(s, validate=False)
        except Exception:
            return data if isinstance(data, (bytes, bytearray)) else s.encode("utf-8")

    return data if isinstance(data, (bytes, bytearray)) else s.encode("utf-8")

async def render_one_image(img_bytes: bytes, mime: str) -> bytes:
    model = os.getenv("NANO_BANANA_MODEL", "gemini-2.5-flash-image")

    # ✅ 동기 네트워크 호출이라 이벤트 루프 블로킹 방지: thread로 실행
    def _call():
        return client.models.generate_content(
            model=model,
            contents=[
                {"text": PROMPT},
                {"inline_data": {"mime_type": mime, "data": img_bytes}},
            ],
            config=types.GenerateContentConfig(response_modalities=["Text", "Image"]),
        )

    resp = await anyio.to_thread.run_sync(_call)

    if not resp.candidates:
        raise ValueError("no candidates from model")

    parts = resp.candidates[0].content.parts if resp.candidates[0].content else []
    out_bytes = None

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            out_bytes = inline.data
            break

    if out_bytes is None:
        raise ValueError("no image returned from model")

    out_bytes = _decode_if_base64_image(out_bytes)

    # PNG/JPEG 매직넘버 체크
    if len(out_bytes) >= 2 and out_bytes[0] == 0xFF and out_bytes[1] == 0xD8:
        return out_bytes
    if len(out_bytes) >= 8 and out_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return out_bytes

    # 혹시 아직 base64면 추가 디코드
    try:
        head = out_bytes[:20].decode("utf-8", errors="ignore")
        if head.startswith("iVBOR") or head.startswith("/9j/"):
            out_bytes = base64.b64decode(out_bytes, validate=False)
    except Exception:
        pass

    return out_bytes
