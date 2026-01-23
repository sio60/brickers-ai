import os
import base64
from typing import Union, Any

import anyio
from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

PROMPT = """
Create ONE SINGLE IMAGE that is a 2x2 grid collage (four panels inside one image).
Do NOT output separate images. Do NOT output a single-view image.

The subject MUST be a simple LEGO-brick sculpture version of the input (blocky, low detail).
If the input is a line drawing or has a transparent/cutout background, interpret it as a cute toy-like brick sculpture.

Layout (must be visible as 4 distinct panels with clear spacing/borders):
- Top-left: front view
- Top-right: left 3/4 view
- Bottom-left: right 3/4 view
- Bottom-right: back view

Style rules (VERY IMPORTANT):
- Matte, non-reflective ABS plastic look (NO glossy/shiny, NO specular highlights, NO reflections).
- Flat diffuse lighting (UNLIT / toon-like shading is OK). Avoid gradients and baked lighting.
- Minimal shadow only (very soft, very low contrast). NO strong AO/dark creases.
- No textures, no patterns, no decals, no metal, no glass, no transparency.
- Keep the exact same model consistent across all 4 panels, same scale.
- Clean plain white background in every panel.
- Add thin dividers or margins between panels so the 2x2 grid is obvious.

Color rules:
- Prefer colors that match the input object. Do NOT invent random extra colors.
- If the input looks like monochrome line art (mostly black lines on white), keep the sculpture mostly WHITE,
  with only minimal BLACK details for eyes/mouth/outline; avoid multicolor.
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


def _call_gemini_generate(mime: str, img_bytes: bytes) -> Any:
    model = os.getenv("NANO_BANANA_MODEL", "gemini-2.5-flash-image")
    return client.models.generate_content(
        model=model,
        contents=[
            {"text": PROMPT},
            {"inline_data": {"mime_type": mime, "data": img_bytes}},
        ],
        config=types.GenerateContentConfig(response_modalities=["Image"]),
    )


async def render_one_image(img_bytes: bytes, mime: str) -> bytes:
    resp = await anyio.to_thread.run_sync(_call_gemini_generate, mime, img_bytes)

    if not getattr(resp, "candidates", None):
        raise ValueError("no candidates from model")

    cand0 = resp.candidates[0]
    parts = cand0.content.parts if getattr(cand0, "content", None) else []
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

    try:
        head = out_bytes[:20].decode("utf-8", errors="ignore")
        if head.startswith("iVBOR") or head.startswith("/9j/"):
            out_bytes = base64.b64decode(out_bytes, validate=False)
    except Exception:
        pass

    return out_bytes
