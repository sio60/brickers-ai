# service/gemini_image.py
"""Gemini 이미지 보정 + 메타데이터(Subject/Tags) 파싱"""
from __future__ import annotations

import os
import base64

import anyio
from google import genai
from google.genai import types as genai_types

PROMPT_NANO_BANANA = """
Create a high-quality, vibrant image suitable for 3D modeling.

Requirements:
- Single view from the best angle to capture the subject's key features
- Clean white studio background
- Soft, even lighting with subtle shadows
- Vivid, saturated colors with clear color separation
- Sharp, well-defined edges and contours
- High contrast between subject and background
- Professional product photography style
- Remove any text, logos, patterns, or decorative details
- Simplify complex textures into solid, uniform colors
- Maintain the overall shape and proportions of the subject

[METADATA_REQUEST]
Also, identify the subject in a single word and provide 3-5 relevant hashtags.
Format: SUBJECT: <word> | TAGS: <tag1>, <tag2>, ...
Example: SUBJECT: Pikachu | TAGS: Pokemon, Kids, Brick, Yellow
"""


def _decode_if_base64_image(data: bytes | str) -> bytes:
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
        s.startswith("iVBOR")
        or s.startswith("/9j/")
        or all(c.isalnum() or c in "+/=\n\r" for c in s[:200])
    )

    if looks_like_base64:
        try:
            return base64.b64decode(s, validate=False)
        except Exception:
            return data if isinstance(data, (bytes, bytearray)) else s.encode("utf-8")

    return data if isinstance(data, (bytes, bytearray)) else s.encode("utf-8")


def _render_one_image_sync(img_bytes: bytes, mime: str, language: str = "en") -> tuple[bytes, str, list[str]]:
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    lang_instruction = ""
    if language == "ko":
        lang_instruction = "\n[IMPORTANT] The SUBJECT and TAGS representing the analysis result MUST be in KOREAN."
    elif language == "ja":
        lang_instruction = "\n[IMPORTANT] The SUBJECT and TAGS representing the analysis result MUST be in JAPANESE."
    
    final_prompt = PROMPT_NANO_BANANA + lang_instruction

    model = os.environ.get("NANO_BANANA_MODEL", "gemini-2.5-flash-image")
    client = genai.Client(api_key=gemini_key)

    resp = client.models.generate_content(
        model=model,
        contents=[
            {"text": final_prompt},
            {"inline_data": {"mime_type": mime, "data": img_bytes}},
        ],
        config=genai_types.GenerateContentConfig(response_modalities=["Text", "Image"]),
    )

    if not resp.candidates:
        raise ValueError("no candidates from model")

    parts = resp.candidates[0].content.parts if resp.candidates[0].content else []
    out_bytes = None
    meta_text = ""

    for part in parts:
        if hasattr(part, "inline_data") and part.inline_data and part.inline_data.data:
            out_bytes = part.inline_data.data
        if hasattr(part, "text") and part.text:
            meta_text += part.text

    if out_bytes is None:
        if "data:image" in meta_text or (
            len(meta_text) > 1000
            and any(prefix in meta_text for prefix in ["iVBOR", "/9j/"])
        ):
            out_bytes = meta_text.strip()
        else:
            print(f"[Gemini] Warning: No image returned from model (meta_text len: {len(meta_text)}). Falling back to original image.")
            out_bytes = img_bytes

    out_bytes = _decode_if_base64_image(out_bytes)

    # PNG/JPG magic number check
    is_valid_image = False
    if len(out_bytes) >= 2 and out_bytes[0] == 0xFF and out_bytes[1] == 0xD8:
        is_valid_image = True
    elif len(out_bytes) >= 8 and out_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        is_valid_image = True

    if not is_valid_image:
        try:
            head = out_bytes[:20].decode("utf-8", errors="ignore")
            if head.startswith("iVBOR") or head.startswith("/9j/"):
                out_bytes = base64.b64decode(out_bytes, validate=False)
        except Exception:
            pass

    # Metadata parsing (SUBJECT / TAGS)
    subject = "Object"
    tags = ["Kids", "Brick"]

    try:
        if meta_text:
            print(f"[Gemini Meta] Raw extraction text: {meta_text[:100]}...")

        if "SUBJECT:" in meta_text:
            s_part = meta_text.split("SUBJECT:")[1].split("|")[0].strip()
            if s_part:
                subject = s_part
        if "TAGS:" in meta_text:
            t_part = meta_text.split("TAGS:")[1].strip()
            tags = [t.strip() for t in t_part.replace("#", "").split(",") if t.strip()]
    except Exception as e:
        print(f"[Gemini Meta] Tag parse error: {e}")

    return out_bytes, subject, tags


async def render_one_image_async(img_bytes: bytes, mime: str, language: str = "en") -> tuple[bytes, str, list[str]]:
    """Gemini 호출은 동기라서 thread로 래핑"""
    return await anyio.to_thread.run_sync(_render_one_image_sync, img_bytes, mime, language)
