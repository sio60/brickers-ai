# screenshot-server/service/background_composer.py
import os
import io
import asyncio
import base64
import httpx
from PIL import Image

# Google GenAI
import google.genai as genai
from google.genai import types as genai_types

# Service
# screenshot-server uses s3_client from its own service package
from .s3_client import upload_bytes_to_s3

PROMPT_TEMPLATE = """
Generate a realistic, cinematic background image suitable for a LEGO model of: {subject}.
The background should depict the natural environment or setting where this object would exist in real life.
Examples:
- If subject is "Dinosaur", generate a prehistoric Jurassic jungle.
- If subject is "Race Car", generate a race track or city street.
- If subject is "Spaceship", generate a starry outer space or moon surface.
- If subject is "Castle", generate a medieval landscape.

Style: Realistic photography, cinematic lighting, 8k resolution, highly detailed textures.
Composition: Eye-level perspective. The center area MUST be open and have a flat surface (ground, road, water) to place the model.
Do NOT include the subject itself in the background, only the environment.
No text, no borders.
"""

def _sanitize_text(s: str) -> str:
    if not s:
        return ""
    return s.encode("utf-8", "ignore").decode("utf-8", "ignore")

def _generate_background_sync(subject: str) -> bytes:
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    client = genai.Client(api_key=gemini_key)
    model = os.environ.get("NANO_BANANA_MODEL", "gemini-2.0-flash-exp") # [NOTE] Adjust model name if needed

    safe_subject = _sanitize_text(subject)
    prompt = PROMPT_TEMPLATE.format(subject=safe_subject or "lego creation")

    # Safety settings: allow everything to prevent false positives
    safety_settings = [
        genai_types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
        genai_types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
        genai_types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
        genai_types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
    ]

    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    candidate_count=1,
                    safety_settings=safety_settings,
                ),
            )
            if resp.candidates:
                break
        except Exception as e:
            print(f"[Gemini] Attempt {attempt+1}/{max_retries} failed: {e}")
            last_error = e
            import time
            time.sleep(1)
    else:
        raise last_error or ValueError(f"Failed to generate image after {max_retries} attempts")

    if not resp.candidates:
        raise ValueError("No image candidates returned from Gemini")

    def _looks_like_image(buf: bytes) -> bool:
        return (
            buf.startswith(b"\x89PNG")
            or buf.startswith(b"\xff\xd8\xff")
            or (buf.startswith(b"RIFF") and b"WEBP" in buf[:16])
            or buf.startswith(b"GIF8")
            or (buf[4:8] == b"ftyp" and buf[8:12] in (b"heic", b"heix", b"mif1", b"avif"))
        )

    def _validate_image_bytes(buf: bytes, mime: str | None = None) -> bytes:
        try:
            Image.open(io.BytesIO(buf)).verify()
            return buf
        except Exception:
            if _looks_like_image(buf):
                return buf
            try:
                img = Image.open(io.BytesIO(buf))
                img.load()
                return buf
            except Exception:
                raise

    for part in resp.candidates[0].content.parts:
        inline = getattr(part, "inline_data", None)
        if inline and inline.data:
            mime = (inline.mime_type or "").lower()
            data = inline.data
            if isinstance(data, str):
                if data.startswith("data:"):
                    try:
                        data = data.split(",", 1)[1]
                    except Exception:
                        pass
                try:
                    data = base64.b64decode(data)
                except Exception:
                    try:
                        data = data.encode("utf-8", errors="surrogateescape")
                    except Exception:
                        try:
                            data = data.encode("utf-8", errors="ignore")
                        except Exception:
                            continue
            elif isinstance(data, bytes):
                if not _looks_like_image(data):
                    try:
                        decoded = base64.b64decode(data)
                        if _looks_like_image(decoded):
                            data = decoded
                    except Exception:
                        pass
            if mime and not mime.startswith("image/"):
                continue
            if not mime and not _looks_like_image(data):
                continue
            try:
                return _validate_image_bytes(data, mime)
            except Exception:
                continue

    raise ValueError("No valid image data found in Gemini response")

async def generate_background_async(subject: str) -> bytes:
    return await asyncio.to_thread(_generate_background_sync, subject)
