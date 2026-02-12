
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
from service.s3_client import upload_bytes_to_s3

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

def _generate_background_sync(subject: str) -> bytes:
    """Gemini를 사용하여 배경 이미지 생성 (Sync)"""
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    client = genai.Client(api_key=gemini_key)
    model = os.environ.get("NANO_BANANA_MODEL", "gemini-2.5-flash-image")

    prompt = PROMPT_TEMPLATE.format(subject=subject)

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            response_modalities=["IMAGE"],  # ask for image
            candidate_count=1,
        ),
    )

    if not resp.candidates:
        raise ValueError("No image candidates returned from Gemini")

    def _looks_like_image(buf: bytes) -> bool:
        # quick magic sniff for common formats (PNG, JPEG, WEBP, GIF, HEIC/AVIF)
        return (
            buf.startswith(b"\x89PNG")  # png
            or buf.startswith(b"\xff\xd8\xff")  # jpeg
            or (buf.startswith(b"RIFF") and b"WEBP" in buf[:16])  # webp
            or buf.startswith(b"GIF8")  # gif
            or (buf[4:8] == b"ftyp" and buf[8:12] in (b"heic", b"heix", b"mif1", b"avif"))  # heic/avif
        )

    def _validate_image_bytes(buf: bytes, mime: str | None = None) -> bytes:
        # If Pillow can verify, great
        try:
            Image.open(io.BytesIO(buf)).verify()
            return buf
        except Exception as e_verify:
            # If magic bytes look like an image, accept to let the browser handle formats Pillow lacks (e.g., HEIC/AVIF)
            if _looks_like_image(buf):
                print(f"[Gemini] Pillow verify failed but magic looks like image (mime={mime}): {e_verify}")
                return buf
            # Try load (some formats need load)
            try:
                img = Image.open(io.BytesIO(buf))
                img.load()
                return buf
            except Exception:
                raise

    # Gemini may return inline_data (base64) or file_data (URL)
    for part in resp.candidates[0].content.parts:
        inline = getattr(part, "inline_data", None)
        if inline and inline.data:
            mime = (inline.mime_type or "").lower()
            data = inline.data
            if isinstance(data, str):
                # strip data URL prefix if present
                if data.startswith("data:"):
                    try:
                        data = data.split(",", 1)[1]
                    except Exception:
                        pass
                try:
                    data = base64.b64decode(data)
                except Exception:
                    # Fallback: treat as utf-8 bytes
                    data = data.encode("utf-8", errors="ignore")
            elif isinstance(data, bytes):
                 # Check if it is valid image data
                 if not _looks_like_image(data):
                     # If not, try to decode as base64 (Gemini returns base64 bytes sometimes)
                     try:
                         decoded = base64.b64decode(data)
                         if _looks_like_image(decoded):
                             data = decoded
                     except Exception:
                         pass

            # If mime is present and not image, skip
            if mime and not mime.startswith("image/"):
                print(f"[Gemini] Non-image inline response skipped: mime={mime}")
                continue
            # If mime missing, sniff bytes
            if not mime and not _looks_like_image(data):
                print(f"[Gemini] Inline data without mime does not look like image, skipping (len={len(data)})")
                continue
            try:
                return _validate_image_bytes(data, mime)
            except Exception as e:
                print(f"[Gemini] Inline image validation failed: {e} (len={len(data)}, mime={mime})")
                continue

        file_data = getattr(part, "file_data", None)
        if file_data and getattr(file_data, "file_uri", None):
            try:
                r = httpx.get(file_data.file_uri, timeout=30.0)
                r.raise_for_status()
                buf = r.content
                return _validate_image_bytes(buf, getattr(file_data, "mime_type", None))
            except Exception as e:
                print(f"[Gemini] Failed to fetch/validate file_uri: {e}")
                continue

    raise ValueError("No valid image data found in Gemini response")


async def generate_background_async(subject: str) -> bytes:
    """Gemini 배경 생성 (Async Wrapper)"""
    return await asyncio.to_thread(_generate_background_sync, subject)


def _composite_sync(fg_bytes: bytes, bg_bytes: bytes) -> bytes:
    """Pillow를 사용하여 전경(투명 PNG)을 배경 위에 합성"""
    try:
        fg_img = Image.open(io.BytesIO(fg_bytes)).convert("RGBA")
        bg_img = Image.open(io.BytesIO(bg_bytes)).convert("RGBA")

        # 1. 배경을 전경 크기에 맞게 리사이즈 (Aspect Ratio 유지하며 Fill)
        #    혹은 전경이 너무 크면 줄임. 보통 배경이 1024x1024.
        
        # 전략: 배경을 1024x1024로 가정. 전경을 적절히 배치.
        #      만약 전경이 너무 크면 80% 수준으로 리사이즈.
        
        bg_w, bg_h = bg_img.size
        fg_w, fg_h = fg_img.size
        
        # 전경이 배경보다 크거나, 너무 꽉 차면 리사이즈 (여백 10%)
        max_w = int(bg_w * 0.8)
        max_h = int(bg_h * 0.8)
        
        scale = min(max_w / fg_w, max_h / fg_h)
        if scale < 1.0:
            new_w = int(fg_w * scale)
            new_h = int(fg_h * scale)
            fg_img = fg_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # 2. 중앙 하단에 배치
        #    전경의 바닥이 배경의 중심보다 약간 아래에 오도록?
        #    중앙 정렬.
        
        fg_w, fg_h = fg_img.size
        x = (bg_w - fg_w) // 2
        y = (bg_h - fg_h) // 2 + int(bg_h * 0.1) # 약간 아래로

        # 합성
        bg_img.paste(fg_img, (x, y), fg_img)

        # 결과 저장
        out_io = io.BytesIO()
        bg_img.convert("RGB").save(out_io, format="PNG")
        return out_io.getvalue()
        
    except Exception as e:
        print(f"[Composite] Error: {e}")
        # 실패 시 배경만이라도 반환? 아니면 에러?
        raise e


async def composite_images_async(fg_bytes: bytes, bg_bytes: bytes) -> bytes:
    return await asyncio.to_thread(_composite_sync, fg_bytes, bg_bytes)
