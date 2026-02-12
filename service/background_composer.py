
import os
import io
import asyncio
import base64
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
            response_modalities=["Image"],
            candidate_count=1
        ),
    )

    if not resp.candidates:
        raise ValueError("No image candidates returned from Gemini")
    
    # Gemini 2.0 Flash Image returns inline data usually
    for part in resp.candidates[0].content.parts:
        if part.inline_data and part.inline_data.data:
            data = part.inline_data.data
            if isinstance(data, str):
                try:
                    return base64.b64decode(data)
                except Exception:
                    # Maybe it's raw bytes in string encoding? or just return as bytes
                    return data.encode('utf-8')
            return data
            
    raise ValueError("No image data found in Gemini response")


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
