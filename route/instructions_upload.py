# routes/instructions_upload.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import time
import re
import config

router = APIRouter(prefix="/api/instructions", tags=["instructions"])

SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9._-]")

UPLOAD_DIR = Path("uploads") / "instructions"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED = {"image/png", "image/jpeg", "image/webp"}
MAX_BYTES = 8 * 1024 * 1024  # 8MB (원하면 조절)

@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED:
        raise HTTPException(status_code=400, detail=f"Unsupported content-type: {file.content_type}")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large: {len(raw)} bytes (max={MAX_BYTES})")

    # 파일명 안전화 + 충돌 방지
    orig = file.filename or "image.png"
    orig = SAFE_NAME_RE.sub("_", orig)

    # 확장자 보정(없으면 content-type 기준으로)
    if "." not in orig:
        if file.content_type == "image/png":
            orig += ".png"
        elif file.content_type == "image/jpeg":
            orig += ".jpg"
        elif file.content_type == "image/webp":
            orig += ".webp"

    ts = int(time.time() * 1000)
    save_name = f"{ts}_{orig}"
    save_path = UPLOAD_DIR / save_name

    try:
        save_path.write_bytes(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # ✅ 브라우저/프론트에서 접근할 URL (정적 mount 필요)
    url = f"{config.API_PUBLIC_BASE_URL}/static/instructions/{save_name}"

    return JSONResponse({
        "ok": True,
        "url": url,
        "bytes": len(raw),
        "contentType": file.content_type,
    })
