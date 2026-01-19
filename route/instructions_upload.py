# route/instructions_upload.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import time
import re

router = APIRouter(prefix="/api/instructions", tags=["instructions"])

SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9._-]")

UPLOAD_DIR = Path("uploads") / "instructions"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED = {"image/png", "image/jpeg", "image/webp"}

@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED:
        raise HTTPException(status_code=400, detail=f"Unsupported content-type: {file.content_type}")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    # 파일명 안전화 + 충돌 방지
    orig = file.filename or "image.png"
    orig = SAFE_NAME_RE.sub("_", orig)
    ts = int(time.time() * 1000)
    save_name = f"{ts}_{orig}"

    save_path = UPLOAD_DIR / save_name
    save_path.write_bytes(raw)

    # ✅ 브라우저/프론트에서 접근할 URL
    url = f"http://localhost:8000/static/instructions/{save_name}"

    return JSONResponse({"ok": True, "url": url, "bytes": len(raw), "contentType": file.content_type})
