#!/usr/bin/env python3
"""brick_judge 서버 (Rust 전용)"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import time

app = FastAPI(title="brick_judge", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

from .physics import full_judge, calc_score_from_issues, get_backend_info
from .parser import parse_ldr_string

ISSUE_COLORS = {
    "top_only": "#0055FF",
    "floating": "#FF0000",
    "isolated": "#FFCC00",
}


@app.get("/")
async def index():
    return {"status": "ok", "backend": get_backend_info()}


@app.post("/api/verify")
async def verify_ldr(file: UploadFile = File(...)):
    """LDR 파일 물리 검증"""
    try:
        content = await file.read()
        ldr_content = content.decode('utf-8')
    except Exception as e:
        raise HTTPException(400, f"파일 읽기 실패: {e}")

    try:
        model = parse_ldr_string(ldr_content)
    except Exception as e:
        raise HTTPException(400, f"LDR 파싱 실패: {e}")

    start = time.perf_counter()
    issues = full_judge(model)
    score = calc_score_from_issues(issues)
    elapsed = (time.perf_counter() - start) * 1000

    brick_colors = {}
    for i in issues:
        if i.brick_id is not None and i.brick_id not in brick_colors:
            brick_colors[i.brick_id] = ISSUE_COLORS.get(i.issue_type.value, "#888888")

    return {
        "model_name": model.model_name,
        "brick_count": len(model.bricks),
        "score": score,
        "stable": score >= 50 and not any(i.severity.value == "critical" for i in issues),
        "issues": [
            {
                "brick_id": i.brick_id,
                "type": i.issue_type.value,
                "severity": i.severity.value,
                "message": i.message,
                "color": ISSUE_COLORS.get(i.issue_type.value, "#888888")
            }
            for i in issues
        ],
        "brick_colors": brick_colors,
        "elapsed_ms": elapsed,
        "backend": get_backend_info()["backend"]
    }


@app.get("/api/info")
async def info():
    return get_backend_info()


if __name__ == "__main__":
    import uvicorn
    print(f"\n🧱 brick_judge 서버")
    print(f"   백엔드: {get_backend_info()['backend'].upper()}")
    print(f"   URL: http://localhost:8888\n")
    uvicorn.run(app, host="0.0.0.0", port=8888)
