#!/usr/bin/env python3
"""brick_judge 서버 (Rust 전용)"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
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

# 3D 뷰어 HTML 경로
VIEWER_PATH = Path(__file__).parent.parent / "demo" / "brick_judge_viewer.html"


class LdrRequest(BaseModel):
    ldr_content: str


TEST_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>🧱 brick_judge 테스트</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        h1 { color: #333; }
        .upload-box { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; border-radius: 10px; }
        .upload-box:hover { border-color: #007bff; background: #f8f9fa; }
        input[type="file"] { margin: 10px 0; }
        button { background: #007bff; color: white; padding: 10px 30px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        button:hover { background: #0056b3; }
        #result { margin-top: 20px; padding: 20px; background: #f5f5f5; border-radius: 10px; display: none; }
        .score { font-size: 48px; font-weight: bold; }
        .score.good { color: #28a745; }
        .score.warn { color: #ffc107; }
        .score.bad { color: #dc3545; }
        .issue { padding: 10px; margin: 5px 0; border-radius: 5px; }
        .issue.critical { background: #f8d7da; border-left: 4px solid #dc3545; }
        .issue.high { background: #fff3cd; border-left: 4px solid #ffc107; }
        .info { color: #666; margin-top: 10px; }
    </style>
</head>
<body>
    <h1>🧱 brick_judge 물리 검증</h1>
    <p>LDR 파일을 업로드하면 브릭 구조의 안정성을 검증합니다.</p>

    <div class="upload-box">
        <input type="file" id="ldrFile" accept=".ldr,.dat">
        <br><br>
        <button onclick="verify()">검증하기</button>
    </div>

    <div id="result"></div>

    <script>
    async function verify() {
        const file = document.getElementById('ldrFile').files[0];
        if (!file) { alert('파일을 선택하세요'); return; }

        const formData = new FormData();
        formData.append('file', file);

        document.getElementById('result').style.display = 'block';
        document.getElementById('result').innerHTML = '검증 중...';

        try {
            const res = await fetch('/api/verify', { method: 'POST', body: formData });
            const data = await res.json();

            let scoreClass = data.score >= 80 ? 'good' : data.score >= 50 ? 'warn' : 'bad';
            let html = `
                <div class="score ${scoreClass}">${data.score}점</div>
                <p><strong>${data.model_name}</strong> - ${data.brick_count}개 브릭</p>
                <p>안정성: ${data.stable ? '✅ 안정' : '❌ 불안정'}</p>
            `;

            if (data.issues.length > 0) {
                html += '<h3>발견된 문제</h3>';
                data.issues.forEach(i => {
                    html += `<div class="issue ${i.severity}">${i.message}</div>`;
                });
            } else {
                html += '<p style="color: #28a745;">✅ 문제 없음!</p>';
            }

            html += `<p class="info">처리시간: ${data.elapsed_ms.toFixed(2)}ms | 백엔드: ${data.backend}</p>`;
            document.getElementById('result').innerHTML = html;
        } catch (e) {
            document.getElementById('result').innerHTML = '<p style="color:red;">오류: ' + e.message + '</p>';
        }
    }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return TEST_HTML


@app.get("/api/status")
async def status():
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


@app.get("/viewer", response_class=HTMLResponse)
async def viewer():
    """3D 뷰어 페이지"""
    if VIEWER_PATH.exists():
        return VIEWER_PATH.read_text(encoding='utf-8')
    else:
        return "<h1>viewer.html not found</h1><p>demo/brick_judge_viewer.html 파일이 없습니다.</p>"


@app.post("/api/test/all")
async def test_all(req: LdrRequest):
    """3D 뷰어용 API (JSON 입력)"""
    try:
        model = parse_ldr_string(req.ldr_content)
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
        "issue_colors": ISSUE_COLORS,
        "elapsed_ms": elapsed,
        "backend": get_backend_info()["backend"]
    }


if __name__ == "__main__":
    import uvicorn
    print(f"\n🧱 brick_judge 서버")
    print(f"   백엔드: {get_backend_info()['backend'].upper()}")
    print(f"   URL: http://localhost:8888\n")
    uvicorn.run(app, host="0.0.0.0", port=8888)
