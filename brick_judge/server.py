#!/usr/bin/env python3
"""brick_judge 서버 - LDR 브릭 구조 물리 검증 API

OpenAPI 스펙을 통해 GPT, Gemini, Claude 등 모든 LLM에서 사용 가능.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from pathlib import Path
import time

# OpenAPI 메타데이터
app = FastAPI(
    title="Brick Judge API",
    description="""
## 🧱 LDR 브릭 구조 물리 검증 API

LEGO/브릭 모델의 구조적 안정성을 검증하는 API입니다.

### 주요 기능
- **물리 검증**: 브릭 구조의 안정성 분석
- **이슈 탐지**: floating(공중 부유), isolated(고립), top_only(위에서만 연결) 등
- **점수 산출**: 0-100점 안정성 점수

### LDR 포맷
LDraw 표준 포맷 (.ldr, .dat)을 사용합니다.
```
0 Model Name
1 <color> <x> <y> <z> <rotation matrix 9개> <part>.dat
```

### 사용 예시 (LLM용)
1. `POST /api/judge` 에 LDR 문자열 전송
2. 점수와 이슈 목록 확인
3. 점수 50 미만이면 구조 수정 필요
""",
    version="1.0.0",
    contact={"name": "Brickers Team", "url": "https://github.com/sio60/brickers-ai"},
    license_info={"name": "MIT"},
    openapi_tags=[
        {"name": "judge", "description": "브릭 구조 물리 검증 (LLM용 메인 API)"},
        {"name": "info", "description": "서버 상태 및 정보"},
        {"name": "viewer", "description": "웹 UI"},
    ]
)
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


# ============================================
# Pydantic 모델 (OpenAPI 스키마 자동 생성)
# ============================================

class LdrRequest(BaseModel):
    """LDR 검증 요청"""
    ldr_content: str = Field(
        ...,
        description="LDraw 포맷의 브릭 모델 문자열",
        example="""0 Simple Tower
1 4 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat
1 4 0 -24 0 1 0 0 0 1 0 0 0 1 3001.dat
1 1 0 -48 0 1 0 0 0 1 0 0 0 1 3003.dat"""
    )


class BrickIssue(BaseModel):
    """브릭 구조 이슈"""
    brick_id: Optional[int] = Field(None, description="문제가 있는 브릭 ID (0부터 시작)")
    type: str = Field(..., description="이슈 타입", example="floating")
    severity: str = Field(..., description="심각도: critical, high, medium, low", example="critical")
    message: str = Field(..., description="이슈 설명", example="브릭 #5 바닥과 연결 안됨")
    color: str = Field(..., description="시각화용 색상 (hex)", example="#FF0000")
    data: Optional[Dict] = Field(None, description="추가 데이터 (전복 벡터 등)")


class JudgeResponse(BaseModel):
    """물리 검증 결과"""
    model_name: str = Field(..., description="모델 이름", example="Simple Tower")
    brick_count: int = Field(..., description="총 브릭 개수", example=15)
    score: int = Field(..., description="안정성 점수 (0-100, 50 이상이면 안정)", example=85)
    stable: bool = Field(..., description="안정 여부 (score >= 50 and no critical)", example=True)
    issues: List[BrickIssue] = Field(default=[], description="발견된 이슈 목록")
    brick_colors: Dict[int, str] = Field(default={}, description="이슈 브릭별 색상 맵")
    elapsed_ms: float = Field(..., description="처리 시간 (밀리초)", example=1.23)
    backend: str = Field(..., description="사용된 백엔드 엔진", example="rust")

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "Simple Tower",
                "brick_count": 15,
                "score": 70,
                "stable": True,
                "issues": [
                    {"brick_id": 5, "type": "top_only", "severity": "medium",
                     "message": "브릭 #5 위에서만 연결됨", "color": "#0055FF"}
                ],
                "brick_colors": {5: "#0055FF"},
                "elapsed_ms": 1.23,
                "backend": "rust"
            }
        }


class BackendInfo(BaseModel):
    """백엔드 정보"""
    backend: str = Field(..., description="백엔드 타입", example="rust")
    version: str = Field(..., description="버전", example="0.1.0")
    module: str = Field(..., description="모듈명", example="brick_judge_rs")


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


@app.get("/api/status", tags=["info"], summary="서버 상태 확인")
async def status():
    """서버 상태와 백엔드 정보를 반환합니다."""
    return {"status": "ok", "backend": get_backend_info()}


@app.post("/api/verify", tags=["viewer"], summary="LDR 파일 업로드 검증")
async def verify_ldr(file: UploadFile = File(..., description="LDR 파일 (.ldr, .dat)")):
    """파일 업로드 방식의 LDR 검증 (웹 UI용)"""
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
    score = calc_score_from_issues(issues, len(model.bricks))
    elapsed = (time.perf_counter() - start) * 1000

    brick_colors = {}
    for i in issues:
        if i.brick_id is not None and i.brick_id not in brick_colors:
            brick_colors[i.brick_id] = ISSUE_COLORS.get(i.issue_type.value, "#888888")

    return {
        "model_name": model.model_name,
        "brick_count": len(model.bricks),
        "score": score,
        "stable": not any(i.issue_type.value in ("unstable_base", "floating", "isolated") for i in issues),
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


@app.get("/api/info", tags=["info"], response_model=BackendInfo, summary="백엔드 정보")
async def info():
    """사용 중인 물리 엔진 백엔드 정보를 반환합니다."""
    return get_backend_info()


@app.get("/viewer", response_class=HTMLResponse)
async def viewer():
    """3D 뷰어 페이지"""
    if VIEWER_PATH.exists():
        return VIEWER_PATH.read_text(encoding='utf-8')
    else:
        return "<h1>viewer.html not found</h1><p>demo/brick_judge_viewer.html 파일이 없습니다.</p>"


@app.post(
    "/api/judge",
    tags=["judge"],
    response_model=JudgeResponse,
    summary="🎯 LDR 브릭 구조 물리 검증 (LLM용 메인 API)",
    response_description="검증 결과: 점수, 안정성, 이슈 목록"
)
async def judge_ldr(req: LdrRequest):
    """
    ## LDR 브릭 구조의 물리적 안정성을 검증합니다.

    ### 입력
    - `ldr_content`: LDraw 포맷 문자열

    ### 출력
    - `score`: 0-100점 (50점 이상이면 안정)
    - `stable`: 안정 여부
    - `issues`: 발견된 문제 목록

    ### 이슈 타입
    - `floating`: 공중에 떠있는 브릭 (critical)
    - `isolated`: 다른 브릭과 연결되지 않음 (high)
    - `top_only`: 위에서만 연결됨, 아래 지지 없음 (medium)

    ### 사용 예시
    ```python
    response = requests.post("/api/judge", json={
        "ldr_content": "0 My Model\\n1 4 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat"
    })
    if response.json()["score"] < 50:
        print("구조 수정 필요!")
    ```
    """
    try:
        model = parse_ldr_string(req.ldr_content)
    except Exception as e:
        raise HTTPException(400, f"LDR 파싱 실패: {e}")

    start = time.perf_counter()
    issues = full_judge(model)
    score = calc_score_from_issues(issues, len(model.bricks))
    elapsed = (time.perf_counter() - start) * 1000

    brick_colors = {}
    for i in issues:
        if i.brick_id is not None and i.brick_id not in brick_colors:
            brick_colors[i.brick_id] = ISSUE_COLORS.get(i.issue_type.value, "#888888")

    return JudgeResponse(
        model_name=model.model_name,
        brick_count=len(model.bricks),
        score=score,
        stable=not any(i.issue_type.value in ("unstable_base", "floating", "isolated") for i in issues),
        issues=[
            BrickIssue(
                brick_id=i.brick_id,
                type=i.issue_type.value,
                severity=i.severity.value,
                message=i.message,
                color=ISSUE_COLORS.get(i.issue_type.value, "#888888"),
                data=i.data
            )
            for i in issues
        ],
        brick_colors=brick_colors,
        elapsed_ms=elapsed,
        backend=get_backend_info()["backend"]
    )


@app.post("/api/test/all", tags=["viewer"], include_in_schema=False)
async def test_all(req: LdrRequest):
    """3D 뷰어용 API (레거시, /api/judge 사용 권장)"""
    try:
        model = parse_ldr_string(req.ldr_content)
    except Exception as e:
        raise HTTPException(400, f"LDR 파싱 실패: {e}")

    start = time.perf_counter()
    issues = full_judge(model)
    
    # 디버그: unstable_base 이슈의 data 확인
    for i in issues:
        if i.issue_type.value == "unstable_base":
            print(f"Python Debug: unstable_base issue data = {i.data}")
    
    score = calc_score_from_issues(issues, len(model.bricks))
    elapsed = (time.perf_counter() - start) * 1000

    brick_colors = {}
    for i in issues:
        if i.brick_id is not None and i.brick_id not in brick_colors:
            brick_colors[i.brick_id] = ISSUE_COLORS.get(i.issue_type.value, "#888888")

    return {
        "model_name": model.model_name,
        "brick_count": len(model.bricks),
        "score": score,
        "stable": not any(i.issue_type.value in ("unstable_base", "floating", "isolated") for i in issues),
        "issues": [
            {
                "brick_id": i.brick_id,
                "type": i.issue_type.value,
                "severity": i.severity.value,
                "message": i.message,
                "color": ISSUE_COLORS.get(i.issue_type.value, "#888888"),
                "data": i.data
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
