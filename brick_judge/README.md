# 🧱 Brick Judge API

**LDR 브릭 구조 물리 검증 API** - LEGO/브릭 모델의 구조적 안정성을 검증합니다.

GPT, Gemini, Claude 등 모든 LLM에서 도구(Tool)로 사용 가능합니다.

---

## 📋 목차

- [빠른 시작](#-빠른-시작)
- [API 엔드포인트](#-api-엔드포인트)
- [LLM 연동 가이드](#-llm-연동-가이드)
- [LDR 포맷 설명](#-ldr-포맷-설명)
- [이슈 타입](#-이슈-타입)
- [예제 코드](#-예제-코드)

---

## 🚀 빠른 시작

### 서버 실행

```bash
cd brickers-ai
python -m brick_judge.server
```

서버가 시작되면:
- **API**: http://localhost:8888
- **Swagger UI**: http://localhost:8888/docs
- **3D 뷰어**: http://localhost:8888/viewer

### 간단한 테스트

```bash
curl -X POST http://localhost:8888/api/judge \
  -H "Content-Type: application/json" \
  -d '{"ldr_content": "0 Test\n1 4 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat"}'
```

---

## 📡 API 엔드포인트

### `POST /api/judge` (메인 API)

LDR 브릭 구조의 물리적 안정성을 검증합니다.

#### 요청

```json
{
  "ldr_content": "0 My Model\n1 4 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat\n1 4 0 -24 0 1 0 0 0 1 0 0 0 1 3001.dat"
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `ldr_content` | string | LDraw 포맷의 브릭 모델 |

> ⚠️ **중요: 여러 줄 입력 가능!**
> JSON에서는 줄바꿈을 `\n`으로 표현합니다.
> Python의 `requests.post(json={...})`는 자동으로 변환해줍니다.
>
> ```python
> # Python에서는 그냥 여러 줄 문자열 사용
> ldr = """0 My Model
> 1 4 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat
> 1 4 0 -24 0 1 0 0 0 1 0 0 0 1 3001.dat"""
>
> requests.post(url, json={"ldr_content": ldr})  # 알아서 \n 처리됨
> ```

#### 응답

```json
{
  "model_name": "My Model",
  "brick_count": 2,
  "score": 100,
  "stable": true,
  "issues": [],
  "brick_colors": {},
  "elapsed_ms": 0.05,
  "backend": "rust"
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `model_name` | string | 모델 이름 |
| `brick_count` | int | 총 브릭 개수 |
| `score` | int | 안정성 점수 (0-100) |
| `stable` | bool | 안정 여부 (`score >= 50` and no critical) |
| `issues` | array | 발견된 문제 목록 |
| `brick_colors` | object | 이슈 브릭별 색상 (시각화용) |
| `elapsed_ms` | float | 처리 시간 (밀리초) |
| `backend` | string | 사용된 백엔드 (rust/python) |

#### 점수 해석

| 점수 | 상태 | 설명 |
|------|------|------|
| 80-100 | ✅ 안정 | 문제 없음 |
| 50-79 | ⚠️ 주의 | 경미한 문제 있음 |
| 0-49 | ❌ 불안정 | 구조 수정 필요 |

---

### 기타 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/api/status` | GET | 서버 상태 확인 |
| `/api/info` | GET | 백엔드 정보 |
| `/api/verify` | POST | 파일 업로드 검증 (웹 UI용) |
| `/viewer` | GET | 3D 시각화 페이지 |
| `/docs` | GET | Swagger UI |
| `/openapi.json` | GET | OpenAPI 스펙 |

---

## 🤖 LLM 연동 가이드

### 방법 1: LangChain Tool (추천)

GPT, Gemini, Claude 모두 동일한 방식으로 사용 가능합니다.

```python
import requests
from langchain_core.tools import tool

@tool
def verify_brick_structure(ldr_content: str) -> dict:
    """
    LDR 브릭 구조의 물리적 안정성을 검증합니다.

    Args:
        ldr_content: LDraw 포맷의 브릭 모델 문자열

    Returns:
        score: 0-100점 (50점 이상이면 안정)
        stable: 안정 여부
        issues: 발견된 문제 목록
    """
    response = requests.post(
        "http://localhost:8888/api/judge",
        json={"ldr_content": ldr_content},
        timeout=10
    )
    return response.json()
```

### 방법 2: GPT-4 Function Calling

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools([verify_brick_structure])

response = llm_with_tools.invoke("이 LDR 구조를 검증해줘: ...")
```

### 방법 3: Gemini Function Calling

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
llm_with_tools = llm.bind_tools([verify_brick_structure])

response = llm_with_tools.invoke("이 LDR 구조를 검증해줘: ...")
```

### 방법 4: OpenAPI 스펙 직접 사용

GPT Actions, Gemini Extensions 등에서 OpenAPI 스펙을 직접 가져와 사용:

```
http://localhost:8888/openapi.json
```

---

## 📐 LDR 포맷 설명

LDraw는 LEGO 모델을 표현하는 텍스트 포맷입니다.

### 기본 구조

```
0 모델 이름
0 Author: 작성자
1 <색상> <X> <Y> <Z> <회전행렬 9개> <파트>.dat
0 STEP
1 <색상> <X> <Y> <Z> <회전행렬 9개> <파트>.dat
```

### 예시

```ldr
0 Simple Tower
0 Author: Claude
1 4 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat
1 4 0 -24 0 1 0 0 0 1 0 0 0 1 3001.dat
1 1 0 -48 0 1 0 0 0 1 0 0 0 1 3003.dat
0 STEP
1 14 0 -72 0 1 0 0 0 1 0 0 0 1 3005.dat
```

### Line Type 1 (브릭 배치)

```
1 <color> <x> <y> <z> <a> <b> <c> <d> <e> <f> <g> <h> <i> <part>
```

| 필드 | 설명 |
|------|------|
| color | LDraw 색상 코드 (4=빨강, 1=파랑, 14=노랑 등) |
| x, y, z | 위치 (LDU 단위, Y축이 위/아래) |
| a~i | 3x3 회전 행렬 |
| part | 파트 파일명 (예: 3001.dat = 2x4 브릭) |

### 자주 쓰는 파트

| 파트 | 이름 | 크기 |
|------|------|------|
| 3001.dat | Brick 2x4 | 80x24x40 LDU |
| 3003.dat | Brick 2x2 | 40x24x40 LDU |
| 3004.dat | Brick 1x2 | 40x24x20 LDU |
| 3005.dat | Brick 1x1 | 20x24x20 LDU |
| 3010.dat | Brick 1x4 | 80x24x20 LDU |

---

## ⚠️ 이슈 타입

검증 시 발견되는 문제 유형입니다.

| 타입 | 심각도 | 설명 | 색상 |
|------|--------|------|------|
| `floating` | critical | 공중에 떠있는 브릭 (바닥과 연결 없음) | 🔴 빨강 |
| `isolated` | high | 다른 브릭과 전혀 연결되지 않음 | 🟡 노랑 |
| `top_only` | medium | 위에서만 연결됨 (아래 지지 없음) | 🔵 파랑 |

### 심각도별 점수 감점

| 심각도 | 감점 |
|--------|------|
| critical | -30점 |
| high | -15점 |
| medium | -5점 |
| low | -2점 |

---

## 💻 예제 코드

### Python - 단순 API 호출

```python
import requests

ldr = """0 My Tower
1 4 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat
1 4 0 -24 0 1 0 0 0 1 0 0 0 1 3001.dat
1 1 0 -48 0 1 0 0 0 1 0 0 0 1 3003.dat"""

response = requests.post(
    "http://localhost:8888/api/judge",
    json={"ldr_content": ldr}
)

result = response.json()
print(f"점수: {result['score']}")
print(f"안정: {result['stable']}")

for issue in result['issues']:
    print(f"  - [{issue['severity']}] {issue['message']}")
```

### Python - LLM 에이전트

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests

@tool
def verify_brick_structure(ldr_content: str) -> dict:
    """LDR 브릭 구조의 물리적 안정성을 검증합니다."""
    response = requests.post(
        "http://localhost:8888/api/judge",
        json={"ldr_content": ldr_content}
    )
    return response.json()

# LLM에 도구 연결
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools([verify_brick_structure])

# LLM에게 검증 요청
response = llm_with_tools.invoke([
    HumanMessage(content="이 브릭 구조가 안정적인지 확인해줘: ...")
])

# 도구 호출 결과 확인
if response.tool_calls:
    for call in response.tool_calls:
        result = verify_brick_structure.invoke(call['args'])
        print(f"검증 결과: {result}")
```

### JavaScript/Node.js

```javascript
const response = await fetch('http://localhost:8888/api/judge', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    ldr_content: `0 Test
1 4 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat`
  })
});

const result = await response.json();
console.log(`Score: ${result.score}, Stable: ${result.stable}`);
```

---

## 🔧 개발자 정보

- **백엔드**: Rust (brick_judge_rs) - 고속 물리 검증
- **API**: FastAPI + OpenAPI 3.0
- **3D 뷰어**: Three.js + LDrawLoader

### 파일 구조

```
brick_judge/
├── __init__.py
├── server.py          # FastAPI 서버 + OpenAPI
├── physics.py         # 물리 검증 로직 (Rust 바인딩)
├── parser.py          # LDR 파싱
├── test_llm_tool.py   # LLM 도구 테스트
├── openapi.json       # OpenAPI 스펙 (자동 생성)
└── README.md          # 이 문서
```

