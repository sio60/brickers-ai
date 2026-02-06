#!/usr/bin/env python3
"""
LangChain으로 GPT/Gemini가 brick_judge API를 도구로 사용하는 테스트
"""

import os
import requests
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# ============================================
# brick_judge Tool 정의
# ============================================

@tool
def verify_brick_structure(ldr_content: str) -> dict:
    """
    LDR 브릭 구조의 물리적 안정성을 검증합니다.

    Args:
        ldr_content: LDraw 포맷의 브릭 모델 문자열

    Returns:
        검증 결과 (score, stable, issues 등)
        - score: 0-100점 (50점 이상이면 안정적인 구조)
        - stable: 안정 여부
        - issues: 발견된 문제 목록 (floating, isolated, top_only 등)

    Example:
        ldr = '''0 Tower
        1 4 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat
        1 4 0 -24 0 1 0 0 0 1 0 0 0 1 3001.dat'''
        result = verify_brick_structure(ldr)
        if result['score'] < 50:
            print("구조 수정 필요!")
    """
    try:
        response = requests.post(
            "http://localhost:8888/api/judge",
            json={"ldr_content": ldr_content},
            timeout=10
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


# 테스트용 LDR 샘플
SAMPLE_LDR_STABLE = """0 Stable Tower
1 4 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat
1 4 0 -24 0 1 0 0 0 1 0 0 0 1 3001.dat
1 1 0 -48 0 1 0 0 0 1 0 0 0 1 3003.dat"""

SAMPLE_LDR_UNSTABLE = """0 Floating Brick Test
1 4 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat
1 4 0 -200 0 1 0 0 0 1 0 0 0 1 3001.dat"""


def test_with_gpt():
    """GPT-4로 테스트"""
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY 환경변수 필요")
        return

    print("\n" + "="*50)
    print("🤖 GPT-4 테스트")
    print("="*50)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools([verify_brick_structure])

    # LLM에게 브릭 구조 검증 요청
    messages = [
        HumanMessage(content=f"""
다음 LDR 브릭 구조를 검증해주세요. 안정적인지 확인하고 문제가 있으면 알려주세요.

```ldr
{SAMPLE_LDR_UNSTABLE}
```
""")
    ]

    response = llm_with_tools.invoke(messages)
    print(f"\n📤 GPT 응답:")
    print(f"   Content: {response.content}")

    if response.tool_calls:
        print(f"\n🔧 Tool Calls:")
        for tc in response.tool_calls:
            print(f"   - {tc['name']}: {tc['args'][:100]}...")

            # 실제 도구 실행
            result = verify_brick_structure.invoke(tc['args'])
            print(f"\n📊 검증 결과:")
            print(f"   Score: {result.get('score', 'N/A')}")
            print(f"   Stable: {result.get('stable', 'N/A')}")
            print(f"   Issues: {len(result.get('issues', []))}개")
            for issue in result.get('issues', []):
                print(f"     - [{issue['severity']}] {issue['message']}")


def test_with_gemini():
    """Gemini로 테스트"""
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY 환경변수 필요")
        return

    print("\n" + "="*50)
    print("💎 Gemini 테스트")
    print("="*50)

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    llm_with_tools = llm.bind_tools([verify_brick_structure])

    messages = [
        HumanMessage(content=f"""
다음 LDR 브릭 구조를 검증해주세요. 안정적인지 확인하고 문제가 있으면 알려주세요.

```ldr
{SAMPLE_LDR_UNSTABLE}
```
""")
    ]

    response = llm_with_tools.invoke(messages)
    print(f"\n📤 Gemini 응답:")
    print(f"   Content: {response.content}")

    if response.tool_calls:
        print(f"\n🔧 Tool Calls:")
        for tc in response.tool_calls:
            print(f"   - {tc['name']}")

            result = verify_brick_structure.invoke(tc['args'])
            print(f"\n📊 검증 결과:")
            print(f"   Score: {result.get('score', 'N/A')}")
            print(f"   Stable: {result.get('stable', 'N/A')}")
            print(f"   Issues: {len(result.get('issues', []))}개")
            for issue in result.get('issues', []):
                print(f"     - [{issue['severity']}] {issue['message']}")


def test_tool_directly():
    """도구 직접 테스트 (API 연결 확인)"""
    print("\n" + "="*50)
    print("🔧 도구 직접 테스트")
    print("="*50)

    result = verify_brick_structure.invoke({"ldr_content": SAMPLE_LDR_STABLE})
    print(f"\n✅ 안정적인 구조:")
    print(f"   Score: {result.get('score')}, Stable: {result.get('stable')}")

    result = verify_brick_structure.invoke({"ldr_content": SAMPLE_LDR_UNSTABLE})
    print(f"\n❌ 불안정한 구조:")
    print(f"   Score: {result.get('score')}, Stable: {result.get('stable')}")
    for issue in result.get('issues', []):
        print(f"   - {issue['message']}")


if __name__ == "__main__":
    print("🧱 brick_judge LLM Tool 테스트")
    print("="*50)

    # 1. 도구 직접 테스트
    test_tool_directly()

    # 2. GPT 테스트
    test_with_gpt()

    # 3. Gemini 테스트
    test_with_gemini()
