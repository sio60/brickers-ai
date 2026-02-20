"""
Admin AI Analyst — LLM 유틸리티
OpenAI/Gemini 호환 API를 통한 LLM 호출 및 JSON 파싱.
"""
from __future__ import annotations

import json
import logging
import os
import re

log = logging.getLogger("admin_analyst.llm")

# ─── LLM 클라이언트 (app.py startup에서 주입) ───
_llm_client = None


def get_llm_client():
    return _llm_client


def set_llm_client(client):
    """app.py 서버 시작 시 호출하여 httpx AsyncClient를 주입."""
    global _llm_client
    _llm_client = client
    log.info("[LLM] 클라이언트 주입 완료")


def _extract_json_from_text(text: str):
    """LLM 응답에서 JSON을 안전하게 추출. 여러 전략을 순차 시도."""
    # 전략 1: ```json ... ``` 블록 (정규식으로 안전 추출)
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return json.loads(match.group(1).strip())

    # 전략 2: 임의 ``` ... ``` 블록
    match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        return json.loads(candidate)

    # 전략 3: 텍스트 전체가 JSON인 경우
    return json.loads(text.strip())


async def call_llm_json(prompt: str):
    """
    LLM을 호출하고 JSON 파싱을 시도합니다.
    ```json ... ``` 블록도 자동 추출.
    실패 시 None 반환.
    """
    client = get_llm_client()
    if not client:
        log.warning("[LLM] 클라이언트 미설정 — LLM 호출 불가")
        return None

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }

    try:
        resp = await client.post("chat/completions", json=body)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return _extract_json_from_text(content)
    except json.JSONDecodeError as e:
        log.warning(f"[LLM] JSON 파싱 실패: {e} | 원본(첫 200자): {content[:200] if 'content' in dir() else 'N/A'}")
        return None
    except Exception as e:
        log.error(f"[LLM] 호출 실패: {e}")
        return None


async def call_llm_text(prompt: str) -> str | None:
    """
    LLM을 호출하고 텍스트 응답을 반환합니다. (JSON 파싱 X)
    """
    client = get_llm_client()
    if not client:
        log.warning("[LLM] 클라이언트 미설정 — LLM 호출 불가")
        return None

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7, 
    }

    try:
        resp = await client.post("chat/completions", json=body)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        log.error(f"[LLM] 텍스트 호출 실패: {e}")
        return None
