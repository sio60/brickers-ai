# service/backend_client.py
"""Backend 연동 (Stage 업데이트, Tags 저장, Agent 로그 전송)"""
from __future__ import annotations

import os

import httpx

BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8080").rstrip("/")


async def update_job_stage(job_id: str, stage: str) -> None:
    """Backend에 Job stage 업데이트 (실패해도 무시)"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.patch(
                f"{BACKEND_URL}/api/kids/jobs/{job_id}/stage",
                json={"stage": stage},
                headers={"X-Internal-Token": os.environ.get("INTERNAL_API_TOKEN", "")},
            )
        print(f"   \u2705 [Stage Update] {stage}")
    except Exception as e:
        print(f"   \u26a0\ufe0f [Stage Update] \uc2e4\ud328 (\ubb34\uc2dc) | stage={stage} | error={str(e)}")


async def update_job_suggested_tags(job_id: str, tags: list[str]) -> None:
    """Backend에 Gemini가 추출한 suggested_tags 저장 (실패해도 무시)"""
    if not tags:
        return
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.patch(
                f"{BACKEND_URL}/api/kids/jobs/{job_id}/suggested-tags",
                json={"suggestedTags": tags},
                headers={"X-Internal-Token": os.environ.get("INTERNAL_API_TOKEN", "")},
            )
            if resp.status_code >= 400:
                print(f"   \u26a0\ufe0f [Suggested Tags] Backend 응답 에러: Status={resp.status_code} | Body={resp.text}")
            else:
                print(f"   \u2705 [Suggested Tags] 저장 성공: Status={resp.status_code} | Tags={tags}")
    except Exception as e:
        print(f"   \u26a0\ufe0f [Suggested Tags] 저장 실패 (통신 오류) | tags={tags} | error={str(e)}")


def make_agent_log_sender(job_id: str):
    """CoScientist 에이전트 로그 전송 콜백 (sync context용)"""
    import requests

    def send_log(step: str, message: str):
        url = f"{BACKEND_URL}/api/kids/jobs/{job_id}/logs"
        token = os.environ.get("INTERNAL_API_TOKEN", "")
        try:
            resp = requests.post(
                url,
                json={"step": step, "message": message[:2000]},
                headers={"X-Internal-Token": token},
                timeout=5.0,
            )
            if resp.status_code == 200:
                print(f"  [AgentLog] \u2705 sent: [{step}] {message[:50]}...")
            else:
                print(f"  [AgentLog] \u26a0\ufe0f HTTP {resp.status_code} | url={url} | body={resp.text[:100]}")
        except Exception as e:
            print(f"  [AgentLog] \u274c failed: {e} | url={url}")

    return send_log


async def send_agent_log(job_id: str, step: str, message: str) -> None:
    """CoScientist 에이전트 로그 전송 (async context용 - kids_render.py 파이프라인)"""
    url = f"{BACKEND_URL}/api/kids/jobs/{job_id}/logs"
    token = os.environ.get("INTERNAL_API_TOKEN", "")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                url,
                json={"step": step, "message": message[:2000]},
                headers={"X-Internal-Token": token},
            )
            if resp.status_code == 200:
                print(f"  [AgentLog] \u2705 sent: [{step}] {message[:50]}...")
            else:
                print(f"  [AgentLog] \u26a0\ufe0f HTTP {resp.status_code} | url={url} | body={resp.text[:100]}")
    except Exception as e:
        print(f"  [AgentLog] \u274c failed: {e} | url={url}")
