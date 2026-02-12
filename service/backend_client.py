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


async def check_job_canceled(job_id: str) -> bool:
    """Backend에서 Job 상태 확인 (취소 여부 체크)"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{BACKEND_URL}/api/kids/jobs/{job_id}",
                headers={"X-Internal-Token": os.environ.get("INTERNAL_API_TOKEN", "")},
            )
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status", "")
                if status == "CANCELED":
                    print(f"   🚫 [Job Status] Job is CANCELED | jobId={job_id}")
                    return True
            return False
    except Exception as e:
        print(f"   ⚠️ [Job Status] 상태 확인 실패 (진행) | jobId={job_id} | error={str(e)}")
        return False  # 확인 실패 시 계속 진행


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


async def get_analytics_summary(days: int = 7) -> dict | None:
    """백엔드로부터 GA4 요약 데이터(Users, Views, Sessions)를 가져옵니다."""
    token = os.environ.get("INTERNAL_API_TOKEN", "")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{BACKEND_URL}/api/admin/analytics/summary",
                params={"days": days},
                headers={"X-Internal-Token": token},
            )
            if resp.status_code == 200:
                return resp.json()
            print(f"  ⚠️ [BackendClient] Analytics Summary Error: {resp.status_code}")
    except Exception as e:
        print(f"  ⚠️ [BackendClient] Analytics Summary Fail: {e}")
    return None


async def get_daily_users(days: int = 30) -> list | None:
    """백엔드로부터 일별 활성 사용자(DAU) 트렌드를 가져옵니다."""
    token = os.environ.get("INTERNAL_API_TOKEN", "")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{BACKEND_URL}/api/admin/analytics/daily-users",
                params={"days": days},
                headers={"X-Internal-Token": token},
            )
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        print(f"  ⚠️ [BackendClient] Daily Users Fail: {e}")
    return None


async def get_event_stats(event_name: str, days: int = 7) -> list | None:
    """특정 이벤트의 발생 통계를 가져옵니다."""
    token = os.environ.get("INTERNAL_API_TOKEN", "")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{BACKEND_URL}/api/admin/analytics/event-stats",
                params={"event": event_name, "days": days},
                headers={"X-Internal-Token": token},
            )
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        print(f"  ⚠️ [BackendClient] Event Stats Fail: {e}")
    return None


async def get_user_activity(user_id: str, days: int = 30) -> list | None:
    """특정 유저의 상호작용(이벤트 발생 내역)을 가져옵니다."""
    token = os.environ.get("INTERNAL_API_TOKEN", "")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{BACKEND_URL}/api/admin/analytics/user-activity",
                params={"userId": user_id, "days": days},
                headers={"X-Internal-Token": token},
            )
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        print(f"  ⚠️ [BackendClient] User Activity Fail: {e}")
    return None


async def get_top_tags(days: int = 30, limit: int = 10) -> list | None:
    """백엔드로부터 실시간 인기 태그 순위를 가져옵니다."""
    token = os.environ.get("INTERNAL_API_TOKEN", "")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{BACKEND_URL}/api/admin/analytics/top-tags",
                params={"days": days, "limit": limit},
                headers={"X-Internal-Token": token},
            )
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        print(f"  ⚠️ [BackendClient] Top Tags Fail: {e}")
    return None


async def get_heavy_users(days: int = 30, limit: int = 10) -> list | None:
    """백엔드로부터 활동량이 많은 상위 유저 리스트를 가져옵니다."""
    token = os.environ.get("INTERNAL_API_TOKEN", "")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{BACKEND_URL}/api/admin/analytics/heavy-users",
                params={"days": days, "limit": limit},
                headers={"X-Internal-Token": token},
            )
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        print(f"  ⚠️ [BackendClient] Heavy Users Fail: {e}")
    return None
