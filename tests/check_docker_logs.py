import asyncio
import logging
import sys
from pathlib import Path

# 경로 설정
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from brick_engine.agent.log_analyzer import app, fetch_logs_node

async def test_real_docker_fetch():
    logging.basicConfig(level=logging.INFO)
    print("\n" + "="*60)
    print("Test: Raw Docker Log Archiving & Fetching")
    print("="*60)
    
    container_name = "brickers-ai-container" # 실제 컨테이너명
    test_job_id = "test-verification-id" # 테스트용 가상 ID (또는 실제 로그 보이면 수정)
    
    print(f"1. [Skipped] Archiving logs (persistence module missing on server)")
    # success = await archive_failed_job_logs(test_job_id, container_name)
    # if success:
    #     print("✅ Archiving call finished")
    # else:
    #     print("❌ Archiving failed")

    print("\n2. [Skipped] Checking DB for archived logs")
    # logs = await get_archived_logs(test_job_id)

    # --- initial_state 정의 복구 ---
    initial_state = {
        "container_name": container_name,
        "job_id": test_job_id,
        "messages": [],
        "iteration": 0,
        "logs": ""
    }

    print("\n3. Testing Agent's real fetch node...")
    try:
        result = await fetch_logs_node(initial_state)
        print("✅ Fetch node execution result:")
        print(f"   Collected Logs Length: {len(result.get('logs', ''))}")
        print(f"   Target Job ID: {result.get('job_id')}")
    except Exception as e:
        print(f"❌ Fetch node failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_real_docker_fetch())
