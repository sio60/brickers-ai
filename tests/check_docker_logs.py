import asyncio
import logging
import sys
from pathlib import Path

# 경로 설정
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from brick_engine.agent.log_analyzer.persistence import archive_failed_job_logs, get_archived_logs
from brick_engine.agent.log_analyzer.agent import app

async def test_real_docker_fetch():
    logging.basicConfig(level=logging.INFO)
    print("\n" + "="*60)
    print("Test: Raw Docker Log Archiving & Fetching")
    print("="*60)
    
    container_name = "brickers-ai-container" # 실제 컨테이너명
    test_job_id = "test-verification-id" # 테스트용 가상 ID (또는 실제 로그 보이면 수정)
    
    print(f"1. Attempting to archive logs for Job [{test_job_id}] from container [{container_name}]...")
    # 실제 로그에 test_job_id가 수동으로는 없을테니, 
    # 실제 로그에 찍힌 jobId 하나를 찾아보거나 전체 로그를 그냥 가져오는 식으로 테스트 유도
    
    success = await archive_failed_job_logs(test_job_id, container_name)
    if success:
        print("✅ Archiving call finished (Check logs to see if ID was found)")
    else:
        print("❌ Archiving failed (Container might be down or ID not found)")

    print("\n2. Checking DB for archived logs...")
    logs = await get_archived_logs(test_job_id)
    if logs:
        print(f"✅ Found archived logs in DB ({len(logs)} bytes)")
    else:
        print("ℹ️ No logs found in DB for this Job ID.")

    print("\n3. Testing Agent's real fetch node...")
    # fetch_logs_node를 직접 호출해보기 위해 initial state 구성
    initial_state = {
        "container_name": container_name,
        "job_id": test_job_id,
        "messages": [],
        "iteration": 0,
        "logs": ""
    }
    
    # 여기서 ainvoke를 통해 첫번째 노드만 실행하거나 전체 실행
    # 간단하게 ainvoke로 결과만 확인
    try:
        # 실제 AI 호출까지 가면 비용/시간 걸리므로 fetch_logs 노드 결과만 확인하는 용도
        from brick_engine.agent.log_analyzer.agent import fetch_logs_node
        result = await fetch_logs_node(initial_state)
        print("✅ Fetch node execution result:")
        print(f"   Collected Logs Length: {len(result.get('logs', ''))}")
        print(f"   Target Job ID: {result.get('job_id')}")
    except Exception as e:
        print(f"❌ Fetch node failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_real_docker_fetch())
