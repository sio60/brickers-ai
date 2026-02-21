# ============================================================================
# 에이전트 트레이싱 및 상태 직렬화 유틸리티
# graph.py의 순수성을 위해 분리된 헬퍼 모듈
# ============================================================================

import time
import asyncio
import anyio


def serialize_state(s) -> dict:
    """
    에이전트 상태를 JSON 직렬화 가능한 형태로 변환합니다.
    Admin UI에서 읽기 쉬운 포맷으로 가공합니다.
    """
    if not isinstance(s, dict):
        return str(s)

    clean_state = {}
    for k, v in s.items():
        if k == 'messages':
            # 메시지 내역을 읽기 쉬운 포맷으로 변환 (최근 5개만)
            clean_state[k] = [
                {
                    "role": "assistant" if "AI" in str(type(m)) else ("user" if "Human" in str(type(m)) else "system"),
                    "content": m.content if hasattr(m, 'content') else str(m)
                } for m in v[-5:]
            ]
        elif k in ['hypothesis_maker', 'verifier']:
            # 직렬화 불가능한 객체 제외
            continue
        elif k in ['verification_raw_result']:
            # 수백 개의 브릭 데이터는 요약정보만 표시 (가독성 보호)
            count = len(v.get('issues', [])) if isinstance(v, dict) else 0
            clean_state[k] = f"[Filtered: {count} items for readability]"
        elif isinstance(v, list) and len(v) > 20:
            # 너무 긴 리스트는 잘라냄
            clean_state[k] = [serialize_state(x) for x in v[:20]] + [f"... and {len(v)-20} more"]
        elif isinstance(v, (str, int, float, bool, list, dict)) or v is None:
            clean_state[k] = v
        else:
            clean_state[k] = str(v)
    return clean_state


async def trace_node(graph_instance, node_name: str, func, state):
    """
    노드 실행 트레이싱 래퍼.
    노드 함수를 실행하면서 입력/출력 스냅샷과 소요 시간을 기록하여
    Admin UI로 트레이스를 전송합니다.
    
    Args:
        graph_instance: RegenerationGraph 인스턴스 (self)
        node_name: 노드 이름 (예: "node_model")
        func: 실행할 노드 함수
        state: 현재 에이전트 상태
    """
    from service.backend_client import send_agent_trace

    start_ts = time.time()
    input_snap = serialize_state(state)

    status = "SUCCESS"
    output_snap = {}

    try:
        # Sync vs Async 자동 판별
        if asyncio.iscoroutinefunction(func):
            result = await func(graph_instance, state)
        else:
            # 동기 노드는 스레드에서 실행하여 이벤트 루프 블로킹 방지
            result = await anyio.to_thread.run_sync(func, graph_instance, state)

        output_snap = serialize_state(result) if isinstance(result, dict) else {"result": str(result)}
        return result

    except Exception as e:
        status = "FAILURE"
        output_snap = {"error": str(e)}
        raise e
    finally:
        duration = int((time.time() - start_ts) * 1000)
        if graph_instance.job_id != "offline":
            # Fire and forget 트레이스 전송
            asyncio.create_task(
                send_agent_trace(
                    graph_instance.job_id,
                    step="TRACE",
                    node_name=node_name,
                    status=status,
                    input_data=input_snap,
                    output_data=output_snap,
                    duration_ms=duration
                )
            )
