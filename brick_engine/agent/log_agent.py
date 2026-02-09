from typing import TypedDict, Annotated, List, Optional, Dict, Any, Union
import os
import docker
import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END

# Logger configuration
logger = logging.getLogger("agent.log_agent")

# LLM Client Import
try:
    from .llm_clients import GeminiClient
    from .agent_tools import ReadFileSnippet, CheckDBStatus, CheckSystemHealth, CheckSQSStatus, execute_read_file, execute_check_db, execute_check_system, execute_check_sqs
except ImportError:
    # Standalone execution support
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent))
    from llm_clients import GeminiClient
    from agent_tools import ReadFileSnippet, CheckDBStatus, CheckSystemHealth, CheckSQSStatus, execute_read_file, execute_check_db, execute_check_system, execute_check_sqs

# --- State Definition ---
class LogAnalysisState(TypedDict):
    container_name: str
    logs: str  # Raw logs
    messages: List[Annotated[str, "History"]] 
    analysis_result: Optional[str] 
    error_count: int
    iteration: int # Loop counter
    job_id: Optional[str] # 특정 Job 추적용 추가

# --- Tool Execution Logic ---
# --- Tool Execution Logic (Moved to agent_tools.py or imported) ---
# execute_read_file is now imported from agent_tools.py
# ensuring we use the centralized definition.

# --- Nodes ---

# --- Nodes (노드 정의) ---



# --- Nodes (노드 정의) ---

async def fetch_logs_node(state: LogAnalysisState):
    """
    [노드 1: 로그 수집]
    Docker SDK 또는 상태값을 활용하여 특정 Job의 로그만 추출하거나 최근 실패한 Job을 찾습니다.
    """
    container_name = state.get("container_name", "brickers-ai-container")
    target_job_id = state.get("job_id") # 특정 Job ID가 지정되었는지 확인
    
    logger.info(f"--- [로그 에이전트] 1단계: 로그 수집 및 Job 분석 시작 ({container_name}) ---")
    
    try:
        # 1. 독커에서 넉넉하게 로그 가져오기 (Job 전체 맥락을 파악하기 위해)
        client = docker.from_env()
        container = client.containers.get(container_name)
        raw_logs = container.logs(tail=2000).decode("utf-8", errors="replace")
        logger.info(f"✅ 원본 로그 수집 완료 ({len(raw_logs)} 바이트).")
    except Exception as e:
        existing_logs = state.get("logs", "")
        if existing_logs:
             logger.warning("⚠️ 독커 연결 실패. 입력된 텍스트 로그를 사용합니다.")
             raw_logs = existing_logs 
        else:
             logger.error(f"❌ 독커 로그 수집 에러: {str(e)}")
             return {"analysis_result": json.dumps({"action": "finish", "analysis": {"error_found": True, "summary": f"로그 수집 불가: {str(e)}"}})}

    # 2. Job ID 기반 로그 추출 로직
    # 실패한 Job ID 찾기 (패턴: ❌ [AI-SERVER] 요청 실패! | jobId=...)
    import re
    
    if not target_job_id:
        # 가장 최근에 '실패'한 Job ID를 찾음
        failure_matches = re.findall(r"요청 실패! \| jobId=([a-f0-9-]+)", raw_logs)
        if failure_matches:
            target_job_id = failure_matches[-1]
            logger.info(f"🕵️ 최근 실패한 Job 발견: {target_job_id}")
        else:
            # 실패건이 없으면 가장 최근 '시작'된 Job ID 추출
            start_matches = re.findall(r"요청 시작 \| jobId=([a-f0-9-]+)", raw_logs)
            if start_matches:
                target_job_id = start_matches[-1]
                logger.info(f"ℹ️ 실패 건은 없으나 최근 Job 분석 진행: {target_job_id}")

    # 3. 해당 Job ID와 관련된 로그만 모으기
    if target_job_id:
        job_logs_list = []
        for line in raw_logs.splitlines():
            if target_job_id in line:
                job_logs_list.append(line)
        
        filtered_logs = "\n".join(job_logs_list)
        logger.info(f"📂 Job [{target_job_id}] 관련 로그 {len(job_logs_list)}줄 필터링 완료.")
    else:
        filtered_logs = raw_logs[-4000:] # Job ID 못 찾으면 기존처럼 마지막 부분 사용
        logger.warning("⚠️ Job ID를 식별하지 못했습니다. 마지막 4000자만 사용합니다.")

    user_prompt = f"""
    [지정된 Job ID: {target_job_id or "알 수 없음"}]
    [해당 Job 관련 로그]
    {filtered_logs} 
    
    이 Job의 로그만 집중적으로 분석하여 오류(Traceback, Exception, Timeout)의 근본 원인을 식별하십시오.
    
    사용 가능한 도구:
    1. `read_file`: 코드 레벨의 오류 확인 시 사용.
    2. `check_db`: DB 연결/상태 점검 시 사용.
    3. `check_sqs`: 메시지 큐 지연/에러 시 사용.
    4. `check_system`: 리소스 부족 의심 시 사용.
    """
    
    return {
        "logs": filtered_logs, 
        "messages": [HumanMessage(content=user_prompt)], 
        "iteration": 0,
        "job_id": target_job_id
    }

async def diagnose_node(state: LogAnalysisState):
    """
    [노드 2: 에러 진단 및 의사결정]
    """
    messages = state.get("messages", [])
    iteration = state.get("iteration", 0)
    logger.info(f"--- [로그 에이전트] 2단계: 에러 진단 중 (반복: {iteration}) ---")
    
    system_prompt = """
    당신은 전문 디버깅 에이전트 및 시스템 아키텍트입니다.
    목표: 특정 Job ID와 관련된 로그를 분석하여 근본 원인을 찾으십시오.
    
    의사결정 프로세스:
    1. **로그 분석**: 코드 에러(`read_file`), DB(`check_db`), SQS(`check_sqs`), 시스템(`check_system`) 중 의심 지점 확인.
    2. **아키텍처 제안**: 단순 수치 조정을 넘어, 특정 로직이 누락되었거나 새로운 함수(도구)가 필요하다고 판단되면 이를 해결책(suggestion)에 구체적인 코드 예시와 함께 포함하십시오.
    
    출력 형식 (JSON):
    - 도구 사용: `{"action": "도구이름", "args": {...}, "reasoning": "이유"}`
    - 종료: `{"action": "finish", "analysis": {"error_found": true, "summary": "요약", "root_cause": "원인", "suggestion": "해결책 (필요시 새로운 로직/함수 설계 포함)"}}`
    
    모든 분석 보고서는 한국어로 작성하십시오.
    """
    
    try:
        llm = GeminiClient()
        # 비동기 호출을 위해 to_thread 사용 (llm_clients가 동기인 경우)
        import asyncio
        response = await asyncio.to_thread(llm.generate_json, messages[-1].content, system_prompt)
        logger.info(f"🤖 AI 결정: {json.dumps(response, ensure_ascii=False)}")
        
        return {"analysis_result": json.dumps(response, ensure_ascii=False), "iteration": iteration + 1}
        
    except Exception as e:
         logger.error(f"❌ AI 진단 실패: {str(e)}")
         return {"analysis_result": json.dumps({"action": "finish", "analysis": {"error_found": True, "summary": f"진단 도중 에러 발생: {str(e)}"}})}

async def tool_execution_node(state: LogAnalysisState):
    """
    [노드 3: 도구 실행]
    """
    raw_result = state.get("analysis_result")
    try:
        decision = json.loads(raw_result)
    except:
        logger.error("❌ 도구 실행 노드에서 JSON 파싱 에러 발생.")
        return {"messages": [HumanMessage(content="JSON 의사결정 파싱 에러.")]}

    action = decision.get("action")
    args = decision.get("args", {})
    logger.info(f"🛠️ [로그 에이전트] 3단계: 도구 '{action}' 실행 중...")
    
    # 도구별 실행 (현재 도구들은 동기 방식이므로 to_thread 권장)
    import asyncio
    tool_output = ""

    if action == "read_file":
        tool_output = await asyncio.to_thread(execute_read_file, args)
    elif action == "check_db":
        tool_output = await asyncio.to_thread(execute_check_db, args)
    elif action == "check_system":
        tool_output = await asyncio.to_thread(execute_check_system, args)
    elif action == "check_sqs":
        tool_output = await asyncio.to_thread(execute_check_sqs, args)
    
    if tool_output:
        logger.info(f"📥 도구 결과 수신 ({len(tool_output)} 바이트).")
        # LLM에게 도구 결과 전달
        feedback_msg = f"""
        [{action} 도구 실행 결과]
        {tool_output}
        
        이 결과를 바탕으로 근본 원인을 파악했습니까?
        파악했다면 finish를, 더 정보가 필요하면 다른 도구를 요청하십시오.
        """
        messages = state.get("messages", [])
        messages.append(AIMessage(content=f"Executed {action}"))
        messages.append(HumanMessage(content=feedback_msg))
        
        return {"messages": messages}
    
    logger.warning("⚠️ 도구가 아무런 결과를 반환하지 않았습니다.")
    return {}

# --- Conditional Edge (분기 조건) ---
def should_continue(state: LogAnalysisState):
    iteration = state.get("iteration", 0)
    raw_result = state.get("analysis_result")
    
    try:
        decision = json.loads(raw_result)
        action = decision.get("action")
        
        # 도구 사용 요청이고, 반복 횟수가 5회 미만이면 계속 진행
        if action in ["read_file", "check_db", "check_system", "check_sqs"] and iteration < 5: 
            logger.info(f"🔄 Routing to 'inspect_code' (Current Iteration: {iteration})")
            return "inspect_code"
        else:
            logger.info(f"🔚 Routing to 'END' (Action: {action}, Iteration: {iteration})")
            return END 
    except:
        logger.error("❌ Error in routing logic, forced to END.")
        return END

# --- Graph Construction (그래프 조립) ---
workflow = StateGraph(LogAnalysisState)

# 1. 노드 추가
workflow.add_node("fetch_logs", fetch_logs_node)      # 로그 수집
workflow.add_node("diagnose_error", diagnose_node)    # 분석 및 판단
workflow.add_node("inspect_code", tool_execution_node) # 코드 조회 (Tool)

# 2. 엣지 연결 (흐름 정의)
workflow.set_entry_point("fetch_logs")                # 시작점
workflow.add_edge("fetch_logs", "diagnose_error")     # 수집 -> 진단

# 3. 분기 및 순환 설정
workflow.add_conditional_edges(
    "diagnose_error",
    should_continue,
    {
        "inspect_code": "inspect_code",  # 코드 더 봐야 하면 -> inspect_code
        END: END                         # 다 봤으면 -> 끝
    }
)

workflow.add_edge("inspect_code", "diagnose_error") # 코드 봤으면 다시 진단 (Loop Back)

app = workflow.compile()

