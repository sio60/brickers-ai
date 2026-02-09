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

# --- Tool Execution Logic ---
# --- Tool Execution Logic (Moved to agent_tools.py or imported) ---
# execute_read_file is now imported from agent_tools.py
# ensuring we use the centralized definition.

# --- Nodes ---

# --- Nodes (노드 정의) ---



# --- Nodes (노드 정의) ---

def fetch_logs_node(state: LogAnalysisState):
    """
    [노드 1: 로그 수집]
    Docker SDK를 사용하여 실행 중인 컨테이너의 최신 로그를 가져옵니다.
    """
    container_name = state.get("container_name", "brickers-ai-container")
    logger.info(f"--- [로그 에이전트] 1단계: 컨테이너 로그 수집 중 ({container_name}) ---")
    try:
        client = docker.from_env()
        container = client.containers.get(container_name)
        logs = container.logs(tail=500).decode("utf-8", errors="replace")
        logger.info(f"✅ 로그 수집 성공 ({len(logs)} 바이트).")
    except Exception as e:
        # 독커 수집 실패 시 시뮬레이션 로그 사용 여부 확인
        existing_logs = state.get("logs", "")
        if existing_logs and ("ERROR" in existing_logs or "Traceback" in existing_logs):
             logger.warning("⚠️ 독커 연결 실패. 상태에 저장된 테스트용 로그를 사용합니다.")
             logs = existing_logs 
        else:
             logger.error(f"❌ 독커 로그 수집 에러: {str(e)}")
             logs = f"Docker에서 로그를 가져오는 중 오류 발생: {str(e)}\n(독커가 실행 중인지 확인하세요)"
    
    user_prompt = f"""
    [시스템 로그]
    {logs[-4000:]} 
    
    로그를 분석하여 오류(Traceback, Exception, Timeout)를 식별하십시오.
    
    사용 가능한 도구:
    1. `read_file`: Traceback에서 파일 경로가 보일 때 코드를 확인하기 위해 사용.
    2. `check_db`: 'ConnectionTimeout', 'MongoError' 등 DB 관련 오류 시 사용.
    3. `check_sqs`: 'Empty Message', 'Boto3Error', 처리 지연 발생 시 사용.
    4. `check_system`: 'MemoryError', 'Kill signal', 전반적인 느려짐 발생 시 사용.
    
    도구를 사용하거나 분석을 종료하기 위한 JSON을 출력하십시오.
    """
    
    return {
        "logs": logs, 
        "messages": [HumanMessage(content=user_prompt)], 
        "iteration": 0,
        "error_count": 0
    }

def diagnose_node(state: LogAnalysisState):
    """
    [노드 2: 에러 진단 및 의사결정]
    LLM이 로그를 분석하여 '도구를 사용할지' 아니면 '분석을 종료할지' 결정합니다.
    """
    messages = state.get("messages", [])
    iteration = state.get("iteration", 0)
    logger.info(f"--- [로그 에이전트] 2단계: 에러 진단 중 (반복: {iteration}) ---")
    
    # 의사결정을 위한 시스템 프롬프트
    system_prompt = """
    당신은 전문 디버깅 에이전트입니다.
    목표: 에러의 근본 원인(코드, DB, SQS 또는 시스템)을 찾으십시오.
    
    의사결정 프로세스:
    1. **로그 분석**: 키워드를 찾으십시오.
       - 코드 에러 -> `read_file` (주의: 인자명은 'file_path'를 사용)
       - DB 에러 (Timeout, Connection) -> `check_db`
       - Queue/AWS 에러 -> `check_sqs`
       - 리소스/크래시 -> `check_system`
    
    2. **정교화**: 도구를 사용했다면, 다음 단계에서 그 결과를 분석하십시오.
    
    출력 형식 (JSON):
    - 도구 사용: `{"action": "도구이름", "args": {...}, "reasoning": "이유"}`
    - 종료: `{"action": "finish", "analysis": {"error_found": true, "summary": "요약(한국어)", "root_cause": "원인(한국어)", "suggestion": "해결책(한국어)"}}`
    
    모든 분석 결과(summary, root_cause, suggestion)는 반드시 한국어로 작성하십시오.
    """
    
    try:
        llm = GeminiClient()
        response = llm.generate_json(messages[-1].content, system_prompt)
        logger.info(f"🤖 AI 결정: {json.dumps(response, ensure_ascii=False)}")
        
        return {"analysis_result": json.dumps(response, ensure_ascii=False), "iteration": iteration + 1}
        
    except Exception as e:
         logger.error(f"❌ AI 진단 실패: {str(e)}")
         return {"analysis_result": json.dumps({"action": "finish", "analysis": {"error_found": True, "summary": f"진단 도중 에러 발생: {str(e)}"}})}

def tool_execution_node(state: LogAnalysisState):
    """
    [노드 3: 도구 실행]
    `diagnose_node`에서 요청한 도구를 실행하고 결과를 반환합니다.
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
    
    tool_output = ""

    if action == "read_file":
        tool_output = execute_read_file(args)
    elif action == "check_db":
        tool_output = execute_check_db(args)
    elif action == "check_system":
        tool_output = execute_check_system(args)
    elif action == "check_sqs":
        tool_output = execute_check_sqs(args)
    
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

