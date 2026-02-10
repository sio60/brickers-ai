import json
import logging
import re
import asyncio
from typing import List, Optional, Dict, Any, Union
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END

from .state import LogAnalysisState
from .persistence import get_archived_logs
from ..llm_clients import GeminiClient
from ..agent_tools import (
    execute_read_file, 
    execute_check_db, 
    execute_check_system, 
    execute_check_sqs
)

# Logger configuration
logger = logging.getLogger("agent.log_analyzer.agent")

# --- Nodes (노드 정의) ---

async def fetch_logs_node(state: LogAnalysisState):
    """
    [노드 1: 로그 수집]
    DB(Persistence)에서 아카이빙된 로그를 가져옵니다. (Docker 직접 접근 제거)
    """
    target_job_id = state.get("job_id")
    raw_logs = state.get("logs", "")
    
    logger.info(f"--- [로그 분석기] 1단계: 로그 수집 시작 (Job: {target_job_id or 'Unknown'}) ---")
    
    # 1. State에 이미 로그가 있으면 사용 (테스트용 등)
    if raw_logs:
        logger.info(f"✅ 입력된 로그 사용 ({len(raw_logs)} bytes)")
        filtered_logs = raw_logs
    
    # 2. Job ID로 DB 아카이브 조회
    elif target_job_id:
        logger.info(f"🔍 DB 아카이브 조회 중... (Job: {target_job_id})")
        archived = await get_archived_logs(target_job_id)
        if archived:
            filtered_logs = archived
            logger.info(f"✅ DB에서 아카이브된 로그 로드 성공 ({len(filtered_logs.splitlines())}줄)")
        else:
            logger.warning(f"⚠️ DB에서 로그를 찾을 수 없습니다. (Job: {target_job_id})")
            filtered_logs = "로그 데이터가 없습니다."
    else:
        logger.warning("⚠️ Job ID가 제공되지 않았습니다.")
        filtered_logs = "분석할 로그 데이터가 없습니다."

    user_prompt = f"""
    [대상 Job ID: {target_job_id or "알 수 없음"}]
    
    [로그 데이터]
    {filtered_logs}
    
    이 로그를 정밀 분석하여 문제의 근본 원인을 찾으십시오. 
    필요하다면 제공된 도구를 사용하여 코드나 시스템 상태를 추가로 확인하십시오.
    """
    
    return {
        "logs": filtered_logs,
        "messages": [HumanMessage(content=user_prompt)],
        "iteration": 0,
        "job_id": target_job_id
    }

async def diagnose_node(state: LogAnalysisState):
    """
    [노드 2: 정밀 진단]
    """
    messages = state.get("messages", [])
    iteration = state.get("iteration", 0)
    
    system_prompt = """
    당신은 고급 시스템 디버깅 에이전트입니다.
    분석 결과는 반드시 한국어로 요약하여 제공하십시오.
    
    1. 원인 파악: Traceback, DB 연결, SQS 상태 등을 종합 분석.
    2. 도구 활용: 구체적인 파일 내용 확인이나 상태 조회가 필요하면 도구를 호출.
    3. 해결책 제안: 단순히 에러메시지를 읽는 게 아니라, 아키텍처 개선안이나 코드 수정안을 구체적으로 제시.
    
    출력 형식 (JSON):
    - 도구 사용: {"action": "read_file", "args": {"file_path": "..."}, "reasoning": "..."}
    - 완료: {"action": "finish", "analysis": {"error_found": true, "summary": "...", "root_cause": "...", "suggestion": "..."}}
    """
    
    try:
        llm = GeminiClient()
        response = await asyncio.to_thread(llm.generate_json, messages[-1].content, system_prompt)
        return {"analysis_result": json.dumps(response, ensure_ascii=False), "iteration": iteration + 1}
    except Exception as e:
        logger.error(f"AI 진단 에러: {e}")
        return {"analysis_result": json.dumps({"action": "finish", "analysis": {"error_found": True, "summary": f"진단 에러: {e}"}})}

async def tool_execution_node(state: LogAnalysisState):
    """
    [노드 3: 도구 실행]
    """
    decision = json.loads(state.get("analysis_result", "{}"))
    action = decision.get("action")
    args = decision.get("args", {})
    
    tool_map = {
        "read_file": execute_read_file,
        "check_db": execute_check_db,
        "check_system": execute_check_system,
        "check_sqs": execute_check_sqs
    }
    
    if action in tool_map:
        result = await asyncio.to_thread(tool_map[action], args)
        feedback = f"[{action} 결과]\n{result}\n\n위 결과를 바탕으로 분석을 계속하거나 종료하십시오."
        
        curr_messages = state.get("messages", [])
        curr_messages.append(AIMessage(content=f"Executed {action}"))
        curr_messages.append(HumanMessage(content=feedback))
        return {"messages": curr_messages}
        
    return {}

def should_continue(state: LogAnalysisState):
    decision = json.loads(state.get("analysis_result", "{}"))
    if decision.get("action") in ["read_file", "check_db", "check_system", "check_sqs"] and state.get("iteration", 0) < 5:
        return "tool_exec"
    return END

# Graph
workflow = StateGraph(LogAnalysisState)
workflow.add_node("fetch", fetch_logs_node)
workflow.add_node("diagnose", diagnose_node)
workflow.add_node("tool_exec", tool_execution_node)

workflow.set_entry_point("fetch")
workflow.add_edge("fetch", "diagnose")
workflow.add_conditional_edges("diagnose", should_continue, {"tool_exec": "tool_exec", END: END})
workflow.add_edge("tool_exec", "diagnose")

app = workflow.compile()
