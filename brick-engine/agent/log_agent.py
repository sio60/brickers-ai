from typing import TypedDict, Annotated, List, Optional
import os
import docker
import json
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

# LLM Client Import
try:
    from .llm_clients import GeminiClient, OpenAIClient
except ImportError:
    # Standalone execution support
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent))
    from llm_clients import GeminiClient, OpenAIClient

# --- State Definition ---
class LogAnalysisState(TypedDict):
    container_name: str
    logs: str  # Raw logs
    analysis_result: Optional[str] # LLM Analysis (JSON string)
    error_count: int

# --- Nodes ---

def fetch_logs_node(state: LogAnalysisState):
    """docker SDK를 사용하여 최신 로그를 가져옵니다."""
    container_name = state.get("container_name", "brickers-ai-container")
    try:
        client = docker.from_env()
        container = client.containers.get(container_name)
        # 최근 300줄 가져옴 (컨텍스트 확보)
        logs = container.logs(tail=300).decode("utf-8", errors="replace")
        return {"logs": logs}
    except Exception as e:
        return {"logs": f"Error fetching logs: {str(e)}", "error_count": 1}

def analyze_logs_node(state: LogAnalysisState):
    """가져온 로그를 LLM에게 분석시킵니다."""
    logs = state.get("logs", "")
    if not logs or "Error fetching logs" in logs:
         return {
             "analysis_result": json.dumps({
                 "error_found": True,
                 "summary": "로그를 가져오는 데 실패했습니다.",
                 "root_cause": logs,
                 "suggestion": "Docker 컨테이너 상태를 확인하세요."
             }), 
             "error_count": 1
         }

    # 간단한 프롬프트 구성
    current_logs = logs[-3000:] # 길이 제한 (토큰 절약)

    system_prompt = """
    당신은 시스템 로그 분석 전문가(SRE)입니다.
    Docker 컨테이너의 로그를 분석하여 잠재적인 문제, 에러, 또는 비정상적인 패턴을 식별해야 합니다.
    
    분석 결과는 항상 **한국어**로 작성해주세요.
    반드시 다음 JSON 형식을 준수해야 합니다:
    {
        "error_found": boolean, // 에러(Exception, Error, Critical) 발견 여부
        "summary": "string",    // 로그 요약 (주요 작업 내용)
        "root_cause": "string", // 에러 원인 (없으면 null)
        "suggestion": "string"  // 해결책 제안 (코드 수정, 재시작 등, 없으면 null)
    }
    """

    user_prompt = f"""
    아래는 'brickers-ai' 컨테이너의 최신 로그입니다.
    
    [LOG START]
    {current_logs}
    [LOG END]
    
    위 로그를 분석해 주세요. 특히 'Traceback', 'Error', 'Exception' 키워드에 주목하세요.
    """
    
    try:
        # Default to Gemini for cost/speed
        llm = GeminiClient() 
        result = llm.generate_json(user_prompt, system_prompt)
        
        return {"analysis_result": json.dumps(result, ensure_ascii=False)}
        
    except Exception as e:
        # Fallback handling
        fallback_result = {
            "error_found": True,
            "summary": "LLM 분석 중 오류가 발생했습니다.",
            "root_cause": str(e),
            "suggestion": "API 키 또는 네트워크 연결을 확인하세요."
        }
        return {"analysis_result": json.dumps(fallback_result, ensure_ascii=False), "error_count": 1}

# --- Graph Construction ---

workflow = StateGraph(LogAnalysisState)

workflow.add_node("fetch_logs", fetch_logs_node)
workflow.add_node("analyze_logs", analyze_logs_node)

workflow.set_entry_point("fetch_logs")
workflow.add_edge("fetch_logs", "analyze_logs")
workflow.add_edge("analyze_logs", END)

app = workflow.compile()
