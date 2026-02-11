"""
Log Analyzer — Node 함수
========================
LangGraph 그래프의 각 노드를 구성하는 비동기 함수들.

Nodes:
  1. fetch_logs_node        — 로그 수집 (Docker SDK + DB 폴백)
  2. no_logs_report_node    — [NEW] 로그 수집 실패 시 빈 리포트 생성
  3. parse_error_node       — Traceback 추출 및 카테고리 분류 (infra vs code)
  4. agent_investigate_node  — ReAct Loop (코드 버그 중심)
  5. investigate_infra_node — [NEW] ReAct Loop (인프라 장애 중심)
  6. simple_summary_node    — 에러 미감지 시 간단 요약
  7. generate_report_node   — 상세 분석 리포트 생성
  8. validate_report_node   — [NEW] 리포트 JSON 검증 및 재시도 제어
  9. alert_admin_node       — [NEW] Critical 에러 시 알림 전송
"""

import json
import logging
import re
import asyncio
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from .state import LogAnalysisState
from .persistence import get_archived_logs
from .config import (
    TOOL_SCHEMAS,
    TOOL_EXECUTOR_MAP,
    MAX_INVESTIGATION_ROUNDS,
    DOCKER_LOG_TAIL_LINES,
    JOB_FAILURE_PATTERN,
    JOB_START_PATTERN,
    INFRA_ERROR_TYPES,
    INFRA_ERROR_KEYWORDS,
    MAX_REPORT_RETRIES,
    REPORT_REQUIRED_FIELDS,
    DEEP_DIVE_THRESHOLD,
)
from .prompts import (
    INVESTIGATE_SYSTEM_PROMPT,
    INVESTIGATE_INFRA_SYSTEM_PROMPT,
    DEEP_DIVE_PROMPT,
    REPORT_SYSTEM_PROMPT,
    SIMPLE_SUMMARY_SYSTEM_PROMPT,
    SIMPLE_SUMMARY_USER_TEMPLATE,
)
from ..llm_clients import GeminiClient

logger = logging.getLogger("agent.log_analyzer.nodes")


# ============================================================
# NODE 1: fetch_logs — 로그 수집
# ============================================================
async def fetch_logs_node(state: LogAnalysisState):
    """로그 확보 (우선순위: State > DB > Docker)"""
    target_job_id = state.get("job_id")
    raw_logs = state.get("logs", "")
    container_name = state.get("container_name", "brickers-ai-container")

    logger.info(f"--- [Node 1: fetch_logs] Job: {target_job_id} ---")

    if raw_logs:
        logger.info(f"✅ State에서 로그 확보 ({len(raw_logs)} chars)")
    elif target_job_id:
        archived = await get_archived_logs(target_job_id)
        if archived:
            raw_logs = archived
            logger.info(f"✅ DB 아카이브에서 로그 로드 ({len(raw_logs)} chars)")
    
    if not raw_logs:
        try:
            import docker
            client = docker.from_env()
            container = client.containers.get(container_name)
            raw_logs = container.logs(tail=DOCKER_LOG_TAIL_LINES).decode("utf-8", errors="replace")
            logger.info(f"✅ Docker SDK 로그 수집 완료 ({len(raw_logs)} chars)")
        except Exception as e:
            logger.warning(f"⚠️ Docker 연결 실패: {e}")
            raw_logs = ""

    # Job ID 자동 탐색 & 필터링
    if not target_job_id and raw_logs:
        failure_matches = re.findall(JOB_FAILURE_PATTERN, raw_logs)
        target_job_id = failure_matches[-1] if failure_matches else None
        if not target_job_id:
            start_matches = re.findall(JOB_START_PATTERN, raw_logs)
            target_job_id = start_matches[-1] if start_matches else None
    
    if target_job_id and raw_logs:
        job_lines = [line for line in raw_logs.splitlines() if target_job_id in line]
        if job_lines:
            filtered = "\n".join(job_lines)
            raw_logs = filtered if len(filtered) > 500 else filtered + "\n\n[=== 원본 로그 ===]\n" + raw_logs[-2000:]

    return {
        "logs": raw_logs,
        "job_id": target_job_id,
        "iteration": 0,
        "investigation_notes": [],
        "report_retry_count": 0,
    }


# ============================================================
# NODE 1-alt: no_logs_report — 로그 수집 실패 시
# ============================================================
async def no_logs_report_node(state: LogAnalysisState):
    """로그가 없을 때 '수집 실패' 리포트 생성"""
    logger.info("--- [Node: no_logs_report] 로그 없음 → 종료 ---")
    fallback = {
        "error_identification": {"error_type": "DataFetchError", "severity": "medium"},
        "root_cause": {"summary": "로그 데이터를 확보하지 못했습니다."},
        "summary": "DB 아카이브와 Docker 컨테이너 모두에서 로그를 찾을 수 없어 분석을 중단합니다.",
    }
    return {"analysis_result": json.dumps(fallback, ensure_ascii=False)}


# ============================================================
# NODE 2: parse_error — Traceback 추출 & 카테고리 분류
# ============================================================
async def parse_error_node(state: LogAnalysisState):
    """에러 파싱 및 infra vs code 카테고리 결정"""
    logs = state.get("logs", "")
    logger.info("--- [Node 2: parse_error] 에러 분석 중 ---")

    # 전체 Traceback 블록 추출
    traceback_pattern = r'(Traceback \(most recent call last\):.*?)(?=\n\S|\Z)'
    traceback_blocks = re.findall(traceback_pattern, logs, re.DOTALL)

    # 에러 타입/메시지 추출
    error_type = ""
    error_message = ""
    for line in reversed(logs.splitlines()):
        err_match = re.match(r'^(\w+(?:Error|Exception|Warning|Timeout))\s*:\s*(.+)', line.strip())
        if err_match:
            error_type = err_match.group(1)
            error_message = err_match.group(2).strip()
            break

    # 카테고리 분류
    category = "code_bug"
    if error_type in INFRA_ERROR_TYPES:
        category = "infra_issue"
    else:
        # 메시지 키워드 검사
        msg_lower = error_message.lower()
        if any(kw in msg_lower for kw in INFRA_ERROR_KEYWORDS):
            category = "infra_issue"

    # 호출 스택 추출
    file_pattern = r'File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<func>\S+)'
    all_frames = []
    user_frames = []
    for m in re.finditer(file_pattern, logs):
        frame = {"file": m.group("file"), "line": int(m.group("line")), "function": m.group("func")}
        all_frames.append(frame)
        if "site-packages" not in frame["file"]:
            user_frames.append(frame)

    primary = user_frames[-1] if user_frames else (all_frames[-1] if all_frames else {})

    error_context = {
        "error_type": error_type,
        "error_message": error_message,
        "call_stack": user_frames,
        "primary_file": primary.get("file", "unknown"),
        "primary_line": primary.get("line", 0),
        "primary_function": primary.get("function", "unknown"),
        "total_frames": len(all_frames),
        "user_code_frames": len(user_frames),
        "traceback_raw": traceback_blocks[-1][:800] if traceback_blocks else "",
    }

    logger.info(f"📊 Category: {category}, Error: {error_type}")
    return {"error_context": error_context, "error_category": category}


# ============================================================
# Core Investigation Node (Shared Logic)
# ============================================================
async def _run_investigation(state: LogAnalysisState, system_prompt: str, node_name: str):
    iteration = state.get("iteration", 0)
    error_ctx = state.get("error_context", {})
    logs = state.get("logs", "")[-3000:]

    logger.info(f"--- [{node_name}] Round {iteration + 1}/{MAX_INVESTIGATION_ROUNDS} ---")

    # ── 메시지 구성 ──
    if iteration == 0:
        initial_context = f"""[에러 정보]
- Type: {error_ctx.get('error_type', 'Unknown')}
- Message: {error_ctx.get('error_message', 'Unknown')}
- File: {error_ctx.get('primary_file', 'Unknown')}:{error_ctx.get('primary_line', '?')}
- Function: {error_ctx.get('primary_function', 'Unknown')}

[호출 스택 (사용자 코드)]
{json.dumps(error_ctx.get('call_stack', []), indent=2, ensure_ascii=False)}

[Traceback]
{error_ctx.get('traceback_raw', '없음')}

[로그 (최근)]
{logs}

위 정보를 바탕으로 도구를 사용하여 조사를 시작하세요. 에러 발생 파일부터 읽어보세요."""
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=initial_context)]
    else:
        current_msgs = state.get("messages", [])
        if iteration >= DEEP_DIVE_THRESHOLD:
            # 3라운드 이상 시 Deep Dive 유도
            deep_dive_msg = HumanMessage(content=DEEP_DIVE_PROMPT.format(iteration=iteration+1, prev_rounds=iteration))
            messages = current_msgs + [deep_dive_msg]
        else:
            messages = current_msgs + [HumanMessage(content=f"[조사 라운드 {iteration + 1}] 이전 조사 결과를 바탕으로, 추가로 확인이 필요한 파일이나 인프라가 있으면 도구를 호출하세요. 충분하다면 도구 호출 없이 응답하세요.")]

    # ── LLM 호출 ──
    llm = GeminiClient()
    model = llm.bind_tools(TOOL_SCHEMAS)
    response = await asyncio.to_thread(model.invoke, messages)

    # ── 도구 실행 ──
    tool_messages = []
    notes = []
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            name, args, tid = tc["name"], tc["args"], tc.get("id", tc["name"])
            executor = TOOL_EXECUTOR_MAP.get(name)
            result = str(executor(args))[:2000] if executor else f"Unknown tool: {name}"
            tool_messages.append(ToolMessage(content=result, tool_call_id=tid))
            notes.append(f"[{name}] {json.dumps(args, ensure_ascii=False)} -> {result[:100]}...")

    # 이번 라운드에 추가된 메시지들만 반환 (operator.add로 누적됨)
    if iteration == 0:
        # 첫 라운드: System + Human + AI + Tool
        new_messages = messages + [response] + tool_messages
    else:
        # 이후 라운드: Human(last) + AI + Tool
        # messages[-1]이 방금 추가한 HumanMessage임
        new_messages = [messages[-1], response] + tool_messages

    note_summary = f"[Round {iteration+1}] " + ("; ".join(notes) if notes else "No tools used.")
    return {
        "messages": new_messages,
        "iteration": iteration + 1,
        "investigation_notes": [note_summary],
    }

# NODE 3: 일반 조사
async def agent_investigate_node(state: LogAnalysisState):
    return await _run_investigation(state, INVESTIGATE_SYSTEM_PROMPT, "Node 3: investigate")

# NODE 3-alt: 인프라 조사
async def investigate_infra_node(state: LogAnalysisState):
    return await _run_investigation(state, INVESTIGATE_INFRA_SYSTEM_PROMPT, "Node 3-infra: invest_infra")


# ============================================================
# NODE 4: simple_summary — 에러 미감지
# ============================================================
async def simple_summary_node(state: LogAnalysisState):
    logs = state.get("logs", "")[-2000:]
    llm = GeminiClient()
    try:
        response = await asyncio.to_thread(llm.generate_json, SIMPLE_SUMMARY_USER_TEMPLATE.format(logs=logs), SIMPLE_SUMMARY_SYSTEM_PROMPT)
        return {"analysis_result": json.dumps(response, ensure_ascii=False)}
    except Exception as e:
        return {"analysis_result": json.dumps({"summary": f"Error summarizing: {e}"})}


# ============================================================
# NODE 5: generate_insight — 관리자용 인사이트 생성
# ============================================================
async def generate_report_node(state: LogAnalysisState):
    """
    기존의 기술 리포트 대신, 관리자용 BIA 인사이트를 생성합니다.
    """
    logger.info("--- [Node 5: generate_insight] 어드민 인사이트 생성 시작 ---")
    error_ctx = state.get("error_context", {})
    notes = state.get("investigation_notes", [])
    logs = state.get("logs", "")[-3000:]

    prompt = f"[에러 정보]\n{json.dumps(error_ctx)}\n\n[조사 기록]\n{chr(10).join(notes)}\n\n[원본 로그]\n{logs}"
    
    try:
        from service.nano_banana import GeminiClient
        llm = GeminiClient()
        # INSIGHT_SYSTEM_PROMPT 사용 (비개발자 관리자 타겟)
        response = await asyncio.to_thread(llm.generate_json, prompt, INSIGHT_SYSTEM_PROMPT)
        
        # 상태에 개별 인사이트 필드 저장
        return {
            "analysis_result": json.dumps(response, ensure_ascii=False),
            "plain_summary": response.get("plain_summary"),
            "user_impact_level": response.get("user_impact_level"),
            "suggested_actions": response.get("suggested_actions"),
            "business_insight": response.get("business_insight")
        }
    except Exception as e:
        logger.error(f"❌ [generate_insight] AI 응답 생성 실패: {e}")
        return {"analysis_result": json.dumps({"plain_summary": f"인사이트를 생성하지 못했습니다: {e}"})}


# ============================================================
# NODE 6: validate_report — [NEW] 리포트 검증
# ============================================================
async def validate_report_node(state: LogAnalysisState):
    """리포트 JSON 유효성 및 필수 필드 검사"""
    result_str = state.get("analysis_result", "{}")
    retry_count = state.get("report_retry_count", 0)
    logger.info(f"--- [Node 6: validate_report] 검증 시작 (시도 {retry_count + 1}) ---")

    try:
        data = json.loads(result_str)
        missing = [f for f in REPORT_REQUIRED_FIELDS if f not in data]
        if not missing:
            logger.info("✅ 리포트 검증 통과")
            return {"report_retry_count": retry_count} # 값 유지
        else:
            logger.warning(f"⚠️ 필수 필드 누락: {missing}")
    except Exception as e:
        logger.warning(f"⚠️ JSON 파싱 실패: {e}")

    return {"report_retry_count": retry_count + 1}


# ============================================================
# NODE 7: alert_admin — [NEW] Critical 알림
# ============================================================
async def alert_admin_node(state: LogAnalysisState):
    """Critical 에러 발생 시 외부 알림 (로깅/Slack)"""
    result_str = state.get("analysis_result", "{}")
    try:
        data = json.loads(result_str)
        severity = data.get("error_identification", {}).get("severity", "unknown")
        summary = data.get("summary", "No summary")
        
        if severity == "critical":
            logger.error(f"🚨 [CRITICAL ALERT] 시스템 장애 감지!\n사유: {summary}")
            # 추후 Slack Webhook 등 연동 지점
    except:
        pass
    
    return {}

