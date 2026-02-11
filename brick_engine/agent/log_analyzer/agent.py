import json
import logging
import os
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
    [Node 1: Fetch] DB/Memory에서 로그 로드
    """
    target_job_id = state.get("job_id")
    raw_logs = state.get("logs", "")
    
    logger.info(f"--- [로그 분석기] 1. 로그 수집 (Job: {target_job_id}) ---")
    
    if not raw_logs and target_job_id:
        archived = await get_archived_logs(target_job_id)
        if archived:
            raw_logs = archived
            logger.info(f"✅ DB 로그 로드 완료 ({len(raw_logs)} chars)")
        else:
            raw_logs = "로그 없음"

    return {"logs": raw_logs, "iteration": 0}


async def parse_error_node(state: LogAnalysisState):
    """
    [Node 2: Parse Error] 로그에서 전체 Traceback 체인 추출
    - 단일 에러가 아니라, 호출 스택 전체를 추출하여 맥락을 보존
    """
    logs = state.get("logs", "")
    logger.info("--- [로그 분석기] 2. 에러 파싱 (Full Traceback) ---")
    
    # 1. 전체 Traceback 블록 추출
    traceback_pattern = r'(Traceback \(most recent call last\):.*?)(?=\n\S|\Z)'
    traceback_blocks = re.findall(traceback_pattern, logs, re.DOTALL)
    
    # 2. 모든 File 참조 추출 (호출 스택 전체)
    file_pattern = r'File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<func>\S+)'
    
    all_frames = []  # 전체 호출 스택
    user_frames = []  # 사용자 코드만
    
    for match in re.finditer(file_pattern, logs):
        file_path = match.group("file")
        line_no = int(match.group("line"))
        func_name = match.group("func")
        
        frame = {
            "file": file_path,
            "line": line_no,
            "function": func_name,
            "is_user_code": "site-packages" not in file_path
        }
        all_frames.append(frame)
        
        if frame["is_user_code"]:
            user_frames.append(frame)
    
    # 3. 에러 메시지 추출 (마지막 Exception 라인)
    error_message = ""
    error_type = ""
    lines = logs.splitlines()
    for line in reversed(lines):
        # Python Exception 패턴: ExceptionType: message
        err_match = re.match(r'^(\w+(?:Error|Exception|Warning|Timeout))\s*:\s*(.+)', line.strip())
        if err_match:
            error_type = err_match.group(1)
            error_message = err_match.group(2).strip()
            break
    
    # 4. 에러 컨텍스트 구성 (가장 중요한 사용자 코드 프레임 우선)
    primary_frame = user_frames[-1] if user_frames else (all_frames[-1] if all_frames else {})
    
    error_context = {
        "primary_file": primary_frame.get("file", "unknown"),
        "primary_line": primary_frame.get("line", 0),
        "primary_function": primary_frame.get("function", "unknown"),
        "error_type": error_type,
        "error_message": error_message,
        "call_stack": user_frames,  # 사용자 코드 호출 스택 전체
        "total_frames": len(all_frames),
        "user_code_frames": len(user_frames),
        "traceback_raw": traceback_blocks[-1][:500] if traceback_blocks else ""
    }
    
    logger.info(f"🎯 에러 감지: {error_type}: {error_message[:80]}")
    logger.info(f"📍 위치: {primary_frame.get('file', '?')}:{primary_frame.get('line', '?')} in {primary_frame.get('function', '?')}")
    logger.info(f"📊 호출 스택: 전체 {len(all_frames)}개 프레임, 사용자 코드 {len(user_frames)}개")

    return {"error_context": error_context}


async def context_retrieval_node(state: LogAnalysisState):
    """
    [Node 3: Context Retrieval] 
    - 에러 발생 파일 + 연관 파일들 (호출 스택 기반) 코드 읽기
    - DB / System / SQS 인프라 종합 조회
    """
    error_ctx = state.get("error_context", {})
    job_id = state.get("job_id")
    logs = state.get("logs", "")
    logger.info("--- [로그 분석기] 3. 문맥 확보 (Multi-File + Infra) ---")
    
    related_code_sections = []
    db_context = ""
    system_info = ""
    sqs_info = ""
    
    # ========================================================
    # 1. Multi-File Code Reading (호출 스택의 모든 사용자 코드 파일)
    # ========================================================
    call_stack = error_ctx.get("call_stack", [])
    
    # 중복 파일 제거 (같은 파일의 다른 라인은 범위를 합침)
    files_to_read = {}
    for frame in call_stack:
        fp = frame["file"]
        ln = frame["line"]
        fn = frame["function"]
        
        if fp not in files_to_read:
            files_to_read[fp] = {"lines": [], "functions": []}
        files_to_read[fp]["lines"].append(ln)
        files_to_read[fp]["functions"].append(fn)
    
    # 최대 5개 파일까지 읽기 (너무 많으면 토큰 낭비)
    for file_path, info in list(files_to_read.items())[:5]:
        try:
            # 각 파일에서 에러 관련 라인 ±15줄 읽기
            min_line = max(1, min(info["lines"]) - 15)
            max_line = max(info["lines"]) + 15
            
            read_result = execute_read_file({
                "file_path": file_path,
                "line_start": min_line,
                "line_end": max_line
            })
            
            if "Error" not in str(read_result):
                section_header = f"📄 File: {file_path} (Lines {min_line}-{max_line})"
                section_header += f"\n   Functions: {', '.join(info['functions'])}"
                section_header += f"\n   Error Lines: {info['lines']}"
                related_code_sections.append(f"{section_header}\n```python\n{read_result}\n```")
                logger.info(f"✅ 코드 읽기 성공: {os.path.basename(file_path)} ({len(info['lines'])} 지점)")
            else:
                related_code_sections.append(f"⚠️ 읽기 실패: {file_path} → {read_result}")
                logger.warning(f"⚠️ 코드 읽기 실패: {file_path}")
                
        except Exception as e:
            related_code_sections.append(f"❌ 읽기 에러: {file_path} → {e}")
            logger.error(f"❌ 코드 읽기 에러: {file_path}: {e}")
    
    # 에러가 발생한 파일이 call_stack에 없는 경우 (primary_file 보충)
    primary_file = error_ctx.get("primary_file", "")
    if primary_file and primary_file not in files_to_read and primary_file != "unknown":
        try:
            primary_line = error_ctx.get("primary_line", 1)
            read_result = execute_read_file({
                "file_path": primary_file,
                "line_start": max(1, primary_line - 20),
                "line_end": primary_line + 20
            })
            if "Error" not in str(read_result):
                related_code_sections.insert(0, f"📄 [PRIMARY] File: {primary_file} (Line {primary_line})\n```python\n{read_result}\n```")
                logger.info(f"✅ Primary 파일 읽기 성공: {primary_file}")
        except Exception as e:
            logger.error(f"❌ Primary 파일 읽기 실패: {e}")
    
    related_code = "\n\n---\n\n".join(related_code_sections) if related_code_sections else "코드 확보 실패"
    
    # ========================================================
    # 2. DB Check (Job Status + 연결 상태)
    # ========================================================
    if job_id:
        try:
            db_res = execute_check_db({"query": {"jobId": job_id}, "collection": "kids_jobs"})
            db_context = f"[DB 조회 결과]\nJob Metadata: {db_res}"
            logger.info("✅ DB 메타데이터 확보")
        except Exception as e:
            db_context = f"[DB 조회 실패] {e}"
            logger.warning(f"⚠️ DB 조회 실패: {e}")
    
    # ========================================================
    # 3. System Health Check (CPU/메모리/디스크)
    # ========================================================
    try:
        sys_res = execute_check_system({"dummy": "ignore"})
        system_info = f"[System Health]\n{sys_res}"
        logger.info("✅ 시스템 상태 확보")
    except Exception as e:
        system_info = f"[System Check 실패] {e}"
        logger.warning(f"⚠️ System Check 실패: {e}")

    # ========================================================
    # 4. SQS Queue Status (조건부 — 로그에 SQS 관련 키워드 있을 때)
    # ========================================================
    sqs_keywords = ["sqs", "boto", "queue", "empty message", "timeout", "connection"]
    if any(kw in logs.lower() for kw in sqs_keywords):
        try:
            sqs_res = execute_check_sqs({"queue_type": "all"})
            sqs_info = f"[SQS Status]\n{sqs_res}"
            logger.info("✅ SQS 상태 확보")
        except Exception as e:
            sqs_info = f"[SQS Check 실패] {e}"
            logger.warning(f"⚠️ SQS Check 실패: {e}")
    else:
        sqs_info = "[SQS] 관련 에러 미감지 → 조회 생략"

    # 모든 인프라 정보 병합
    infra_context = f"{db_context}\n\n{system_info}\n\n{sqs_info}"
    
    logger.info(f"📊 문맥 확보 완료: 코드 {len(related_code_sections)}개 파일, Infra 3개 섹션")

    return {"related_code": related_code, "db_context": infra_context}


async def solution_generation_node(state: LogAnalysisState):
    """
    [Node 4: Solution] 종합 분석 및 상세 리포트 생성
    - 관리자 페이지용: 최대한 세세하고 자세하고 정확한 분석
    """
    logger.info("--- [로그 분석기] 4. 상세 솔루션 생성 (Admin Grade) ---")
    
    logs = state.get("logs", "")[-4000:]  # 관리자용이므로 더 많은 로그 포함
    error_ctx = state.get("error_context", {})
    related_code = state.get("related_code", "코드 확보 실패")
    db_ctx = state.get("db_context", "")
    
    system_prompt = """
    당신은 Brickers AI 시스템의 **수석 디버깅 전문가(Senior Debugging Specialist)**입니다.
    당신의 분석 리포트는 **관리자 대시보드**에 표시되며, 개발팀이 이 리포트만 보고 즉시 문제를 해결할 수 있어야 합니다.
    
    ════════════════════════════════════════
    [핵심 원칙] 
    - "한 줄 요약"으로 끝내지 마십시오. 반드시 **상세한 다단계 분석**을 수행하십시오.
    - 모든 분석은 **한국어**로 작성하십시오.
    - 추상적 조언 금지. "코드를 확인하세요"가 아니라 "315라인의 `last_progress` 변수를 0으로 초기화하세요"처럼 구체적으로.
    ════════════════════════════════════════

    [분석 절차 — 반드시 이 순서대로 수행하십시오]
    
    ■ STEP 1: 에러 식별
    - 어떤 종류의 에러(Exception Type)가 발생했는가?
    - 에러 메시지가 의미하는 바는 무엇인가?
    - 이 에러가 Python 내장 에러인가, 외부 라이브러리 에러인가, 커스텀 에러인가?
    
    ■ STEP 2: 호출 스택 분석
    - Traceback의 호출 스택(Call Stack)을 따라가며 실행 흐름을 설명하시오.
    - 어떤 함수가 어떤 함수를 호출했고, 어디서 실패했는가?
    - `call_stack` 데이터에서 사용자 코드 프레임을 모두 분석하시오.
    
    ■ STEP 3: 근본 원인 분석 (Root Cause)
    - 에러가 발생한 코드(`related_code`)를 정밀 검토하시오.
    - **"왜"** 이 코드가 실패했는가? (변수 미초기화, None 접근, 타입 불일치, 인코딩, 타임아웃 등)
    - 이 에러가 일시적(transient)인가, 구조적(structural)인가?
    
    ■ STEP 4: 연관 코드 검토
    - 에러 발생 파일 외에도, 호출 스택에 포함된 **다른 파일들의 코드**도 검토하시오.
    - 해당 파일들에서 수정이 필요한 부분이 있는가?
    - 함수 간 데이터 전달 과정에서 타입이나 값이 잘못된 곳은 없는가?
    - 비동기(async/await) 처리가 올바르게 되어 있는가? (await 누락, 동기 함수를 비동기 컨텍스트에서 호출 등)
    
    ■ STEP 5: 인프라 상태 점검
    - DB 연결 상태: 정상인가? 타임아웃이 발생할 만한 상태인가? 값이 예상과 다른가?
    - System Health: CPU/메모리가 부족하여 OOM Kill이 발생할 수 있는가?
    - SQS 큐: 메시지가 쌓여서 처리가 지연되고 있는가? Dead Letter Queue에 빠진 메시지가 있는가?
    
    ■ STEP 6: 수정안 제시
    - **Before (문제 코드)**: 현재 문제가 되는 코드를 그대로 보여주시오.
    - **After (수정 코드)**: 수정된 코드를 완전한 형태로 보여주시오.
    - 수정이 필요한 **모든 파일**에 대해 각각 Before/After를 제시하시오.
    - 왜 이렇게 수정해야 하는지 이유를 설명하시오.
    
    ■ STEP 7: 추가 권장 사항
    - 이 에러를 방지하기 위한 예방적 조치 (예: 입력 검증, try-except 추가, 타임아웃 설정 등)
    - 프롬프트 튜닝이 필요한 경우 구체적 문구 제안 (Gemini/Tripo)
    - 파라미터 조정이 필요한 경우 구체적 수치 제안 (ex: timeout 60s → 120s)

    ════════════════════════════════════════
    [출력 형식 — JSON]
    반드시 다음 JSON 형식을 엄수하십시오. 모든 필드를 빠짐없이 채우십시오.
    ════════════════════════════════════════
    {
        "error_identification": {
            "error_type": "에러 타입 (ex: RuntimeError)",
            "error_message": "에러 메시지 전문",
            "error_category": "code_bug | api_timeout | infra_issue | data_mismatch | async_issue | config_error",
            "severity": "critical | high | medium | low"
        },
        "call_stack_analysis": "호출 스택을 따라가며 실행 흐름을 설명 (어떤 함수 → 어떤 함수 → 실패 지점)",
        "root_cause": {
            "summary": "근본 원인 한 줄 요약",
            "detail": "근본 원인 상세 설명 (코드 흐름, 변수 상태, 외부 의존성 등)",
            "is_transient": false
        },
        "investigation_steps": [
            "1단계: [어떤 파일]의 [어떤 함수]를 확인함 → [발견한 사실]",
            "2단계: [어떤 DB/API]를 조회함 → [발견한 사실]",
            "3단계: ..."
        ],
        "code_patches": [
            {
                "file_path": "수정 대상 파일 경로",
                "function_name": "수정 대상 함수명",
                "line_range": "수정 범위 (ex: 310-320)",
                "before_code": "현재 문제 코드 (원본)",
                "after_code": "수정된 코드",
                "reason": "이렇게 수정해야 하는 이유"
            }
        ],
        "related_issues": [
            {
                "file_path": "연관 파일 경로",
                "issue": "발견된 문제",
                "suggestion": "수정 제안"
            }
        ],
        "infra_diagnosis": {
            "db_status": "정상 | 이상 | 미확인",
            "db_detail": "DB 관련 상세 소견",
            "system_status": "정상 | 이상 | 미확인",
            "system_detail": "CPU/메모리/디스크 상세 소견",
            "sqs_status": "정상 | 이상 | 미확인",
            "sqs_detail": "SQS 큐 상세 소견"
        },
        "async_check": {
            "has_issue": false,
            "detail": "비동기 처리 관련 소견 (await 누락, 동기/비동기 혼용 등)"
        },
        "recommendations": [
            "예방적 조치 1",
            "예방적 조치 2"
        ],
        "summary": "전체 분석을 3-5문장으로 요약 (관리자가 빠르게 읽을 수 있도록)"
    }
    """
    
    user_prompt = f"""
    ════════════════ 분석 대상 ════════════════
    
    [Job ID]
    {state.get("job_id", "Unknown")}
    
    [Error Context (파싱 결과)]
    - Error Type: {error_ctx.get("error_type", "Unknown")}
    - Error Message: {error_ctx.get("error_message", "Unknown")}
    - Primary File: {error_ctx.get("primary_file", "Unknown")}:{error_ctx.get("primary_line", "?")}
    - Primary Function: {error_ctx.get("primary_function", "Unknown")}
    - Call Stack ({error_ctx.get("user_code_frames", 0)} user frames / {error_ctx.get("total_frames", 0)} total):
    {json.dumps(error_ctx.get("call_stack", []), indent=2, ensure_ascii=False)}
    
    [Raw Traceback]
    {error_ctx.get("traceback_raw", "없음")}
    
    ════════════════ 소스 코드 ════════════════
    
    [Related Code (호출 스택 기반 다중 파일)]
    {related_code}
    
    ════════════════ 로그 전문 ════════════════
    
    [Log Snippet (최근 4000자)]
    {logs}
    
    ════════════════ 인프라 상태 ════════════════
    
    [Infra Info (DB / System / SQS)]
    {db_ctx}
    """
    
    try:
        llm = GeminiClient()
        response = await asyncio.to_thread(llm.generate_json, user_prompt, system_prompt)
        result = json.dumps(response, ensure_ascii=False)
        logger.info(f"✅ 상세 분석 리포트 생성 완료 ({len(result)} chars)")
        return {"analysis_result": result}
    except Exception as e:
        logger.error(f"❌ AI 분석 에러: {e}")
        fallback = {
            "error_identification": {"error_type": "AnalysisError", "error_message": str(e)},
            "root_cause": {"summary": f"AI 분석 자체가 실패함: {e}", "detail": str(e)},
            "summary": f"AI 분석 에러 발생: {e}"
        }
        return {"analysis_result": json.dumps(fallback, ensure_ascii=False)}


# ============================================================
# Graph Construction
# ============================================================
workflow = StateGraph(LogAnalysisState)

workflow.add_node("fetch", fetch_logs_node)
workflow.add_node("parse_error", parse_error_node)
workflow.add_node("retrieve", context_retrieval_node)
workflow.add_node("solve", solution_generation_node)

workflow.set_entry_point("fetch")
workflow.add_edge("fetch", "parse_error")
workflow.add_edge("parse_error", "retrieve")
workflow.add_edge("retrieve", "solve")
workflow.add_edge("solve", END)

app = workflow.compile()
