# ============================================================================
# 메인 오케스트레이션: regeneration_loop
# ============================================================================

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from ..core.llm_clients import BaseLLMClient
from ..core.llm_state import AgentState
from ..core.memory_utils import memory_manager

from .constants import DEFAULT_PARAMS
from .graph import RegenerationGraph
from .evolver_runner import run_evolver

logger = logging.getLogger("agent.regeneration.pipeline")


# ============================================================================
# Memory & DB Helper Functions
# ============================================================================

def load_memory_from_db(model_id: str):
    """Legacy 로드 비활성화 (RAG로 대체)"""
    return {}


def save_memory_to_db(model_id: str, memory: Dict):
    """학습 데이터를 MongoDB에 저장"""
    try:
        import os
        from pymongo import MongoClient
        from datetime import datetime

        mongo_uri = os.getenv("MONGODB_URI")
        if not mongo_uri:
            logger.warning("[Memory] MONGODB_URI not set, skip save")
            return

        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
        col = client["brickers"]["regeneration_memory"]

        doc = {
            "model_id": model_id,
            "failed_approaches": memory.get("failed_approaches", []),
            "successful_patterns": memory.get("successful_patterns", []),
            "lessons": memory.get("lessons", []),
            "consecutive_failures": memory.get("consecutive_failures", 0),
            "updated_at": datetime.utcnow(),
        }

        col.update_one(
            {"model_id": model_id},
            {"$set": doc},
            upsert=True,
        )
        logger.info(f"[Memory] Saved to DB: {model_id} (lessons={len(doc['lessons'])})")

    except Exception as e:
        logger.error(f"[Memory] DB save failed: {e}")


def _is_truthy(value: Optional[str]) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


# ============================================================================
# 메인 루프
# ============================================================================

async def regeneration_loop(
    glb_path: str,
    output_ldr_path: str,
    subject_name: str = "Unknown Object",
    llm_client: Optional[BaseLLMClient] = None,
    max_retries: int = 11,  # 도구별 한도: TuneParameters(5) + MergeBricks(5) + RemoveBricks(1)
    acceptable_failure_ratio: float = 0.1,
    gui: bool = False,
    params: Optional[Dict[str, Any]] = None,
):
    logger.info("=" * 60)
    logger.info("Co-Scientist Agent (Tool-Use Ver.)")
    logger.info("=" * 60)

    # 로그 콜백 추출 (kids_render.py에서 주입)
    log_callback = params.pop("log_callback", None) if params else None

    def _log(step, msg):
        if log_callback:
            try:
                log_callback(step, msg)
            except Exception:
                pass

    _log("ANALYZE", "입력 이미지를 구조 관점에서 다시 해석하고 있어요.")

    # job_id 추출
    job_id = params.get("job_id", "offline") if params else "offline"

    graph_builder = RegenerationGraph(llm_client, log_callback=log_callback, job_id=job_id)
    app = graph_builder.build()

    system_msg = SystemMessage(content=graph_builder.SYSTEM_PROMPT)

    # DB에서 Memory 로드
    initial_memory = {
        "failed_approaches": [],
        "successful_patterns": [],
        "lessons": [],
        "consecutive_failures": 0
    }
    try:
        model_id = Path(glb_path).name
        loaded_mem = load_memory_from_db(model_id)
        if loaded_mem:
            initial_memory.update(loaded_mem)
    except Exception as e:
        logger.error(f"⚠️ [Memory] 초기 로드 실패: {e}")

    # 파라미터 병합
    merged_params = DEFAULT_PARAMS.copy()
    if params:
        merged_params.update(params)
        logger.info(f"⚙️  Custom Params Applied: {list(params.keys())}")

    initial_state = AgentState(
        glb_path=glb_path,
        ldr_path=output_ldr_path,
        subject_name=subject_name,
        params=merged_params,
        attempts=0,
        session_id=memory_manager.start_session(Path(glb_path).name, "main_agent") if memory_manager else "offline",
        max_retries=max_retries,
        acceptable_failure_ratio=acceptable_failure_ratio,
        verification_duration=2.0,
        gui=gui,
        messages=merged_params.get("override_messages") or [
            system_msg,
            HumanMessage(content=f"'{subject_name}' 모델의 물리적 안정성을 최적화하고 LDR 파일을 설계하세요.")
        ],
        verification_raw_result={},
        floating_bricks_ids=[],
        verification_errors=0,
        tool_usage_count={},
        last_tool_used=None,
        
        # [Best of 3]
        modification_attempts=0,
        best_score=-1,
        best_ldr_content=None,
        hallucination_count=0,
        
        # [Hypothesis]
        observation="",

        # [시스템 컨텍스트]
        job_id=job_id,
        consecutive_same_tool=0,
        previous_metrics={},
        current_metrics={},
        final_report={},
        memory=initial_memory,
        hypothesis_maker=graph_builder.hypothesis_maker,
        round_count=0,
        internal_score=0,
        debate_history=[],
        merged=False,
        next_action="generate"
    )

    # 실행
    _log("GENERATE", "브릭 배치를 미세 조정하고 있어요.")
    # [ASYNC CHANGE] invoke -> ainvoke
    final_state = await app.ainvoke(initial_state, config={"recursion_limit": 100})

    # [NEW] Token Usage Injection (Cost calculation moved to router)
    if llm_client and hasattr(llm_client, "usage"):
        usage = llm_client.usage
        report = final_state.get("final_report", {})
        if "final_metrics" not in report:
            report["final_metrics"] = {}
        
        # Inject Tokens & Model Name
        report["final_metrics"]["token_usage"] = usage
        report["model_name"] = getattr(llm_client, "model_name", "gemini-1.5-flash")
        
        # Update state
        final_state["final_report"] = report
        
    _log("VERIFY", "현 설계가 반복 조립에도 안정적인지 확인 중이에요.")

    # Evolver Post-Processing
    report = final_state.get("final_report", {}) or {}
    final_metrics = report.get("final_metrics", {}) or {}

    # Evolver mode:
    # - auto (default): run only when result is still unstable
    # - always: always run
    # - off: never run
    evolver_mode = str(os.environ.get("COSCIENTIST_EVOLVER_MODE", "auto")).strip().lower()
    if _is_truthy(os.environ.get("COSCIENTIST_DISABLE_EVOLVER")):
        evolver_mode = "off"
    if evolver_mode not in {"auto", "always", "off"}:
        evolver_mode = "auto"

    final_success = bool(report.get("success", False))
    failure_ratio = float(final_metrics.get("failure_ratio", 1.0))
    should_run_evolver = True

    if Path(output_ldr_path).exists():
        file_size = Path(output_ldr_path).stat().st_size
        logger.debug(f"[DEBUG] LDR File exists before Evolver: {output_ldr_path} (Size: {file_size} bytes)")
    else:
        logger.warning(f"[DEBUG] LDR File MISSING before Evolver: {output_ldr_path}")
        should_run_evolver = False

    if should_run_evolver:
        # Pre-Evolver merge pass (optional)
        try:
            from ..merger import merge_small_bricks
            logger.info("[Pre-Processing] Try merging 1x1 bricks before Evolver...")
            merge_stats = merge_small_bricks(output_ldr_path, min_merge_count=2)
            if merge_stats.get("merged", 0) > 0:
                logger.info(
                    "[Pre-Processing] Merged "
                    f"{merge_stats['merged']} groups "
                    f"(Total: {merge_stats['original_count']} -> {merge_stats['new_count']})"
                )
                pass  # SSE 제거 (유저 불필요)
            else:
                logger.info("[Pre-Processing] No mergeable 1x1 groups")
        except Exception as e:
            logger.error(f"[Pre-Processing] Merge failed (continue): {e}")

        logger.info("[Evolver] Running Evolver post-processing...")
        evolver_result = await run_evolver(output_ldr_path, glb_path, log_callback=log_callback)
        if evolver_result.get("success"):
            logger.info("[Evolver] Post-processing completed")
        else:
            reason = evolver_result.get("reason", "unknown")
            logger.warning(f"[Evolver] Skipped/failed: {reason}")
    else:
        skip_reason = f"mode={evolver_mode}, success={final_success}, failure_ratio={failure_ratio:.3f}"
        logger.info(f"[Evolver] Skipped ({skip_reason})")

    # 최종 리포트
    logger.info("=" * 60)
    logger.info("📋 최종 결과 리포트")
    logger.info("=" * 60)

    report = final_state.get('final_report', {})
    if report:
        success = report.get('success', False)
        status = "✅ 성공" if success else "❌ 실패"
        logger.info(f"상태: {status}")
        logger.info(f"총 시도: {report.get('total_attempts', final_state['attempts'])}회")

        tool_usage = report.get('tool_usage', {})
        if tool_usage:
            logger.info("도구 사용 현황:")
            for tool, count in tool_usage.items():
                logger.info(f"  - {tool}: {count}회")

        metrics = report.get('final_metrics', {})
        if metrics:
            logger.info("최종 메트릭:")
            logger.info(f"  - 실패율: {metrics.get('failure_ratio', 0) * 100:.1f}%")
            logger.info(f"  - 1x1 비율: {metrics.get('small_brick_ratio', 0) * 100:.1f}%")
            logger.info(f"  - 총 브릭: {metrics.get('total_bricks', 0)}개")

        logger.info(f"메시지: {report.get('message', '')}")
    else:
        logger.info(f"총 시도: {final_state['attempts']}회")

    logger.info("=" * 60)

    # 세션 피드백 보고서
    if memory_manager:
        try:
            session_id = final_state.get('session_id', '')
            if session_id and session_id != 'offline':
                feedback_report = memory_manager.generate_session_report(session_id)
                if 'error' not in feedback_report:
                    logger.info("📊 [Co-Scientist] 세션 피드백 보고서 생성 완료")
                    logger.info(f"   - 총 반복: {feedback_report.get('statistics', {}).get('total_iterations', 0)}회")
                    logger.info(f"   - 성공률: {feedback_report.get('statistics', {}).get('success_rate', 0)}%")
                    logger.info(f"   - 권장사항: {feedback_report.get('final_recommendation', '')}")
        except Exception as e:
            logger.error(f"⚠️ [Co-Scientist] 보고서 생성 실패: {e}")

    # 학습 데이터 DB 저장
    try:
        model_id = Path(glb_path).name
        mem = final_state.get("memory", {})
        report_data = final_state.get("final_report", {})
        mem["final_report"] = {
            "success": report_data.get("success", False),
            "total_attempts": report_data.get("total_attempts", final_state.get("attempts", 0)),
            "final_metrics": report_data.get("final_metrics", {}),
            # [NEW] Pre-Processing & Evolver Stats
            "pre_process_merge": merge_stats if 'merge_stats' in locals() else None,
            "evolver_result": evolver_result if 'evolver_result' in locals() else None,
        }
        save_memory_to_db(model_id, mem)
    except Exception as e:
        logger.error(f"⚠️ [Memory] 저장 중 오류: {e}")

    # COMPLETE SSE 제거 (kids_render.py의 complete와 중복)

    # [NEW] 초기 모델 경로를 최종 리포트에 포함 (프론트엔드 비교용)
    if final_state.get("initial_ldr_path"):
        if "final_report" not in final_state:
            final_state["final_report"] = {}
        final_state["final_report"]["initial_model_path"] = final_state["initial_ldr_path"]
        logger.info(f"[Pipeline] Initial Model Path added to report: {final_state['initial_ldr_path']}")

    return final_state
