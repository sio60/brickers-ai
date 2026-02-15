# ============================================================================
# 메인 오케스트레이션: regeneration_loop
# ============================================================================

from pathlib import Path
from typing import Dict, Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from ..llm_clients import BaseLLMClient
from ..llm_state import AgentState
from ..memory_utils import memory_manager

from .constants import DEFAULT_PARAMS
from .graph import RegenerationGraph
from .evolver_runner import run_evolver_subprocess


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
            print("  [Memory] MONGODB_URI not set, skip save")
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
        print(f"  [Memory] Saved to DB: {model_id} (lessons={len(doc['lessons'])})")

    except Exception as e:
        print(f"  [Memory] DB save failed: {e}")


# ============================================================================
# 메인 루프
# ============================================================================

async def regeneration_loop(
    glb_path: str,
    output_ldr_path: str,
    subject_name: str = "Unknown Object",
    llm_client: Optional[BaseLLMClient] = None,
    max_retries: int = 5,
    acceptable_failure_ratio: float = 0.1,
    gui: bool = False,
    params: Optional[Dict[str, Any]] = None,
):
    print("=" * 60)
    print("Co-Scientist Agent (Tool-Use Ver.)")
    print("=" * 60)

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
        print(f"⚠️ [Memory] 초기 로드 실패: {e}")

    # 파라미터 병합
    merged_params = DEFAULT_PARAMS.copy()
    if params:
        merged_params.update(params)
        print(f"⚙️  Custom Params Applied: {list(params.keys())}")

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
        messages=[
            system_msg,
            HumanMessage(content=f"'{subject_name}' 모델의 물리적 안정성을 최적화하고 LDR 파일을 설계하세요.")
        ],
        verification_raw_result=None,
        floating_bricks_ids=[],
        verification_errors=0,
        tool_usage_count={},
        last_tool_used=None,
        
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
    final_state = await app.ainvoke(initial_state)

    _log("VERIFY", "현 설계가 반복 조립에도 안정적인지 확인 중이에요.")

    # Evolver Post-Processing
    if Path(output_ldr_path).exists():
        file_size = Path(output_ldr_path).stat().st_size
        print(f"[DEBUG] LDR File exists before Evolver: {output_ldr_path} (Size: {file_size} bytes)")

        _log("EVOLVE", "형태와 효율 사이의 균형을 맞추고 있어요.")
        print("\n[Evolver] 형태 개선 에이전트 실행 중...")
        evolver_result = run_evolver_subprocess(output_ldr_path, glb_path)
        if evolver_result.get("success"):
            print("[Evolver] ✅ 형태 개선 완료")
            _log("EVOLVE", "형태 개선이 완료됐어요. 최종 검토에 들어갈게요.")
        else:
            reason = evolver_result.get("reason", "unknown")
            print(f"[Evolver] ⚠️ 형태 개선 스킵: {reason}")
            _log("EVOLVE", "형태 개선을 건너뛰고 다음 단계로 넘어갈게요.")
    else:
        print(f"[DEBUG] ❌ LDR File MISSING before Evolver: {output_ldr_path}")

    _log("REFLECT", "현재 결과를 기준으로 최종 정리 중이에요.")

    # 최종 리포트
    print("\n" + "=" * 60)
    print("📋 최종 결과 리포트")
    print("=" * 60)

    report = final_state.get('final_report', {})
    if report:
        success = report.get('success', False)
        status = "✅ 성공" if success else "❌ 실패"
        print(f"상태: {status}")
        print(f"총 시도: {report.get('total_attempts', final_state['attempts'])}회")

        tool_usage = report.get('tool_usage', {})
        if tool_usage:
            print(f"도구 사용 현황:")
            for tool, count in tool_usage.items():
                print(f"  - {tool}: {count}회")

        metrics = report.get('final_metrics', {})
        if metrics:
            print(f"최종 메트릭:")
            print(f"  - 실패율: {metrics.get('failure_ratio', 0) * 100:.1f}%")
            print(f"  - 1x1 비율: {metrics.get('small_brick_ratio', 0) * 100:.1f}%")
            print(f"  - 총 브릭: {metrics.get('total_bricks', 0)}개")

        print(f"메시지: {report.get('message', '')}")
    else:
        print(f"총 시도: {final_state['attempts']}회")

    print("=" * 60)

    # 세션 피드백 보고서
    if memory_manager:
        try:
            session_id = final_state.get('session_id', '')
            if session_id and session_id != 'offline':
                feedback_report = memory_manager.generate_session_report(session_id)
                if 'error' not in feedback_report:
                    print("\n📊 [Co-Scientist] 세션 피드백 보고서 생성 완료")
                    print(f"   - 총 반복: {feedback_report.get('statistics', {}).get('total_iterations', 0)}회")
                    print(f"   - 성공률: {feedback_report.get('statistics', {}).get('success_rate', 0)}%")
                    print(f"   - 권장사항: {feedback_report.get('final_recommendation', '')}")
        except Exception as e:
            print(f"⚠️ [Co-Scientist] 보고서 생성 실패: {e}")

    # 학습 데이터 DB 저장
    try:
        model_id = Path(glb_path).name
        mem = final_state.get("memory", {})
        report_data = final_state.get("final_report", {})
        mem["final_report"] = {
            "success": report_data.get("success", False),
            "total_attempts": report_data.get("total_attempts", final_state.get("attempts", 0)),
            "final_metrics": report_data.get("final_metrics", {}),
        }
        save_memory_to_db(model_id, mem)
    except Exception as e:
        print(f"⚠️ [Memory] 저장 중 오류: {e}")

    _log("COMPLETE", "설계가 완료됐어요. 다음 단계로 넘어가도 좋아요.")

    return final_state
