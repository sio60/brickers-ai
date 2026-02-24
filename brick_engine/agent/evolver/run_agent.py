"""
Evolver Agent Runner
Usage: python run_agent.py <ldr_path>
"""
import os
import sys
import copy
import json
from pathlib import Path
from dotenv import load_dotenv

# Setup paths
EVOLVER_DIR = Path(__file__).parent  # agent/evolver
AGENT_DIR = EVOLVER_DIR.parent       # agent
PROJECT_ROOT = AGENT_DIR.parent      # brick_engine
EXPORTER_DIR = PROJECT_ROOT / "exporter"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPORTER_DIR))

load_dotenv(PROJECT_ROOT / ".env")

from ldr_converter import ldr_to_brick_model, model_to_ldr  # noqa: E402
from evolver_agent import build_graph, AgentState  # noqa: E402
from evolver_agent.tools import get_model_state, analyze_glb  # noqa: E402
from evolver_agent.config import init_config  # noqa: E402

# Memory Utils Import (agent.memory 참조)
from agent.memory.manager import memory_manager
from agent.memory.builders import build_hypothesis, build_experiment, build_verification, build_improvement

import logging
logger = logging.getLogger(__name__)

def run_agent(ldr_path: str, glb_path: str = None):
    # ... (print headers) ...
    logger.info("=" * 60)
    logger.info("LangGraph + CoScientist Evolver Agent")
    logger.info("=" * 60)

    # Initialize - parts_db 로드
    parts_db = {}
    cache = EXPORTER_DIR / "data" / "parts_cache.json"
    if not cache.exists():
        # Fallback for legacy path
        cache = EXPORTER_DIR / "parts_cache.json"
        
    if cache.exists():
        with open(cache, 'r', encoding='utf-8') as f:
            parts_db = json.load(f)

    if not parts_db:
        logger.error("ERROR: parts_cache.json not found in %s!", cache)
        sys.exit(1)

    init_config(parts_db, EXPORTER_DIR)

    model = ldr_to_brick_model(ldr_path)
    model.name = Path(ldr_path).stem

    # 색상 분포 디버깅 출력
    from collections import Counter
    color_dist = Counter(b.color_code for b in model.bricks)
    logger.debug("[COLOR DEBUG] 입력 LDR 색상 분포:")
    for color, count in color_dist.most_common():
        logger.debug("  color_code=%s: %s개", color, count)
    if len(color_dist) == 1 and 15 in color_dist:
        logger.warning("  ⚠️ 모든 브릭이 흰색(15)! → 입력 LDR에 이미 색상 정보 없음")
        logger.warning("  → GLB→LDR 변환 시 색상 추출 실패 가능성 높음")

    # GLB 분석 (있으면)
    glb_ref = None
    if glb_path and Path(glb_path).exists():
        logger.info("[GLB Reference] Analyzing %s...", glb_path)
        glb_ref = analyze_glb(glb_path)
        if glb_ref.get("available"):
            logger.info("  Model: %s (%s)", glb_ref.get('name'), glb_ref.get('model_type'))
            logger.info("  Legs: %s", glb_ref.get('legs'))
            logger.info("  Features: %s", glb_ref.get('key_features'))

    # Session ID 생성
    session_id = "offline_evolver"
    if memory_manager:
        session_id = memory_manager.start_session(model.name, "evolver")

    initial_state: AgentState = {
        "model": model,
        "model_backup": copy.deepcopy(model),
        # parts_db는 config에서 가져옴 (7MB 데이터 state에서 제거)
        "original_brick_count": len(model.bricks),
        "glb_reference": glb_ref,
        "floating_count": 0,
        "collision_count": 0,
        "floating_bricks": [],
        "verification_result": None,
        "verification_score": 100.0,
        "verification_evidence": [],
        # Vision & Symmetry
        "vision_quality_score": None,
        "vision_problems": [],
        "symmetry_issues": [],
        "model_type": "unknown",
        # Tracking
        "iteration": 0,
        "session_id": session_id,
        "total_removed": 0,
        "action_history": [],
        "strategy": "",
        "target_brick": None,
        "proposals": [],
        "selected_proposal": None,
        "memory": {
            "failed_approaches": [],
            "successful_patterns": [],
            "lessons": [],
            "consecutive_failures": 0
        },
        "should_finish": False,
        "finish_reason": "",
        "messages": []  # LangSmith 트레이싱용 대화 기록
    }

    logger.info("Loaded: %s", ldr_path)
    logger.info("Bricks: %s", initial_state['original_brick_count'])
    logger.info("Removal limit: %s (10%%)", int(initial_state['original_brick_count'] * 0.10))

    # Run graph
    logger.info("[실행중...] 에이전트 시작")
    graph = build_graph()
    logger.info("[실행중...] LLM 호출 대기중...")

    # Checkpointer 사용 시 thread_id 필요
    import uuid
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    final_state = graph.invoke(initial_state, config=config)
    logger.info("[완료] 에이전트 종료")

    # Save result
    output = Path(ldr_path).parent / f"{Path(ldr_path).stem}_evolved.ldr"
    ldr = model_to_ldr(final_state["model"], parts_db, skip_validation=True, skip_physics=True, step_mode='layer')
    with open(output, 'w', encoding='utf-8') as f:
        f.write(ldr)

    # 에이전트 실행 후 색상 분포 출력
    after_color_dist = Counter(b.color_code for b in final_state["model"].bricks)
    logger.debug("[COLOR DEBUG] 에이전트 실행 후 색상 분포:")
    for color, count in after_color_dist.most_common():
        logger.debug("  color_code=%s: %s개", color, count)

    # Print summary
    initial_floating = get_model_state(initial_state["model_backup"], parts_db)["floating_count"]

    logger.info("=" * 60)
    logger.info("RESULT")
    logger.info("=" * 60)
    logger.info("Floating: %s -> %s", initial_floating, final_state['floating_count'])
    logger.info("Removed: %s", final_state['total_removed'])
    logger.info("Iterations: %s", final_state['iteration'])
    logger.info("Reason: %s", final_state['finish_reason'])
    logger.info("Saved: %s", output)

    # Log Success if fixed
    if final_state['floating_count'] == 0 and final_state['finish_reason'] == "All fixed!":
        success_lesson = "SUCCESS: Model is valid. No floating bricks or collisions found."
        logger.info("[Lessons Learned]\n  - %s", success_lesson)
        
        # Log to DB
        if memory_manager:
            try:
                # Log final success experiment
                memory_manager.log_experiment(
                    session_id=session_id,
                    model_id=model.name,
                    agent_type="evolver",
                    iteration=final_state["iteration"],
                    hypothesis=build_hypothesis(
                        observation=f"Final State: floating={final_state['floating_count']}",
                        hypothesis="Verification of final state",
                        reasoning="All checks passed",
                        prediction="Valid model"
                    ),
                    experiment=build_experiment(
                        tool="finish",
                        parameters={},
                        model_name="rules"
                    ),
                    verification=build_verification(
                        passed=True,
                        metrics_before={},
                        metrics_after={"floating": 0, "collisions": 0},
                        numerical_analysis="Validation Success"
                    ),
                    improvement=build_improvement(
                        lesson_learned=success_lesson,
                        next_hypothesis="Task Complete"
                    )
                )
            except Exception as e:
                logger.warning("⚠️ [Run Agent] Log failed: %s", e)

    logger.info("[Lessons Learned History]")
    for lesson in final_state["memory"]["lessons"]:
        logger.info("  - %s", lesson)

    logger.info("=" * 60)

def run_evolver_direct(ldr_path: str, glb_path: str = None, log_callback=None) -> dict:
    """직접 호출용 (subprocess 없이). pipeline.py에서 사용.

    Returns:
        {"success": True} or {"success": False, "reason": "..."}
    """
    try:
        # parts_db 로드
        parts_db = {}
        cache = EXPORTER_DIR / "parts_cache.json"
        if cache.exists():
            with open(cache, 'r', encoding='utf-8') as f:
                parts_db = json.load(f)

        if not parts_db:
            return {"success": False, "reason": "parts_cache.json not found"}

        init_config(parts_db, EXPORTER_DIR, log_callback=log_callback)

        model = ldr_to_brick_model(ldr_path)
        model.name = Path(ldr_path).stem

        # GLB 분석
        glb_ref = None
        if glb_path and Path(glb_path).exists():
            glb_ref = analyze_glb(glb_path)

        # Session
        session_id = "offline_evolver"
        if memory_manager:
            session_id = memory_manager.start_session(model.name, "evolver")

        initial_state: AgentState = {
            "model": model,
            "model_backup": copy.deepcopy(model),
            "original_brick_count": len(model.bricks),
            "glb_reference": glb_ref,
            "floating_count": 0,
            "collision_count": 0,
            "floating_bricks": [],
            "verification_result": None,
            "verification_score": 100.0,
            "verification_evidence": [],
            "vision_quality_score": None,
            "vision_problems": [],
            "symmetry_issues": [],
            "model_type": "unknown",
            "iteration": 0,
            "session_id": session_id,
            "total_removed": 0,
            "action_history": [],
            "strategy": "",
            "target_brick": None,
            "proposals": [],
            "selected_proposal": None,
            "memory": {
                "failed_approaches": [],
                "successful_patterns": [],
                "lessons": [],
                "consecutive_failures": 0
            },
            "should_finish": False,
            "finish_reason": "",
            "messages": []
        }

        import uuid
        graph = build_graph()
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        final_state = graph.invoke(initial_state, config=config)

        # Save evolved LDR
        output = Path(ldr_path).parent / f"{Path(ldr_path).stem}_evolved.ldr"
        ldr = model_to_ldr(final_state["model"], parts_db, skip_validation=True, skip_physics=True, step_mode='layer')
        with open(output, 'w', encoding='utf-8') as f:
            f.write(ldr)

        logger.info("[Evolver Direct] Saved: %s", output)
        logger.info("[Evolver Direct] Floating: %s, Removed: %s", final_state.get('floating_count', '?'), final_state.get('total_removed', 0))

        return {"success": True}

    except Exception as e:
        logger.error("[Evolver Direct] Exception occurred: %s", e, exc_info=True)
        return {"success": False, "reason": str(e)}


def print_mermaid():
    """LangGraph 그래프를 Mermaid 코드로 출력"""
    graph = build_graph()
    mermaid = graph.get_graph().draw_mermaid()
    logger.info("%s", mermaid)
    logger.info("https://mermaid.live 에 붙여넣기")


if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    if len(sys.argv) >= 2 and sys.argv[1] == "--graph":
        print_mermaid()
        sys.exit(0)

    if len(sys.argv) < 2:
        print("Usage: python run_agent.py <ldr_path> [glb_path]")
        print("       python run_agent.py --graph  (Mermaid 그래프 출력)")
        sys.exit(1)

    ldr = sys.argv[1]
    if not Path(ldr).exists():
        print(f"Error: {ldr} not found")
        sys.exit(1)

    glb = sys.argv[2] if len(sys.argv) > 2 else None
    run_agent(ldr, glb)
