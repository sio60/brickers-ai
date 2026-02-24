"""OBSERVE Node - Analyze current model state using brick_judge (Rust)"""
import logging
from ..state import AgentState
from ..tools import get_model_state, analyze_symmetry
from ..config import get_config

logger = logging.getLogger(__name__)
# from ..path_utils import setup_db_paths, setup_vision_paths, get_evolver_dir
from ..constants import (
    MAX_REMOVAL_RATIO,
    SKIP_SYMMETRY_TYPES
)


def _load_memory_from_db(model_id: str) -> dict | None:
    """MongoDB에서 기존 학습 메모리 로드"""
    try:
        import os
        from pymongo import MongoClient

        mongo_uri = os.getenv("MONGODB_URI")
        if not mongo_uri:
            logger.info("  [Memory] MONGODB_URI not set")
            return None

        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
        db = client["brickers"]
        col = db["evolver_memory"]

        doc = col.find_one({"_id": model_id})
        if doc:
            logger.info("  [Memory] Loaded: %s (lessons=%s)", model_id, len(doc.get('lessons', [])))
            return {
                "failed_approaches": doc.get("failed_approaches", []),
                "successful_patterns": doc.get("successful_patterns", []),
                "lessons": doc.get("lessons", []),
                "consecutive_failures": doc.get("consecutive_failures", 0)
            }
        else:
            logger.info("  [Memory] No previous data for: %s", model_id)
            return None

    except ImportError as e:
        logger.info("  [Memory] Skip load (DB module not found): %s", e)
        return None
    except Exception as e:
        logger.warning("  [Memory] Load failed: %s", e)
        return None


# NOTE: _analyze_symmetry() 중복 코드 제거됨 (tools.py의 analyze_symmetry 사용)

def _setup_vision_paths():
    """path_utils 함수 대체: Vision 출력 경로 생성"""
    import os
    from pathlib import Path
    config = get_config()
    export_dir = config.exporter_dir if config.exporter_dir else Path(os.getcwd()) / "evolver_output"
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir



def node_observe(state: AgentState) -> AgentState:
    """Observe current model state with brick_judge + 대칭 분석"""
    from ..config import log_sse
    log_sse("EVOLVER_OBSERVE", "구조가 안정됐어요. 이제 전체 형태를 살펴볼게요.")
    logger.info("[OBSERVE] Iteration %s", state['iteration'])

    # 첫 iteration에서 MongoDB 메모리 로드
    memory = state.get("memory", {
        "failed_approaches": [],
        "successful_patterns": [],
        "lessons": [],
        "consecutive_failures": 0
    })

    if state['iteration'] == 0:
        # 모델 ID 생성
        model_id = getattr(state["model"], "name", "unknown")
        if model_id == "unknown" and state["model"].bricks:
            model_id = f"model_{state['model'].bricks[0].id[:8]}"

        loaded_memory = _load_memory_from_db(model_id)
        if loaded_memory:
            memory = loaded_memory

    # ========================================
    # Vision 분석 먼저 (model_type 판단 필수)
    # ========================================
    vision_quality_score = state.get("vision_quality_score")
    vision_problems = state.get("vision_problems", [])
    model_type = state.get("model_type", "unknown")

    if state['iteration'] == 0 and vision_quality_score is None:
        logger.info("  [Vision] Running Vision analysis (FIRST - model_type 판단)...")
        try:
            evolver_dir = _setup_vision_paths()

            from ..tools.vision_analyzer import find_problems, analyze_multi_angle
            from ..tools.ldr_renderer import render_model_multi_angle

            VISION_ANGLES = ["FRONT", "BACK", "LEFT", "RIGHT", "BOTTOM"]
            images = render_model_multi_angle(state["model"], get_config().parts_db, angles=VISION_ANGLES)
            if images:
                import base64
                render_dir = evolver_dir / "renders"
                render_dir.mkdir(exist_ok=True)
                for angle, b64_data in images.items():
                    if b64_data:
                        img_path = render_dir / f"{angle.lower()}.png"
                        with open(img_path, "wb") as f:
                            f.write(base64.b64decode(b64_data))
                logger.info("  [Vision] 렌더링 저장됨: %s", render_dir)

                # Vision LLM으로 모델 분석 (model_type 추출) - 가장 먼저!
                model_name = getattr(state["model"], "name", "unknown")
                analysis = analyze_multi_angle(images, model_name)
                model_type = analysis.get("model_type", "unknown")
                logger.info("  [Vision] Model type: %s", model_type)

                # Vision LLM으로 문제점 찾기
                result = find_problems(images, model_name)
                vision_quality_score = result.get("overall_quality", 50)
                all_problems = result.get("problems", [])
                vision_problems = [p for p in all_problems if p.get("severity", "").lower() == "high"]

                logger.info("  [Vision] Quality: %s/100", vision_quality_score)
                logger.info("  [Vision] Problems: %s high-severity (total: %s)", len(vision_problems), len(all_problems))
                for p in vision_problems[:3]:
                    logger.info("    - %s: %s", p.get('location'), p.get('issue'))
            else:
                logger.info("  [Vision] Render failed, skipping")
                vision_quality_score = 50

        except ImportError as e:
            logger.info("  [Vision] Skipped - 렌더러 미설치 (%s), 물리 검증만 진행합니다.", e)
            vision_quality_score = 50
        except Exception as e:
            logger.error("  [Vision] Error: %s", e)
            vision_quality_score = 50
    elif vision_quality_score is not None:
        logger.info("  Vision: %s/100 (%s problems), model_type: %s", vision_quality_score, len(vision_problems), model_type)

    model_state = get_model_state(state["model"], get_config().parts_db)

    logger.info("  Bricks: %s", model_state['total_bricks'])
    logger.info("  Score: %s", model_state.get('score', 'N/A'))
    logger.info("  Valid: %s", model_state.get('is_valid', 'N/A'))
    logger.info("  Floating: %s", model_state['floating_count'])
    logger.info("  Collisions: %s", model_state['collision_count'])
    logger.info("  Removed: %s/%s", state['total_removed'], int(state['original_brick_count'] * MAX_REMOVAL_RATIO))

    # Show evidence from brick_judge
    evidence = model_state.get('evidence', [])
    if evidence:
        logger.info("  Evidence (%s issues):", len(evidence))
        for ev in evidence[:5]:  # Show first 5
            logger.info("    - [%s] %s: %s...", ev.severity.value, ev.issue_type.value, ev.message[:80])

    # Symmetry analysis (비대칭이 자연스러운 모형은 스킵)
    # model_type은 위에서 Vision 분석으로 이미 결정됨
    symmetry_issues = []

    if model_type not in SKIP_SYMMETRY_TYPES:
        symmetry_issues = analyze_symmetry(state["model"], get_config().parts_db)
        if symmetry_issues:
            logger.info("  Symmetry: %s missing bricks detected", len(symmetry_issues))
        else:
            logger.info("  Symmetry: OK")
    else:
        logger.info("  Symmetry: Skipped (%s is naturally asymmetric)", model_type)


    return {
        **state,
        "floating_count": model_state["floating_count"],
        "collision_count": model_state["collision_count"],
        "floating_bricks": model_state["floating_bricks"],
        "verification_result": model_state.get("verification_result"),
        "verification_score": model_state.get("score", 100.0),
        "verification_evidence": evidence,
        "symmetry_issues": symmetry_issues,
        "memory": memory,
        "vision_quality_score": vision_quality_score,
        "vision_problems": vision_problems,
        "model_type": model_type
    }
