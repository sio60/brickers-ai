# ============================================================================
# MemoryUtils: 통합 메모리 관리 클래스 (Coordinator)
# ============================================================================

import os
import sys
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np

# LangSmith Tracing (Optional)
try:
    from langsmith import traceable
except ImportError:
    def traceable(**kwargs):
        def decorator(func):
            return func
        return decorator

# DB Connection (yang_db.py는 agent/ 디렉토리에 위치)
try:
    from yang_db import get_db
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))  # agent/
    try:
        from yang_db import get_db
    except ImportError:
        get_db = None

# Config (config.py는 프로젝트 루트에 위치)
try:
    import config
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))  # brickers-ai/
    try:
        import config
    except ImportError:
        config = None

from .background import bg_saver
from .embeddings import get_embedding
from .search import (
    format_context_for_embedding,
    search_similar_cases as _search_similar_cases,
    search_success_and_failure as _search_success_and_failure,
)
from .reporting import generate_session_report as _generate_session_report

logger = logging.getLogger("CoScientistMemory")


class MemoryUtils:
    """
    Co-Scientist 통합 메모리 관리 클래스.
    - MongoDB 구조화된 실험 데이터 저장
    - 세션 관리
    - RAG를 위한 Vector Search
    """

    def __init__(self):
        self.db = get_db() if get_db else None
        self.collection_exps = self.db["co_scientist_experiments"] if self.db is not None else None
        self.collection_sessions = self.db["co_scientist_sessions"] if self.db is not None else None

        # Vector Search 설정
        self.vector_index_name = getattr(config, "ATLAS_VECTOR_INDEX_MEMORY", "co_scientist_memory_index") if config else "co_scientist_memory_index"
        self.use_vector = bool(getattr(config, "MONGODB_URI", "") if config else "") and bool(self.vector_index_name)

    def calculate_model_metrics(self, plan: Any, verification_result: Any = None) -> Dict[str, Any]:
        """LDR 모델의 물리적 특성 추출"""
        metrics = {
            "total_bricks": 0,
            "total_volume": 0.0,
            "bounding_box": {"min": [0,0,0], "max": [0,0,0]},
            "dimensions": {"width": 0.0, "height": 0.0, "depth": 0.0},
            "aspect_ratio": 1.0,
            "collision_count": 0,
            "floating_count": 0,
        }

        bricks = plan.get_all_bricks() if hasattr(plan, 'get_all_bricks') else []
        metrics["total_bricks"] = len(bricks)

        if bricks:
            origins = []
            total_vol = 0.0
            for b in bricks:
                if b.origin:
                    origins.append(b.origin)
                if hasattr(b, 'volume'):
                    total_vol += b.volume
                else:
                    total_vol += 1.0

            metrics["total_volume"] = total_vol

            if origins:
                arr = np.array(origins)
                min_xyz = np.min(arr, axis=0).tolist()
                max_xyz = np.max(arr, axis=0).tolist()
                metrics["bounding_box"] = {"min": min_xyz, "max": max_xyz}

                dx = abs(max_xyz[0] - min_xyz[0])
                dy = abs(max_xyz[1] - min_xyz[1])
                dz = abs(max_xyz[2] - min_xyz[2])
                metrics["dimensions"] = {"width": dx, "height": dy, "depth": dz}
                metrics["aspect_ratio"] = dx / dz if dz > 0 else 1.0

        if verification_result:
            metrics["floating_count"] = getattr(verification_result, 'floating_bricks', 0)
            evidences = getattr(verification_result, 'evidence', [])
            metrics["collision_count"] = len([ev for ev in evidences if getattr(ev, 'type', '') == 'COLLISION'])

        return metrics

    @traceable(name="MemoryUtils.search_success_and_failure")
    def search_success_and_failure(
        self,
        observation: str,
        limit: int = 5,
        min_score: float = 0.5,
        verification_metrics: Dict[str, Any] = None,
        shape_metrics: Dict[str, Any] = None,
        subject_name: str = None,
    ) -> Dict[str, List[Dict]]:
        """성공/실패 사례를 구분하여 검색"""
        return _search_success_and_failure(
            self.collection_exps, self.vector_index_name, self.use_vector,
            observation, limit, min_score, verification_metrics, shape_metrics,
            subject_name=subject_name
        )

    @traceable(name="MemoryUtils.log_experiment")
    def log_experiment(
        self,
        session_id: str,
        model_id: str,
        agent_type: str,
        iteration: int,
        hypothesis: Dict[str, str],
        experiment: Dict[str, Any],
        verification: Dict[str, Any],
        improvement: Dict[str, Any],
        async_save: bool = True,
    ) -> str:
        """단일 실험 결과를 구조화하여 저장"""
        if self.collection_exps is None:
            logger.warning("DB connection not available. Skipping log.")
            return ""

        exp_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        # State Diff Calculation
        metrics_before = verification.get("metrics_before", {})
        metrics_after = verification.get("metrics_after", {})
        state_diff = {}
        for k, v in metrics_after.items():
            if k in metrics_before and isinstance(v, (int, float)) and isinstance(metrics_before[k], (int, float)):
                state_diff[k] = v - metrics_before[k]

        doc = {
            "_id": exp_id,
            "session_id": session_id,
            "model_id": model_id,
            "agent_type": agent_type,
            "iteration": iteration,
            "timestamp": timestamp,
            "hypothesis": hypothesis,
            "experiment": experiment,
            "verification": verification,
            "improvement": improvement,
            "result_success": verification.get("passed", False),
            "algorithm": experiment.get("tool", "unknown"),
            "state_before": metrics_before,
            "state_after": metrics_after,
            "state_diff": state_diff,
            "shape_metrics": experiment.get("shape_metrics", {}),
        }

        search_text = format_context_for_embedding(hypothesis.get('observation', ''), verification)

        summary_text = (
            f"Observation: {hypothesis.get('observation', '')} "
            f"Algorithm: {experiment.get('tool', '')} "
            f"Params: {experiment.get('parameters', {})} "
            f"Result: {'Success' if doc['result_success'] else 'Failure'} "
            f"Metrics Diff: {state_diff} "
            f"Lesson: {improvement.get('lesson_learned', '')}"
        )

        collection = self.collection_exps
        use_vector = self.use_vector

        def _do_save():
            nonlocal doc
            try:
                if use_vector:
                    doc["embedding"] = get_embedding(summary_text)
                    doc["observation_embedding"] = get_embedding(search_text)
                    doc["summary_text"] = summary_text
                    doc["search_text"] = search_text

                collection.insert_one(doc)
                logger.info(f"Experiment logged: {exp_id}")
            except Exception as e:
                logger.error(f"Failed to log experiment: {e}")

        if async_save:
            bg_saver.enqueue(_do_save)
        else:
            _do_save()

        return exp_id

    def start_session(self, model_id: str, agent_type: str) -> str:
        """새로운 세션 시작"""
        session_id = str(uuid.uuid4())

        if self.collection_sessions is None:
            return session_id

        doc = {
            "_id": session_id,
            "model_id": model_id,
            "agent_type": agent_type,
            "start_time": datetime.utcnow().isoformat(),
            "status": "RUNNING"
        }

        try:
            self.collection_sessions.insert_one(doc)
        except Exception as e:
            logger.error(f"Failed to start session: {e}")

        return session_id

    def end_session(self, session_id: str, final_status: str, summary: Dict[str, Any]):
        """세션 종료 처리"""
        if self.collection_sessions is None:
            return

        try:
            self.collection_sessions.update_one(
                {"_id": session_id},
                {"$set": {
                    "end_time": datetime.utcnow().isoformat(),
                    "status": final_status,
                    "summary": summary
                }}
            )
        except Exception as e:
            logger.error(f"Failed to end session: {e}")

    @traceable(name="MemoryUtils.search_similar_cases")
    def search_similar_cases(
        self,
        observation: str,
        limit: int = 10,
        min_score: float = 0.5,
        verification_metrics: Dict[str, Any] = None,
        subject_name: str = None,
    ) -> List[Dict]:
        """RAG: 유사 후보군 검색"""
        return _search_similar_cases(
            self.collection_exps, self.vector_index_name, self.use_vector,
            observation, limit, min_score, verification_metrics, subject_name,
        )

    def generate_session_report(self, session_id: str) -> Dict[str, Any]:
        """세션 피드백 보고서 생성"""
        return _generate_session_report(
            self.collection_exps, self.collection_sessions, session_id,
        )


# Singleton Instance
memory_manager = MemoryUtils()
