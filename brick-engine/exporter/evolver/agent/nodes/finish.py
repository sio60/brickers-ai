"""FINISH Node - Complete and save"""
from ..state import AgentState
from ..path_utils import setup_db_paths


def node_finish(state: AgentState) -> AgentState:
    """Finish the agent run and save memory to MongoDB"""
    print(f"\n[FINISH] {state['finish_reason']}")

    # MongoDB에 학습 결과 저장
    try:
        from datetime import datetime
        import uuid
        import os
        from pymongo import MongoClient

        # MongoDB 직접 연결
        try:
            mongo_uri = os.getenv("MONGODB_URI")
            if not mongo_uri:
                print(f"  [MongoDB] MONGODB_URI not set, skip save")
                return state

            client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
            db = client["brickers"]
            col = db["evolver_memory"]

            # 모델 ID 생성 (모델명 또는 첫 브릭 ID)
            model_id = getattr(state["model"], "name", "unknown")
            if model_id == "unknown" and state["model"].bricks:
                model_id = f"model_{state['model'].bricks[0].id[:8]}"

            # model_type 추출 (state에서 가져오거나 기본값)
            model_type = state.get("model_type", "unknown")

            # session_id 생성 (UUID)
            session_id = str(uuid.uuid4())

            # 저장할 데이터
            memory_doc = {
                "_id": model_id,
                "model_type": model_type,
                "session_id": session_id,
                "failed_approaches": state["memory"]["failed_approaches"],
                "successful_patterns": state["memory"]["successful_patterns"],
                "lessons": state["memory"]["lessons"],
                "consecutive_failures": state["memory"]["consecutive_failures"],
                "total_iterations": state["iteration"],
                "total_removed": state["total_removed"],
                "original_brick_count": state["original_brick_count"],
                "final_floating_count": state["floating_count"],
                "finish_reason": state["finish_reason"],
                "last_updated": datetime.utcnow()
            }

            # Vision 데이터 있으면 추가
            if "vision_quality_score" in state:
                memory_doc["vision_quality_score"] = state["vision_quality_score"]
                memory_doc["vision_problems_count"] = state.get("vision_problems_count", 0)

            # Upsert
            col.update_one(
                {"_id": model_id},
                {"$set": memory_doc},
                upsert=True
            )

            print(f"  [MongoDB] Memory saved: {model_id}")
            print(f"    - Lessons: {len(state['memory']['lessons'])}")
            print(f"    - Success: {len(state['memory']['successful_patterns'])}")
            print(f"    - Failed: {len(state['memory']['failed_approaches'])}")

        except ImportError as e:
            print(f"  [MongoDB] Skip save (DB module not found): {e}")
        except Exception as e:
            print(f"  [MongoDB] Save failed: {e}")

    except Exception as e:
        print(f"  [FINISH] Warning - Memory save error: {e}")

    return state
