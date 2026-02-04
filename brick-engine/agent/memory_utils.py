"""
Co-Scientist 통합 메모리 관리 모듈 (v2)
- 비동기 저장 (Background Thread)
- 재시도 로직 (Retry with Backoff)
- HuggingFace 로컬 임베딩 (Fallback)
- Vector Search 인덱스 체크
"""

import os
import sys
import uuid
import logging
import threading
from queue import Queue
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

# LangSmith Tracing (Optional)
try:
    from langsmith import traceable
except ImportError:
    def traceable(func): return func

# DB Connection
try:
    from yang_db import get_db
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from yang_db import get_db
    except ImportError:
        get_db = None

# Config
try:
    import config
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    import config

# Logging
logger = logging.getLogger("CoScientistMemory")

# ============================================================================
# Background Worker for Async Save
# ============================================================================

class BackgroundSaver:
    """비동기 DB 저장을 위한 백그라운드 워커"""
    
    def __init__(self):
        self.queue: Queue = Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self.running = False
        
    def start(self):
        if self.running:
            return
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        logger.info("🚀 Background saver started")
        
    def stop(self):
        self.running = False
        if self.worker_thread:
            self.queue.put(None)  # Poison pill
            self.worker_thread.join(timeout=5)
            
    def enqueue(self, task: callable):
        """저장 작업을 큐에 추가"""
        if not self.running:
            self.start()
        self.queue.put(task)
        
    def _worker(self):
        while self.running:
            try:
                task = self.queue.get(timeout=1)
                if task is None:
                    break
                task()  # Execute the save task
            except Exception as e:
                if "Empty" not in str(type(e).__name__):
                    logger.error(f"Background save failed: {e}")


# Global background saver instance
_bg_saver = BackgroundSaver()


# ============================================================================
# Helper Functions
# ============================================================================

def calculate_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """metrics_before와 metrics_after의 차이 자동 계산"""
    delta = {}
    for key in before:
        if key in after:
            try:
                if isinstance(before[key], (int, float)) and isinstance(after[key], (int, float)):
                    change = after[key] - before[key]
                    delta[key] = round(change, 4) if isinstance(change, float) else change
            except:
                pass
    return delta


def ensure_not_empty(value: Any, default: Any) -> Any:
    """빈 값이면 기본값 반환"""
    if value is None or value == "" or value == {} or value == []:
        return default
    return value


def build_hypothesis(observation: str, hypothesis: str = None, reasoning: str = None, prediction: str = None) -> Dict[str, str]:
    """표준화된 hypothesis 객체 생성 (빈값 방지)"""
    return {
        "observation": ensure_not_empty(observation, "No observation"),
        "hypothesis": ensure_not_empty(hypothesis, "No explicit hypothesis"),
        "reasoning": ensure_not_empty(reasoning, "Automatic tool selection"),
        "prediction": ensure_not_empty(prediction, "Improvement expected")
    }


def build_experiment(tool: str, parameters: Dict = None, model_name: str = None, duration_sec: float = None) -> Dict[str, Any]:
    """표준화된 experiment 객체 생성 (빈값 방지)"""
    return {
        "tool": ensure_not_empty(tool, "unknown"),
        "parameters": ensure_not_empty(parameters, {}),
        "model_name": ensure_not_empty(model_name, "gemini-2.5-flash"),
        "duration_sec": ensure_not_empty(duration_sec, 0.0)
    }


def build_verification(passed: bool, metrics_before: Dict, metrics_after: Dict, numerical_analysis: str = None) -> Dict[str, Any]:
    """표준화된 verification 객체 생성 (delta 자동 계산)"""
    delta = calculate_delta(metrics_before, metrics_after)
    return {
        "passed": passed,
        "metrics_before": ensure_not_empty(metrics_before, {}),
        "metrics_after": ensure_not_empty(metrics_after, {}),
        "delta": delta,
        "numerical_analysis": ensure_not_empty(numerical_analysis, f"Delta: {delta}")
    }


def build_improvement(lesson_learned: str, next_hypothesis: str = None) -> Dict[str, str]:
    """표준화된 improvement 객체 생성 (빈값 방지)"""
    return {
        "lesson_learned": ensure_not_empty(lesson_learned, "No lesson recorded"),
        "next_hypothesis": ensure_not_empty(next_hypothesis, "Continue current strategy")
    }


# ============================================================================
# Memory Utils Class
# ============================================================================

class MemoryUtils:
    """
    Co-Scientist 통합 메모리 관리 클래스.
    - MongoDB 구조화된 실험 데이터 저장 (Experiments)
    - 세션 및 오케스트레이션 데이터 저장 (Sessions)
    - 실시간 RAG를 위한 Vector Embedding 및 Indexing
    """
    
    def __init__(self):
        self.db = get_db() if get_db else None
        self.collection_exps = self.db["co_scientist_experiments"] if self.db is not None else None
        self.collection_sessions = self.db["co_scientist_sessions"] if self.db is not None else None
        
        # Vector Search 설정
        self.vector_index_name = getattr(config, "ATLAS_VECTOR_INDEX_MEMORY", "co_scientist_memory_index")
        self.use_vector = bool(getattr(config, "MONGODB_URI", "")) and bool(self.vector_index_name)
        self._vector_index_verified = False  # 인덱스 존재 여부 캐시
        
        # HuggingFace 임베딩 모델 (Lazy Loading)
        self._hf_model = None
        self._hf_tokenizer = None
        self._embed_lock = threading.Lock()  # Thread-safe 모델 로딩
        
    def _load_hf_model(self):
        """HuggingFace 임베딩 모델 로드 (Lazy)"""
        if self._hf_model is not None:
            return
            
        with self._embed_lock:
            if self._hf_model is not None:
                return
                
            try:
                from transformers import AutoTokenizer, AutoModel
                model_name = getattr(config, "HF_EMBED_MODEL", "intfloat/multilingual-e5-small")
                logger.info(f"📦 Loading HF embedding model: {model_name}")
                self._hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._hf_model = AutoModel.from_pretrained(model_name)
                logger.info("✅ HF embedding model loaded")
            except Exception as e:
                logger.error(f"Failed to load HF model: {e}")
        
    def _get_embedding(self, text: str, max_retries: int = 2) -> List[float]:
        """텍스트를 벡터로 변환 (HuggingFace 우선, Gemini Fallback)"""
        if not text:
            return []
        
        # 1차: HuggingFace 로컬 모델 (무료, 오프라인 가능)
        try:
            self._load_hf_model()
            if self._hf_model and self._hf_tokenizer:
                import torch
                inputs = self._hf_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = self._hf_model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
                return embedding
        except Exception as e:
            logger.warning(f"HF embedding failed: {e}")
        
        # Fallback: Gemini API (재시도 로직 포함)
        for attempt in range(max_retries):
            try:
                if getattr(config, "GEMINI_API_KEY", ""):
                    import google.generativeai as genai
                    genai.configure(api_key=config.GEMINI_API_KEY)
                    result = genai.embed_content(
                        model="models/text-embedding-004",
                        content=text,
                        task_type="retrieval_document"
                    )
                    return result['embedding']
            except Exception as e:
                logger.warning(f"Gemini embedding attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(0.5 * (attempt + 1))
            
        return []

    def _format_context_for_embedding(self, observation: str, verification: Dict[str, Any] = None) -> str:
        """[신규] 임베딩용 문맥 포맷팅 (정확도 향상용)"""
        context_parts = []
        
        if verification:
            # 1. 실패 유형 명시
            if not verification.get("stable", True):
                context_parts.append("Status: Unstable")
            elif verification.get("floating_bricks_count", 0) > 0:
                context_parts.append(f"Status: Floating Bricks ({verification.get('floating_bricks_count')} bricks)")
            
            # 2. 핵심 수치 정보
            if "small_brick_ratio" in verification:
                ratio = float(verification["small_brick_ratio"])
                context_parts.append(f"SmallBrickRatio: {ratio:.2f}")
                
        # 3. 관찰 텍스트
        context_parts.append(f"Observation: {observation}")
        
        return " | ".join(context_parts)

    def _verify_vector_index(self) -> bool:
        """Vector Search 인덱스 존재 여부 확인 (캐시됨)"""
        if self._vector_index_verified:
            return True
            
        if not self.collection_exps:
            return False
            
        try:
            # 간단한 테스트 쿼리로 인덱스 확인
            test_pipeline = [
                {"$vectorSearch": {
                    "index": self.vector_index_name,
                    "path": "embedding",
                    "queryVector": [0.0] * 384,  # 더미 벡터
                    "numCandidates": 1,
                    "limit": 1
                }}
            ]   
            list(self.collection_exps.aggregate(test_pipeline))
            self._vector_index_verified = True
            logger.info(f"✅ Vector index '{self.vector_index_name}' verified")
            return True
        except Exception as e:
            if "index not found" in str(e).lower() or "Atlas" in str(e):
                logger.warning(f"⚠️ Vector index '{self.vector_index_name}' not found. RAG disabled.")
            else:
                logger.warning(f"Vector index check failed: {e}")
            return False

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
        async_save: bool = True,  # 비동기 저장 옵션
    ) -> str:
        """단일 실험 결과를 구조화하여 저장 (비동기 가능)"""
        if self.collection_exps is None:
            logger.warning("DB connection not available. Skipping log.")
            return ""

        exp_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        # Document 구조 생성
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
            "result_success": verification.get("passed", False)
        }
        
        # RAG 검색용 텍스트 (Observation + 상세 메트릭 임베딩)
        # "상황 유사도" 정확도 향상을 위해 구조화된 포맷 사용
        search_text = self._format_context_for_embedding(hypothesis.get('observation', ''), verification)
        
        # LLM에게 보여줄 전체 요약
        summary_text = (
            f"Observation: {hypothesis.get('observation', '')} "
            f"Hypothesis: {hypothesis.get('hypothesis', '')} "
            f"Action: {experiment.get('tool', '')} "
            f"Result: {verification.get('numerical_analysis', '')} "
            f"Lesson: {improvement.get('lesson_learned', '')}"
        )
        
        def _do_save():
            """실제 저장 작업 (임베딩 포함)"""
            nonlocal doc
            try:
                # Vector Embedding (Dual Strategy)
                if self.use_vector:
                    # 1. Main Embedding (Full Context): 나중에 분석용
                    doc["embedding"] = self._get_embedding(summary_text)
                    
                    # 2. Scoring Embedding (Observation Only): 검색/점수산정용
                    # "상황이 얼마나 비슷한가?"를 판단하기 위해 사용
                    doc["observation_embedding"] = self._get_embedding(search_text)
                    
                    doc["summary_text"] = summary_text
                    doc["search_text"] = search_text 
                    
                # DB Insert
                self.collection_exps.insert_one(doc)
                logger.info(f"✅ Experiment logged: {exp_id}")
            except Exception as e:
                logger.error(f"Failed to log experiment: {e}")
        
        # 비동기 또는 동기 저장
        if async_save:
            _bg_saver.enqueue(_do_save)
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
    def search_similar_cases(self, observation: str, limit: int = 10, min_score: float = 0.5, verification_metrics: Dict[str, Any] = None) -> List[Dict]:
        """RAG: 유사 후보군 검색 (Re-ranking을 위해 넉넉히 검색)"""
        if not self.use_vector or self.collection_exps is None:
            return []
            
        # 인덱스 확인 (첫 호출 시만)
        if not self._verify_vector_index():
            return []
            
        # 검색용 텍스트 생성 (메트릭 포함)
        query_text = self._format_context_for_embedding(observation, verification_metrics)
        query_vector = self._get_embedding(query_text)
        if not query_vector:
            return []
            
        # 1. 점수 기반 필터링 검색 (Observation 기준)
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.vector_index_name,
                    "path": "observation_embedding",  # [변경] 상황 유사도 기준 검색
                    "queryVector": query_vector,
                    "numCandidates": limit * 10,
                    "limit": limit * 2
                }
            },
            {
                "$addFields": {
                    "similarity_score": {"$meta": "vectorSearchScore"}
                }
            },
            {
                "$match": {
                    "similarity_score": {"$gte": min_score}
                }
            },
            {
                "$limit": limit
            },
            {
                "$project": {
                    "_id": 0,
                    # embedding, observation_embedding은 결과 분석을 위해 포함
                    "session_id": 0,
                }
            }
        ]
        
        try:
            results = list(self.collection_exps.aggregate(pipeline))
            
            # 2. 결과가 없으면 Fallback (점수 무시하고 가장 유사한 것 1개 검색)
            if not results:
                logger.info(f"🔍 No matches with score >= {min_score}. Trying fallback...")
                fallback_pipeline = [
                    {
                        "$vectorSearch": {
                            "index": self.vector_index_name,
                            "path": "observation_embedding",
                            "queryVector": query_vector,
                            "numCandidates": 10,
                            "limit": 1
                        }
                    },
                    {
                        "$project": {
                            "_id": 0,
                            # embedding 포함
                            "session_id": 0,
                        }
                    }
                ]
                results = list(self.collection_exps.aggregate(fallback_pipeline))
                if results:
                    logger.info("🔍 Fallback successful: Found 1 similar case.")
            else:
                logger.info(f"🔍 Found {len(results)} similar cases (score >= {min_score})")
                
            return results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def generate_session_report(self, session_id: str) -> Dict[str, Any]:
        """
        세션의 모든 실험을 분석하여 피드백 보고서 생성
        
        Returns:
            Dict containing:
            - model_id: 모델 파일명
            - total_iterations: 총 반복 횟수
            - successful_count: 성공한 실험 수
            - failed_count: 실패한 실험 수
            - success_rate: 성공률
            - tools_used: 사용된 도구 통계
            - key_lessons: 핵심 교훈
            - timeline: 실험 타임라인
            - final_recommendation: 최종 권장사항
        """
        if self.collection_exps is None:
            return {"error": "Database not connected"}
        
        try:
            # 세션의 모든 실험 조회
            experiments = list(self.collection_exps.find(
                {"session_id": session_id}
            ).sort("iteration", 1))
            
            if not experiments:
                return {"error": "No experiments found for session"}
            
            # 기본 정보
            model_id = experiments[0].get("model_id", "unknown")
            agent_type = experiments[0].get("agent_type", "unknown")
            
            # 통계 계산
            total = len(experiments)
            successful = sum(1 for e in experiments if e.get("result_success", False))
            failed = total - successful
            
            # 도구 사용 통계
            tools_used = {}
            for exp in experiments:
                tool = exp.get("experiment", {}).get("tool", "unknown")
                if tool not in tools_used:
                    tools_used[tool] = {"count": 0, "success": 0, "fail": 0}
                tools_used[tool]["count"] += 1
                if exp.get("result_success"):
                    tools_used[tool]["success"] += 1
                else:
                    tools_used[tool]["fail"] += 1
            
            # 도구별 성공률 계산
            for tool in tools_used:
                stats = tools_used[tool]
                stats["success_rate"] = round(stats["success"] / stats["count"] * 100, 1) if stats["count"] > 0 else 0
            
            # 핵심 교훈 추출
            lessons = []
            for exp in experiments:
                lesson = exp.get("improvement", {}).get("lesson_learned", "")
                if lesson and lesson not in lessons:
                    lessons.append(lesson)
            
            # 타임라인 (간략화)
            timeline = []
            for exp in experiments:
                timeline.append({
                    "iteration": exp.get("iteration", 0),
                    "tool": exp.get("experiment", {}).get("tool", "unknown"),
                    "success": exp.get("result_success", False),
                    "analysis": exp.get("verification", {}).get("numerical_analysis", ""),
                    "delta": exp.get("verification", {}).get("delta", {})
                })
            
            # 초기/최종 상태 비교 (detailed_metrics)
            first_exp = experiments[0]
            last_exp = experiments[-1]
            initial_metrics = first_exp.get("verification", {}).get("metrics_before", {})
            final_metrics = last_exp.get("verification", {}).get("metrics_after", {})
            
            improvement_by_metric = {}
            for key in initial_metrics:
                if key in final_metrics:
                    try:
                        start_val = initial_metrics[key]
                        end_val = final_metrics[key]
                        if isinstance(start_val, (int, float)) and isinstance(end_val, (int, float)):
                            delta = end_val - start_val
                            pct = round((start_val - end_val) / start_val * 100, 1) if start_val != 0 else 0
                            improvement_by_metric[key] = {
                                "start": start_val,
                                "end": end_val,
                                "delta": delta,
                                "improvement_pct": pct  # 양수 = 개선
                            }
                    except:
                        pass
            
            # 성공/실패 패턴 분석
            tool_sequence = [t["tool"] for t in timeline]
            successful_sequences = []
            failed_sequences = []
            
            # 연속된 도구 조합 패턴 추출
            for i in range(len(timeline) - 1):
                pair = [timeline[i]["tool"], timeline[i+1]["tool"]]
                if timeline[i+1]["success"]:
                    if pair not in successful_sequences:
                        successful_sequences.append(pair)
                else:
                    if pair not in failed_sequences:
                        failed_sequences.append(pair)
            
            # 메트릭별 최적 도구 분석
            best_tool_by_metric = {}
            for metric in improvement_by_metric:
                best_delta = 0
                best_tool_for_metric = "none"
                for exp in experiments:
                    exp_delta = exp.get("verification", {}).get("delta", {}).get(metric, 0)
                    if exp_delta < best_delta:  # 감소가 개선 (floating 등)
                        best_delta = exp_delta
                        best_tool_for_metric = exp.get("experiment", {}).get("tool", "unknown")
                if best_tool_for_metric != "none":
                    best_tool_by_metric[metric] = best_tool_for_metric
            
            # 최종 권장사항 생성
            best_tool = max(tools_used.items(), key=lambda x: x[1]["success_rate"])[0] if tools_used else "none"
            worst_tool = min(tools_used.items(), key=lambda x: x[1]["success_rate"])[0] if tools_used else "none"
            
            recommendation = f"가장 효과적인 도구: {best_tool} ({tools_used.get(best_tool, {}).get('success_rate', 0)}% 성공률)"
            if worst_tool != best_tool:
                recommendation += f" | 피해야 할 도구: {worst_tool} ({tools_used.get(worst_tool, {}).get('success_rate', 0)}% 성공률)"
            
            # RAG용 임베딩 요약 생성
            success_pct = round(successful / total * 100) if total > 0 else 0
            tools_summary = " ".join([f"{t}:{s['success']}/{s['count']}" for t, s in tools_used.items()])
            embedding_summary = f"{model_id} {agent_type} iter={total} success={success_pct}% {tools_summary}"
            
            # 보고서 생성
            report = {
                "session_id": session_id,
                "model_id": model_id,
                "agent_type": agent_type,
                "generated_at": datetime.utcnow().isoformat(),
                "statistics": {
                    "total_iterations": total,
                    "successful_count": successful,
                    "failed_count": failed,
                    "success_rate": round(successful / total * 100, 1) if total > 0 else 0
                },
                "detailed_metrics": {
                    "initial_state": initial_metrics,
                    "final_state": final_metrics,
                    "improvement_by_metric": improvement_by_metric
                },
                "tools_analysis": tools_used,
                "patterns": {
                    "tool_sequence": tool_sequence,
                    "successful_sequences": successful_sequences[-3:],  # 최근 3개
                    "failed_sequences": failed_sequences[-3:],
                    "best_tool_by_metric": best_tool_by_metric
                },
                "key_lessons": lessons[-5:],  # 최근 5개
                "timeline": timeline,
                "final_recommendation": recommendation,
                "embedding_summary": embedding_summary
            }
            
            # DB에 보고서 저장 (sessions 컬렉션 업데이트)
            if self.collection_sessions:
                self.collection_sessions.update_one(
                    {"_id": session_id},
                    {"$set": {
                        "report": report,
                        "status": "COMPLETED",
                        "end_time": datetime.utcnow().isoformat()
                    }}
                )
            
            logger.info(f"📊 Session report generated: {model_id} - {successful}/{total} success")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate session report: {e}")
            return {"error": str(e)}


# Singleton Instance
memory_manager = MemoryUtils()
