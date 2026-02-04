# ============================================================================
# LLM 재생성 에이전트 (Tool Calling & History 기반)
# GLB → LDR 변환 후 물리 검증 실패 시 LLM이 '도구'를 사용해 해결책을 제시하는 시스템
#
# 아키텍처 (LangGraph):
# 1. Generator Node: GLB 변환 (TuneParameters 도구 결과 반영)
# 2. Verifier Node: 물리 검증 및 결과 피드백 생성 (History에 추가)
# 3. Model Node: LLM이 History를 보고 도구 선택 (TuneParameters / FixFloatingBricks)
# 4. Tool Node: 선택된 도구 실행
# ============================================================================

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Literal, TypedDict, Union
from dataclasses import dataclass, field, asdict
import json

# LangGraph & LangChain imports
try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
    from typing import Annotated
except ImportError:
    print("❌ LangGraph 또는 LangChain이 설치되지 않았습니다. 'pip install langgraph langchain-core'를 실행하세요.")
    sys.exit(1)

# 모듈 경로 설정
_THIS_DIR = Path(__file__).resolve().parent
_BRICK_ENGINE_DIR = _THIS_DIR.parent
_PROJECT_ROOT = _BRICK_ENGINE_DIR.parent
_PHYSICAL_VERIFICATION_DIR = _PROJECT_ROOT / "physical_verification"

for p in (_THIS_DIR, _BRICK_ENGINE_DIR, _PROJECT_ROOT, _PHYSICAL_VERIFICATION_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


# LLM 클라이언트 & 도구 임포트
try:
    from .llm_clients import BaseLLMClient, GroqClient, GeminiClient
    from .agent_tools import TuneParameters, FixFloatingBricks, MergeBricks
    from .memory_utils import memory_manager, build_hypothesis, build_experiment, build_verification, build_improvement
except ImportError:
    from llm_clients import BaseLLMClient, GroqClient, GeminiClient
    from agent_tools import TuneParameters, FixFloatingBricks, MergeBricks
    from memory_utils import memory_manager, build_hypothesis, build_experiment, build_verification, build_improvement

# DB 연결
try:
    from yang_db import get_db
except ImportError:
    print("⚠️ yang_db.py를 찾을 수 없습니다. Memory 영속화 기능이 비활성화됩니다.")
    get_db = None

# ============================================================================
# Memory & DB Helper Functions
# ============================================================================

try:
    from memory_utils import memory_manager, build_hypothesis, build_experiment, build_verification, build_improvement
except ImportError:
    # 경로 문제 시 현재 폴더 추가 후 재시도
    sys.path.append(str(_THIS_DIR))
    try:
        from memory_utils import memory_manager, build_hypothesis, build_experiment, build_verification, build_improvement
    except ImportError:
        print("⚠️ memory_utils.py를 찾을 수 없습니다.")
        memory_manager = None
        build_hypothesis = build_experiment = build_verification = build_improvement = None

# Legacy functions (하위 호환성을 위해 남겨두거나 삭제 가능)
def get_memory_collection(): return memory_manager.collection_exps if memory_manager else None
def load_memory_from_db(model_id: str): return {} # Legacy 로드 비활성화 (RAG로 대체)
def save_memory_to_db(model_id: str, memory: Dict): pass # Legacy 저장 비활성화


# ============================================================================
# 기본 파라미터 정의
# ============================================================================

DEFAULT_PARAMS = {
    "target": 60,              # 목표 스터드 크기 (400 브릭 기준 60 정도가 적절)
    "min_target": 5,           # 최소 스터드 크기
    "budget": 400,             # 최대 브릭 수 (Kids L1 기준)
    "shrink": 0.85,            # 축소 비율 (0.85)
    "search_iters": 6,         # 이진 탐색 반복 횟수
    "flipx180": False,         # X축 180도 회전
    "flipy180": False,         # Y축 180도 회전
    "flipz180": False,         # Z축 180도 회전
    "kind": "brick",           # 브릭 종류 (brick/plate)
    "plates_per_voxel": 3,     # 복셀당 플레이트 수
    "interlock": True,         # 인터락 활성화
    "max_area": 20,            # 최대 영역
    "solid_color": 4,          # 단색 색상 ID
    "use_mesh_color": True,    # 메시 색상 사용
    "invert_y": False,         # Y축 반전
    "smart_fix": True,         # 스마트 보정 활성화
    # 추가 파라미터 (Legacy Match)
    "span": 4,
    "max_new_voxels": 12000,
    "refine_iters": 8,
    "ensure_connected": True,
    "min_embed": 2,
    "erosion_iters": 1,        # 노이즈 제거
    "fast_search": True,
    "extend_catalog": True,
    "max_len": 8,
    "fill": True,              # 내부 채움 활성화 (안정적 모델 생성)
    "step_order": "bottomup",  # 조립 순서
    "auto_remove_1x1": True,   # 기본값: 안전하게 1x1 삭제
    "support_ratio": 0.3,      # 기본 지지 비율 복구
    "small_side_contact": True, # 작은 브릭 사이드 접촉 허용
}


# ============================================================================
# 데이터 구조 및 헬퍼 함수
# ============================================================================

@dataclass
class VerificationFeedback:
    """PyBullet 검증 결과를 LLM에게 전달하기 위한 구조화된 피드백"""
    stable: bool = True
    total_bricks: int = 0
    fallen_bricks: int = 0
    floating_bricks: int = 0
    floating_brick_ids: List[str] = field(default_factory=list)  # 공중부양 브릭 ID 목록
    fallen_brick_ids: List[str] = field(default_factory=list)    # 떨어진 브릭 ID 목록
    failure_ratio: float = 0.0
    first_failure_brick: Optional[str] = None
    max_drift: float = 0.0
    collision_count: int = 0
    # 1x1 브릭 비율 정보 (MergeBricks 판단용)
    small_brick_count: int = 0      # 1x1 브릭 개수
    small_brick_ratio: float = 0.0  # 1x1 브릭 비율 (0.0 ~ 1.0)

def extract_verification_feedback(result, total_bricks: int) -> VerificationFeedback:
    """PyBullet VerificationResult를 LLM 피드백 형식으로 변환"""
    feedback = VerificationFeedback()
    feedback.total_bricks = total_bricks
    feedback.stable = result.is_valid
    
    fallen_bricks = set()
    floating_bricks = set()
    first_failure = None
    collision_count = 0
    
    for ev in result.evidence:
        if ev.type == "FIRST_FAILURE":
            if ev.brick_ids:
                first_failure = ev.brick_ids[0]
                fallen_bricks.update(ev.brick_ids)
        elif ev.type == "COLLAPSE_AFTERMATH":
            if ev.brick_ids:
                fallen_bricks.update(ev.brick_ids)
        elif ev.type == "FLOATING_BRICK":
            if ev.brick_ids:
                floating_bricks.update(ev.brick_ids)
        elif ev.type == "COLLISION":
            collision_count += 1
    
    feedback.fallen_bricks = len(fallen_bricks)
    feedback.floating_bricks = len(floating_bricks)
    feedback.floating_brick_ids = list(floating_bricks)  # ID 목록 저장
    feedback.fallen_brick_ids = list(fallen_bricks)      # ID 목록 저장
    feedback.first_failure_brick = first_failure
    feedback.collision_count = collision_count
    
    if total_bricks > 0:
        feedback.failure_ratio = (len(fallen_bricks) + len(floating_bricks)) / total_bricks
    
    return feedback

def _format_feedback(feedback: VerificationFeedback) -> str:
    # 상태 판정 로직 강화
    if feedback.stable and feedback.floating_bricks == 0:
        status = "✅ 안정"
    elif feedback.stable and feedback.floating_bricks > 0:
        status = "⚠️ 부분 안정 (공중부양 존재)"
    else:
        status = "❌ 불안정"
        
    lines = [
        f"검증 결과:",
        f"- 상태: {status}",
        f"- 총 브릭 수: {feedback.total_bricks}개",
    ]
    
    # 1x1 브릭 비율 정보 (MergeBricks 판단용)
    if feedback.small_brick_count > 0:
        lines.append(f"- 1x1 브릭: {feedback.small_brick_count}개 ({feedback.small_brick_ratio * 100:.1f}%)")
        if feedback.small_brick_ratio > 0.3:  # 30% 이상이면 권장
            lines.append(f"  → 💡 1x1 브릭 비율이 높습니다. MergeBricks로 연결 강화를 권장합니다.")
    
    # 상세 불안정 정보 (공중부양 포함)
    if not feedback.stable or feedback.floating_bricks > 0:
        lines.extend([
            f"- 떨어진 브릭: {feedback.fallen_bricks}개",
            f"- 공중부양 브릭: {feedback.floating_bricks}개",
            f"- 실패율: {feedback.failure_ratio * 100:.1f}%",
        ])
        if feedback.first_failure_brick:
            lines.append(f"- 최초 붕괴 브릭: {feedback.first_failure_brick}")
        # LLM이 FixFloatingBricks 사용 시 명확히 알 수 있도록 ID 목록 제공
        if feedback.floating_brick_ids:
            lines.append(f"- 공중부양 브릭 ID 목록: {feedback.floating_brick_ids}")
        if feedback.fallen_brick_ids:
            lines.append(f"- 떨어진 브릭 ID 목록: {feedback.fallen_brick_ids}")
            
    if feedback.collision_count > 0:
        lines.append(f"- 충돌 감지: {feedback.collision_count}건")
    
    return "\n".join(lines)


# ============================================================================
# LangGraph State 정의
# ============================================================================

class AgentState(TypedDict):
    # 입력 및 설정
    glb_path: str
    ldr_path: str
    params: Dict[str, Any]
    max_retries: int
    acceptable_failure_ratio: float
    verification_duration: float
    gui: bool
    
    # 실행 상태
    attempts: int
    session_id: str # 오케스트레이션용 세션 ID
    messages: Annotated[List[BaseMessage], add_messages] # 대화 기록 (History)
    
    # 검증 결과 캐시 (Tool 실행 시 참조용)
    verification_raw_result: Any 
    floating_bricks_ids: List[str] # 공중부양 브릭 ID 목록 캐시
    verification_errors: int  # 검증 에러 재시도 카운터

    # 무한 루프 방지용 도구 사용 추적
    tool_usage_count: Dict[str, int]  # {"TuneParameters": 2, "MergeBricks": 1, ...}
    last_tool_used: Optional[str]     # 마지막 사용 도구
    consecutive_same_tool: int        # 같은 도구 연속 사용 횟수
    
    # 도구 효과 측정용 상태 저장
    previous_metrics: Dict[str, Any]  # 도구 실행 전 메트릭
    current_metrics: Dict[str, Any]   # 검증 후 현재 메트릭 (Reflect에서 비교용)
    
    # 최종 결과 리포트
    final_report: Dict[str, Any]  # 최종 결과 요약
    
    # Co-Scientist Memory (학습 메모리)
    memory: Dict[str, Any]  # {
    #     "failed_approaches": ["MergeBricks on layer 5 failed"],
    #     "successful_patterns": ["FixFloatingBricks after MergeBricks"],
    #     "lessons": ["1x1 비율 30% 이상일 때 MergeBricks 효과적"],
    #     "consecutive_failures": 0
    # }

    # [v2] Co-Scientist 아키텍처 추가 필드
    current_hypothesis: Optional[Dict[str, Any]]  # node_hypothesize 결과
    strategy_plan: Optional[Dict[str, Any]]       # node_strategy 결과
    llm_config: Optional[Dict[str, str]]          # {"model": "gpt-4o"}

    # 다음 노드 제어
    next_action: Literal["generate", "verify", "model", "tool", "reflect", "hypothesize", "strategy", "end"]


# ============================================================================
# LangGraph Agent Logic
# ============================================================================

class RegenerationGraph:
    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        # 기본 클라이언트는 Gemini (비용 효율성)
        self.gemini_client = GeminiClient()
        self.default_client = llm_client if llm_client else self.gemini_client
        
        # [Rollback] GPT Client는 현재 사용하지 않음 (User Request)
        self.gpt_client = None
            
        # 초기 시스템 프롬프트 (Tool 사용 권장)
        self.SYSTEM_PROMPT = """당신은 레고 브릭 구조물 설계 및 안정화 전문가(Co-Scientist)입니다.
주어진 3D 모델(GLB)을 레고(LDR)로 변환하는 과정에서 발생하는 구조적 불안정성 문제를 해결해야 합니다.

당신에게는 세 가지 도구가 있습니다:
1. `TuneParameters`: 전체적인 구조적 결함(와르르 무너짐, 연결 없음 등)을 해결하기 위해 변환 파라미터를 조정하여 처음부터 다시 생성합니다.
2. `FixFloatingBricks`: 전체적으로는 괜찮지만 일부 공중부양하거나 불안정한 브릭이 있을 때, 해당 브릭을 *삭제*하여 정리합니다.
3. `MergeBricks`: 같은 색상의 인접한 1x1 브릭들을 큰 브릭(1x2~1x8)으로 병합합니다. 연결이 강화되어 안정성이 향상됩니다. 색상이 다른 브릭은 병합되지 않습니다.

**의사결정 알고리즘 (Decision Logic):**
1. **1x1 브릭 처리 전략 (Smart 1x1 Strategy)**:
   - **기본 상태**: 안전을 위해 `auto_remove_1x1=True` (삭제)로 시작합니다.
   - **디테일 복구**: 만약 눈, 코, 입 등 중요 디테일이 사라졌다면 `TuneParameters`를 호출하여 `auto_remove_1x1=False`로 변경하세요.
   - **조건부 유지**: `auto_remove_1x1=False`로 할 경우, 반드시 `MergeBricks` 사용을 염두에 두어야 합니다. (유지 후 합치기)

2. **실패율(Failure Ratio) 확인**:
   - **20% 미만 (Low Risk)**: 전체 구조는 튼튼합니다. `TuneParameters`로 다시 만들면 오히려 더 나쁜 결과가 나올 위험이 큽니다.
   - **20% ~ 50% (Medium Risk)**: 상황을 판단하세요. 중요 부위가 무너졌다면 재생성, 외곽만 무너졌다면 삭제.
   - **50% 이상 (High Risk)**: 현재 파라미터로는 불가능합니다. `TuneParameters`로 설정을 변경하여 다시 시도하세요.

2. **MergeBricks vs FixFloatingBricks 선택 기준** (실패율 < 20%일 때):
   - **공중부양/떨어진 브릭 ID가 명확히 있으면** → `FixFloatingBricks`로 해당 브릭 삭제
   - **1x1 브릭이 많아 연결이 약하다는 징후가 있으면** → `MergeBricks`로 보강 (1x1들을 큰 브릭으로 통합)
   - **둘 다 해당되면** → 먼저 `MergeBricks`로 보강 → 재검증 후 필요시 `FixFloatingBricks`

목표: 물리적으로 안정적(Stable)인 레고 구조물을 만드는 것.
이전 시도의 실패 원인과 통계(실패율, 부동 브릭 수)를 분석하고, 위 논리에 따라 가장 합리적인 도구를 선택하세요.
"""

        self.verifier = None
        
    # --- Nodes ---

    def _rerank_and_filter_cases(self, observation: str, cases: List[Dict]) -> List[Dict]:
        """[신규] LLM 기반 RAG Re-ranking (Semantic Scoring)"""
        if not cases:
            return []
            
        print(f"  🔍 Re-ranking: {len(cases)}개 후보 분석 중...")
        
        # 후보군 텍스트 변환
        candidates_text = ""
        for i, case in enumerate(cases):
            candidates_text += f"""
[Case {i}]
- Observation: {case['hypothesis'].get('observation', '')[:200]}...
- Action: {case['experiment'].get('tool', '')}
- Result: {case['verification'].get('numerical_analysis', '')}
--------------------------------------------------"""

        prompt = f"""
현재 상황(Current Observation)과 가장 전략적으로 유사한 과거 사례를 선별하세요.
단순 키워드 매칭이 아니라, '실패/성공 원인'과 '구조적 문제'가 유사한지 분석해야 합니다.

현재 상황:
{observation}

후보 사례 목록:
{candidates_text}

분석 후 가장 참고 가치가 높은 Top 3 사례를 다음 JSON 포맷으로 선정하세요:
{{
    "top_cases": [
        {{
            "case_index": 0,
            "relevance_score": 0.95, (0.0~1.0)
            "reason": "현재 상황(~한 문제)과 동일한 실패 패턴을 보임"
        }},
        ...
    ]
}}
"""
        try:
            response = self.default_client.generate_json(prompt)
            top_cases = response.get("top_cases", [])
            
            # 인덱스로 원본 찾아서 반환 (점수순 정렬)
            reranked_results = []
            for item in sorted(top_cases, key=lambda x: x.get('relevance_score', 0), reverse=True):
                idx = item.get("case_index")
                if 0 <= idx < len(cases):
                    case = cases[idx]
                    # 메타데이터에 Re-ranking 점수와 이유 추가
                    case['_rerank_score'] = item.get('relevance_score')
                    case['_rerank_reason'] = item.get('reason')
                    reranked_results.append(case)
            
            print(f"  ✨ Re-ranking 완료: Top {len(reranked_results)} 선정 (Max Score: {reranked_results[0]['_rerank_score'] if reranked_results else 0})")
            return reranked_results
            
        except Exception as e:
            print(f"  ⚠️ Re-ranking 실패 (Fallback to raw vector rank): {e}")
            return cases[:3]  # 실패 시 그냥 벡터 상위 3개 반환

    def node_hypothesize(self, state: AgentState) -> Dict[str, Any]:
        """[신규] 가설 생성 노드: RAG 검색 및 구체적 가설 수립"""
        print("\n[Hypothesize] 가설 수립 및 RAG 검색 중...")
        
        # 1. RAG 검색
        current_observation = ""
        last_msg = state['messages'][-1]
        if isinstance(last_msg, HumanMessage):
            current_observation = str(last_msg.content)[:500]
            
        similar_cases = []
        if memory_manager:
            # 1. 넓은 범위 검색 (Top 10) - 메트릭 정보 포함하여 검색 정확도 향상
            verification_metrics = state.get("verification_result")
            raw_cases = memory_manager.search_similar_cases(
                current_observation, 
                limit=10, 
                min_score=0.5,
                verification_metrics=verification_metrics
            )
            # 2. LLM Re-ranking (Top 3 선별)
            similar_cases = self._rerank_and_filter_cases(current_observation, raw_cases)
            print(f"  📚 유사 실패 사례 {len(similar_cases)}건 선정 (Re-ranked)")
            
        # 2. 가설 생성 (Gemini Fast 사용)
        rag_context = ""
        for case in similar_cases:
            rag_context += f"- {case.get('model_id')}: {case['experiment'].get('tool')} 사용 -> {case['verification'].get('numerical_analysis')}\n"
            
        prompt = f"""
현재 상황:
{current_observation}

유사 과거 사례:
{rag_context}

위 상황을 분석하여 다음 JSON 형식으로 가설을 수립하세요:
{{
    "observation": "현재 문제 상황 요약 (1문장)",
    "hypothesis": "구체적인 해결 가설 (어떤 도구가 왜 효과적일지)",
    "reasoning": "가설의 근거 (과거 사례 또는 논리적 추론)",
    "difficulty": "Easy|Medium|Hard" (문제 난이도 평가)
}}
"""
        try:
            # 가설 생성은 빠른 Gemini 사용
            response = self.gemini_client.generate_json(prompt)
            print(f"  💭 가설: {response.get('hypothesis')}")
            print(f"  📊 난이도: {response.get('difficulty')}")
            
            return {
                "current_hypothesis": response,
                "next_action": "strategy"
            }
        except Exception as e:
            print(f"  ⚠️ 가설 생성 실패: {e}")
            # 실패 시 기본 가설로 진행
            return {
                "current_hypothesis": {"observation": "분석 실패", "difficulty": "Medium"},
                "next_action": "strategy"
            }

    def node_strategy(self, state: AgentState) -> Dict[str, Any]:
        """[신규] 전략 결정 노드: 난이도에 따른 LLM 모델 선택"""
        hypothesis = state.get("current_hypothesis", {})
        difficulty = hypothesis.get("difficulty", "Medium")
        
        # [Rollback] GPT 사용 안 함 -> 무조건 Gemini 선택
        model_selection = "gemini-2.5-flash"
        
        if difficulty == "Hard":
             reason = "난이도 높음 (Hard) - Gemini 집중 모드 권장"
        elif difficulty == "Easy":
            reason = "난이도 낮음 (Easy)"
        else:
            reason = "일반 난이도"
                
        print(f"\n[Strategy] 전략 결정: {model_selection} ({reason})")
        
        return {
            "llm_config": {"model": model_selection},
            "strategy_plan": {"selected_model": model_selection, "reason": reason},
            "next_action": "model"
        }

    def node_generator(self, state: AgentState) -> Dict[str, Any]:
        """GLB -> LDR 변환 노드"""
        from glb_to_ldr_embedded import convert_glb_to_ldr
        
        print(f"\n[Generator] 변환 시도 {state['attempts'] + 1}/{state['max_retries']}")
        print(f"  Params: target={state['params'].get('target')}, budget={state['params'].get('budget')}")
        
        try:
            # 고수준 API 호출 (내부적으로 budget-finding 루프 포함)
            result = convert_glb_to_ldr(
                state['glb_path'],
                state['ldr_path'],
                **state['params']
            )
            
            brick_count = result.get('parts', 0)
            final_target = result.get('final_target', 0)
            
            print(f"  ✅ 변환 완료: {brick_count}개 브릭 (Final Target: {final_target})")
            return {"attempts": state['attempts'] + 1, "next_action": "verify"}
            
        except Exception as e:
            print(f"  ❌ 변환 실패: {e}")
            # 변환 자체가 실패하면 에러 메시지를 history에 추가하고 Model에게 도움 요청
            error_msg = f"변환 중 치명적 오류 발생: {e}. 파라미터를 크게 변경해야 합니다."
            return {
                "attempts": state['attempts'] + 1,
                "messages": [HumanMessage(content=error_msg)],
                "next_action": "model"
            }

    def node_verifier(self, state: AgentState) -> Dict[str, Any]:
        """물리 검증 노드"""
        from physical_verification.pybullet_verifier import PyBulletVerifier
        from physical_verification.ldr_loader import LdrLoader
        
        print("\n[Verifier] 물리 검증 수행 중...")
        
        if not os.path.exists(state['ldr_path']):
            return {"messages": [HumanMessage(content="LDR 파일이 생성되지 않았습니다.")], "next_action": "model"}
            
        try:
            loader = LdrLoader()
            plan = loader.load_from_file(state['ldr_path'])
            total_bricks = len(plan.bricks)
            
            # 1x1 브릭 비율 계산 (MergeBricks 판단용)
            small_brick_parts = {"3005.dat", "3024.dat"}  # 1x1 브릭, 1x1 플레이트
            small_brick_count = 0
            for b in plan.bricks:
                # 브릭 객체의 part_id 속성 안전하게 접근
                part_id = getattr(b, 'part_id', None) or (b.get('part') if isinstance(b, dict) else None)
                if part_id in small_brick_parts:
                    small_brick_count += 1
            small_brick_ratio = small_brick_count / total_bricks if total_bricks > 0 else 0.0
            
            # 이전 verifier가 있으면 세션 닫기 (PyBullet 상태 충돌 방지)
            if self.verifier is not None:
                try:
                    self.verifier._close_simulation()
                except:
                    pass
            
            # 항상 새 verifier 생성 (LDR 파일 수정 후에도 깨끗한 상태 유지)
            verifier = PyBulletVerifier(plan, gui=state['gui'])
            self.verifier = verifier
            
            stab_result = verifier.run_stability_check(duration=state['verification_duration'], auto_close=False)
            
            feedback = extract_verification_feedback(stab_result, total_bricks)
            # 1x1 브릭 비율 정보 추가
            feedback.small_brick_count = small_brick_count
            feedback.small_brick_ratio = small_brick_ratio
            
            feedback_text = _format_feedback(feedback)
            
            # 상태 메시지 결정
            if feedback.stable and feedback.floating_bricks == 0:
                short_status = "✅ 안정"
            elif feedback.stable and feedback.floating_bricks > 0:
                short_status = "⚠️ 부분 안정 (공중부양 존재)"
            else:
                short_status = "❌ 불안정"
            
            print(f"  결과: {short_status}")
            
            # 불안정하거나 부분 안정이면 상세 내용 출력 (디버깅용)
            if not feedback.stable or feedback.floating_bricks > 0:
                 summary_text = feedback_text.replace('\n', ', ').replace('\r', '')
                 if len(summary_text) > 200:
                     summary_text = summary_text[:200] + "..."
                 print(f"  요약: {summary_text}")
            
            # 공중부양 브릭 ID 캐싱 (Tool에서 사용)
            floating_ids = []
            for ev in stab_result.evidence:
                if ev.type == "FLOATING_BRICK" and ev.brick_ids:
                    floating_ids.extend(ev.brick_ids)
            
            # 현재 메트릭 저장 (도구 효과 측정용)
            current_metrics = {
                "failure_ratio": feedback.failure_ratio,
                "small_brick_ratio": small_brick_ratio,
                "small_brick_count": small_brick_count,
                "total_bricks": total_bricks,
                "floating_count": feedback.floating_bricks,
                "fallen_count": feedback.fallen_bricks,
            }
            
            # 성공 판정: 
            # 1. 물리적으로 안정적이거나 실패율이 허용치 이내여야 함
            is_physically_okay = feedback.stable or (feedback.failure_ratio <= state['acceptable_failure_ratio'])
            # 2. 단, 공중부양(Floating) 브릭은 절대 없어야 함 (Zero Tolerance)
            # 3. 예산(Budget) 초과 체크
            budget = state['params'].get('budget', 500)
            is_over_budget = total_bricks > budget
            
            is_success = is_physically_okay and (feedback.floating_bricks == 0) and (not is_over_budget)
            
            if is_success and not feedback.stable:
                 print(f"  (참고: 불안정 판정이나 실패율 {feedback.failure_ratio*100:.1f}%가 허용치 이내이며 공중부양 없음 -> 성공 간주)")
            
            if is_success:
                print("🎉 목표 달성! 프로세스를 종료합니다.")
                # 최종 리포트 생성
                final_report = {
                    "success": True,
                    "total_attempts": state['attempts'],
                    "tool_usage": state.get('tool_usage_count', {}),
                    "final_metrics": current_metrics,
                    "message": "안정적인 구조물 생성 완료"
                }
                return {"next_action": "end", "final_report": final_report}
            
            if state['attempts'] >= state['max_retries']:
                print("💥 최대 시도 횟수 초과.")
                # 최종 리포트 생성 (실패)
                final_report = {
                    "success": False,
                    "total_attempts": state['attempts'],
                    "tool_usage": state.get('tool_usage_count', {}),
                    "final_metrics": current_metrics,
                    "message": "최대 시도 횟수 초과로 종료"
                }
                return {"next_action": "end", "final_report": final_report}

            # 결과를 Reflect 노드로 전달 (실제 결과 분석용)
            # 주의: previous_metrics는 Reflect에서 비교 후 업데이트해야 하므로 여기선 건드리지 않음
            
            # LLM에게 전달할 메시지 보강
            custom_feedback = feedback_text
            
            if is_over_budget:
                custom_feedback += f"\n\n🚨 **중요: 예산 초과! 현재 {total_bricks}개 브릭입니다. 목표 예산은 {budget}개입니다. `TuneParameters` 도구를 사용하여 `target` 값을 줄여야 합니다.**"

            elif feedback.floating_bricks > 0:
                custom_feedback += "\n\n⚠️ **중요: 아직 공중부양(Floating) 브릭이 남아있습니다. 이 상태로는 절대 작업을 완료할 수 없습니다. 반드시 FixFloatingBricks 도구를 사용하거나 파라미터를 조정하여 해결하세요.**"
            
            return {
                "verification_raw_result": stab_result,
                "floating_bricks_ids": floating_ids,
                "messages": [HumanMessage(content=custom_feedback)],
                "current_metrics": current_metrics,   # Reflect에서 실제 결과 분석용
                "next_action": "reflect"  # Verify 후 Reflect로 이동
            }
            
        except Exception as e:
            print(f"  ❌ 검증 중 에러: {e}")
            # 검증 에러 시 LLM에게 맡기지 않고 재시도 (FixFloatingBricks 결과 보존)
            verification_errors = state.get('verification_errors', 0) + 1
            if verification_errors >= 3:
                # 3회 이상 실패 시 재생성으로 전환
                print(f"  ⚠️ 검증 에러 {verification_errors}회 - 재생성으로 전환합니다.")
                return {
                    "messages": [HumanMessage(content=f"검증 시스템 에러가 반복됨: {e}")],
                    "verification_errors": 0,
                    "next_action": "model"
                }
            else:
                # 재시도
                print(f"  🔄 검증 재시도 ({verification_errors}/3)...")
                import time
                time.sleep(1)  # PyBullet 안정화 대기
                return {"verification_errors": verification_errors, "next_action": "verifier"}

    def node_model(self, state: AgentState) -> Dict[str, Any]:
        """LLM이 상황을 분석하고 도구를 선택하는 노드"""
        import time
        # API Rate Limit (429) 방지를 위한 짧은 딜레이 (특히 Free Tier 사용 시)
        time.sleep(2) 
        
        print("\n[Co-Scientist] 상황 분석 중...")
        
        # 사용 가능한 도구 정의
        tools = [TuneParameters, FixFloatingBricks, MergeBricks]
    
        # --- [전략 가이드 주입] ---
        # 실패율이 낮으면 FixFloatingBricks를 권장하는 힌트 메시지 추가 (강제 X)
        messages_to_send = state['messages'][:]
        
        # --- [Memory 정보 주입 (RAG)] ---
        # Vector Search를 통해 현재 상황과 가장 유사한 과거 사례를 검색
        
        # 현재 관찰 요약 (검색 쿼리용)
        last_human_msg = next((m for m in reversed(messages_to_send) if isinstance(m, HumanMessage)), None)
        current_observation = last_human_msg.content if last_human_msg else ""
        
        if memory_manager:
            # 1. 넓은 범위 검색 (Top 10) - 메트릭 포함
            verification_metrics = state.get("verification_result")
            raw_cases = memory_manager.search_similar_cases(
                current_observation, 
                limit=10, 
                min_score=0.5,
                verification_metrics=verification_metrics
            )
            # 2. LLM Re-ranking (Top 3 선별)
            similar_cases = self._rerank_and_filter_cases(current_observation, raw_cases)
            
            if similar_cases:
                memory_info = "\n**📚 유사한 과거 실험 사례 (RAG):**\n"
                for i, case in enumerate(similar_cases, 1):
                    # RAG 검색 결과 포맷팅
                    tool = case['experiment'].get('tool', 'Unknown')
                    result = case['verification'].get('numerical_analysis', 'N/A')
                    lesson = case['improvement'].get('lesson_learned', 'No lesson')
                    outcome = "성공" if case.get('result_success') else "실패"
                    
                    memory_info += f"[{i}] {outcome} 사례 (도구: {tool})\n"
                    memory_info += f"    결과: {result}\n"
                    memory_info += f"    교훈: {lesson}\n"
                
                memory_info += "\n위 사례를 참고하여 성공 확률이 높은 전략을 수립하세요.\n"
                messages_to_send.append(SystemMessage(content=memory_info))
                print(f"  📚 RAG 검색 결과 {len(similar_cases)}건 주입됨")
        
        # Legacy Memory (Fallback)
        memory = state.get('memory', {})
        lessons = memory.get('lessons', [])
        failed_approaches = memory.get('failed_approaches', [])
        
        if lessons or failed_approaches:
            memory_info = "\n**📚 이전 경험 (Memory):**\n"
            if lessons:
                memory_info += "- 최근 교훈: " + "; ".join(lessons[-3:]) + "\n"
            if failed_approaches:
                memory_info += "- 피해야 할 접근법: " + "; ".join(failed_approaches[-3:]) + "\n"
            
            memory_msg = SystemMessage(content=memory_info)
            messages_to_send.append(memory_msg)
            print(f"  📚 Memory 정보 {len(lessons)}개 교훈 전달됨")
        
        # 직전 검증 결과 확인
        last_msg = messages_to_send[-1]
        
        if isinstance(last_msg, HumanMessage) and "검증 결과" in str(last_msg.content):
            content = str(last_msg.content)
            floating_ids = state.get('floating_bricks_ids', [])
            
            # 실패율 파싱 (간이)
            ratio = 0.0
            import re
            match = re.search(r"실패율: ([\d.]+)%", content)
            if match:
                ratio = float(match.group(1))

            # 상황별 전략 힌트 (Strategy Hint) - 논리적 설득 강화
            hints = []
            
            # 1. 공중부양 브릭이 있는 경우 (가장 중요)
            if floating_ids:
                print(f"  💡 [Strategy Hint] 공중부양 브릭({len(floating_ids)}개) 감지 -> 정밀 수리 권장")
                advice = f"""**⚠️ 상황 분석 및 도구 추천:**
현재 공중부양 브릭이 {len(floating_ids)}개 발견되었습니다. (ID: {floating_ids})
전체 실패율은 {ratio}%입니다.

**논리적 판단 가이드:**
1. **실패율이 낮음 (<20%)**: 전체적인 구조는 튼튼합니다.
2. **국소적 문제**: 문제는 일부 브릭의 뜸(Floating) 현상뿐입니다.
3. **도구 선택**:
   - `TuneParameters`: 전체를 다시 만듭니다. 현재의 튼튼한 구조를 잃을 위험이 있습니다. (비권장 ❌)
   - `FixFloatingBricks`: 문제 있는 브릭만 정확히 제거합니다. 현재 구조를 유지하며 안정성을 확보합니다. (강력 권장 ✅)
   
따라서 **FixFloatingBricks**를 사용하여 해결하세요."""
                hints.append(advice)
            
            # 2. 실패율이 낮은 경우
            elif ratio > 0 and ratio < 20.0:
                print(f"  💡 [Strategy Hint] 낮은 실패율({ratio}%) 감지 -> 부분 수정 권장")
                hints.append(f"현재 실패율이 {ratio}%로 매우 낮습니다. 구조가 거의 완성되었으므로 무분별한 재생성보다는 정밀한 도구 사용을 고려하세요.")
            
            if hints:
                hint_msg = SystemMessage(content="\n".join(hints))
                messages_to_send.append(hint_msg)

        # 모델 바인딩 및 호출
        try:
            # [Rollback] 무조건 Gemini 사용
            client_to_use = self.gemini_client
            print(f"  🤖 Active Model: Gemini-2.5-Flash (Fixed)")
            
            model_with_tools = client_to_use.bind_tools(tools)
            response = model_with_tools.invoke(messages_to_send)
            
            # 응답 확인
            if response.tool_calls:
                print(f"  🔨 도구 선택: {[tc['name'] for tc in response.tool_calls]}")
                return {"messages": [response], "next_action": "tool"}
            else:
                # 도구를 선택하지 않은 경우 (끝났다고 판단)
                print(f"  💭 LLM 의견: {response.content}")
                
                # 실제 성공 여부 재확인
                current_metrics = state.get('current_metrics', {})
                floating_count = current_metrics.get('floating_count', 0)
                failure_ratio = current_metrics.get('failure_ratio', 0)
                
                # 공중부양이 없고 실패율이 낮으면 종료 허용
                if floating_count == 0 and failure_ratio <= state['acceptable_failure_ratio']:
                    print("🎉 모든 조건 충족. 종료합니다.")
                    return {"messages": [response], "next_action": "end"}
                else:
                    # 아직 문제가 남았는데 종료하려고 하면 힌트를 주고 재시도
                    print(f"⚠️ 경고: 문제가 남았는데({floating_count}개 공중부양) 종료 시도함. 재지시 중...")
                    error_feedback = f"아직 완료되지 않았습니다. {floating_count}개의 공중부양 브릭이 남아있습니다. 'FixFloatingBricks' 도구를 사용하거나 파라미터를 교체하여 모든 브릭이 연결되도록 하세요."
                    hint = HumanMessage(content=error_feedback)
                    return {"messages": [response, hint], "next_action": "model"}
                
        except Exception as e:
            print(f"  ⚠️ LLM 호출 에러: {e}")
            if "429" in str(e):
                print("  💤 API 할당량 초과. 잠시 대기 후 재시도합니다...")
                time.sleep(10)
                return {"next_action": "model"}
            return {"next_action": "end"}

    def node_tool_executor(self, state: AgentState) -> Dict[str, Any]:
        """선택된 도구를 실행하는 노드"""
        last_message = state['messages'][-1]
        
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {"next_action": "model"}
        
        tool_results = []
        next_step = "model" # 기본값
        
        # 도구 사용 추적 초기화
        tool_usage_count = state.get('tool_usage_count', {})
        last_tool_used = state.get('last_tool_used', None)
        consecutive_same_tool = state.get('consecutive_same_tool', 0)
        previous_metrics = state.get('previous_metrics', {})
        
        for tool_call in last_message.tool_calls:
            tool_name = tool_call['name']
            args = tool_call['args']
            tool_call_id = tool_call['id']
            
            # 무한 루프 방지: 같은 도구 연속 사용 체크
            if tool_name == last_tool_used:
                consecutive_same_tool += 1
            else:
                consecutive_same_tool = 1
            
            # 3회 연속 같은 도구 사용 시 경고
            if consecutive_same_tool >= 3:
                print(f"  ⚠️ 경고: {tool_name}을(를) {consecutive_same_tool}회 연속 사용 중!")
                warning_msg = f"'{tool_name}'을(를) 3회 연속 사용했습니다. 다른 전략을 고려해주세요."
                tool_results.append(ToolMessage(
                    content=warning_msg,
                    tool_call_id=tool_call_id
                ))
                return {
                    "messages": tool_results,
                    "next_action": "model",
                    "tool_usage_count": tool_usage_count,
                    "last_tool_used": tool_name,
                    "consecutive_same_tool": consecutive_same_tool,
                }
            
            # 도구 사용 횟수 업데이트
            tool_usage_count[tool_name] = tool_usage_count.get(tool_name, 0) + 1
            
            print(f"\n[Tool Execution] {tool_name} 실행... (총 {tool_usage_count[tool_name]}회)")
            
            result_content = ""
            
            if tool_name == "TuneParameters":
                # 파라미터 업데이트
                new_params = state['params'].copy()
                new_params.update(args)
                # shrink는 내부 최적화 파라미터이므로 고정값 사용 (LLM이 조정 불가)
                new_params['shrink'] = 0.7
                result_content = f"파라미터가 업데이트되었습니다. ({args})"
                
                # 파라미터가 바뀌었으니 재생성(Generator)으로 이동
                next_step = "generator"
                
                # 업데이트된 파라미터 반환
                state['params'] = new_params
                
            elif tool_name == "FixFloatingBricks":
                # 브릭 삭제 로직 수행
                from ldr_modifier import apply_llm_decisions
                
                # 삭제 요청된 브릭 처리
                bricks_to_delete = args.get('bricks_to_delete', [])
                if not bricks_to_delete:
                    result_content = "삭제할 브릭 목록이 비어있습니다."
                else:
                    # 'decisions' 포맷으로변환
                    decisions = [{"action": "delete", "brick_id": bid} for bid in bricks_to_delete]
                    
                    try:
                        stats = apply_llm_decisions(state['ldr_path'], decisions)
                        result_content = f"수정 완료: {stats['deleted']}개 브릭 삭제됨."
                        # 이전 메트릭 대비 효과 예상 표시
                        if previous_metrics:
                            prev_floating = previous_metrics.get('floating_count', 0)
                            result_content += f" (이전 공중부양: {prev_floating}개 → 삭제 후 재검증 필요)"
                        # 수정했으니 다시 검증(Verifier)으로 이동 parameter 조정 불필요
                        next_step = "verifier"
                    except Exception as e:
                        result_content = f"수정 실패: {e}"
            
            elif tool_name == "MergeBricks":
                # 브릭 병합 로직 수행
                from ldr_modifier import merge_small_bricks
                
                target_brick_ids = args.get('target_brick_ids', None)
                min_merge_count = args.get('min_merge_count', 2)
                
                try:
                    stats = merge_small_bricks(
                        state['ldr_path'],
                        target_brick_ids=target_brick_ids,
                        min_merge_count=min_merge_count
                    )
                    result_content = f"병합 완료: {stats['merged']}개 그룹 병합됨 (원본 {stats['original_count']}개 -> 신규 {stats['new_count']}개)"
                    # 이전 메트릭 대비 효과 표시
                    if previous_metrics:
                        prev_ratio = previous_metrics.get('small_brick_ratio', 0)
                        result_content += f" (이전 1x1 비율: {prev_ratio*100:.1f}% → 병합 후 재검증 필요)"
                    # 병합했으니 다시 검증(Verifier)으로 이동
                    next_step = "verifier"
                except Exception as e:
                    result_content = f"병합 실패: {e}"
            
            else:
                result_content = f"알 수 없는 도구: {tool_name}"
            
            print(f"  결과: {result_content}")
            
            tool_results.append(ToolMessage(
                content=result_content,
                tool_call_id=tool_call_id
            ))
            
        # ToolMessage들을 History에 추가하고, 다음 단계로 이동
        # verifier로 바로 이동 (Verify 후 Reflect에서 실제 결과 분석)
        
        return {
            "messages": tool_results, 
            "next_action": next_step,  # verifier 또는 generator로 직접 이동
            "params": state['params'],
            "tool_usage_count": tool_usage_count,
            "last_tool_used": tool_name,
            "consecutive_same_tool": consecutive_same_tool,
        }

    def node_reflect(self, state: AgentState) -> Dict[str, Any]:
        """
        회고 노드: 검증 결과를 분석하고 성공/실패를 Memory에 기록합니다.
        Co-Scientist의 핵심 학습 메커니즘입니다.
        
        이제 Verify 후에 호출되므로 실제 결과를 알 수 있습니다.
        """
        print("\n[Reflect] 실제 결과 분석 중...")
        
        # Memory 초기화 (없으면)
        memory = state.get('memory', {
            "failed_approaches": [],
            "successful_patterns": [],
            "lessons": [],
            "consecutive_failures": 0
        })
        
        previous_metrics = state.get('previous_metrics', {})
        current_metrics = state.get('current_metrics', {})
        last_tool = state.get('last_tool_used', 'unknown')
        
        # 이전 메트릭이 없으면 첫 실행
        # 이전 메트릭이 없으면 첫 실행 (비교 대상 없음)
        if not previous_metrics:
            print("  (첫 검증 - 기준점 설정)")
            return {
                "memory": memory, 
                "previous_metrics": current_metrics, # 기준점 설정
                "next_action": "hypothesize"
            }
        
        # 메트릭 비교
        prev_failure = previous_metrics.get('failure_ratio', 0)
        curr_failure = current_metrics.get('failure_ratio', 0)
        prev_floating = previous_metrics.get('floating_count', 0)
        curr_floating = current_metrics.get('floating_count', 0)
        prev_small_ratio = previous_metrics.get('small_brick_ratio', 0)
        curr_small_ratio = current_metrics.get('small_brick_ratio', 0)
        
        # 효과 판정
        failure_improved = curr_failure < prev_failure
        floating_improved = curr_floating < prev_floating
        small_ratio_improved = curr_small_ratio < prev_small_ratio
        
        overall_improved = failure_improved or floating_improved
        
        # 도구별 결과 분석 및 기록
        # 2. 결과 분석 및 통합 로그 저장 (Unified Log)
        if memory_manager:
            try:
                # 관찰 (Observation)
                observation = f"ratio={prev_small_ratio:.2f}, floating={prev_floating}, failure={prev_failure:.2f}"
                
                current_hypothesis = state.get('current_hypothesis', {})
                hyp_text = current_hypothesis.get('hypothesis', 'No hypothesis')

                # 간단한 성공/실패 판정 및 메시지
                if overall_improved:
                    lesson = f"✅ {last_tool} 성공: {hyp_text} (Gained Improvement)"
                    memory["successful_patterns"].append(f"{last_tool}: 효과 있음")
                    memory["consecutive_failures"] = 0
                    print(f"  {lesson}")
                else:
                    lesson = f"❌ {last_tool} 실패: {hyp_text} (No Improvement)"
                    memory["failed_approaches"].append(f"{last_tool}: 효과 미미")
                    memory["consecutive_failures"] += 1
                    print(f"  {lesson}")
                
                memory["lessons"].append(lesson)
                
                # 리스트 관리
                memory["lessons"] = memory["lessons"][-10:]
                memory["failed_approaches"] = memory["failed_approaches"][-5:]
                memory["successful_patterns"] = memory["successful_patterns"][-5:]

                # DB & Vector Store 저장 (표준화된 헬퍼 함수 사용)
                memory_manager.log_experiment(
                    session_id=state.get('session_id', 'unknown_session'),
                    model_id=Path(state['glb_path'] or state['ldr_path']).name,
                    agent_type="main_agent",
                    iteration=state['attempts'],
                    hypothesis=build_hypothesis(
                        observation=observation,
                        hypothesis=hyp_text,
                        reasoning=f"Based on memory lessons: {memory.get('lessons', [])[-1] if memory.get('lessons') else 'None'}",
                        prediction=f"floating: {prev_floating}→{curr_floating}, ratio: {prev_small_ratio:.2f}→?"
                    ) if build_hypothesis else {"observation": observation},
                    experiment=build_experiment(
                        tool=last_tool,
                        parameters=state.get('params', {}),
                        model_name="gemini-2.5-flash"
                    ) if build_experiment else {"tool": last_tool},
                    verification=build_verification(
                        passed=overall_improved,
                        metrics_before=previous_metrics,
                        metrics_after=current_metrics,
                        numerical_analysis=f"floating {prev_floating}→{curr_floating}, ratio {prev_small_ratio:.2f}→{curr_small_ratio:.2f}, failure {prev_failure:.2f}→{curr_failure:.2f}"
                    ) if build_verification else {"passed": overall_improved},
                    improvement=build_improvement(
                        lesson_learned=lesson,
                        next_hypothesis="Maintain strategy" if overall_improved else "Change strategy"
                    ) if build_improvement else {"lesson_learned": lesson}
                )
            except Exception as e:
                print(f"⚠️ [Memory] 통합 로그 저장 실패: {e}")
        
        return {
            "memory": memory, 
            "previous_metrics": current_metrics, # 다음 턴을 위해 현재 메트릭 승격
            "next_action": "hypothesize"
        }


    # --- Build Graph ---

    def build(self):
        workflow = StateGraph(AgentState)
        
        # 노드 추가
        workflow.add_node("generator", self.node_generator)
        workflow.add_node("verifier", self.node_verifier)
        workflow.add_node("model", self.node_model)
        workflow.add_node("tool_executor", self.node_tool_executor)
        workflow.add_node("reflect", self.node_reflect)      # 회고 (학습)
        workflow.add_node("hypothesize", self.node_hypothesize)  # [v2] 가설 생성
        workflow.add_node("strategy", self.node_strategy)        # [v2] 전략 결정
        
        # 라우팅 로직
        def route_next(state: AgentState):
            return state['next_action']
            
        # 엣지 정의
        # 1. Generator -> Verify
        workflow.add_conditional_edges("generator", route_next, {"verify": "verifier", "model": "model"})
        
        # 2. Verifier -> Reflect (성공 시 End)
        workflow.add_conditional_edges("verifier", route_next, {
            "model": "model",       # 에러 등 예외 시
            "end": END,             # 성공 또는 포기 시
            "verifier": "verifier", # 재시도 시
            "reflect": "reflect"    # 검증 완료 후 회고로 이동
        })
        
        # 3. Reflect -> Hypothesize (v2 핵심: 회고 후 바로 모델이 아니라 가설 수립으로)
        # 단, 첫 실행이라 비교할 게 없으면 바로 Strategy나 Model로 갈 수도 있음
        workflow.add_conditional_edges("reflect", route_next, {
            "model": "model",             # 바로 모델로 가는 경우 (Legacy)
            "hypothesize": "hypothesize"  # 보통 가설 생성으로 이동
        })
        
        # 4. Hypothesize -> Strategy
        workflow.add_conditional_edges("hypothesize", route_next, {"strategy": "strategy"})
        
        # 5. Strategy -> Model (모델 설정 후 도구 선택)
        workflow.add_conditional_edges("strategy", route_next, {"model": "model"})
        
        # 6. Model -> Tool
        workflow.add_conditional_edges("model", route_next, {"tool": "tool_executor", "model": "model", "end": END})
        
        # 7. Tool -> Generator or Verifier
        workflow.add_conditional_edges("tool_executor", route_next, {
            "generator": "generator", 
            "verifier": "verifier", 
            "model": "model",
        })
        
        workflow.set_entry_point("generator")
        
        return workflow.compile()


# ============================================================================
# 실행 함수
# ============================================================================

def regeneration_loop(
    glb_path: str,
    output_ldr_path: str,
    llm_client: Optional[BaseLLMClient] = None,
    max_retries: int = 5,
    acceptable_failure_ratio: float = 0.1,
    gui: bool = False,
    params: Optional[Dict[str, Any]] = None,  # [수정] 외부 파라미터 주입 허용
):
    print("=" * 60)
    print("🤖 Co-Scientist Agent (Tool-Use Ver.)")
    print("=" * 60)
    
    graph_builder = RegenerationGraph(llm_client)
    app = graph_builder.build()
    
    # 시스템 메시지 및 초기 설정
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

    # 파라미터 병합 (기본값 + 사용자 입력)
    merged_params = DEFAULT_PARAMS.copy()
    if params:
        merged_params.update(params)
        print(f"⚙️  Custom Params Applied: {list(params.keys())}")

    initial_state = AgentState(
        glb_path=glb_path,
        ldr_path=output_ldr_path,
        params=merged_params,
        attempts=0,
        session_id=memory_manager.start_session(Path(glb_path).name, "main_agent") if memory_manager else "offline",
        max_retries=max_retries,
        acceptable_failure_ratio=acceptable_failure_ratio,
        verification_duration=2.0,
        gui=gui,
        messages=[system_msg], # History 시작
        verification_raw_result=None,
        floating_bricks_ids=[],
        verification_errors=0,  # 검증 에러 카운터 초기화
        # 새로 추가된 필드들
        tool_usage_count={},      # 도구 사용 횟수 추적
        last_tool_used=None,      # 마지막 사용 도구
        consecutive_same_tool=0,  # 같은 도구 연속 사용 횟수
        previous_metrics={},      # 이전 메트릭 (효과 측정용)
        current_metrics={},       # 현재 메트릭 (Reflect에서 비교용)
        final_report={},          # 최종 결과 리포트
        # Co-Scientist Memory 초기화 (DB 로드 반영)
        memory=initial_memory,
        next_action="generate" 
    )
    
    # 실행
    final_state = app.invoke(initial_state)
    
    print("\n" + "=" * 60)
    print("📋 최종 결과 리포트")
    print("=" * 60)
    
    # 최종 리포트 출력
    report = final_state.get('final_report', {})
    if report:
        success = report.get('success', False)
        status = "✅ 성공" if success else "❌ 실패"
        print(f"상태: {status}")
        print(f"총 시도: {report.get('total_attempts', final_state['attempts'])}회")
        
        # 도구 사용 통계
        tool_usage = report.get('tool_usage', {})
        if tool_usage:
            print(f"도구 사용 현황:")
            for tool, count in tool_usage.items():
                print(f"  - {tool}: {count}회")
        
        # 최종 메트릭
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
    
    # 📊 세션 피드백 보고서 생성
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
    
    return final_state


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("glb", help="입력 GLB 파일")
    parser.add_argument("--out", default="output.ldr", help="출력 LDR")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--api-key", help="API Key")
    
    args = parser.parse_args()
    
    # 툴 바인딩을 위해 GeminiClient 사용 (LangChain 호환)
    client = GeminiClient(api_key=args.api_key)
    
    regeneration_loop(
        args.glb,
        args.out,
        llm_client=client,
        max_retries=args.max_retries,
        gui=args.gui
    )
