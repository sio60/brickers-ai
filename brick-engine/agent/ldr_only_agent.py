# ============================================================================
# LDR-ONLY Co-Scientist Agent
# 원본 llm_regeneration_agent.py의 수정본으로, GLB 변환 없이 LDR 파일만으로 최적화를 수행합니다.
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
except ImportError:
    from llm_clients import BaseLLMClient, GroqClient, GeminiClient
    from agent_tools import TuneParameters, FixFloatingBricks, MergeBricks

# DB 연결
try:
    from yang_db import get_db
except ImportError:
    print("⚠️ yang_db.py를 찾을 수 없습니다. Memory 영속화 기능이 비활성화됩니다.")
    get_db = None

# ============================================================================
# Memory & DB Helper Functions
# ============================================================================

import config  # This registers AGENT_DIR in sys.path
from memory_utils import memory_manager, build_hypothesis, build_experiment, build_verification, build_improvement

# Legacy functions (kept for compatibility)
def get_memory_collection(): return memory_manager.collection_exps if memory_manager else None
def load_memory_from_db(model_id: str): return {}
def save_memory_to_db(model_id: str, memory: Dict): pass

# ============================================================================
# 기본 파라미터 정의
# ============================================================================

DEFAULT_PARAMS = {
    "target": 25,
    "min_target": 5,
    "budget": 150,
    "shrink": 0.7,
    "search_iters": 6,
    "flipx180": False,
    "flipy180": False,
    "flipz180": False,
    "kind": "brick",
    "plates_per_voxel": 3,
    "interlock": True,
    "max_area": 20,
    "solid_color": 4,
    "use_mesh_color": True,
    "invert_y": False,
    "smart_fix": True,
    "fill": True,
    "step_order": "bottomup",
}

# ============================================================================
# 데이터 구조 및 헬퍼 함수
# ============================================================================

@dataclass
class VerificationFeedback:
    stable: bool = True
    total_bricks: int = 0
    fallen_bricks_count: int = 0
    floating_bricks_count: int = 0
    floating_brick_ids: List[str] = field(default_factory=list)
    fallen_brick_ids: List[str] = field(default_factory=list)
    failure_ratio: float = 0.0
    first_failure_brick: Optional[str] = None
    max_drift: float = 0.0
    collision_count: int = 0
    small_brick_count: int = 0
    small_brick_ratio: float = 0.0

def extract_verification_feedback(result, total_bricks: int) -> VerificationFeedback:
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
    
    feedback.fallen_bricks_count = len(fallen_bricks)
    feedback.floating_bricks_count = len(floating_bricks)
    feedback.floating_brick_ids = list(floating_bricks)
    feedback.fallen_brick_ids = list(fallen_bricks)
    feedback.first_failure_brick = first_failure
    feedback.collision_count = collision_count
    
    if total_bricks > 0:
        feedback.failure_ratio = (len(fallen_bricks) + len(floating_bricks)) / total_bricks
    
    return feedback

def _format_feedback(feedback: VerificationFeedback) -> str:
    if feedback.stable and feedback.floating_bricks_count == 0:
        status = "✅ 안정"
    elif feedback.stable and feedback.floating_bricks_count > 0:
        status = "⚠️ 부분 안정 (공중부양 존재)"
    else:
        status = "❌ 불안정"
        
    lines = [
        f"검증 결과:",
        f"- 상태: {status}",
        f"- 총 브릭 수: {feedback.total_bricks}개",
    ]
    
    if feedback.small_brick_count > 0:
        lines.append(f"- 1x1 브릭: {feedback.small_brick_count}개 ({feedback.small_brick_ratio * 100:.1f}%)")
        if feedback.small_brick_ratio > 0.3:
            lines.append(f"  → 💡 1x1 브릭 비율이 높습니다. MergeBricks로 연결 강화를 권장합니다.")
    
    if not feedback.stable or feedback.floating_bricks_count > 0:
        lines.extend([
            f"- 떨어진 브릭: {feedback.fallen_bricks_count}개",
            f"- 공중부양 브릭: {feedback.floating_bricks_count}개",
            f"- 실패율: {feedback.failure_ratio * 100:.1f}%",
        ])
        if feedback.first_failure_brick:
            lines.append(f"- 최초 붕괴 브릭: {feedback.first_failure_brick}")
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
    glb_path: Optional[str]  # Optional for LDR-only mode
    ldr_path: str
    params: Dict[str, Any]
    max_retries: int
    acceptable_failure_ratio: float
    verification_duration: float
    gui: bool
    
    attempts: int
    session_id: str
    messages: Annotated[List[BaseMessage], add_messages]
    
    verification_raw_result: Any 
    floating_bricks_ids: List[str]
    verification_errors: int
    
    tool_usage_count: Dict[str, int]
    last_tool_used: Optional[str]
    consecutive_same_tool: int
    
    previous_metrics: Dict[str, Any]
    current_metrics: Dict[str, Any]
    
    final_report: Dict[str, Any]
    memory: Dict[str, Any]
    next_action: str


# ============================================================================
# LangGraph Agent Logic (LDR-ONLY MODIFIED)
# ============================================================================

class RegenerationGraph:
    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        if llm_client is None:
            self.llm_client = GeminiClient()
        else:
            self.llm_client = llm_client
            
        # LDR 전용에 맞게 프롬프트 수정
        self.SYSTEM_PROMPT = """당신은 레고 브릭 구조물 설계 및 안정화 전문가(Co-Scientist)입니다.
LDR 3D 모델의 구조적 불안정성 문제를 해결해야 합니다.

당신에게는 두 가지 주요 수리 도구가 있습니다:
1. `FixFloatingBricks`: 전체적으로는 괜찮지만 일부 공중부양하거나 불안정한 브릭이 있을 때, 해당 브릭을 *삭제*하여 정리합니다. (강력 권장)
2. `MergeBricks`: 같은 색상의 인접한 1x1 브릭들을 큰 브릭(1x2~1x8)으로 병합합니다. 연결이 강화되어 안정성이 향상됩니다.
3. `TuneParameters`: (주의) 이 도구는 GLB 파일이 필요하므로 현재 모드에서는 사용할 수 없습니다.

**의사결정 알고리즘 (Decision Logic):**
1. **공중부양/떨어진 브릭 ID가 명확히 있으면** → `FixFloatingBricks`로 해당 브릭 삭제
2. **1x1 브릭이 많아 연결이 약하다는 징후가 있으면** → `MergeBricks`로 보강 (1x1들을 큰 브릭으로 통합)
3. **둘 다 해당되면** → 먼저 `MergeBricks`로 보강 → 재검증 후 필요시 `FixFloatingBricks`

목표: 물리적으로 안정적(Stable)인 레고 구조물을 만드는 것.
공중부양 브릭이 0개가 되어야 합니다.
"""

        self.verifier = None
        
    # --- Nodes ---

    def node_generator(self, state: AgentState) -> Dict[str, Any]:
        """GLB -> LDR 변환 노드 (LDR-only 모드에서는 Dummy 역할)"""
        print(f"\n[Generator] LDR-only 모드이므로 변환 단계를 건너뜁니다.")
        return {"next_action": "verify"}

    def node_verifier(self, state: AgentState) -> Dict[str, Any]:
        """물리 검증 노드"""
        from physical_verification.pybullet_verifier import PyBulletVerifier
        from physical_verification.ldr_loader import LdrLoader
        
        print("\n[Verifier] 물리 검증 수행 중...")
        
        if not os.path.exists(state['ldr_path']):
            return {"messages": [HumanMessage(content="LDR 파일이 존재하지 않습니다.")], "next_action": "end"}
            
        try:
            loader = LdrLoader()
            plan = loader.load_from_file(state['ldr_path'])
            total_bricks = len(plan.bricks)
            
            small_brick_parts = {"3005.dat", "3024.dat"}
            small_brick_count = 0
            for b in plan.bricks:
                part_id = getattr(b, 'part_id', None) or (b.get('part') if isinstance(b, dict) else None)
                if part_id in small_brick_parts:
                    small_brick_count += 1
            small_brick_ratio = small_brick_count / total_bricks if total_bricks > 0 else 0.0
            
            if self.verifier is not None:
                try:
                    self.verifier._close_simulation()
                except:
                    pass
            
            verifier = PyBulletVerifier(plan, gui=state['gui'])
            self.verifier = verifier
            
            stab_result = verifier.run_stability_check(duration=state['verification_duration'], auto_close=False)
            
            feedback = extract_verification_feedback(stab_result, total_bricks)
            feedback.small_brick_count = small_brick_count
            feedback.small_brick_ratio = small_brick_ratio
            
            feedback_text = _format_feedback(feedback)
            
            if feedback.stable and feedback.floating_bricks_count == 0:
                short_status = "✅ 안정"
            elif feedback.stable and feedback.floating_bricks_count > 0:
                short_status = "⚠️ 부분 안정 (공중부양 존재)"
            else:
                short_status = "❌ 불안정"
            
            print(f"  결과: {short_status}")
            
            if not feedback.stable or feedback.floating_bricks_count > 0:
                 summary_text = feedback_text.replace('\n', ', ').replace('\r', '')
                 if len(summary_text) > 200:
                     summary_text = summary_text[:200] + "..."
                 print(f"  요약: {summary_text}")
            
            floating_ids = []
            for ev in stab_result.evidence:
                if ev.type == "FLOATING_BRICK" and ev.brick_ids:
                    floating_ids.extend(ev.brick_ids)
            
            current_metrics = {
                "failure_ratio": feedback.failure_ratio,
                "small_brick_ratio": small_brick_ratio,
                "small_brick_count": small_brick_count,
                "total_bricks": total_bricks,
                "floating_count": feedback.floating_bricks_count,
                "fallen_count": feedback.fallen_bricks_count,
            }
            
            is_physically_okay = feedback.stable or (feedback.failure_ratio <= state['acceptable_failure_ratio'])
            is_success = is_physically_okay and (feedback.floating_bricks_count == 0)
            
            if is_success:
                print("🎉 목표 달성! 프로세스를 종료합니다.")
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
                final_report = {
                    "success": False,
                    "total_attempts": state['attempts'],
                    "tool_usage": state.get('tool_usage_count', {}),
                    "final_metrics": current_metrics,
                    "message": "최대 시도 횟수 초과로 종료"
                }
                return {"next_action": "end", "final_report": final_report}

            custom_feedback = feedback_text
            if feedback.floating_bricks_count > 0:
                custom_feedback += "\n\n⚠️ **중요: 아직 공중부양(Floating) 브릭이 남아있습니다. 이 상태로는 절대 작업을 완료할 수 없습니다. 반드시 FixFloatingBricks 도구를 사용하거나 파라미터를 조정하여 해결하세요.**"
            
            return {
                "verification_raw_result": stab_result,
                "floating_bricks_ids": floating_ids,
                "messages": [HumanMessage(content=custom_feedback)],
                "current_metrics": current_metrics,
                "next_action": "reflect"
            }
            
        except Exception as e:
            print(f"  ❌ 검증 중 에러: {e}")
            verification_errors = state.get('verification_errors', 0) + 1
            if verification_errors >= 3:
                return {"next_action": "end"} # 재생성 불가하므로 종료
            else:
                import time
                time.sleep(1)
                return {"verification_errors": verification_errors, "next_action": "verifier"}

    def node_model(self, state: AgentState) -> Dict[str, Any]:
        """LLM이 상황을 분석하고 도구를 선택하는 노드"""
        import time
        time.sleep(2) 
        
        print("\n[Co-Scientist] 상황 분석 중...")
        
        # Tools definitions
        # TuneParameters는 제외하거나 경고 처리할 수도 있지만, 일단 포함하되 프롬프트에서 제한
        # 여기서는 안전하게 FixFloatingBricks, MergeBricks만 활성화
        tools = [FixFloatingBricks, MergeBricks]
    
        messages_to_send = state['messages'][:]
        
        if memory_manager:
            last_msg = messages_to_send[-1]
            obs = last_msg.content if isinstance(last_msg, HumanMessage) else ""
            similar_cases = memory_manager.search_similar_cases(obs, limit=3)
            
            if similar_cases:
                memory_info = "\n**📚 유사한 과거 실험 사례 (RAG):**\n"
                for i, case in enumerate(similar_cases, 1):
                    tool = case['experiment'].get('tool', 'Unknown')
                    result = case['verification'].get('numerical_analysis', 'N/A')
                    outcome = "성공" if case.get('result_success') else "실패"
                    memory_info += f"[{i}] {outcome} ({tool}): {result}\n"
                messages_to_send.append(SystemMessage(content=memory_info))

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
            messages_to_send.append(SystemMessage(content=memory_info))
        
        # 힌트 주입
        last_msg = messages_to_send[-1]
        if isinstance(last_msg, HumanMessage) and "검증 결과" in str(last_msg.content):
            floating_ids = state.get('floating_bricks_ids', [])
            if floating_ids:
                advice = f"⚠️ 공중부양 브릭 {len(floating_ids)}개를 해결하기 위해 `FixFloatingBricks`를 사용하세요."
                messages_to_send.append(SystemMessage(content=advice))

        try:
            model_with_tools = self.llm_client.bind_tools(tools)
            response = model_with_tools.invoke(messages_to_send)
            
            if response.tool_calls:
                print(f"  🔨 도구 선택: {[tc['name'] for tc in response.tool_calls]}")
                return {"messages": [response], "next_action": "tool"}
            else:
                print(f"  💭 LLM 의견: {response.content}")
                
                # 강제 진행 유도
                retry_msg = "도구를 선택하지 않았습니다. 문제를 해결하려면 반드시 도구를 사용해야 합니다."
                return {"messages": [response, HumanMessage(content=retry_msg)], "next_action": "model"}
                
        except Exception as e:
            print(f"  ⚠️ LLM 호출 에러: {e}")
            if "429" in str(e):
                time.sleep(10)
                return {"next_action": "model"}
            return {"next_action": "end"}

    def node_tool_executor(self, state: AgentState) -> Dict[str, Any]:
        """선택된 도구를 실행하는 노드"""
        last_message = state['messages'][-1]
        
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {"next_action": "model"}
        
        tool_results = []
        next_step = "verify" # 기본적으로 다시 검증으로 이동
        
        tool_usage_count = state.get('tool_usage_count', {})
        previous_metrics = state.get('previous_metrics', {})
        
        for tool_call in last_message.tool_calls:
            tool_name = tool_call['name']
            args = tool_call['args']
            tool_call_id = tool_call['id']
            
            tool_usage_count[tool_name] = tool_usage_count.get(tool_name, 0) + 1
            print(f"\n[Tool Execution] {tool_name} 실행... (총 {tool_usage_count[tool_name]}회)")
            
            result_content = ""
            
            if tool_name == "FixFloatingBricks":
                from ldr_modifier import apply_llm_decisions
                bricks_to_delete = args.get('bricks_to_delete', [])
                if not bricks_to_delete:
                    result_content = "삭제할 브릭 목록이 비어있습니다."
                else:
                    decisions = [{"action": "delete", "brick_id": bid} for bid in bricks_to_delete]
                    try:
                        stats = apply_llm_decisions(state['ldr_path'], decisions)
                        result_content = f"수정 완료: {stats['deleted']}개 브릭 삭제됨."
                    except Exception as e:
                        result_content = f"수정 실패: {e}"
            
            elif tool_name == "MergeBricks":
                from ldr_modifier import merge_small_bricks
                target_brick_ids = args.get('target_brick_ids', None)
                min_merge_count = args.get('min_merge_count', 2)
                try:
                    stats = merge_small_bricks(
                        state['ldr_path'],
                        target_brick_ids=target_brick_ids,
                        min_merge_count=min_merge_count
                    )
                    result_content = f"병합 완료: {stats['merged']}개 그룹 병합됨"
                except Exception as e:
                    result_content = f"병합 실패: {e}"
            else:
                result_content = f"지원하지 않는 도구: {tool_name}"
            
            print(f"  결과: {result_content}")
            
            tool_results.append(ToolMessage(content=result_content, tool_call_id=tool_call_id))
            
        return {
            "messages": tool_results, 
            "next_action": "verifier", # 무조건 검증으로
            "tool_usage_count": tool_usage_count,
        }

    def node_reflect(self, state: AgentState) -> Dict[str, Any]:
        """회고 노드"""
        print("\n[Reflect] 실제 결과 분석 중...")
        
        memory = state.get('memory', {"failed_approaches": [], "successful_patterns": [], "lessons": [], "consecutive_failures": 0})
        current_metrics = state.get('current_metrics', {})
        
        # 이전 메트릭 가져오기
        previous_metrics = state.get('previous_metrics', {})
        if not previous_metrics:
            return {"memory": memory, "previous_metrics": current_metrics, "next_action": "model"}

        # 메트릭 비교
        prev_floating = previous_metrics.get('floating_count', 0)
        curr_floating = current_metrics.get('floating_count', 0)
        floating_improved = curr_floating < prev_floating
        
        last_tool = state.get('last_tool_used', 'unknown')
        
        # 간단한 성공 판정
        success = floating_improved
        lesson = f"{last_tool}: 공중부양 {prev_floating}->{curr_floating} ({'성공' if success else '실패'})"
        
        if success:
             memory["successful_patterns"].append(f"{last_tool}: 효과 확인")
             memory["consecutive_failures"] = 0
        else:
             memory["failed_approaches"].append(f"{last_tool}: 효과 미미")
             memory["consecutive_failures"] += 1
             
        memory["lessons"].append(lesson)

        # Unified Logging (표준화된 헬퍼 함수 사용)
        if memory_manager:
            try:
                # 상세 observation 생성
                detailed_obs = f"floating={prev_floating}, ratio={previous_metrics.get('small_brick_ratio', 0):.2f}, total={previous_metrics.get('total_bricks', 0)}"
                
                memory_manager.log_experiment(
                    session_id=state.get('session_id', 'ldr_session'),
                    model_id=Path(state['ldr_path']).name,
                    agent_type="ldr_only",
                    iteration=state['attempts'],
                    hypothesis=build_hypothesis(
                        observation=detailed_obs,
                        hypothesis=f"{last_tool} 적용으로 floating 감소 기대",
                        reasoning=f"Memory lessons: {memory.get('lessons', [])[-1] if memory.get('lessons') else 'None'}",
                        prediction=f"floating: {prev_floating}→{curr_floating} 예상"
                    ),
                    experiment=build_experiment(
                        tool=last_tool,
                        parameters=state.get('params', {}),
                        model_name="gemini-2.5-flash"
                    ),
                    verification=build_verification(
                        passed=success,
                        metrics_before=previous_metrics,
                        metrics_after=current_metrics,
                        numerical_analysis=f"floating {prev_floating}→{curr_floating} ({curr_floating - prev_floating:+d}), ratio {previous_metrics.get('small_brick_ratio', 0):.2f}→{current_metrics.get('small_brick_ratio', 0):.2f}"
                    ),
                    improvement=build_improvement(
                        lesson_learned=lesson,
                        next_hypothesis="Continue" if success else "Try different tool"
                    )
                )
            except Exception as e:
                print(f"⚠️ [Memory] 로그 저장 실패: {e}")

        # Legacy Save (Fallback)
        try:
             model_id = Path(state['ldr_path']).name
             save_memory_to_db(model_id, memory)
        except: pass
        
        return {
            "memory": memory, 
            "previous_metrics": current_metrics,
            "next_action": "model"
        }

    def build(self):
        workflow = StateGraph(AgentState)
        
        workflow.add_node("generator", self.node_generator)
        workflow.add_node("verifier", self.node_verifier)
        workflow.add_node("model", self.node_model)
        workflow.add_node("tool_executor", self.node_tool_executor)
        workflow.add_node("reflect", self.node_reflect)
        
        def route_next(state: AgentState):
            return state['next_action']
            
        workflow.add_conditional_edges("generator", route_next, {"verify": "verifier"})
        workflow.add_conditional_edges("verifier", route_next, {"model": "model", "end": END, "verifier": "verifier", "reflect": "reflect"})
        workflow.add_conditional_edges("model", route_next, {"tool": "tool_executor", "model": "model", "end": END})
        workflow.add_conditional_edges("tool_executor", route_next, {"verifier": "verifier"})
        workflow.add_conditional_edges("reflect", route_next, {"model": "model"})
        
        # START POINT CHANGED TO VERIFIER
        workflow.set_entry_point("verifier")
        
        return workflow.compile()

# ============================================================================
# 실행 함수
# ============================================================================

def regeneration_loop(
    ldr_path: str,
    llm_client: Optional[BaseLLMClient] = None,
    max_retries: int = 5,
    gui: bool = False,
):
    print("=" * 60)
    print("🤖 Co-Scientist Agent (LDR-Only Ver.)")
    print("=" * 60)
    
    graph_builder = RegenerationGraph(llm_client)
    app = graph_builder.build()
    
    system_msg = SystemMessage(content=graph_builder.SYSTEM_PROMPT)
    
    # Memory Load
    initial_memory = {"failed_approaches": [], "successful_patterns": [], "lessons": [], "consecutive_failures": 0}
    try:
        model_id = Path(ldr_path).name
        loaded_mem = load_memory_from_db(model_id)
        if loaded_mem:
            initial_memory.update(loaded_mem)
    except:
        pass

    initial_state = AgentState(
        glb_path=None,
        ldr_path=ldr_path,
        params=DEFAULT_PARAMS.copy(),
        attempts=0,
        session_id=memory_manager.start_session(Path(ldr_path).name, "ldr_only") if memory_manager else "offline",
        max_retries=max_retries,
        acceptable_failure_ratio=0.1,
        verification_duration=2.0,
        gui=gui,
        messages=[system_msg],
        verification_raw_result=None,
        floating_bricks_ids=[],
        verification_errors=0,
        tool_usage_count={},
        last_tool_used=None,
        consecutive_same_tool=0,
        previous_metrics={},
        current_metrics={},
        final_report={},
        memory=initial_memory,
        next_action="verifier" # START ACTION
    )
    
    final_state = app.invoke(initial_state)
    
    print("\n" + "=" * 60)
    print("📋 최종 결과")
    print("=" * 60)
    if 'final_report' in final_state and final_state['final_report'].get('success'):
        print("✅ 성공")
    else:
        print("❌ 실패 또는 중단됨")
    
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
    parser.add_argument("ldr", help="최적화할 LDR 파일 경로")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--api-key", help="API Key")
    
    args = parser.parse_args()
    
    client = GeminiClient(api_key=args.api_key)
    
    regeneration_loop(
        args.ldr,
        llm_client=client,
        max_retries=args.max_retries,
        gui=args.gui
    )
