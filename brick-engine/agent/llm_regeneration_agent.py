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
    from .agent_tools import TuneParameters, FixFloatingBricks
except ImportError:
    from llm_clients import BaseLLMClient, GroqClient, GeminiClient
    from agent_tools import TuneParameters, FixFloatingBricks


# ============================================================================
# 기본 파라미터 정의
# ============================================================================

DEFAULT_PARAMS = {
    "target": 25,              # 목표 스터드 크기 (150 브릭 기준 25 정도가 적절)
    "min_target": 5,           # 최소 스터드 크기
    "budget": 150,             # 최대 브릭 수
    "shrink": 0.7,             # 축소 비율 (빠른 수렴을 위해 0.85 -> 0.7)
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
    "fill": True,              # 내부 채움 활성화
    "step_order": "bottomup",  # 조립 순서
}


# ============================================================================
# 데이터 구조 및 헬퍼 함수
# ============================================================================

@dataclass
class VerificationFeedback:
    """PyBullet 검증 결과를 LLM에게 전달하기 위한 구조화된 피드백"""
    stable: bool = True
    total_bricks: int = 0
    fallen_bricks_count: int = 0
    floating_bricks_count: int = 0
    floating_brick_ids: List[str] = field(default_factory=list)  # 공중부양 브릭 ID 목록
    fallen_brick_ids: List[str] = field(default_factory=list)    # 떨어진 브릭 ID 목록
    failure_ratio: float = 0.0
    first_failure_brick: Optional[str] = None
    max_drift: float = 0.0
    collision_count: int = 0

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
    
    feedback.fallen_bricks_count = len(fallen_bricks)
    feedback.floating_bricks_count = len(floating_bricks)
    feedback.floating_brick_ids = list(floating_bricks)  # ID 목록 저장
    feedback.fallen_brick_ids = list(fallen_bricks)      # ID 목록 저장
    feedback.first_failure_brick = first_failure
    feedback.collision_count = collision_count
    
    if total_bricks > 0:
        feedback.failure_ratio = (len(fallen_bricks) + len(floating_bricks)) / total_bricks
    
    return feedback

def _format_feedback(feedback: VerificationFeedback) -> str:
    status = "✅ 안정" if feedback.stable else "❌ 불안정"
    lines = [
        f"검증 결과:",
        f"- 상태: {status}",
        f"- 총 브릭 수: {feedback.total_bricks}개",
    ]
    if not feedback.stable:
        lines.extend([
            f"- 떨어진 브릭: {feedback.fallen_bricks_count}개",
            f"- 공중부양 브릭: {feedback.floating_bricks_count}개",
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
    messages: Annotated[List[BaseMessage], add_messages] # 대화 기록 (History)
    
    # 검증 결과 캐시 (Tool 실행 시 참조용)
    verification_raw_result: Any 
    floating_bricks_ids: List[str] # 공중부양 브릭 ID 목록 캐시
    verification_errors: int  # 검증 에러 재시도 카운터

    # 다음 노드 제어
    next_action: Literal["generate", "verify", "model", "tool", "end"]


# ============================================================================
# LangGraph Agent Logic
# ============================================================================

class RegenerationGraph:
    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        if llm_client is None:
            self.llm_client = GeminiClient()
        else:
            self.llm_client = llm_client
            
        # 초기 시스템 프롬프트 (Tool 사용 권장)
        self.SYSTEM_PROMPT = """당신은 레고 브릭 구조물 설계 및 안정화 전문가(Co-Scientist)입니다.
주어진 3D 모델(GLB)을 레고(LDR)로 변환하는 과정에서 발생하는 구조적 불안정성 문제를 해결해야 합니다.

당신에게는 두 가지 도구가 있습니다:
1. `TuneParameters`: 전체적인 구조적 결함(와르르 무너짐, 연결 없음 등)을 해결하기 위해 변환 파라미터를 조정하여 처음부터 다시 생성합니다.
2. `FixFloatingBricks`: 전체적으로는 괜찮지만 일부 공중부양하거나 불안정한 브릭이 있을 때, 해당 브릭을 *삭제*하여 정리합니다.

**의사결정 알고리즘 (Decision Logic):**
1. **실패율(Failure Ratio) 확인**:
   - **20% 미만 (Low Risk)**: 전체 구조는 튼튼합니다. `TuneParameters`로 다시 만들면 오히려 더 나쁜 결과가 나올 위험이 큽니다. 무조건 `FixFloatingBricks`를 선택하여 불안정한 브릭만 제거하세요.
   - **20% ~ 50% (Medium Risk)**: 상황을 판단하세요. 중요 부위가 무너졌다면 재생성, 외곽만 무너졌다면 삭제.
   - **50% 이상 (High Risk)**: 현재 파라미터로는 불가능합니다. `TuneParameters`로 설정을 변경(shrink 증가, interlock 활성화 등)하여 다시 시도하세요.

목표: 물리적으로 안정적(Stable)인 레고 구조물을 만드는 것.
이전 시도의 실패 원인과 통계(실패율, 부동 브릭 수)를 분석하고, 위 논리에 따라 가장 합리적인 도구를 선택하세요.
"""

        self.verifier = None
        
    # --- Nodes ---

    def node_generator(self, state: AgentState) -> Dict[str, Any]:
        """GLB -> LDR 변환 노드"""
        from glb_to_ldr_embedded_copy import convert_glb_to_ldr
        
        print(f"\n[Generator] 변환 시도 {state['attempts'] + 1}/{state['max_retries']}")
        print(f"  Params: target={state['params'].get('target')}, shrink={state['params'].get('shrink')}")
        
        try:
            conv_result = convert_glb_to_ldr(
                state['glb_path'],
                state['ldr_path'],
                auto_remove_1x1=False,
                **state['params']
            )
            print(f"  ✅ 변환 완료: {conv_result.get('parts', 0)}개 브릭")
            # 변환 후에는 반드시 검증으로 감
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
            feedback_text = _format_feedback(feedback)
            
            print(f"  결과: {'✅ 안정' if feedback.stable else '❌ 불안정'}")
            if not feedback.stable:
                 print(f"  요약: {feedback_text.replace(chr(10), ', ')}")
            
            # 공중부양 브릭 ID 캐싱 (Tool에서 사용)
            floating_ids = []
            for ev in stab_result.evidence:
                if ev.type == "FLOATING_BRICK" and ev.brick_ids:
                    floating_ids.extend(ev.brick_ids)
            
            # 성공 판정
            is_success = (
                feedback.stable or 
                (feedback.failure_ratio <= state['acceptable_failure_ratio'] and feedback.floating_bricks_count == 0)
            )
            
            if is_success:
                print("🎉 목표 달성! 프로세스를 종료합니다.")
                return {"next_action": "end"}
            
            if state['attempts'] >= state['max_retries']:
                print("💥 최대 시도 횟수 초과.")
                return {"next_action": "end"}

            # 결과를 LLM에게 피드백으로 전달
            return {
                "verification_raw_result": stab_result,
                "floating_bricks_ids": floating_ids,
                "messages": [HumanMessage(content=feedback_text)],
                "next_action": "model"
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
        """LLM 의사결정 노드 (Tool Binding)"""
        print("\n[Co-Scientist] 상황 분석 중...")
        
        # 사용 가능한 도구 정의
        tools = [TuneParameters, FixFloatingBricks]
    
        # --- [전략 가이드 주입] ---
        # 실패율이 낮으면 FixFloatingBricks를 권장하는 힌트 메시지 추가 (강제 X)
        messages_to_send = state['messages'][:]
        
        # 직전 검증 결과 확인
        last_msg = messages_to_send[-1]
        
        if isinstance(last_msg, HumanMessage) and "검증 결과" in str(last_msg.content):
            content = str(last_msg.content)
            if "❌ 불안정" in content and "실패율" in content:
                try:
                    # 실패율 파싱 (간이)
                    import re
                    match = re.search(r"실패율: ([\d.]+)%", content)
                    if match:
                        ratio = float(match.group(1))
                        # 20% 미만이면 부분 수정 권장
                        if ratio < 20.0: 
                            print(f"  💡 [Strategy Hint] 낮은 실패율({ratio}%) 감지 -> FixFloatingBricks 권장")
                            hint_msg = SystemMessage(content=f"현재 실패율이 {ratio}%로 낮습니다. 전체 재생성보다는 `FixFloatingBricks`로 문제 브릭만 정리하는 것이 효율적일 수 있습니다.")
                            messages_to_send.append(hint_msg)
                except Exception:
                    pass

        # 모델 바인딩 및 호출
            
        # 모델 바인딩 및 호출
        try:
            model_with_tools = self.llm_client.bind_tools(tools)
            response = model_with_tools.invoke(messages_to_send)
            
            # 응답 확인
            if response.tool_calls:
                print(f"  🔨 도구 선택: {[tc['name'] for tc in response.tool_calls]}")
                return {"messages": [response], "next_action": "tool"}
            else:
                print(f"  💭 LLM 의견: {response.content}")
                # 도구를 안 불렀으면 그냥 메시지만 추가하고 다시 Model로 가거나(무한루프 위험), 힌트를 줌
                # 여기서는 힌트를 주고 다시 Model 호출
                hint = HumanMessage(content="도구를 사용하여 문제를 해결하세요. 파라미터를 조정하거나 브릭을 삭제하세요.")
                return {"messages": [response, hint], "next_action": "model"}
                
        except Exception as e:
            print(f"  ⚠️ LLM 호출 에러: {e}")
            return {"next_action": "end"}

    def node_tool_executor(self, state: AgentState) -> Dict[str, Any]:
        """선택된 도구를 실행하는 노드"""
        last_message = state['messages'][-1]
        
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {"next_action": "model"}
        
        tool_results = []
        next_step = "model" # 기본값
        
        for tool_call in last_message.tool_calls:
            tool_name = tool_call['name']
            args = tool_call['args']
            tool_call_id = tool_call['id']
            
            print(f"\n[Tool Execution] {tool_name} 실행...")
            
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
                        # 수정했으니 다시 검증(Verifier)으로 이동 parameter 조정 불필요
                        next_step = "verifier"
                    except Exception as e:
                        result_content = f"수정 실패: {e}"
            
            else:
                result_content = f"알 수 없는 도구: {tool_name}"
            
            print(f"  결과: {result_content}")
            
            tool_results.append(ToolMessage(
                content=result_content,
                tool_call_id=tool_call_id
            ))
            
        # ToolMessage들을 History에 추가하고, 다음 단계로 이동
        # params가 업데이트 된 경우 state에 반영되어야 함 (RegenerationGraph는 state 업데이트 방식이 return dict merge임)
        return {
            "messages": tool_results, 
            "next_action": next_step, 
            "params": state['params'] # 갱신된 파라미터 전달
        }


    # --- Build Graph ---

    def build(self):
        workflow = StateGraph(AgentState)
        
        # 노드 추가
        workflow.add_node("generator", self.node_generator)
        workflow.add_node("verifier", self.node_verifier)
        workflow.add_node("model", self.node_model)
        workflow.add_node("tool_executor", self.node_tool_executor)
        
        # 라우팅 로직
        def route_next(state: AgentState):
            return state['next_action']
            
        # 엣지 정의
        workflow.add_conditional_edges("generator", route_next, {"verify": "verifier", "model": "model"})
        workflow.add_conditional_edges("verifier", route_next, {"model": "model", "end": END})
        workflow.add_conditional_edges("model", route_next, {"tool": "tool_executor", "model": "model", "end": END})
        workflow.add_conditional_edges("tool_executor", route_next, {"generator": "generator", "verifier": "verifier", "model": "model"})
        
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
):
    print("=" * 60)
    print("🤖 Co-Scientist Agent (Tool-Use Ver.)")
    print("=" * 60)
    
    graph_builder = RegenerationGraph(llm_client)
    app = graph_builder.build()
    
    # 시스템 메시지 및 초기 설정
    system_msg = SystemMessage(content=graph_builder.SYSTEM_PROMPT)
    
    initial_state = AgentState(
        glb_path=glb_path,
        ldr_path=output_ldr_path,
        params=DEFAULT_PARAMS.copy(),
        attempts=0,
        max_retries=max_retries,
        acceptable_failure_ratio=acceptable_failure_ratio,
        verification_duration=2.0,
        gui=gui,
        messages=[system_msg], # History 시작
        verification_raw_result=None,
        floating_bricks_ids=[],
        verification_errors=0,  # 검증 에러 카운터 초기화
        next_action="generate" 
    )
    
    # 실행
    final_state = app.invoke(initial_state)
    
    print("\n" + "=" * 60)
    print("📋 최종 결과")
    print("=" * 60)
    
    print(f"총 시도: {final_state['attempts']}회")
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
