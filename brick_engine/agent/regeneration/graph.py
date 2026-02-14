# ============================================================================
# RegenerationGraph: LangGraph 워크플로우 정의
# ============================================================================

from typing import Optional

from langgraph.graph import StateGraph, END

from ..llm_state import AgentState
from ..llm_clients import BaseLLMClient, GeminiClient
from ..hypothesis_maker.core import HypothesisMaker
from ..hypothesis_maker import build_hypothesis_graph
from ..memory_utils import memory_manager

from .prompts import SYSTEM_PROMPT
from .nodes.hypothesize import node_hypothesize
from .nodes.strategy import node_strategy
from .nodes.generator import node_generator
from .nodes.verifier import node_verifier
from .nodes.model import node_model
from .nodes.tool_executor import node_tool_executor
from .nodes.reflect import node_reflect


class RegenerationGraph:
    """LangGraph 기반 재생성 에이전트 그래프"""

    SYSTEM_PROMPT = SYSTEM_PROMPT

    def __init__(self, llm_client: Optional[BaseLLMClient] = None, log_callback=None, job_id: str = "offline"):
        self.gemini_client = GeminiClient()
        self.default_client = llm_client if llm_client else self.gemini_client
        self.hypothesis_maker = HypothesisMaker(memory_manager, self.gemini_client)

        # [Rollback] GPT Client는 현재 사용하지 않음
        self.gpt_client = None

        # SSE 로그 콜백 (Kids 모드용)
        self._log_callback = log_callback
        self.job_id = job_id 
        self.verifier = None

    def _log(self, step: str, message: str):
        """SSE 로그 전송 헬퍼"""
        if self._log_callback:
            try:
                self._log_callback(step, message)
            except Exception:
                pass

    async def _trace(self, node_name: str, func, state):
        """노드 실행 트레이싱 래퍼"""
        import time
        import anyio
        import asyncio
        from service.backend_client import send_agent_trace

        start_ts = time.time()
        # Input snapshot (Enriched for Admin UI)
        def serialize_state(s):
            """에이전트 상태를 JSON 직렬화 가능한 형태로 변환"""
            if not isinstance(s, dict):
                return str(s)
            
            clean_state = {}
            for k, v in s.items():
                if k == 'messages':
                    # 메시지 내역을 읽기 쉬운 포맷으로 변환
                    clean_state[k] = [
                        {
                            "role": "assistant" if "AI" in str(type(m)) else ("user" if "Human" in str(type(m)) else "system"),
                            "content": m.content if hasattr(m, 'content') else str(m)
                        } for m in v[-5:] # 최근 5개 메시지만
                    ]
                elif k in ['hypothesis_maker']: # 직렬화 불가능한 객체 제외
                    continue
                elif isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                    clean_state[k] = v
                else:
                    clean_state[k] = str(v)
            return clean_state

        input_snap = serialize_state(state)
        
        status = "SUCCESS"
        output_snap = {}
        
        try:
            # Sync vs Async Check
            if asyncio.iscoroutinefunction(func):
                result = await func(self, state)
            else:
                # Run sync node in thread to avoid blocking loop
                result = await anyio.to_thread.run_sync(func, self, state)
            
            output_snap = serialize_state(result) if isinstance(result, dict) else {"result": str(result)}
            return result
        
        except Exception as e:
            status = "FAILURE"
            output_snap = {"error": str(e)}
            raise e
        finally:
            duration = int((time.time() - start_ts) * 1000)
            if self.job_id != "offline":
                # Fire and forget trace sending
                asyncio.create_task(
                    send_agent_trace(
                        self.job_id,
                        step="TRACE",
                        node_name=node_name,
                        status=status,
                        input_data=input_snap,
                        output_data=output_snap,
                        duration_ms=duration
                    )
                )

    # --- Node method wrappers ---
    # 각 노드 로직은 nodes/ 패키지에 분리되어 있고,
    # graph 인스턴스(self)를 첫 인자로 전달

    async def node_hypothesize(self, state):
        return await self._trace("node_hypothesize", node_hypothesize, state)

    async def node_strategy(self, state):
        return await self._trace("node_strategy", node_strategy, state)

    async def node_generator(self, state):
        return await self._trace("node_generator", node_generator, state)

    async def node_verifier(self, state):
        return await self._trace("node_verifier", node_verifier, state)

    async def node_model(self, state):
        return await self._trace("node_model", node_model, state)

    async def node_tool_executor(self, state):
        return await self._trace("node_tool_executor", node_tool_executor, state)

    async def node_reflect(self, state):
        return await self._trace("node_reflect", node_reflect, state)

    # --- Build Graph ---

    def build(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("generator", self.node_generator)
        workflow.add_node("verifier", self.node_verifier)
        workflow.add_node("model", self.node_model)
        workflow.add_node("tool_executor", self.node_tool_executor)
        workflow.add_node("reflect", self.node_reflect)
        workflow.add_node("strategy", self.node_strategy)

        hyp_graph = build_hypothesis_graph()
        workflow.add_node("hypothesize", hyp_graph)

        def route_next(state: AgentState):
            return state['next_action']

        workflow.add_conditional_edges("generator", route_next, {"verify": "verifier", "model": "model"})
        workflow.add_conditional_edges("verifier", route_next, {
            "model": "model",
            "end": END,
            "verifier": "verifier",
            "reflect": "reflect"
        })
        workflow.add_conditional_edges("reflect", route_next, {
            "model": "model",
            "hypothesize": "hypothesize"
        })
        workflow.add_conditional_edges("hypothesize", route_next, {"strategy": "strategy"})
        workflow.add_conditional_edges("strategy", route_next, {"model": "model"})
        workflow.add_conditional_edges("model", route_next, {"tool": "tool_executor", "model": "model", "end": END})
        workflow.add_conditional_edges("tool_executor", route_next, {
            "generator": "generator",
            "verifier": "verifier",
            "model": "model",
        })

        workflow.set_entry_point("generator")

        return workflow.compile()
