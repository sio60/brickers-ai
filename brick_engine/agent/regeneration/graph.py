# ============================================================================
# RegenerationGraph: LangGraph 워크플로우 정의
# ============================================================================

from typing import Optional

from langgraph.graph import StateGraph, END

from ..core.llm_state import AgentState
from ..core.llm_clients import BaseLLMClient, GeminiClient
from ..hypothesis_maker.core import HypothesisMaker
from ..hypothesis_maker import build_hypothesis_graph
from ..core.memory_utils import memory_manager

from .prompts import SYSTEM_PROMPT
from .utils import trace_node
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
        """노드 실행 트레이싱 래퍼 (utils.trace_node에 위임)"""
        return await trace_node(self, node_name, func, state)

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
            "hypothesize": "hypothesize",
            "end": END
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
