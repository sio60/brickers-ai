"""DEBATE Node - Multi-Agent 의견 수렴 (CoScientist Debate)

핵심 원칙:
1. 두 에이전트가 각각 의견 제시 (FIDELITY, CREATIVE)
2. 가중치 기반 최종 결정 (FIDELITY 50%, CREATIVE 50%)
3. 반대 의견도 기록 (학습용)

Enhanced: Structured Output (Pydantic)
Note: 1라운드만 (성능 최적화 - 2라운드 토론 제거)
"""
import os
import json
import sys
from typing import Literal, List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from ..state import AgentState
from ..prompts import DEBATE_SYSTEM, DEBATE_PROMPT, FIDELITY_SYSTEM, CREATIVE_SYSTEM
from ..constants import AGENT_WEIGHTS, LLM_MODEL, LLM_TIMEOUT

llm = ChatOpenAI(model=LLM_MODEL, temperature=0.3, timeout=LLM_TIMEOUT)


# ===== Structured Output Models =====
class AgentOpinion(BaseModel):
    """에이전트 의견 구조화 출력"""
    preferred: str = Field(description="선호하는 제안 ID")
    ranking: List[str] = Field(description="순위 리스트 (1위부터)")
    opinion: str = Field(description="선호 이유 (한국어)")
    concerns: str = Field(description="다른 제안들의 우려사항 (한국어)")


# Structured output 적용된 LLM
opinion_llm = llm.with_structured_output(AgentOpinion)


# 에이전트별 시스템 프롬프트 (prompts.py에서 import)
AGENT_PROMPTS = {
    "FIDELITY": FIDELITY_SYSTEM,
    "CREATIVE": CREATIVE_SYSTEM
}


def _flush_print(msg: str):
    """실시간 출력"""
    print(msg)
    sys.stdout.flush()


def _get_agent_opinion(agent_type: str, proposals: list, state: AgentState) -> dict:
    """에이전트 의견 제시 (Structured Output) - DEBATE_PROMPT 템플릿 활용"""
    system_prompt = AGENT_PROMPTS.get(agent_type, AGENT_PROMPTS["FIDELITY"])

    # 제안 목록
    proposals_text = "\n".join([
        f"- {p.get('id')}: {p.get('type')} - {p.get('description')} (risk: {p.get('risk')})"
        for p in proposals
    ])

    # 브릭 목록 (간략화)
    brick_list = f"모델 타입: {state.get('model_type', 'unknown')}, Vision 품질: {state.get('vision_quality_score', 'N/A')}/100"

    # Vision 분석 결과
    vision_problems = state.get('vision_problems', [])
    vision_analysis = "\n".join([f"- {p.get('location')}: {p.get('issue')}" for p in vision_problems[:3]]) if vision_problems else "문제 없음"

    # DEBATE_PROMPT 템플릿 사용
    context = DEBATE_PROMPT.format(
        brick_list=brick_list,
        vision_analysis=vision_analysis,
        proposals=proposals_text
    ) + f"\n\n이전 실패: {state['memory']['failed_approaches'][-3:]}\n선호하는 제안, 순위, 이유, 우려사항을 알려줘."

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=context)
        ]
        opinion: AgentOpinion = opinion_llm.invoke(
            messages,
            config={"run_name": f"의견_{agent_type}", "tags": ["debate", "opinion", agent_type.lower()]}
        )
        return {
            "preferred": opinion.preferred,
            "ranking": opinion.ranking,
            "opinion": opinion.opinion,
            "concerns": opinion.concerns
        }
    except Exception as e:
        _flush_print(f"    [{agent_type}] 의견 실패: {e}")

    # fallback
    fallback_id = proposals[0].get('id') if proposals else "unknown"
    fallback_ranking = [p.get('id') for p in proposals] if proposals else []
    return {
        "preferred": fallback_id,
        "ranking": fallback_ranking,
        "opinion": f"LLM 오류로 기본 선택 ({fallback_id})",
        "concerns": "평가 불가",
        "is_fallback": True
    }


def _final_decision(fidelity_opinion: dict, creative_opinion: dict, proposals: list) -> str:
    """최종 결정: 가중치 기반"""

    # 같은 선택이면 바로 결정
    if fidelity_opinion.get('preferred') == creative_opinion.get('preferred'):
        return fidelity_opinion.get('preferred')

    # 다르면 ranking 기반 점수 계산
    # 1위 = 100점, 2위 = 60점, 3위 = 30점
    def calc_score(opinion: dict, proposal_id: str) -> int:
        ranking = opinion.get('ranking', [])
        if proposal_id in ranking:
            rank = ranking.index(proposal_id)
            return [100, 60, 30, 10][min(rank, 3)]
        return 0

    best_id = None
    best_score = -1

    for p in proposals:
        pid = p.get('id')
        fidelity_score = calc_score(fidelity_opinion, pid) * AGENT_WEIGHTS.get("FIDELITY", 0.5)
        creative_score = calc_score(creative_opinion, pid) * AGENT_WEIGHTS.get("CREATIVE", 0.5)
        total = fidelity_score + creative_score

        if total > best_score:
            best_score = total
            best_id = pid

    return best_id or proposals[0].get('id')


def node_debate(state: AgentState) -> AgentState:
    """Multi-Agent Debate - FIDELITY vs CREATIVE"""
    _flush_print(f"\n[DEBATE] {len(state['proposals'])}개 제안 평가")

    if not state["proposals"]:
        _flush_print("  제안 없음")
        return {**state, "selected_proposal": None}

    if len(state["proposals"]) == 1:
        selected = state["proposals"][0]
        _flush_print(f"  단일 제안 선택: {selected['id']}")
        return {**state, "selected_proposal": selected}

    proposals = state["proposals"]

    # ===== 에이전트 의견 수렴 =====
    _flush_print(f"\n  === 에이전트 의견 ===")

    _flush_print(f"    [FIDELITY] 평가 중...")
    fidelity_opinion = _get_agent_opinion("FIDELITY", proposals, state)
    _flush_print(f"      선호: {fidelity_opinion.get('preferred')}")
    _flush_print(f"      이유: {fidelity_opinion.get('opinion', '')[:200]}...")

    _flush_print(f"    [CREATIVE] 평가 중...")
    creative_opinion = _get_agent_opinion("CREATIVE", proposals, state)
    _flush_print(f"      선호: {creative_opinion.get('preferred')}")
    _flush_print(f"      이유: {creative_opinion.get('opinion', '')[:200]}...")

    # ===== 최종 결정 =====
    _flush_print(f"\n  === 결정 ===")

    consensus = fidelity_opinion.get('preferred') == creative_opinion.get('preferred')

    if consensus:
        final_choice_id = fidelity_opinion.get('preferred')
        _flush_print(f"  만장일치: {final_choice_id}")
    else:
        final_choice_id = _final_decision(fidelity_opinion, creative_opinion, proposals)
        _flush_print(f"  가중 투표: FIDELITY({fidelity_opinion.get('preferred')}) vs CREATIVE({creative_opinion.get('preferred')}) → {final_choice_id}")

    selected = next((p for p in proposals if p.get('id') == final_choice_id), proposals[0])

    # 결과 기록
    debate_summary = {
        "fidelity_preferred": fidelity_opinion.get('preferred'),
        "creative_preferred": creative_opinion.get('preferred'),
        "consensus": consensus
    }
    selected["debate_summary"] = debate_summary

    _flush_print(f"  선택됨: {selected['id']}")

    # 학습용 기록 (immutable 업데이트)
    lesson_entry = f"Debate: FIDELITY→{fidelity_opinion.get('preferred')}, CREATIVE→{creative_opinion.get('preferred')} = {selected['id']}"
    updated_memory = {
        **state["memory"],
        "lessons": state["memory"]["lessons"] + [lesson_entry]
    }

    # 대화 기록 추가 (LangSmith 트레이싱용)
    new_messages = [
        HumanMessage(content=f"[DEBATE] {len(proposals)}개 제안 평가"),
        AIMessage(content=f"FIDELITY: {fidelity_opinion.get('preferred')} - {fidelity_opinion.get('opinion', '')[:100]}"),
        AIMessage(content=f"CREATIVE: {creative_opinion.get('preferred')} - {creative_opinion.get('opinion', '')[:100]}"),
        AIMessage(content=f"최종 선택: {selected['id']} ({'만장일치' if consensus else '가중투표'})")
    ]

    return {**state, "selected_proposal": selected, "proposals": proposals, "memory": updated_memory, "messages": new_messages}
