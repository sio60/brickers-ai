"""GENERATE Node - Multi-Agent 제안 생성 (Map 패턴)

핵심 원칙:
1. 제안 2-3개 생성 (CoScientist: 다양한 접근법)
2. RELOCATE/ROTATE/REBUILD가 우선 (형태 개선)
3. ADD_SUPPORT는 최후의 수단
4. 알고리즘이 좌표 계산, LLM은 전략 판단만

Enhanced: Send() API 지원 구조 (병렬 제안 평가용)
- 각 제안은 독립적인 Proposal 객체로 생성
- DEBATE 노드에서 Send()로 병렬 평가 가능
"""
import os
import json
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from ..state import AgentState, Proposal
from ..tools import (
    find_nearby_stable_bricks, generate_support_candidates,
    find_best_placement, _build_occupancy_set
)
from ..prompts import GENERATE_SYSTEM, GENERATE_PROMPT
from ..constants import LLM_MODEL, LLM_TIMEOUT
from ..config import get_config

import sys

llm = ChatOpenAI(model=LLM_MODEL, temperature=0.7, timeout=LLM_TIMEOUT)


# ===== Structured Output for Proposals =====
class ProposalItem(BaseModel):
    """단일 제안 구조"""
    id: str = Field(description="제안 ID (예: relocate_front_1)")
    type: str = Field(description="제안 유형: relocate, rotate, rebuild, remove, bridge, add_support")
    description: str = Field(description="제안 설명 (한국어)")
    approach: str = Field(description="접근 방식: conservative, moderate, aggressive")


class ProposalList(BaseModel):
    """제안 목록 구조화 출력"""
    proposals: List[ProposalItem] = Field(description="2-3개의 제안 목록")


# Structured output 적용된 LLM
proposal_llm = llm.with_structured_output(ProposalList)

def _flush_print(msg: str):
    """실시간 출력"""
    print(msg)
    sys.stdout.flush()


def _generate_multiple_proposals_llm(state: AgentState, base_context: str) -> list:
    """LLM으로 여러 제안 생성 (Structured Output - 파싱 에러 없음)"""
    prompt = f"""{base_context}

2-3개의 서로 다른 제안을 생성해줘.

⚠️ 핵심 규칙: 각 제안은 반드시 모델의 서로 다른 위치/부품을 대상으로 해야 해.
- 같은 위치(예: "왼쪽 다리")에 대해 여러 제안 금지
- 이미 다룬 위치는 다른 제안에서 반복하지 말 것
- 예시 (좋음): 1번=왼쪽 다리 수정, 2번=꼬리 재배치, 3번=오른쪽 날개 회전
- 예시 (나쁨): 1번=왼쪽 다리 이동, 2번=왼쪽 다리 회전, 3번=왼쪽 다리 삭제

접근 방식은 각각 다르게:
1. conservative: 최소 변경, 낮은 위험
2. moderate: 균형 잡힌 접근
3. aggressive: 큰 변경, 더 나은 결과 가능성

사용 가능한 type: relocate, rotate, rebuild, remove, bridge, add_support

모든 응답은 한국어로 해."""

    try:
        messages = [
            SystemMessage(content=GENERATE_SYSTEM),
            HumanMessage(content=prompt)
        ]
        result: ProposalList = proposal_llm.invoke(
            messages,
            config={"run_name": "제안생성", "tags": ["generate", "proposals"]}
        )

        return [
            {
                "id": p.id,
                "type": p.type,
                "description": p.description,
                "approach": p.approach
            }
            for p in result.proposals
        ]
    except Exception as e:
        print(f"  [WARNING] LLM 제안 생성 실패: {e}")

    return []


def _generate_for_strategy(state: AgentState, strategy: str, target) -> list:
    """단일 전략에 대한 제안 생성"""
    proposals = []
    strategy_upper = strategy.upper()

    if strategy_upper == "ROLLBACK":
        proposals.append({
            "id": "rollback",
            "type": "rollback",
            "description": "원본으로 복원",
            "brick_id": None,
            "candidates": None,
            "risk": "none",
            "score": None
        })

    # ===== 형태 개선 전략 - Multi-Proposal 생성 =====
    elif strategy_upper in ["RELOCATE", "ROTATE", "REBUILD"] and target:
        problem = target
        location = problem.get('location', 'unknown')
        issue = problem.get('issue', 'unknown')
        strategy_lower = strategy.lower()

        # LLM으로 여러 접근법 제안 받기
        supervisor_decision = f"전략: {strategy_upper}\n위치: {location}\n문제: {issue}"
        brick_list = f"총 브릭: {state['original_brick_count']}개, 부유: {state['floating_count']}개, 타입: {state.get('model_type', 'unknown')}"
        vision_problems = state.get('vision_problems', [])
        vision_analysis = "\n".join([f"- {p.get('location')}: {p.get('issue')}" for p in vision_problems[:5]]) if vision_problems else "문제 없음"
        memory = state.get('memory', {})
        lessons = memory.get('lessons', [])[-3:]
        lessons_text = "\n".join(lessons) if lessons else "없음"
        success_text = str(memory.get('successful_patterns', [])[-3:]) if memory.get('successful_patterns') else "없음"
        memory_text = f"이전 교훈 (더 나은 제안을 위한 참고):\n{lessons_text}\n성공 패턴: {success_text}"

        context = GENERATE_PROMPT.format(
            supervisor_decision=supervisor_decision,
            brick_list=brick_list,
            vision_analysis=vision_analysis,
            memory=memory_text
        )
        llm_proposals = _generate_multiple_proposals_llm(state, context)

        approach_risk_map = {'conservative': 'low', 'moderate': 'medium', 'aggressive': 'high'}

        if llm_proposals:
            for i, llm_prop in enumerate(llm_proposals[:3]):
                prop_type = llm_prop.get('type', strategy_lower)
                prop_approach = llm_prop.get('approach', 'unknown')
                proposals.append({
                    "id": f"{prop_type}_{location}_{i+1}",
                    "type": prop_type,
                    "description": llm_prop.get('description', f"{strategy_upper}: {location}"),
                    "problem": problem,
                    "approach": prop_approach,
                    "risk": approach_risk_map.get(prop_approach, 'medium'),
                    "score": None
                })
        else:
            proposals.append({
                "id": f"{strategy_lower}_{location}",
                "type": strategy_lower,
                "description": f"{strategy_upper}: {location} - {issue}",
                "problem": problem,
                "approach": "default",
                "risk": "medium",
                "score": None
            })

    elif strategy_upper == "SYMMETRY_FIX":
        symmetry_issues = state.get("symmetry_issues", [])
        if symmetry_issues:
            issue = symmetry_issues[0]
            proposals.append({
                "id": f"symmetry_{issue.get('missing_side', 'unknown')}",
                "type": "symmetry_fix",
                "description": f"대칭 브릭 추가: {issue.get('missing_side')}",
                "symmetry_issue": issue,
                "risk": "low",
                "score": None
            })

    elif strategy_upper == "SELECTIVE_REMOVE":
        if target:
            proposals.append({
                "id": f"remove_{target['id']}",
                "type": "remove",
                "description": f"부유 브릭 삭제: {target['id']}",
                "brick_id": target["id"],
                "candidates": None,
                "risk": "low",
                "score": None
            })

    elif strategy_upper == "BRIDGE" and target:
        floating_brick = next((b for b in state["model"].bricks if b.id == target["id"]), None)
        if floating_brick:
            parts_db = get_config().parts_db
            nearby_stable = find_nearby_stable_bricks(state["model"], floating_brick, parts_db)
            candidates = generate_support_candidates(floating_brick, nearby_stable, parts_db)

            if candidates:
                occupied = _build_occupancy_set(state["model"], parts_db)
                best = find_best_placement(candidates, parts_db, occupied)
                if best:
                    candidates = [best] + [c for c in candidates if c.get("x") != best["x"] or c.get("z") != best["z"]]

            proposals.append({
                "id": f"bridge_{target['id']}",
                "type": "bridge",
                "description": f"연결 브릭 추가: {target['id']} ({len(candidates)}개 후보)",
                "brick_id": target["id"],
                "candidates": candidates,
                "risk": "medium",
                "score": None
            })

    elif strategy_upper == "ADD_SUPPORT" and target:
        floating_brick = next((b for b in state["model"].bricks if b.id == target["id"]), None)
        if floating_brick:
            parts_db = get_config().parts_db
            nearby_stable = find_nearby_stable_bricks(state["model"], floating_brick, parts_db)
            kids_mode = state["model"].mode == "kids" if hasattr(state["model"], "mode") else False
            candidates = generate_support_candidates(floating_brick, nearby_stable, parts_db, kids_mode=kids_mode)

            if candidates:
                occupied = _build_occupancy_set(state["model"], parts_db)
                best = find_best_placement(candidates, parts_db, occupied)
                if best:
                    candidates = [best] + [c for c in candidates if c.get("x") != best["x"] or c.get("z") != best["z"]]

            glb_context = ""
            if state.get("glb_reference") and state["glb_reference"].get("available"):
                ref = state["glb_reference"]
                glb_context = f"모델: {ref.get('name')} ({ref.get('model_type')}), 다리: {ref.get('legs')}"

            proposals.append({
                "id": f"support_{target['id']}",
                "type": "add_support",
                "description": f"지지대 추가: {target['id']} ({len(candidates)}개 후보)",
                "brick_id": target["id"],
                "candidates": candidates,
                "risk": "low",
                "score": None,
                "glb_context": glb_context
            })

    return proposals


def node_generate(state: AgentState) -> AgentState:
    """수정 제안 생성 - 단일 전략"""
    strategy = state.get("strategy", "")
    target = state.get("target_brick")

    if not strategy:
        print("\n[GENERATE] 전략 없음")
        return {**state, "proposals": []}

    print(f"\n[GENERATE] 전략: {strategy}")

    proposals = _generate_for_strategy(state, strategy, target)
    print(f"  {len(proposals)}개 제안 생성됨")
    for p in proposals:
        print(f"    - {p['id']}: {p['description']}")

    # 대화 기록
    proposals_summary = "\n".join([f"- {p['id']}: {p['description']}" for p in proposals])
    new_messages = [
        HumanMessage(content=f"[GENERATE] 전략: {strategy}"),
        AIMessage(content=f"제안 {len(proposals)}개 생성:\n{proposals_summary}")
    ]

    return {**state, "proposals": proposals, "messages": new_messages}
