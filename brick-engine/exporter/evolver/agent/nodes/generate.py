"""GENERATE Node - Create repair proposals"""
import os
import json
from openai import OpenAI
from ..state import AgentState, Proposal
from ..tools import find_nearby_stable_bricks, generate_support_candidates

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def node_generate(state: AgentState) -> AgentState:
    """Generate repair proposals - LLM decides strategy, algorithm calculates positions"""
    print(f"\n[GENERATE] Strategy: {state['strategy']}")

    proposals = []

    if state["strategy"] == "AUTO_EVOLVE":
        proposals.append({
            "id": "auto_evolve",
            "type": "auto_evolve",
            "description": "Run CoScientist Evolver",
            "brick_id": None,
            "candidates": None,
            "risk": "medium",
            "score": None
        })

    elif state["strategy"] == "ROLLBACK":
        proposals.append({
            "id": "rollback",
            "type": "rollback",
            "description": "Restore to original",
            "brick_id": None,
            "candidates": None,
            "risk": "none",
            "score": None
        })

    elif state["strategy"] in ["ADD_SUPPORT", "BRIDGE"] and state["target_brick"]:
        target = state["target_brick"]

        # 부유 브릭 찾기
        floating_brick = next(
            (b for b in state["model"].bricks if b.id == target["id"]),
            None
        )

        if floating_brick:
            # 알고리즘으로 주변 안정 브릭 탐색
            nearby_stable = find_nearby_stable_bricks(
                state["model"],
                floating_brick,
                state["parts_db"]
            )

            # 알고리즘으로 후보 위치 생성
            candidates = generate_support_candidates(
                floating_brick,
                nearby_stable,
                state["parts_db"]
            )

            print(f"  Found {len(nearby_stable)} nearby stable bricks")
            print(f"  Generated {len(candidates)} candidate positions")

            # LLM은 전략 판단만
            glb_context = ""
            if state.get("glb_reference") and state["glb_reference"].get("available"):
                ref = state["glb_reference"]
                glb_context = f"Model: {ref.get('name')} ({ref.get('model_type')}), legs: {ref.get('legs')}"

            # 제안 생성 (좌표는 알고리즘이 계산한 candidates 사용)
            proposals.append({
                "id": f"support_{target['id']}",
                "type": "add_support",
                "description": f"Add support bricks for {target['id']} ({len(candidates)} candidates)",
                "brick_id": target["id"],
                "candidates": candidates,  # 알고리즘이 계산한 후보 위치들
                "risk": "low",
                "score": None,
                "glb_context": glb_context
            })
        else:
            print(f"  Warning: Could not find brick {target['id']}")

    elif state["strategy"] == "SELECTIVE_REMOVE" and state["target_brick"]:
        target = state["target_brick"]
        proposals.append({
            "id": f"remove_{target['id']}",
            "type": "remove",
            "description": f"Remove floating brick {target['id']}",
            "brick_id": target["id"],
            "candidates": None,
            "risk": "low",
            "score": None
        })

    print(f"  Generated {len(proposals)} proposals")
    for p in proposals:
        print(f"    - {p['id']}: {p['description']}")

    return {**state, "proposals": proposals}
