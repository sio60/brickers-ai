"""DEBATE Node - Evaluate and select best proposal"""
import os
import json
from openai import OpenAI
from ..state import AgentState
from ..prompts import DEBATE_SYSTEM

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def node_debate(state: AgentState) -> AgentState:
    """Evaluate proposals and select the best one"""
    print(f"\n[DEBATE] Evaluating {len(state['proposals'])} proposals...")

    if not state["proposals"]:
        print("  No proposals to evaluate")
        return {**state, "selected_proposal": None}

    if len(state["proposals"]) == 1:
        selected = state["proposals"][0]
        print(f"  Selected (only option): {selected['id']}")
        return {**state, "selected_proposal": selected}

    # LLM debate
    prompt = f"""
Current state:
- Floating: {state['floating_count']}
- Removed: {state['total_removed']}
- Failed approaches: {state['memory']['failed_approaches'][-3:]}

Proposals:
{json.dumps(state['proposals'], indent=2)}

Which proposal ID should we select?"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": DEBATE_SYSTEM},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    selected_id = response.choices[0].message.content.strip()
    selected = next(
        (p for p in state["proposals"] if p["id"] in selected_id),
        state["proposals"][0]
    )

    print(f"  Selected: {selected['id']}")

    return {**state, "selected_proposal": selected}
