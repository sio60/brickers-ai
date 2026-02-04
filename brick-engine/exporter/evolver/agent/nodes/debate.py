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

    # RAG: Retrieve similar cases acting as 'The Critic'
    critic_context = ""
    try:
        import config  # This registers AGENT_DIR in sys.path
        from memory_utils import memory_manager
        
        if memory_manager:
            obs = f"Floating: {state['floating_count']}, Removed: {state['total_removed']}"
            similar_cases = memory_manager.search_similar_cases(obs, limit=2)
            if similar_cases:
                critic_context = "\n[CRITIC (Past Experience)]:\n"
                for case in similar_cases:
                    outcome = "SUCCESS" if case.get('result_success') else "FAILURE"
                    tool = case['experiment'].get('tool', 'Unknown')
                    result = case['verification'].get('numerical_analysis', 'N/A')
                    critic_context += f"- {outcome} with {tool}: {result}\n"
    except Exception as e:
        print(f"  (Critic unavailable: {e})")

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
