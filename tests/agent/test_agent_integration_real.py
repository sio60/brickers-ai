
import asyncio
import os
import sys
import logging
import json
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../brick-engine')))

# Load environment variables
load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../.env')))

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AgentIntegrationTest")

async def test_agent_integration():
    print("\n🧪 Testing LLMRegenerationAgent Integration (Real)...")
    
    try:
        from agent.llm_regeneration_agent import RegenerationGraph, AgentState
    except ImportError as e:
        print(f"❌ Import Failed: {e}")
        return

    # 1. Initialize Agent
    print("   - Initializing RegenerationGraph...")
    agent = RegenerationGraph()
    
    if not hasattr(agent, 'hypothesis_maker'):
        print("❌ Agent does not have 'hypothesis_maker' attribute!")
        return
    print("✅ Agent initialized with HypothesisMaker.")

    # 2. Simulate State
    state = {
        "messages": [], # Mock messages if needed
        "observation": "Top-heavy structure collapses immediately after simulation starts.",
        "verification_result": {
            "metrics_after": {
                "total_bricks": 200,
                "stability_score": 0.1
            }
        },
        "params": {},
        "attempts": 0,
        "max_retries": 15
    }
    
    # Mock HumanMessage for node_hypothesize logic
    from langchain_core.messages import HumanMessage
    state['messages'].append(HumanMessage(content=state['observation']))

    print(f"\n[Scenario] Observation: {state['observation']}")
    
    # 3. Call node_hypothesize directly
    print("   - Calling node_hypothesize()...")
    try:
        result = await agent.node_hypothesize(state)
        
        # 4. Output Results
        print("\n" + "="*50)
        print("   🧪 AGENT NODE RESULT")
        print("="*50)
        hypothesis = result.get('current_hypothesis', {})
        print(f"Hypothesis : {hypothesis.get('hypothesis')}")
        print(f"Reasoning  : {hypothesis.get('reasoning')}")
        print(f"Params     : {json.dumps(hypothesis.get('proposed_params'), indent=2)}")
        print(f"Next Action: {result.get('next_action')}")
        print("="*50 + "\n")
        
        if hypothesis.get('proposed_params'):
            print("✅ Proposed Parameters detected!")
        else:
            print("⚠️ No proposed_params found in result.")
        
        if result.get('next_action') == "strategy":
            print("✅ Agent Logic Flow Verified!")
        else:
            print("⚠️ Unexpected Next Action.")
            
    except Exception as e:
        print(f"\n❌ Execution Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_agent_integration())
