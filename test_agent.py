
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from current directory
load_dotenv()

# Add brick-engine to path
current_dir = Path(__file__).resolve().parent
brick_engine_dir = current_dir / "brick-engine"
sys.path.append(str(brick_engine_dir))

# Also add brick-engine/agent to path if needed (for internal imports)
sys.path.append(str(brick_engine_dir / "agent"))

try:
    from agent.llm_regeneration_agent import regeneration_loop
    from agent.llm_clients import GeminiClient
except ImportError as e:
    print(f"Import Error: {e}")
    print("Trying alternative import...")
    sys.path.append(str(brick_engine_dir / "agent"))
    from llm_regeneration_agent import regeneration_loop
    from llm_clients import GeminiClient

def test_kids_generation():
    # 1. Inputs
    # You need a sample GLB file. Change this path to a real GLB file on your system.
    # If users have one in uploads, use that.
    # Searching for a GLB file...
    glb_path = "uploads/test.glb" 
    # If no file exists, we can't test. User needs to provide one.
    # But for the script, I'll assume one exists or ask user to provide path.
    
    if len(sys.argv) > 1:
        glb_path = sys.argv[1]
    
    if not os.path.exists(glb_path):
        print(f"❌ GLB file not found: {glb_path}")
        print("Usage: python test_agent.py <path_to_glb>")
        return

    output_ldr = "output_test.ldr"

    # 2. Params (Kids Mode L1 Config)
    params = {
        "target": 60,
        "budget": 400,
        "shrink": 0.85,
        "flipx180": False,
        "flipy180": False, # Adjust if needed
        "invert_y": False,
        "smart_fix": True,
        "smart_fix": True,
        "support_ratio": 0.3,      # default
        "small_side_contact": True,
        "fill": True,              # [IMPORTANT] Solid
        "erosion_iters": 1,        # default
        # Force v3 logic
        "span": 4,
        "max_new_voxels": 12000,
    }

    print("="*60)
    print(f"🚀 Testing Agent with Kids Params: Target={params['target']}, Budget={params['budget']}")
    print("="*60)

    # 3. initialize LLM (optional, falls back if None but better to have)
    # Assumes GENAI_API_KEY or GEMINI_API_KEY is set in env
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GENAI_API_KEY")
    client = None
    if api_key:
        client = GeminiClient(api_key=api_key)
    else:
        print("⚠️ No API Key found. Agent might fail at Model phase.")

    # 4. Run Loop
    final_state = regeneration_loop(
        glb_path=glb_path,
        output_ldr_path=output_ldr,
        llm_client=client,
        max_retries=3,
        params=params,
        gui=False
    )

    # 5. Check Result
    print("\n" + "="*60)
    if os.path.exists(output_ldr) and os.path.getsize(output_ldr) > 0:
        print(f"✅ Success: LDR generated at {output_ldr}")
        print(f"   Size: {os.path.getsize(output_ldr)} bytes")
    else:
        print("❌ Failure: LDR not generated or empty.")

if __name__ == "__main__":
    test_kids_generation()
