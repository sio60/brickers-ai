import sys
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load .env from current directory
load_dotenv()

# Add brick_engine to path
current_dir = Path(__file__).resolve().parent
brick_engine_dir = current_dir / "brick_engine"
sys.path.append(str(brick_engine_dir))
sys.path.append(str(brick_engine_dir / "agent"))

try:
    from agent.llm_regeneration_agent import regeneration_loop
    from agent.llm_clients import GeminiClient
except ImportError as e:
    sys.path.append(str(brick_engine_dir / "agent"))
    from llm_regeneration_agent import regeneration_loop
    from llm_clients import GeminiClient

def test_kids_generation():
    parser = argparse.ArgumentParser(description="Test AI Agent for Lego Generation")
    parser.add_argument("glb_path", nargs="?", default="uploads/test.glb", help="Path to input GLB file")
    parser.add_argument("budget_pos", nargs="?", type=int, help="Target brick count budget (positional)")
    parser.add_argument("--budget", type=int, help="Target brick count budget (option)")
    parser.add_argument("--out", default="output_test.ldr", help="Path to output LDR file")
    parser.add_argument("--target", type=int, default=60, help="Resolution target (studs)")
    
    args = parser.parse_args()
    
    # Priority: --budget > positional budget > default 200
    effective_budget = args.budget if args.budget is not None else (args.budget_pos if args.budget_pos is not None else 200)

    if not os.path.exists(args.glb_path):
        print(f"FAILED: GLB file not found: {args.glb_path}")
        parser.print_help()
        return

    # 2. Params (Kids Mode L1 Config)
    params = {
        "target": args.target,
        "budget": effective_budget,
        "shrink": 0.8,
        "search_iters": 12,
        "flipx180": False,
        "flipy180": False,
        "invert_y": False,
        "smart_fix": True,
        "support_ratio": 0.3,
        "small_side_contact": True,
        "fill": True,
        "erosion_iters": 1,
        "span": 4,
        "max_new_voxels": 12000,
    }

    print("="*60)
    print(f"Testing Agent: {args.glb_path}")
    print(f"   Target Studs: {params['target']}, Budget: {params['budget']}")
    print("="*60)

    # 3. initialize LLM
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GENAI_API_KEY")
    client = None
    if api_key:
        client = GeminiClient(api_key=api_key)
    else:
        print("⚠️ No API Key found. Agent might fail at Model phase.")

    # 4. Run Loop
    final_state = regeneration_loop(
        glb_path=args.glb_path,
        output_ldr_path=args.out,
        llm_client=client,
        max_retries=3,
        params=params,
        gui=False
    )

    # 5. Check Result
    print("\n" + "="*60)
    if os.path.exists(args.out) and os.path.getsize(args.out) > 0:
        print(f"SUCCESS: LDR generated at {args.out}")
        print(f"   Size: {os.path.getsize(args.out)} bytes")
    else:
        print("FAILURE: LDR not generated or empty.")

if __name__ == "__main__":
    test_kids_generation()
