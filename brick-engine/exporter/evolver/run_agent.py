"""
Evolver Agent Runner
Usage: python run_agent.py <ldr_path>
"""
import os
import sys
import copy
import json
from pathlib import Path
from dotenv import load_dotenv

# Setup paths
EVOLVER_DIR = Path(__file__).parent
EXPORTER_DIR = EVOLVER_DIR.parent
sys.path.insert(0, str(EXPORTER_DIR))

load_dotenv(EXPORTER_DIR.parent.parent / ".env")

from ldr_converter import ldr_to_brick_model, model_to_ldr  # noqa: E402
from agent import build_graph, AgentState  # noqa: E402
from agent.tools import init_tools, load_parts_db, get_model_state, analyze_glb  # noqa: E402

def run_agent(ldr_path: str, glb_path: str = None):
    """Run the evolver agent on an LDR file"""
    print("=" * 60)
    print("LangGraph + CoScientist Evolver Agent")
    print("=" * 60)

    # Initialize
    parts_db = load_parts_db()
    if not parts_db:
        cache = EXPORTER_DIR / "parts_cache.json"
        if cache.exists():
            with open(cache, 'r', encoding='utf-8') as f:
                parts_db = json.load(f)

    init_tools(parts_db, EXPORTER_DIR)

    model = ldr_to_brick_model(ldr_path)
    model.name = Path(ldr_path).stem

    # GLB 분석 (있으면)
    glb_ref = None
    if glb_path and Path(glb_path).exists():
        print(f"\n[GLB Reference] Analyzing {glb_path}...")
        glb_ref = analyze_glb(glb_path)
        if glb_ref.get("available"):
            print(f"  Model: {glb_ref.get('name')} ({glb_ref.get('model_type')})")
            print(f"  Legs: {glb_ref.get('legs')}")
            print(f"  Features: {glb_ref.get('key_features')}")

    initial_state: AgentState = {
        "model": model,
        "model_backup": copy.deepcopy(model),
        "parts_db": parts_db,
        "original_brick_count": len(model.bricks),
        "glb_reference": glb_ref,
        "floating_count": 0,
        "collision_count": 0,
        "floating_bricks": [],
        "verification_result": None,
        "verification_score": 100.0,
        "verification_evidence": [],
        "iteration": 0,
        "total_removed": 0,
        "action_history": [],
        "strategy": "",
        "target_brick": None,
        "proposals": [],
        "selected_proposal": None,
        "memory": {
            "failed_approaches": [],
            "successful_patterns": [],
            "lessons": [],
            "consecutive_failures": 0
        },
        "should_finish": False,
        "finish_reason": ""
    }

    print(f"\nLoaded: {ldr_path}")
    print(f"Bricks: {initial_state['original_brick_count']}")
    print(f"Removal limit: {int(initial_state['original_brick_count'] * 0.10)} (10%)")

    # Run graph
    graph = build_graph()
    final_state = graph.invoke(initial_state)

    # Save result
    output = Path(ldr_path).parent / f"{Path(ldr_path).stem}_evolved.ldr"
    ldr = model_to_ldr(final_state["model"], parts_db, skip_validation=True, skip_physics=True)
    with open(output, 'w', encoding='utf-8') as f:
        f.write(ldr)

    # Print summary
    initial_floating = get_model_state(initial_state["model_backup"], parts_db)["floating_count"]

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Floating: {initial_floating} -> {final_state['floating_count']}")
    print(f"Removed: {final_state['total_removed']}")
    print(f"Iterations: {final_state['iteration']}")
    print(f"Reason: {final_state['finish_reason']}")
    print(f"Saved: {output}")

    print("\n[Lessons Learned]")
    for lesson in final_state["memory"]["lessons"]:
        print(f"  - {lesson}")

    print("=" * 60)

if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    if len(sys.argv) < 2:
        print("Usage: python run_agent.py <ldr_path> [glb_path]")
        sys.exit(1)

    ldr = sys.argv[1]
    if not Path(ldr).exists():
        print(f"Error: {ldr} not found")
        sys.exit(1)

    glb = sys.argv[2] if len(sys.argv) > 2 else None
    run_agent(ldr, glb)
