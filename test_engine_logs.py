
import os
import sys
from pathlib import Path

# Add brick_engine to path
curr_dir = Path(__file__).parent.absolute()
sys.path.append(str(curr_dir / "brick_engine"))

from brick_engine.glb_to_ldr_embedded import convert_glb_to_ldr

def test_log_callback():
    print("Starting test_log_callback...")
    
    logs = []
    def log_cb(step, msg):
        print(f"[{step}] {msg}")
        logs.append((step, msg))
    
    # Simple dummy GLB Path (using a dummy so it fails early but we check _log)
    dummy_glb = "dummy.glb"
    with open(dummy_glb, "w") as f:
        f.write("dummy")
        
    try:
        # We expect this to fail eventually because dummy.glb is not a real GLB,
        # but we want to see if the first _log calls work.
        convert_glb_to_ldr(
            dummy_glb, 
            "out.ldr", 
            target=5, 
            budget=10, 
            log_callback=log_cb,
            search_iters=1
        )
    except Exception as e:
        print(f"Expected failure: {e}")
        
    # Clean up
    if os.path.exists(dummy_glb):
        os.remove(dummy_glb)
    if os.path.exists("out.ldr"):
        os.remove("out.ldr")
        
    if any(step == "brickify" for step, msg in logs):
        print("SUCCESS: Log callback received 'brickify' logs.")
    else:
        print("FAILURE: No 'brickify' logs received.")

if __name__ == "__main__":
    test_log_callback()
