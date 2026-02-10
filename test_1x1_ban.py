
import sys
import os
from pathlib import Path

# Add brick_engine to path
sys.path.append(str(Path(__file__).parent / "brick_engine"))

try:
    from pylego3d.optimizer import optimize_bricks
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def test_optimization():
    # 2x1 area (two voxels)
    bricks_data = [
        {"x": 0, "y": 0, "z": 0, "color": 4},
        {"x": 1, "y": 0, "z": 0, "color": 4},
    ]

    print("--- Test 1: Default (Should merge into 1x2 or use two 1x1 if small) ---")
    res1 = optimize_bricks(bricks_data, avoid_1x1=False)
    for p in res1:
        print(f"Part: {p['part']}, Size: {p['w']}x{p['l']}")

    print("\n--- Test 2: Avoid 1x1 (Should use 1x2 or fail to 1x1 fallback if impossible) ---")
    # Let's try 3 disconnected voxels
    bricks_data_disconnected = [
        {"x": 0, "y": 0, "z": 0, "color": 4},
        {"x": 2, "y": 0, "z": 0, "color": 4},
        {"x": 4, "y": 0, "z": 0, "color": 4},
    ]
    
    # Normally these would be three 1x1 bricks
    res2 = optimize_bricks(bricks_data_disconnected, avoid_1x1=True)
    print(f"Result count for disconnected with avoid_1x1=True: {len(res2)}")
    for p in res2:
        print(f"Part: {p['part']}, Size: {p['w']}x{p['l']}")

    # Let's try adjacent voxels that CAN be merged
    bricks_data_connected = [
        {"x": 0, "y": 0, "z": 0, "color": 4},
        {"x": 1, "y": 0, "z": 0, "color": 4},
    ]
    res3 = optimize_bricks(bricks_data_connected, avoid_1x1=True)
    print(f"\nResult for connected 2x1 with avoid_1x1=True:")
    for p in res3:
        print(f"Part: {p['part']}, Size: {p['w']}x{p['l']}")

if __name__ == "__main__":
    test_optimization()
