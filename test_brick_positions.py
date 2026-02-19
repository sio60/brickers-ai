
import sys
import os
import traceback

if __name__ == "__main__":
    with open("test_log.txt", "w") as log:
        original_stdout = sys.stdout
        sys.stdout = log
        try:
            print("Starting test script...")
            # Create a dummy ldr_modifier context or import it
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            
            # Import here to catch import errors in the log
            print("Importing ldr_modifier...")
            from brick_engine.agent.ldr_modifier import _get_brick_stud_positions, BRICK_DIMENSIONS
            print("Import successful.")

            def test_positions():
                print("Testing 1x1 Brick (3005.dat)...")
                b1 = {
                    "part": "3005.dat",
                    "x": 0, "y": 0, "z": 0,
                    "matrix": [1, 0, 0, 0, 1, 0, 0, 0, 1]
                }
                pos1 = _get_brick_stud_positions(b1)
                print(f"1x1 positions ({len(pos1)}): {pos1}")
                assert len(pos1) == 1
                assert pos1[0] == (0, 0, 0)

                print("\nTesting 2x4 Brick (3001.dat)...")
                # LDraw coords: 2x4 brick defined at 0,0,0
                b2 = {
                    "part": "3001.dat",
                    "x": 0, "y": 0, "z": 0,
                    "matrix": [1, 0, 0, 0, 1, 0, 0, 0, 1]
                }
                pos2 = _get_brick_stud_positions(b2)
                print(f"2x4 positions ({len(pos2)}): {pos2}")
                
                assert len(pos2) == 8
                
                # Check for expected coordinates
                expected_x = [-30, -10, 10, 30]
                expected_z = [-10, 10]
                
                found_coords = set(pos2)
                print("Found coords:", found_coords)
                
                for z in expected_z:
                    for x in expected_x:
                        assert (x, 0, z) in found_coords, f"Missing ({x}, 0, {z})"
                        
                print("\nSUCCESS: All positions match expected grid!")

            test_positions()
            
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc(file=log)
        finally:
            sys.stdout = original_stdout
