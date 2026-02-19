
import sys
import os
from pathlib import Path
from collections import defaultdict

def parse_ldr_line(line):
    parts = line.strip().split()
    if not parts or parts[0] != "1":
        return None
    
    try:
        color = int(parts[1])
        x = float(parts[2])
        y = float(parts[3])
        z = float(parts[4])
        matrix = [float(p) for p in parts[5:14]]
        part_name = " ".join(parts[14:])
        return {
            "color": color,
            "x": x, "y": y, "z": z,
            "matrix": matrix,
            "part": part_name
        }
    except:
        return None

def audit_duplicates(ldr_path):
    print(f"Auditing file: {ldr_path}")
    path = Path(ldr_path)
    if not path.exists():
        print(f"Error: File not found: {ldr_path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    position_counts = defaultdict(list)
    total_bricks = 0
    
    for i, line in enumerate(lines):
        parsed = parse_ldr_line(line)
        if parsed is None:
            continue
            
        total_bricks += 1
        # Key: (x, y, z, part, color) - Exact Duplicate Check
        # Also check for overlapping different bricks at same POS
        key = (int(round(parsed["x"])), int(round(parsed["y"])), int(round(parsed["z"])))
        position_counts[key].append((i + 1, parsed["part"], parsed["color"]))

    duplicates = 0
    for pos, bricks in position_counts.items():
        if len(bricks) > 1:
            print(f"Duplicate at {pos}: {len(bricks)} bricks")
            for line_num, part, color in bricks:
                print(f"  Line {line_num}: {part} (Color {color})")
            duplicates += 1

    print(f"Total Bricks: {total_bricks}")
    print(f"Total Duplicate Groups: {duplicates}")
    if duplicates == 0:
        print("No duplicates found.")
    print("-" * 50)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python audit_duplicates.py <ldr_files...>")
    else:
        with open("audit_report.txt", "w") as log:
            # Also print to console for debugging
            original_stdout = sys.stdout
            
            # Simple Tee
            class Tee(object):
                def __init__(self, *files):
                    self.files = files
                def write(self, obj):
                    for f in self.files:
                        f.write(obj)
                        f.flush() # Ensure immediate write
                def flush(self):
                    for f in self.files:
                        f.flush()

            sys.stdout = Tee(sys.stdout, log)
            
            try:
                for fpath in sys.argv[1:]:
                    audit_duplicates(fpath)
            except Exception as e:
                print(f"ERROR: {e}")
            finally:
                sys.stdout = original_stdout
