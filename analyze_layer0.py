
import sys
import os

def analyze_blacks_y0(ldr_path):
    print(f"Analysing {ldr_path}")
    with open(ldr_path, "r") as f:
        lines = f.readlines()
    
    found = []
    for i, line in enumerate(lines):
        if line.startswith("1 "):
            parts = line.split()
            color = parts[1]
            x = float(parts[2])
            y = float(parts[3])
            z = float(parts[4])
            part = parts[14]
            if color == "0" and y == 0:
                found.append((i+1, x, z, part))
    
    found.sort(key=lambda x: (x[2], x[1])) # Sort by Z, then X
    for line_num, x, z, part in found:
        print(f"Line {line_num}: Pos({x}, {z}), Part={part}")

if __name__ == "__main__":
    analyze_blacks_y0("shiba.ldr")
    print("-" * 20)
    analyze_blacks_y0("shiba_merged.ldr")
