
import sys

def check_layer0(ldr_path):
    print(f"Checking {ldr_path}")
    with open(ldr_path, "r") as f:
        for line in f:
            if line.startswith("1 "):
                parts = line.split()
                y = float(parts[3])
                if y == 0:
                    color = parts[1]
                    part = parts[14]
                    x = parts[2]
                    z = parts[4]
                    print(f"Y=0: Color={color}, Pos=({x}, {z}), Part={part}")

if __name__ == "__main__":
    check_layer0("shiba_merged.ldr")
