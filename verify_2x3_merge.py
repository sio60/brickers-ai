
import sys

def count_2x3_black_y0(filename):
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return 0

    count = 0
    for line in lines:
        if line.startswith('1 '):
            parts = line.split()
            # Color 0 (Black), Y=0 (Layer 0), Part 3002 (2x3 Brick)
            if parts[1] == '0' and float(parts[3]) == 0 and '3002.dat' in parts[14]:
                count += 1
                print(f"Found in {filename}: {line.strip()}")
    
    print(f"Total 2x3 black bricks at Y=0 in {filename}: {count}")
    return count

if __name__ == "__main__":
    print("-" * 20)
    c1 = count_2x3_black_y0("shiba.ldr")
    print("-" * 20)
    c2 = count_2x3_black_y0("shiba_merged.ldr")
    print("-" * 20)
    
    if c2 >= c1:
        print("SUCCESS: 2x3 bricks preserved or restored.")
    else:
        print("WARNING: 2x3 brick count decreased (Broken and not restored).")
        
    sys.stdout.flush()
