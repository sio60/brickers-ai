import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Set, Optional

try:
    from .lego_physics import BRICK_SIZES
except ImportError:
    from lego_physics import BRICK_SIZES

class StructureFixer:
    def __init__(self, ldr_path: str):
        self.ldr_path = ldr_path
        self.bricks: List[Dict] = [] # {'id': '...', 'part': '...', 'pos': ...}
        self.graph = nx.Graph()
        self.load_ldr(ldr_path)
        self.build_graph()

    def load_ldr(self, path: str):
        """Parse LDR file and load bricks."""
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('1 '):
                    parts = line.strip().split()
                    # Basic parsing (color, x, y, z, matrix..., part_id)
                    # This is a simplified parser - we might need a more robust one or reuse existing
                    color = int(parts[1])
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    part_id = parts[-1]
                    
                    brick_data = {
                        'id': f"{part_id}_{len(self.bricks)}",
                        'part': part_id,
                        'color': color,
                        'pos': np.array([x, y, z]),
                        # Matrix parsing omitted for brevity in skeleton
                    }
                    self.bricks.append(brick_data)

    def build_graph(self):
        """Build connectivity graph based on stud-tube alignment."""
        self.graph.clear()
        # Add nodes
        for i, b in enumerate(self.bricks):
            self.graph.add_node(b['id'], data=b)
            
        # Add edges (simplified proximity check for now, ideally use lego_physics)
        # In a real implementation, we would use grid alignment checks
        for i, b1 in enumerate(self.bricks):
            for j, b2 in enumerate(self.bricks):
                if i >= j: continue
                # Check vertical connection (y-diff ~ 24 or 8) and x/z overlap
                p1, p2 = b1['pos'], b2['pos']
                # LDU coordinates: Y is up axis? No, LDraw Y is down.
                # Assuming standard LDraw: Y is vertical.
                y_diff = abs(p1[1] - p2[1])
                if 20 <= y_diff <= 28: # Approx 24 LDU (1 brick height)
                     # Check horizontal overlap
                     # Simple distance check for now (assuming 1x1 or similar centers)
                     xz_dist = np.linalg.norm(p1[[0,2]] - p2[[0,2]])
                     if xz_dist < 20: # Within 1 stud spacing
                         self.graph.add_edge(b1['id'], b2['id'])

    def is_weak_column(self, brick_id: str) -> bool:
        """Check if the brick is part of a fragile vertical column (1x1 stack)."""
        # Traverse up and down
        visited = {brick_id}
        stack_height = 1
        
        # Check above
        curr = brick_id
        while True:
            neighbors = list(self.graph.neighbors(curr))
            upper = [n for n in neighbors if self.bricks[self._get_idx(n)]['pos'][1] < self.bricks[self._get_idx(curr)]['pos'][1]] # Y decreases upwards in LDraw
            if not upper: break
            curr = upper[0] # Assume single column
            if curr in visited: break
            visited.add(curr)
            stack_height += 1
            
        # Check below
        curr = brick_id
        while True:
            neighbors = list(self.graph.neighbors(curr))
            lower = [n for n in neighbors if self.bricks[self._get_idx(n)]['pos'][1] > self.bricks[self._get_idx(curr)]['pos'][1]]
            if not lower: break
            curr = lower[0]
            if curr in visited: break
            visited.add(curr)
            stack_height += 1
            
        return stack_height >= 3

    def is_single_support(self, brick_id: str) -> bool:
        """Check if the brick is supported by only one connection below."""
        neighbors = list(self.graph.neighbors(brick_id))
        brick_y = self.bricks[self._get_idx(brick_id)]['pos'][1]
        # Count neighbors below this brick
        params_below = [n for n in neighbors if self.bricks[self._get_idx(n)]['pos'][1] > brick_y]
        return len(params_below) == 1

    def apply_fix(self, fix_instruction: Dict) -> bool:
        """Apply a specific fix to the LDR structure."""
        target_id = fix_instruction['target']
        idx = self._get_idx(target_id)
        if idx is None: return False
        
        target_brick = self.bricks[idx]
        print(f"[Fixer] Applying {fix_instruction['type']} to {target_id}")

        if fix_instruction['type'] == 'reinforce_column':
            # Add a 1x2 plate next to the target to bond it
            # Simplified: finding a free spot at x+20 or z+20
            new_pos = target_brick['pos'].copy()
            new_pos[0] += 20.0 # Shift X by 1 stud
            
            self.bricks.append({
                'id': f"3024.dat_fix_{len(self.bricks)}",
                'part': "3024.dat", # 1x1 plate for simplicity, ideally 1x2
                'color': 5, # Dark Pink for visibility
                'pos': new_pos
            })
            return True
            
        elif fix_instruction['type'] == 'reinforce_base':
            # Add a larger plate below
            new_pos = target_brick['pos'].copy()
            new_pos[1] += 24.0 # One brick height down (or 8 for plate)
            
            self.bricks.append({
                'id': f"3020.dat_fix_{len(self.bricks)}",
                'part': "3020.dat", # 2x4 plate
                'color': 5,
                'pos': new_pos
            })
            return True

        return False

    def _get_idx(self, brick_id: str) -> Optional[int]:
        for i, b in enumerate(self.bricks):
            if b['id'] == brick_id: return i
        return None

    def save_fixed_ldr(self, output_path: str):
        """Save the modified bricks to a new LDR file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("0 Fixed by StructureFixer\n")
            for b in self.bricks:
                # 1 <color> <x> <y> <z> <matrices> <part>
                # Using identity matrix for simplicity in this proto
                f.write(f"1 {b['color']} {b['pos'][0]:.2f} {b['pos'][1]:.2f} {b['pos'][2]:.2f} 1 0 0 0 1 0 0 0 1 {b['part']}\n")
