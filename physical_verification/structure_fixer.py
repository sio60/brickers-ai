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
                    color = int(parts[1])
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    # Parse rotation matrix (9 values)
                    matrix = [float(p) for p in parts[5:14]]
                    part_id = " ".join(parts[14:]) # Handle filenames with spaces if any
                    
                    brick_data = {
                        'id': f"{part_id}_{len(self.bricks)}",
                        'part': part_id,
                        'color': color,
                        'pos': np.array([x, y, z]),
                        'matrix': matrix
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
        current_idx = self._get_idx(brick_id)
        if current_idx is None: return False
        
        visited = {brick_id}
        stack_height = 1
        
        # Check above
        curr = brick_id
        while True:
            neighbors = list(self.graph.neighbors(curr))
            # Y decreases upwards in LDraw (neg Y is up). So upper brick has LOWER Y value.
            curr_y = self.bricks[self._get_idx(curr)]['pos'][1]
            upper = [n for n in neighbors if self.bricks[self._get_idx(n)]['pos'][1] < curr_y] 
            if not upper: break
            curr = upper[0] # Assume single column
            if curr in visited: break
            visited.add(curr)
            stack_height += 1
            
        # Check below
        curr = brick_id
        while True:
            neighbors = list(self.graph.neighbors(curr))
            curr_y = self.bricks[self._get_idx(curr)]['pos'][1]
            lower = [n for n in neighbors if self.bricks[self._get_idx(n)]['pos'][1] > curr_y]
            if not lower: break
            curr = lower[0]
            if curr in visited: break
            visited.add(curr)
            stack_height += 1
            
        return stack_height >= 3

    def is_single_support(self, brick_id: str) -> bool:
        """Check if the brick is supported by only one connection below."""
        if not self.graph.has_node(brick_id): return False
        neighbors = list(self.graph.neighbors(brick_id))
        brick_y = self.bricks[self._get_idx(brick_id)]['pos'][1]
        # Count neighbors below this brick (Y value is larger)
        params_below = [n for n in neighbors if self.bricks[self._get_idx(n)]['pos'][1] > brick_y]
        return len(params_below) == 1

    def analyze_failure(self, failure_report: Dict) -> List[Dict]:
        """Analyze the failure report and suggest fixes."""
        fixes = []
        target_id = failure_report.get('first_failure_id')
        
        if not target_id:
            return []
            
        print(f"[Fixer] Analyzing failure at {target_id}...")
        
        # Check for Weak Column
        if self.is_weak_column(target_id):
            print(f"  -> Detected Weak Column pattern.")
            fixes.append({
                'type': 'reinforce_column',
                'target': target_id
            })
            
        # Check for Single Support
        elif self.is_single_support(target_id):
            print(f"  -> Detected Single Support pattern.")
            fixes.append({
                'type': 'reinforce_base',
                'target': target_id
            })
            
        # Default fallback (if simply disconnected/floating, maybe add base?)
        else:
             print(f"  -> Unknown pattern, trying base reinforcement as fallback.")
             fixes.append({
                'type': 'reinforce_base',
                'target': target_id
            })
            
        return fixes

    def apply_fix(self, fix_instruction: Dict) -> bool:
        """Apply a specific fix to the LDR structure."""
        target_id = fix_instruction['target']
        idx = self._get_idx(target_id)
        if idx is None: return False
        
        target_brick = self.bricks[idx]
        print(f"[Fixer] Applying {fix_instruction['type']} to {target_id}")

        if fix_instruction['type'] == 'reinforce_column':
            # Side support (Buttress)
            # Add a vertical stack next to it to thicken the column
            new_pos = target_brick['pos'].copy()
            new_pos[0] += 20.0 # Shift X by 1 stud
            
            self.bricks.append({
                'id': f"3001.dat_fix_col_{len(self.bricks)}",
                'part': "3001.dat", # Use a full brick (2x4) for strong support
                'color': 5, # Dark Pink
                'pos': new_pos,
                'matrix': target_brick['matrix'][:]
            })
            return True
            
        elif fix_instruction['type'] == 'reinforce_base':
            # "Big Foot" Strategy: Widen the base area
            # Add 2x4 bricks around the target to create a platform
            offsets = [
                (20.0, 0, 0),   # +X
                (-20.0, 0, 0),  # -X
                (0, 0, 20.0),   # +Z
                (0, 0, -20.0)   # -Z
            ]
            
            for i, (dx, dy, dz) in enumerate(offsets):
                new_pos = target_brick['pos'].copy()
                new_pos[0] += dx
                new_pos[1] += dy # Same height
                new_pos[2] += dz
                
                self.bricks.append({
                    'id': f"3020.dat_fix_base_{i}_{len(self.bricks)}",
                    'part': "3020.dat", # 2x4 Plate
                    'color': 5, # Dark Pink
                    'pos': new_pos,
                    'matrix': target_brick['matrix'][:]
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
                # Format: 1 <color> <x> <y> <z> <rotation matrix 9 values> <part>
                matrix_str = " ".join(f"{v:.6f}" for v in b['matrix'])
                f.write(f"1 {b['color']} {b['pos'][0]:.2f} {b['pos'][1]:.2f} {b['pos'][2]:.2f} {matrix_str} {b['part']}\n")
