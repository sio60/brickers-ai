# 이 파일은 물리 검증 결과(불안정성)를 분석하여 자동으로 구조를 보강(기둥 보강, 바닥 보강 등)하는 로직을 담당합니다.
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
        """LDR 파일을 파싱하고 브릭을 로드합니다."""
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('1 '):
                    parts = line.strip().split()
                    # 기본 파싱 (색상, x, y, z, 행렬..., 부품ID)
                    color = int(parts[1])
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    # 회전 행렬 파싱 (9개 값)
                    matrix = [float(p) for p in parts[5:14]]
                    part_id = " ".join(parts[14:]) # 파일명에 공백이 있는 경우 처리
                    
                    brick_data = {
                        'id': f"{part_id}_{len(self.bricks)}",
                        'part': part_id,
                        'color': color,
                        'pos': np.array([x, y, z]),
                        'matrix': matrix
                    }
                    self.bricks.append(brick_data)

    def build_graph(self):
        """스터드-튜브 정렬을 기반으로 연결성 그래프를 구축합니다."""
        self.graph.clear()
        # 노드 추가
        for i, b in enumerate(self.bricks):
            self.graph.add_node(b['id'], data=b)
            
        # 간선 추가 (현재는 단순 근접 확인, 이상적으로는 lego_physics 사용)
        # 실제 구현에서는 그리드 정렬 검사를 사용해야 함
        for i, b1 in enumerate(self.bricks):
            for j, b2 in enumerate(self.bricks):
                if i >= j: continue
                # 수직 연결 (y차이 ~ 24 또는 8) 및 x/z 겹침 확인
                p1, p2 = b1['pos'], b2['pos']
                # LDU 좌표: Y는 위쪽? 아님, LDraw Y는 아래쪽
                # 표준 LDraw 가정: Y는 수직
                y_diff = abs(p1[1] - p2[1])
                if 20 <= y_diff <= 28: # 약 24 LDU (브릭 1개 높이)
                     # 수평 겹침 확인
                     # 현재는 단순 거리 확인 (1x1 또는 유사한 중심 가정)
                     xz_dist = np.linalg.norm(p1[[0,2]] - p2[[0,2]])
                     if xz_dist < 20: # 1 스터드 간격 이내
                         self.graph.add_edge(b1['id'], b2['id'])

    def is_weak_column(self, brick_id: str) -> bool:
        """브릭이 약한 수직 기둥(1x1 적층)의 일부인지 확인합니다."""
        # 위아래로 탐색
        current_idx = self._get_idx(brick_id)
        if current_idx is None: return False
        
        visited = {brick_id}
        stack_height = 1
        
        # 위쪽 확인
        curr = brick_id
        while True:
            neighbors = list(self.graph.neighbors(curr))
            # LDraw에서 Y는 위로 갈수록 감소(마이너스 Y가 위). 따라서 위쪽 브릭은 더 낮은 Y값을 가짐.
            curr_y = self.bricks[self._get_idx(curr)]['pos'][1]
            upper = [n for n in neighbors if self.bricks[self._get_idx(n)]['pos'][1] < curr_y] 
            if not upper: break
            curr = upper[0] # 단일 기둥 가정
            if curr in visited: break
            visited.add(curr)
            stack_height += 1
            
        # 아래쪽 확인
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
        """브릭이 아래쪽에서 단 하나의 연결로만 지지되는지 확인합니다."""
        if not self.graph.has_node(brick_id): return False
        neighbors = list(self.graph.neighbors(brick_id))
        brick_y = self.bricks[self._get_idx(brick_id)]['pos'][1]
        # 이 브릭 아래의 이웃 수 계산 (Y 값이 더 큼)
        params_below = [n for n in neighbors if self.bricks[self._get_idx(n)]['pos'][1] > brick_y]
        return len(params_below) == 1

    def analyze_failure(self, failure_report: Dict) -> List[Dict]:
        """실패 보고서를 분석하고 수정을 제안합니다."""
        fixes = []
        target_id = failure_report.get('first_failure_id')
        
        if not target_id:
            return []
            
        print(f"[Fixer] {target_id}에서 실패 분석 중...")
        
        # 약한 기둥 확인
        if self.is_weak_column(target_id):
            print(f"  -> 약한 기둥 패턴 감지됨.")
            fixes.append({
                'type': 'reinforce_column',
                'target': target_id
            })
            
        # 단일 지지 확인
        elif self.is_single_support(target_id):
            print(f"  -> 단일 지지 패턴 감지됨.")
            fixes.append({
                'type': 'reinforce_base',
                'target': target_id
            })
            
        # 기본 대체 (단순히 끊어지거나 떠있는 경우 바닥 추가?)
        else:
             print(f"  -> 알 수 없는 패턴, 대체 방법으로 바닥 보강 시도.")
             fixes.append({
                'type': 'reinforce_base',
                'target': target_id
            })
            
        return fixes

    def apply_fix(self, fix_instruction: Dict) -> bool:
        """LDR 구조에 특정 수정을 적용합니다."""
        target_id = fix_instruction['target']
        idx = self._get_idx(target_id)
        if idx is None: return False
        
        target_brick = self.bricks[idx]
        print(f"[Fixer] {target_id}에 {fix_instruction['type']} 적용 중")

        if fix_instruction['type'] == 'reinforce_column':
            # 측면 지지 (버팀목)
            # 기둥을 두껍게 하기 위해 옆에 수직 스택 추가
            new_pos = target_brick['pos'].copy()
            new_pos[0] += 20.0 # X축으로 1 스터드 이동
            
            self.bricks.append({
                'id': f"3001.dat_fix_col_{len(self.bricks)}",
                'part': "3001.dat", # 강력한 지지를 위해 전체 브릭(2x4) 사용
                'color': 5, # 어두운 분홍
                'pos': new_pos,
                'matrix': target_brick['matrix'][:]
            })
            return True
            
        elif fix_instruction['type'] == 'reinforce_base':
            # "빅풋" 전략: 바닥 면적 넓히기
            # 플랫폼을 만들기 위해 타겟 주위에 2x4 브릭 추가
            offsets = [
                (20.0, 0, 0),   # +X
                (-20.0, 0, 0),  # -X
                (0, 0, 20.0),   # +Z
                (0, 0, -20.0)   # -Z
            ]
            
            for i, (dx, dy, dz) in enumerate(offsets):
                new_pos = target_brick['pos'].copy()
                new_pos[0] += dx
                new_pos[1] += dy # 높이 동일
                new_pos[2] += dz
                
                self.bricks.append({
                    'id': f"3020.dat_fix_base_{i}_{len(self.bricks)}",
                    'part': "3020.dat", # 2x4 플레이트
                    'color': 5, # 어두운 분홍
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
        """수정된 브릭을 새 LDR 파일로 저장합니다."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("0 Fixed by StructureFixer\n")
            for b in self.bricks:
                # 형식: 1 <색상> <x> <y> <z> <회전 행렬 9값> <부품>
                matrix_str = " ".join(f"{v:.6f}" for v in b['matrix'])
                f.write(f"1 {b['color']} {b['pos'][0]:.2f} {b['pos'][1]:.2f} {b['pos'][2]:.2f} {matrix_str} {b['part']}\n")
