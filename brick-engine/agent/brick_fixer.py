# ============================================================================
# 공중부양 브릭 수정 모듈
# 공중부양(Floating) 브릭의 연결 가능 후보 위치를 계산하는 알고리즘
# LLM이 최종 결정을 내리기 위한 정보를 제공
# ============================================================================

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

# LDU 상수 (lego_physics.py와 동일)
STUD_SPACING = 20.0  # X/Z 그리드 간격
BRICK_HEIGHT = 24.0  # 일반 브릭 높이
PLATE_HEIGHT = 8.0   # 플레이트 높이


@dataclass
class ConnectionCandidate:
    """연결 가능 후보 위치"""
    target_brick_id: str          # 연결 대상 브릭 ID
    new_position: Tuple[float, float, float]  # 이동할 새 위치 (x, y, z)
    connection_type: str          # "top" (위에 쌓기) / "bottom" (아래에 놓기)
    distance: float               # 현재 위치에서의 거리
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_brick": self.target_brick_id,
            "new_position": list(self.new_position),
            "type": self.connection_type,
            "distance": round(self.distance, 1)
        }


def get_brick_height(part_file: str) -> float:
    """브릭의 높이 반환 (LDU)"""
    # 플레이트 판별 (간단 버전)
    plate_parts = ["3020", "3021", "3022", "3023", "3024", "3795", "3034", "3832", "3710"]
    part_num = part_file.replace(".dat", "").lower()
    
    for plate in plate_parts:
        if plate in part_num:
            return PLATE_HEIGHT
    return BRICK_HEIGHT


def find_connection_candidates(
    floating_brick,
    all_bricks: list,
    max_distance: float = 60.0,
    max_candidates: int = 3
) -> List[ConnectionCandidate]:
    """
    공중부양 브릭이 연결될 수 있는 후보 위치를 계산합니다.
    
    Args:
        floating_brick: 공중부양 브릭 객체
        all_bricks: 전체 브릭 리스트
        max_distance: 최대 탐색 거리 (LDU)
        max_candidates: 반환할 최대 후보 수
        
    Returns:
        연결 가능한 후보 위치 리스트 (거리순 정렬)
    """
    if floating_brick.origin is None:
        return []
    
    candidates = []
    floating_pos = floating_brick.origin
    floating_height = get_brick_height(floating_brick.part_file or "3001.dat")
    
    for brick in all_bricks:
        if brick.id == floating_brick.id:
            continue
        if brick.origin is None:
            continue
        
        brick_pos = brick.origin
        brick_height = get_brick_height(brick.part_file or "3001.dat")
        
        # 수평 거리 계산 (X, Z)
        dx = brick_pos[0] - floating_pos[0]
        dz = brick_pos[2] - floating_pos[2]
        horizontal_dist = np.sqrt(dx**2 + dz**2)
        
        if horizontal_dist > max_distance:
            continue
        
        # 후보 1: 이 브릭 위에 쌓기
        # 새 위치 = 대상 브릭 위치, Y는 대상 브릭의 상단
        new_pos_top = (
            brick_pos[0],  # X: 대상 브릭과 같은 위치
            brick_pos[1] - floating_height,  # Y: 대상 브릭 위 (Y-down이므로 빼기)
            brick_pos[2]   # Z: 대상 브릭과 같은 위치
        )
        
        top_dist = np.sqrt(
            (new_pos_top[0] - floating_pos[0])**2 +
            (new_pos_top[1] - floating_pos[1])**2 +
            (new_pos_top[2] - floating_pos[2])**2
        )
        
        candidates.append(ConnectionCandidate(
            target_brick_id=brick.id,
            new_position=new_pos_top,
            connection_type="top",
            distance=top_dist
        ))
        
        # 후보 2: 이 브릭 아래에 놓기
        new_pos_bottom = (
            brick_pos[0],
            brick_pos[1] + brick_height,  # Y: 대상 브릭 아래
            brick_pos[2]
        )
        
        bottom_dist = np.sqrt(
            (new_pos_bottom[0] - floating_pos[0])**2 +
            (new_pos_bottom[1] - floating_pos[1])**2 +
            (new_pos_bottom[2] - floating_pos[2])**2
        )
        
        candidates.append(ConnectionCandidate(
            target_brick_id=brick.id,
            new_position=new_pos_bottom,
            connection_type="bottom",
            distance=bottom_dist
        ))
    
    # 거리순 정렬 후 상위 N개만 반환
    candidates.sort(key=lambda c: c.distance)
    return candidates[:max_candidates]


def analyze_floating_bricks(
    floating_brick_ids: List[str],
    all_bricks: list,
    max_candidates_per_brick: int = 3
) -> Dict[str, List[Dict[str, Any]]]:
    """
    모든 공중부양 브릭에 대한 연결 후보를 분석합니다.
    
    Args:
        floating_brick_ids: 공중부양 브릭 ID 리스트
        all_bricks: 전체 브릭 리스트
        max_candidates_per_brick: 브릭당 최대 후보 수
        
    Returns:
        {
            "3005.dat_0": [
                {"target_brick": "3004.dat_10", "new_position": [...], "type": "top", "distance": 24.0},
                ...
            ],
            ...
        }
    """
    # ID로 브릭 찾기
    brick_by_id = {b.id: b for b in all_bricks}
    
    result = {}
    for brick_id in floating_brick_ids:
        if brick_id not in brick_by_id:
            continue
        
        floating_brick = brick_by_id[brick_id]
        candidates = find_connection_candidates(
            floating_brick, all_bricks, max_candidates=max_candidates_per_brick
        )
        
        result[brick_id] = [c.to_dict() for c in candidates]
    
    return result


def format_for_llm(analysis: Dict[str, List[Dict[str, Any]]]) -> str:
    """
    LLM에게 전달할 형식으로 분석 결과를 포맷합니다.
    """
    lines = ["## 공중부양 브릭 분석 결과\n"]
    
    for brick_id, candidates in analysis.items():
        lines.append(f"### {brick_id}")
        lines.append(f"- 현재 상태: 연결 없음 (공중부양)")
        
        if candidates:
            lines.append("- 연결 가능 후보:")
            for i, c in enumerate(candidates, 1):
                lines.append(f"  {i}. {c['target_brick']}의 {c['type']} → 거리 {c['distance']} LDU")
        else:
            lines.append("- ⚠️ 연결 가능한 후보 없음 (삭제 권장)")
        
        lines.append("")
    
    return "\n".join(lines)
