# ============================================================================
# LDR 파일 수정 모듈
# LLM의 결정에 따라 LDR 파일에서 브릭을 이동하거나 삭제
# ============================================================================

import re
from typing import Tuple, Optional
from pathlib import Path


def parse_ldr_line(line: str) -> Optional[dict]:
    """
    LDR 라인을 파싱하여 브릭 정보 추출
    
    LDR 형식: 1 <color> <x> <y> <z> <a> <b> <c> <d> <e> <f> <g> <h> <i> <part>
    """
    line = line.strip()
    if not line.startswith("1 "):
        return None
    
    parts = line.split()
    if len(parts) < 15:
        return None
    
    try:
        return {
            "type": int(parts[0]),
            "color": int(parts[1]),
            "x": float(parts[2]),
            "y": float(parts[3]),
            "z": float(parts[4]),
            "matrix": [float(p) for p in parts[5:14]],
            "part": parts[14],
            "original_line": line
        }
    except (ValueError, IndexError):
        return None


def build_ldr_line(
    color: int,
    x: float, y: float, z: float,
    matrix: list,
    part: str
) -> str:
    """LDR 라인 재구성"""
    matrix_str = " ".join(str(m) for m in matrix)
    return f"1 {color} {x} {y} {z} {matrix_str} {part}"


def modify_brick_position(
    ldr_path: str,
    brick_id: str,
    new_position: Tuple[float, float, float]
) -> bool:
    """
    LDR 파일에서 특정 브릭의 위치를 수정합니다.
    
    Args:
        ldr_path: LDR 파일 경로
        brick_id: 브릭 ID (예: "3005.dat_0")
        new_position: 새 위치 (x, y, z)
        
    Returns:
        성공 여부
    """
    path = Path(ldr_path)
    if not path.exists():
        print(f"[LDR] 파일 없음: {ldr_path}")
        return False
    
    # 브릭 ID에서 part와 index 추출
    # 예: "3005.dat_0" -> part="3005.dat", index=0
    match = re.match(r"(.+\.dat)_(\d+)", brick_id)
    if not match:
        print(f"[LDR] 잘못된 브릭 ID 형식: {brick_id}")
        return False
    
    target_part = match.group(1)
    target_index = int(match.group(2))
    
    # 파일 읽기
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # 해당 브릭 찾기 및 수정
    part_counter = {}
    modified = False
    
    for i, line in enumerate(lines):
        parsed = parse_ldr_line(line)
        if parsed is None:
            continue
        
        part = parsed["part"]
        if part not in part_counter:
            part_counter[part] = 0
        else:
            part_counter[part] += 1
        
        current_index = part_counter[part]
        
        if part == target_part and current_index == target_index:
            # 위치 수정
            new_line = build_ldr_line(
                parsed["color"],
                new_position[0], new_position[1], new_position[2],
                parsed["matrix"],
                parsed["part"]
            )
            lines[i] = new_line + "\n"
            modified = True
            print(f"[LDR] 브릭 이동: {brick_id} -> ({new_position[0]}, {new_position[1]}, {new_position[2]})")
            break
    
    if not modified:
        print(f"[LDR] 브릭을 찾지 못함: {brick_id}")
        return False
    
    # 파일 쓰기
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    
    return True


def remove_brick(ldr_path: str, brick_id: str) -> bool:
    """
    LDR 파일에서 특정 브릭을 삭제합니다.
    
    Args:
        ldr_path: LDR 파일 경로
        brick_id: 브릭 ID (예: "3005.dat_0")
        
    Returns:
        성공 여부
    """
    path = Path(ldr_path)
    if not path.exists():
        print(f"[LDR] 파일 없음: {ldr_path}")
        return False
    
    # 브릭 ID에서 part와 index 추출
    match = re.match(r"(.+\.dat)_(\d+)", brick_id)
    if not match:
        print(f"[LDR] 잘못된 브릭 ID 형식: {brick_id}")
        return False
    
    target_part = match.group(1)
    target_index = int(match.group(2))
    
    # 파일 읽기
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # 해당 브릭 찾기 및 삭제
    part_counter = {}
    new_lines = []
    removed = False
    
    for line in lines:
        parsed = parse_ldr_line(line)
        if parsed is None:
            new_lines.append(line)
            continue
        
        part = parsed["part"]
        if part not in part_counter:
            part_counter[part] = 0
        else:
            part_counter[part] += 1
        
        current_index = part_counter[part]
        
        if part == target_part and current_index == target_index:
            # 이 라인 삭제 (추가하지 않음)
            removed = True
            print(f"[LDR] 브릭 삭제: {brick_id}")
            continue
        
        new_lines.append(line)
    
    if not removed:
        print(f"[LDR] 브릭을 찾지 못함: {brick_id}")
        return False
    
    # 파일 쓰기
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    
    return True


def apply_llm_decisions(
    ldr_path: str,
    decisions: list
) -> dict:
    """
    LLM의 결정을 일괄 적용합니다.
    
    Args:
        ldr_path: LDR 파일 경로
        decisions: LLM 결정 리스트
            [
                {"brick_id": "3005.dat_0", "action": "move", "position": [x, y, z]},
                {"brick_id": "3005.dat_1", "action": "delete"},
                {"brick_id": "3005.dat_2", "action": "keep"},
            ]
            
    Returns:
        {"moved": 2, "deleted": 1, "kept": 1, "failed": 0}
    """
    stats = {"moved": 0, "deleted": 0, "kept": 0, "failed": 0}
    
    for decision in decisions:
        brick_id = decision.get("brick_id")
        action = decision.get("action", "keep")
        
        if action == "move":
            position = decision.get("position")
            if position and len(position) == 3:
                success = modify_brick_position(ldr_path, brick_id, tuple(position))
                if success:
                    stats["moved"] += 1
                else:
                    stats["failed"] += 1
            else:
                print(f"[LDR] 잘못된 위치 정보: {brick_id}")
                stats["failed"] += 1
                
        elif action == "delete":
            success = remove_brick(ldr_path, brick_id)
            if success:
                stats["deleted"] += 1
            else:
                stats["failed"] += 1
                
        elif action == "keep":
            stats["kept"] += 1
    
    print(f"[LDR] 적용 완료: 이동 {stats['moved']}개, 삭제 {stats['deleted']}개, 유지 {stats['kept']}개, 실패 {stats['failed']}개")
    return stats
