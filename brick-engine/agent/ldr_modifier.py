# ============================================================================
# LDR 파일 수정 모듈 (개선판)
# LLM의 결정에 따라 LDR 파일에서 브릭을 이동하거나 삭제
# 인덱스 밀림 및 ID 불일치 문제를 해결함 (LdrLoader의 글로벌 인덱스 방식 채택)
# ============================================================================

import re
import os
from typing import Tuple, Optional, List, Dict, Any
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


def apply_llm_decisions(
    ldr_path: str,
    decisions: list
) -> dict:
    """
    LLM의 결정을 일괄 적용합니다. (메모리 내 단일 패스 처리)
    
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
    stats = {"moved": 0, "deleted": 0, "kept": 0, "failed": 0, "added": 0}
    path = Path(ldr_path)
    
    if not path.exists():
        print(f"[LDR] 파일 없음: {ldr_path}")
        return stats

    # 1. 파일 전체 읽기
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 2. 브릭 ID -> 라인 인덱스 매핑 구축 (LdrLoader 로직과 일치: 글로벌 카운터 사용)
    id_to_index = {}
    brick_counter = 0
    for i, line in enumerate(lines):
        parsed = parse_ldr_line(line)
        if parsed is None:
            continue
        
        # LdrLoader와 동일한 ID 생성 방식
        brick_id = f"{parsed['part']}_{brick_counter}"
        id_to_index[brick_id] = i
        brick_counter += 1

    # 3. 결정 적용 (메모리 상의 lines 리스트 수정)
    # 삭제 시에는 해당 인덱스 값을 None으로 설정하여 인덱스 밀림 방지
    for decision in decisions:
        brick_id = decision.get("brick_id")
        action = decision.get("action", "keep")
        
        if brick_id not in id_to_index:
            print(f"[LDR] 찾을 수 없는 브릭 ID: {brick_id} (건너뜜)")
            stats["failed"] += 1
            continue
            
        line_idx = id_to_index[brick_id]
        
        if action == "move":
            position = decision.get("position")
            if position and len(position) == 3:
                parsed = parse_ldr_line(lines[line_idx])
                if parsed:
                    new_line = build_ldr_line(
                        parsed["color"],
                        position[0], position[1], position[2],
                        parsed["matrix"],
                        parsed["part"]
                    )
                    lines[line_idx] = new_line + "\n"
                    stats["moved"] += 1
                else:
                    stats["failed"] += 1
            else:
                stats["failed"] += 1
        
        elif action == "add":
            position = decision.get("position")
            part = decision.get("part", "3005.dat")
            if position and len(position) == 3:
                # 공중부양 브릭의 색상을 가져와 동일한 색상으로 보강
                parsed = parse_ldr_line(lines[line_idx])
                color = parsed["color"] if parsed else 4
                
                new_line = build_ldr_line(
                    color,
                    position[0], position[1], position[2],
                    [1, 0, 0, 0, 1, 0, 0, 0, 1], # 기본 매트릭스
                    part
                )
                lines.append(new_line + "\n")
                stats["added"] += 1
                print(f"[LDR] 지지 브릭 추가 완료: {part} at {position}")
            else:
                stats["failed"] += 1
                
        elif action == "delete":
            lines[line_idx] = None  # 삭제 표시
            stats["deleted"] += 1
            print(f"[LDR] 브릭 삭제 처리 완료: {brick_id}")
            
        elif action == "keep":
            stats["kept"] += 1
    
    # 4. 결과 저장 (None이 아닌 라인만 쓰기)
    new_lines = [l for l in lines if l is not None]
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
        
    print(f"[LDR] 수정 완료: {ldr_path}")
    print(f"      - 결과: 추가 {stats['added']}, 이동 {stats['moved']}, 삭제 {stats['deleted']}, 유지 {stats['kept']}, 실패 {stats['failed']}")
    
    return stats


# 하위 호환성을 위한 래퍼 함수들
def modify_brick_position(ldr_path: str, brick_id: str, new_position: Tuple[float, float, float]) -> bool:
    res = apply_llm_decisions(ldr_path, [{"brick_id": brick_id, "action": "move", "position": list(new_position)}])
    return res["moved"] > 0

def remove_brick(ldr_path: str, brick_id: str) -> bool:
    res = apply_llm_decisions(ldr_path, [{"brick_id": brick_id, "action": "delete"}])
    return res["deleted"] > 0
