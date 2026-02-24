# ============================================================================
# LLM 결정 적용 모듈
# ============================================================================

import logging
from typing import Tuple
from pathlib import Path

from .parser import parse_ldr_line, build_ldr_line

logger = logging.getLogger(__name__)


def apply_llm_decisions(
    ldr_path: str,
    decisions: list
) -> dict:
    stats = {"moved": 0, "deleted": 0, "kept": 0, "failed": 0, "added": 0}
    path = Path(ldr_path)
    
    if not path.exists():
        logger.warning(f"파일 없음: {ldr_path}")
        return stats

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    id_to_index = {}
    brick_counter = 0
    for i, line in enumerate(lines):
        parsed = parse_ldr_line(line)
        if parsed is None:
            continue
        
        brick_id = f"{parsed['part']}_{brick_counter}"
        id_to_index[brick_id] = i
        brick_counter += 1

    for decision in decisions:
        brick_id = decision.get("brick_id")
        action = decision.get("action", "keep")
        
        if brick_id not in id_to_index:
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
                parsed = parse_ldr_line(lines[line_idx])
                color = parsed["color"] if parsed else 4
                
                new_line = build_ldr_line(
                    color,
                    position[0], position[1], position[2],
                    [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    part
                )
                lines.append(new_line + "\n")
                stats["added"] += 1
            else:
                stats["failed"] += 1
                
        elif action == "delete":
            lines[line_idx] = None
            stats["deleted"] += 1
            
        elif action == "keep":
            stats["kept"] += 1
    
    new_lines = [line for line in lines if line is not None]
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
        
    return stats


def modify_brick_position(ldr_path: str, brick_id: str, new_position: Tuple[float, float, float]) -> bool:
    res = apply_llm_decisions(ldr_path, [{"brick_id": brick_id, "action": "move", "position": list(new_position)}])
    return res["moved"] > 0

def remove_brick(ldr_path: str, brick_id: str) -> bool:
    res = apply_llm_decisions(ldr_path, [{"brick_id": brick_id, "action": "delete"}])
    return res["deleted"] > 0
