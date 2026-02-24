# ============================================================================
# 1xN 선형 병합 모듈
# ============================================================================

import logging
from typing import Optional
from pathlib import Path
from collections import defaultdict, Counter

from .constants import (
    STUD_SPACING, SMALL_BRICK_PARTS,
    MERGE_TARGET_BRICKS, PLATE_MERGE_TARGETS,
    is_plate
)
from .parser import parse_ldr_line, build_ldr_line
from .merge_2d import merge_rect_bricks

logger = logging.getLogger(__name__)


def merge_small_bricks(
    ldr_path: str, 
    target_brick_ids: Optional[list] = None, 
    min_merge_count: int = 2, 
    max_len: Optional[int] = None, 
    group_by_color: bool = True
) -> dict:
    stats = {
        "original_count": 0, "new_count": 0, "merged": 0, 
        "total_original_count": 0, "total_new_count": 0,
        "small_brick_count": 0, "small_brick_new_count": 0
    }
    path = Path(ldr_path)
    if not path.exists():
        return stats
    
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    bricks = []
    total_brick_count = 0
    brick_counter = 0
    for i, line in enumerate(lines):
        parsed = parse_ldr_line(line)
        if parsed is None:
            continue
        
        total_brick_count += 1
        brick_id = f"{parsed['part']}_{brick_counter}"
        parsed["line_idx"] = i
        
        if target_brick_ids is None or brick_id in target_brick_ids:
            if parsed["part"] in SMALL_BRICK_PARTS:
                bricks.append(parsed)
        brick_counter += 1
    
    stats["total_original_count"] = total_brick_count
    stats["small_brick_count"] = len(bricks)
    stats["original_count"] = len(bricks) 

    if len(bricks) < min_merge_count:
        stats["new_count"] = len(bricks)
        stats["total_new_count"] = total_brick_count
        return stats
    
    new_lines_content, merged_indices, merge_count = merge_all_1x1(
        bricks, min_merge_count, group_by_color=group_by_color, max_len=max_len
    )
    
    if merged_indices:
        final_lines = []
        merged_line_indices = {bricks[idx]["line_idx"] for idx in merged_indices}
        for i, line in enumerate(lines):
            if i not in merged_line_indices:
                final_lines.append(line)
        final_lines.extend(new_lines_content)
        if final_lines and not final_lines[-1].endswith("\n"):
            final_lines[-1] += "\n"
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(final_lines)
            
        stats["merged"] = merge_count
        stats["total_new_count"] = total_brick_count - len(merged_indices) + len(new_lines_content)
        stats["new_count"] = stats["total_new_count"]
        
    return stats


def merge_all_1x1(
    bricks: list, 
    min_merge_count: int = 2, 
    group_by_color: bool = False, 
    max_len: Optional[int] = None, 
    anchor_indices: Optional[set] = None
) -> tuple:
    generated_bricks = []
    merged_indices = set()
    merge_count = 0
    all_merged_indices = set()

    groups = defaultdict(list)
    for i, b in enumerate(bricks):
        is_p = is_plate(b["part"])
        groups[(b["y"], b["color"] if group_by_color else "all", is_p)].append((i, b))

    for key, group in groups.items():
        if len(group) < min_merge_count:
            continue
            
        already_merged = set()
        is_p = key[-1]
        target_mapping = PLATE_MERGE_TARGETS if is_p else MERGE_TARGET_BRICKS

        # X 방향
        z_groups = defaultdict(list)
        for idx, brick in group:
            z_groups[brick["z"]].append((idx, brick))

        for z, z_items in z_groups.items():
            z_items.sort(key=lambda item: item[1]["x"])
            _merge_direction(
                z_items, "x", target_mapping, min_merge_count, max_len,
                group_by_color, anchor_indices,
                already_merged, merged_indices, all_merged_indices,
                generated_bricks
            )

        # Z 방향
        x_groups = defaultdict(list)
        for idx, brick in group:
            if idx in already_merged: continue
            x_groups[brick["x"]].append((idx, brick))

        for x, x_items in x_groups.items():
            x_items.sort(key=lambda item: item[1]["z"])
            _merge_direction(
                x_items, "z", target_mapping, min_merge_count, max_len,
                group_by_color, anchor_indices,
                already_merged, merged_indices, all_merged_indices,
                generated_bricks
            )

    for idx, brick in enumerate(bricks):
        if idx not in all_merged_indices:
            generated_bricks.append(brick)

    final_bricks, count_rect = merge_rect_bricks(generated_bricks, group_by_color)
    merge_count += count_rect
    
    new_brick_lines = []
    for b in final_bricks:
        line = build_ldr_line(b["color"], b["x"], b["y"], b["z"], b["matrix"], b["part"])
        new_brick_lines.append(line + "\n")

    return new_brick_lines, merged_indices, merge_count


def _merge_direction(items, direction, target_mapping, min_merge_count, max_len, group_by_color, anchor_indices, already_merged, merged_indices, all_merged_indices, generated_bricks):
    is_z_dir = (direction == "z")
    coord_key = "x" if not is_z_dir else "z"
    i = 0
    while i < len(items):
        idx_i, brick_i = items[i]
        if idx_i in already_merged:
            i += 1
            continue

        sequence = [(idx_i, brick_i)]
        j = i + 1
        while j < len(items):
            idx_j, brick_j = items[j]
            if idx_j in already_merged:
                j += 1
                continue
            if abs(brick_j[coord_key] - sequence[-1][1][coord_key] - STUD_SPACING) < 0.1:
                sequence.append((idx_j, brick_j))
                j += 1
            else:
                break

        seq_len = len(sequence)
        eff_max = int(max_len) if max_len is not None else 4
        seq_len = min(seq_len, eff_max)
        merged_any = False
        
        while seq_len >= min_merge_count:
            if seq_len in target_mapping:
                if not group_by_color and anchor_indices:
                    has_anchor = any(orig_idx in anchor_indices for orig_idx, _ in sequence[:seq_len])
                    if not has_anchor:
                        if len(set(rb["color"] for _, rb in sequence[:seq_len])) > 1:
                            seq_len -= 1
                            continue
                
                first = sequence[0][1]
                last = sequence[seq_len - 1][1]
                center_x = (first["x"] + last["x"]) / 2
                center_z = (first["z"] + last["z"]) / 2
                
                mat = [0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0] if is_z_dir else [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
                
                max_vol = max(rb.get("_orig_vol", 1) for _, rb in sequence[:seq_len])
                final_color = sequence[0][1]["color"] # 단순화 (위 로직 참조 가능)
                
                new_brick = {
                    "part": target_mapping[seq_len], "color": final_color,
                    "x": center_x, "y": first["y"], "z": center_z,
                    "matrix": mat, "_orig_vol": max_vol
                }
                generated_bricks.append(new_brick)
                for idx_s, _ in sequence[:seq_len]:
                    already_merged.add(idx_s)
                    merged_indices.add(idx_s)
                    all_merged_indices.add(idx_s)
                merged_any = True
                break
            seq_len -= 1
        i = j if not merged_any else i
        if merged_any:
            while i < len(items) and items[i][0] in already_merged: i += 1
