# ============================================================================
# 2D 병합 모듈 (2x2, 2xN 직사각형 병합)
# ============================================================================

from collections import defaultdict

from .constants import (
    STUD_SPACING, BRICK_2X2_MAPPING, BRICK_DIMENSIONS,
    TARGET_RECT_SIZES, PLATE_RECT_SIZES, is_plate
)


def merge_2x2_bricks(bricks: list, group_by_color: bool = True) -> tuple:
    candidates = []
    others = []

    for b in bricks:
        if b["part"] in BRICK_2X2_MAPPING:
            candidates.append(b)
        else:
            others.append(b)

    if not candidates:
        return bricks, 0

    merge_count = 0
    final_bricks = others
    merged_indices = set()

    groups = defaultdict(list)

    for i, b in enumerate(candidates):
        a = b["matrix"][0]
        axis = 'x' if abs(a) > 0.5 else 'z'

        if group_by_color:
            key = (b["y"], b["part"], b["color"], axis)
        else:
            key = (b["y"], b["part"], axis)

        groups[key].append((i, b))

    for key, items in groups.items():
        if len(items) < 2:
            continue

        axis = key[-1]

        if axis == 'x':
            items.sort(key=lambda x: (x[1]["x"], x[1]["z"]))
        else:
            items.sort(key=lambda x: (x[1]["z"], x[1]["x"]))

        used = set()

        for i in range(len(items)):
            if i in used: continue

            idx_i, b_i = items[i]
            best_mate_idx = -1

            for j in range(i + 1, len(items)):
                if j in used: continue

                idx_j, b_j = items[j]

                if axis == 'x':
                    if abs(b_i["x"] - b_j["x"]) < 1.0:
                        dist = abs(b_i["z"] - b_j["z"])
                        if abs(dist - STUD_SPACING) < 1.0:
                            best_mate_idx = j
                            break
                else:
                    if abs(b_i["z"] - b_j["z"]) < 1.0:
                        dist = abs(b_i["x"] - b_j["x"])
                        if abs(dist - STUD_SPACING) < 1.0:
                            best_mate_idx = j
                            break

            if best_mate_idx != -1:
                idx_j, b_j = items[best_mate_idx]
                used.add(i)
                used.add(best_mate_idx)
                merged_indices.add(idx_i)
                merged_indices.add(idx_j)

                new_x = int(round((b_i["x"] + b_j["x"]) / 2))
                new_y = int(round(b_i["y"])) 
                new_z = int(round((b_i["z"] + b_j["z"]) / 2))

                new_part = BRICK_2X2_MAPPING[b_i["part"]]
                
                vol_i = b_i.get("_orig_vol", 1)
                vol_j = b_j.get("_orig_vol", 1)
                
                if vol_i > vol_j:
                    final_color = b_i["color"]
                    final_vol = vol_i
                elif vol_j > vol_i:
                    final_color = b_j["color"]
                    final_vol = vol_j
                else:
                    p_i = b_i.get("_priority_color", False)
                    p_j = b_j.get("_priority_color", False)
                    if p_i and not p_j:
                        final_color = b_i["color"]
                    elif p_j and not p_i:
                        final_color = b_j["color"]
                    else:
                        final_color = b_i["color"]
                    final_vol = vol_i 
                
                has_priority = b_i.get("_priority_color") or b_j.get("_priority_color")

                new_brick = {
                    "part": new_part,
                    "color": final_color,
                    "x": int(round(new_x)), 
                    "y": int(round(new_y)), 
                    "z": int(round(new_z)),
                    "matrix": b_i["matrix"],
                    "_orig_vol": final_vol
                }
                if has_priority:
                    new_brick["_priority_color"] = True
                    
                final_bricks.append(new_brick)
                merge_count += 1
            else:
                final_bricks.append(b_i)
                used.add(i)

    return final_bricks, merge_count


def merge_rect_bricks(bricks: list, group_by_color: bool = True) -> tuple:
    current_bricks = bricks
    total_merge_count = 0
    
    for _ in range(3):
        merged_bricks, count = _merge_rect_pass(current_bricks, group_by_color)
        if count == 0:
            break
        current_bricks = merged_bricks
        total_merge_count += count
        
    return current_bricks, total_merge_count


def _merge_rect_pass(bricks: list, group_by_color: bool) -> tuple:
    groups = defaultdict(list)
    
    final_bricks = []
    for i, b in enumerate(bricks):
        rows, cols = BRICK_DIMENSIONS.get(b["part"], (1, 1))
        
        if rows > 2 or cols > 10: 
             final_bricks.append(b)
             continue
        
        mat_tuple = tuple(b["matrix"])
        key = (b["y"], b["color"] if group_by_color else -1, mat_tuple)
        groups[key].append((i, b))
        
    local_used_global = set()
    merge_count = 0
    
    for key, items in groups.items():
        if len(items) < 2:
            for _, b in items:
                if b not in final_bricks: final_bricks.append(b)
            continue
            
        items.sort(key=lambda x: (x[1]["z"], x[1]["x"]))
        
        local_used = set()
        
        for i in range(len(items)):
            if i in local_used: continue
            
            idx_i, b_i = items[i]
            r1, c1 = BRICK_DIMENSIONS.get(b_i["part"], (1, 1))
            
            best_mate = -1
            
            for j in range(i + 1, len(items)):
                if j in local_used: continue
                
                idx_j, b_j = items[j]
                r2, c2 = BRICK_DIMENSIONS.get(b_j["part"], (1, 1))
                
                if abs(b_i["z"] - b_j["z"]) < 1.0:
                    dist_x = abs(b_i["x"] - b_j["x"])
                    target_dist = (c1 + c2) * 10.0
                    if abs(dist_x - target_dist) < 1.0:
                        if r1 != r2: continue
                        
                        target_key = (r1, c1 + c2)
                        target_part = _get_target_part(target_key, b_i["part"])
                        
                        if target_part:
                            new_x = (b_i["x"] + b_j["x"]) / 2.0
                            new_brick = b_i.copy()
                            new_brick["x"] = new_x
                            new_brick["part"] = target_part
                            local_used.add(i)
                            local_used.add(j)
                            final_bricks.append(new_brick)
                            merge_count += 1
                            best_mate = j
                            break
                            
                if abs(b_i["x"] - b_j["x"]) < 1.0:
                    dist_z = abs(b_i["z"] - b_j["z"])
                    target_dist = (r1 + r2) * 10.0
                    if abs(dist_z - target_dist) < 1.0:
                        if c1 != c2: continue
                        
                        target_key = (r1 + r2, c1)
                        target_part = _get_target_part(target_key, b_i["part"])
                        
                        if target_part:
                            new_z = (b_i["z"] + b_j["z"]) / 2.0
                            new_brick = b_i.copy()
                            new_brick["z"] = new_z
                            new_brick["part"] = target_part
                            local_used.add(i)
                            local_used.add(j)
                            final_bricks.append(new_brick)
                            merge_count += 1
                            best_mate = j
                            break
                            
            if best_mate == -1:
                pass

        for i in range(len(items)):
            if i not in local_used:
                final_bricks.append(items[i][1])
                
    return final_bricks, merge_count


def _get_target_part(size_key: tuple, original_part: str) -> str:
    if is_plate(original_part):
        return PLATE_RECT_SIZES.get(size_key)
    else:
        return TARGET_RECT_SIZES.get(size_key)
