# ============================================================================
# 1xN 선형 병합 모듈 + merge_small_bricks (메인 병합 엔트리포인트)
# 1x1 브릭들을 X/Z 방향으로 그루핑하여 1xN 브릭으로 병합
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
    """
    LDR 파일 내의 1x1 브릭/플레이트들을 찾아서 병합 가능한 경우 더 큰 브릭으로 교체합니다.
    
    Args:
        ldr_path: LDR 파일 경로
        target_brick_ids: 병합 대상 브릭 ID 리스트 (None이면 전체)
        min_merge_count: 최소 병합 단위
        max_len: 최대 병합 길이
        group_by_color: 색상별 병합 여부 (False이면 색상 무관 병합)
    """
    stats = {
        "original_count": 0, 
        "new_count": 0, 
        "merged": 0, 
        "total_original_count": 0,
        "total_new_count": 0,
        "small_brick_count": 0,
        "small_brick_new_count": 0
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
    
    # 통합 엔진 호출
    new_lines_content, merged_indices, merge_count = merge_all_1x1(
        bricks, 
        min_merge_count, 
        group_by_color=group_by_color,
        max_len=max_len
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
        stats["small_brick_new_count"] = len(bricks) - len(merged_indices) + len(new_lines_content)
        stats["total_new_count"] = total_brick_count - len(merged_indices) + len(new_lines_content)
        stats["new_count"] = stats["total_new_count"]
    else:
        stats["small_brick_new_count"] = len(bricks)
        stats["total_new_count"] = total_brick_count
        stats["new_count"] = total_brick_count
        
    return stats


def merge_all_1x1(
    bricks: list, 
    min_merge_count: int = 2, 
    group_by_color: bool = False, 
    max_len: Optional[int] = None, 
    anchor_indices: Optional[set] = None
) -> tuple:
    """
    1x1 브릭/플레이트들을 병합합니다.
    1단계: 1xN (선형) 병합
    2단계: 2xN (면) 병합
    
    Args:
        bricks: 1x1 브릭 객체 리스트
        min_merge_count: 최소 병합 수
        group_by_color: True이면 같은 색상끼리만 병합, False이면 색상 무관 병합
        max_len: 최대 병합 길이 제한 (None이면 4 적용)
        anchor_indices: 안정 브릭의 인덱스 집합 (색상 무관 병합 시 권장)
    """
    generated_bricks = []
    merged_indices = set()
    merge_count = 0
    all_merged_indices = set()

    # 그룹화 (Y좌표별, 색상별/전체, Plate 여부)
    groups = defaultdict(list)
    for i, b in enumerate(bricks):
        is_p = is_plate(b["part"])
        if group_by_color:
            groups[(b["y"], b["color"], is_p)].append((i, b))
        else:
            groups[(b["y"], "all", is_p)].append((i, b))

    for key, group in groups.items():
        if len(group) < min_merge_count:
            continue
            
        already_merged = set()
        
        is_p = key[-1]
        target_mapping = PLATE_MERGE_TARGETS if is_p else MERGE_TARGET_BRICKS

        # --- X 방향 병합 (같은 Z에서) ---
        z_groups = defaultdict(list)
        for idx, brick in group:
            z_groups[brick["z"]].append((idx, brick))

        for z, z_items in z_groups.items():
            if len(z_items) < min_merge_count:
                continue

            z_items.sort(key=lambda item: item[1]["x"])
            _merge_direction(
                z_items, "x", target_mapping, min_merge_count, max_len,
                group_by_color, anchor_indices,
                already_merged, merged_indices, all_merged_indices,
                generated_bricks
            )

        # --- Z 방향 병합 (같은 X에서, 아직 병합 안 된 것만) ---
        x_groups = defaultdict(list)
        for idx, brick in group:
            if idx in already_merged:
                continue
            x_groups[brick["x"]].append((idx, brick))

        for x, x_items in x_groups.items():
            if len(x_items) < min_merge_count:
                continue

            x_items.sort(key=lambda item: item[1]["z"])
            _merge_direction(
                x_items, "z", target_mapping, min_merge_count, max_len,
                group_by_color, anchor_indices,
                already_merged, merged_indices, all_merged_indices,
                generated_bricks
            )
                
        if not group_by_color:
             logger.info("      [Z-Merge] %s rows processed.", len(z_groups))

    # 병합되지 않은 나머지 브릭들도 결과에 포함
    for idx, brick in enumerate(bricks):
        if idx not in all_merged_indices:
            generated_bricks.append(brick)

    # 2단계: 2xN 범용 병합
    final_bricks, count_rect = merge_rect_bricks(generated_bricks, group_by_color)
    merge_count += count_rect
    
    if count_rect > 0:
        logger.info("   [Rectangular Merge] %s pairs of bricks merged into larger rectangles.", count_rect)

    # 최종 LDR 라인 생성
    new_brick_lines = []
    for b in final_bricks:
        line = build_ldr_line(
            b["color"],
            b["x"], b["y"], b["z"],
            b["matrix"],
            b["part"]
        )
        new_brick_lines.append(line + "\n")

    return new_brick_lines, merged_indices, merge_count


def _merge_direction(
    items: list, 
    direction: str, 
    target_mapping: dict,
    min_merge_count: int,
    max_len: Optional[int],
    group_by_color: bool,
    anchor_indices: Optional[set],
    already_merged: set,
    merged_indices: set,
    all_merged_indices: set,
    generated_bricks: list
):
    """
    X 또는 Z 한 방향으로 1xN 선형 병합을 수행하는 내부 함수.
    
    Args:
        direction: "x" 또는 "z" (병합 방향)
    """
    # Z방향이면 매트릭스를 RotY90으로 설정
    is_z_dir = (direction == "z")

    coord_key = "x" if not is_z_dir else "z"
    merge_count = 0

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
            prev_coord = sequence[-1][1][coord_key]
            if abs(brick_j[coord_key] - prev_coord - STUD_SPACING) < 0.1:
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
                first_brick = sequence[0][1]
                last_brick = sequence[seq_len - 1][1]

                # [ANCHOR CHECK] 색상 무관 병합 시에만 적용
                if not group_by_color and anchor_indices:
                    has_anchor = any(orig_idx in anchor_indices for orig_idx, _ in sequence[:seq_len])
                    if not has_anchor:
                        colors = [rb["color"] for _, rb in sequence[:seq_len]]
                        all_same_color = len(set(colors)) == 1
                        if not all_same_color:
                            seq_len -= 1
                            continue
                
                center_x = (first_brick["x"] + last_brick["x"]) / 2
                center_y = first_brick["y"]
                center_z = (first_brick["z"] + last_brick["z"]) / 2
                
                if is_z_dir:
                    mat = [0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
                else:
                    mat = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

                # 색상 결정: 원본 부피가 큰 쪽, priority_color 우선
                max_vol = 0
                for _, rb in sequence[:seq_len]:
                    v = rb.get("_orig_vol", 1)
                    if v > max_vol:
                        max_vol = v
                        
                candidates = [rb for _, rb in sequence[:seq_len] if rb.get("_orig_vol", 1) == max_vol]
                priority_candidates = [rb for rb in candidates if rb.get("_priority_color")]
                target_group = priority_candidates if priority_candidates else candidates
                
                target_colors = [rb["color"] for rb in target_group]
                final_color = Counter(target_colors).most_common(1)[0][0]
                
                has_priority = any(rb.get("_priority_color") for _, rb in sequence[:seq_len])

                new_brick = {
                    "part": target_mapping[seq_len],
                    "color": final_color,
                    "x": int(round(center_x)) if is_z_dir else center_x, 
                    "y": int(round(center_y)), 
                    "z": int(round(center_z)) if is_z_dir else center_z,
                    "matrix": mat,
                    "_orig_vol": max_vol
                }
                if has_priority:
                    new_brick["_priority_color"] = True

                generated_bricks.append(new_brick)
                
                for idx_s, _ in sequence[:seq_len]:
                    already_merged.add(idx_s)
                    merged_indices.add(idx_s)
                    all_merged_indices.add(idx_s)
                merge_count += 1
                merged_any = True
                break
            seq_len -= 1

        if not merged_any:
            i = j if j > i + 1 else i + 1
        else:
            while i < len(items):
                idx_next, _ = items[i]
                if idx_next not in already_merged:
                     break
                i += 1
