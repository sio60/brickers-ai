# ============================================================================
# 구조적 병합 모듈 (Structural Merge)
# 불안정 브릭과 안정 브릭의 경계를 분해 후 재병합하여 구조적 연결 강화
# ============================================================================

import logging
from typing import Optional
from pathlib import Path

from .constants import (
    STUD_SPACING, SMALL_BRICK_PARTS,
    BRICK_STUD_COUNT, BRICK_DIMENSIONS,
    MERGE_TARGET_BRICKS,
    is_plate
)
from .parser import parse_ldr_line
from .merge_linear import merge_all_1x1

logger = logging.getLogger(__name__)


def get_brick_stud_positions(brick: dict) -> list:
    """
    브릭의 모든 스터드 위치를 반환합니다 (2xN 브릭 포함).
    회전 행렬을 분석하여 2차원 그리드(Row, Col)의 실제 월드 좌표를 계산합니다.
    """
    part = brick["part"]
    rows, cols = BRICK_DIMENSIONS.get(part, (1, 1))
    
    if rows == 1 and cols == 1:
        return [(brick["x"], brick["y"], brick["z"])]

    matrix = brick["matrix"]
    
    # 현재 브릭의 로컬 X축(Col방향) 단위 벡터
    local_col_vec = (matrix[0], matrix[1], matrix[2])
    # 현재 브릭의 로컬 Z축(Row방향) 단위 벡터
    local_row_vec = (matrix[6], matrix[7], matrix[8])

    # 공식: StartOffset = -((Count - 1) * 20) / 2
    col_start_offset = -((cols - 1) * STUD_SPACING) / 2.0
    row_start_offset = -((rows - 1) * STUD_SPACING) / 2.0
    
    positions = []
    
    for r in range(rows):
        for c in range(cols):
            local_x = col_start_offset + (c * STUD_SPACING)
            local_z = row_start_offset + (r * STUD_SPACING)
            
            # 월드 좌표 변환
            wx = brick["x"] + (local_x * local_col_vec[0]) + (local_z * local_row_vec[0])
            wy = brick["y"] + (local_x * local_col_vec[1]) + (local_z * local_row_vec[1])
            wz = brick["z"] + (local_x * local_col_vec[2]) + (local_z * local_row_vec[2])
            
            positions.append((int(round(wx)), int(round(wy)), int(round(wz))))

    return positions


def _split_brick_to_1x1(brick: dict, priority_color: bool = False) -> list:
    """
    큰 브릭/플레이트를 1x1 단위로 분해합니다.
    [중요] 원본이 플레이트면 1x1 플레이트로, 브릭이면 1x1 브릭으로 분해하여 높이 충돌 방지.
    """
    positions = get_brick_stud_positions(brick)
    if len(positions) <= 1:
        b = brick.copy()
        if priority_color:
            b["_priority_color"] = True
        return [b]

    target_part = "3024.dat" if is_plate(brick["part"]) else "3005.dat"
    orig_vol = len(positions)

    result_bricks = []
    for x, y, z in positions:
        new_brick = {
            "type": 1,
            "color": brick["color"],
            "x": int(round(x)),
            "y": int(round(y)),
            "z": int(round(z)),
            "matrix": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            "part": target_part,
            "_orig_vol": orig_vol
        }
        if priority_color:
            new_brick["_priority_color"] = True
        result_bricks.append(new_brick)

    return result_bricks


def structural_merge(ldr_path: str, unstable_ids: list) -> dict:
    """
    구조적 병합 (개선된 1회 병합 로직)
    """
    path = Path(ldr_path)
    if not path.exists():
        return {"merged": 0, "split": 0, "rounds": 0}

    # 1. 파일 읽기 및 브릭 파싱
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    all_bricks = []
    brick_counter = 0
    for i, line in enumerate(lines):
        parsed = parse_ldr_line(line)
        if parsed is None:
            continue
        parsed["line_idx"] = i
        parsed["brick_idx"] = brick_counter
        all_bricks.append(parsed)
        brick_counter += 1

    if not all_bricks:
        return {"merged": 0, "split": 0, "rounds": 0}

    unstable_set = set(int(uid) for uid in unstable_ids if uid is not None)

    # 2. 분해 대상 선정 (불안정 + 수평 인접 안정)
    pos_to_brick_idx = {}
    idx_to_brick = {b["brick_idx"]: b for b in all_bricks}
    for b in all_bricks:
        positions = get_brick_stud_positions(b)
        for x, y, z in positions:
            key = (round(x, 1), round(y, 1), round(z, 1))
            pos_to_brick_idx[key] = b["brick_idx"]

    # 안정 브릭 중 경계면에 있는 것 찾기
    stable_boundary_indices = set()
    
    for b in all_bricks:
        if b["brick_idx"] not in unstable_set:
            continue
            
        positions = get_brick_stud_positions(b)
        for bx, by, bz in positions:
            neighbors = [
                (bx + STUD_SPACING, by, bz),
                (bx - STUD_SPACING, by, bz),
                (bx, by, bz + STUD_SPACING),
                (bx, by, bz - STUD_SPACING),
            ]
            
            for nx, ny, nz in neighbors:
                n_key = (round(nx, 1), round(ny, 1), round(nz, 1))
                if n_key in pos_to_brick_idx:
                    n_idx = pos_to_brick_idx[n_key]
                    if n_idx not in unstable_set:
                        neighbor_brick = idx_to_brick.get(n_idx)
                        if neighbor_brick:
                            rows, cols = BRICK_DIMENSIONS.get(neighbor_brick["part"], (0, 0))
                            if rows == 1: 
                                stable_boundary_indices.add(n_idx)

    # 3. 분해 대상 선정
    indices_to_split = unstable_set | stable_boundary_indices

    # 4. 분해 실행
    lines_to_delete = set()
    new_1x1_bricks = []
    anchor_indices = set()
    split_count = 0

    max_y = max(b["y"] for b in all_bricks) if all_bricks else 0

    for brick in all_bricks:
        if brick["brick_idx"] not in indices_to_split:
            continue
        
        is_anchor = (brick["brick_idx"] not in unstable_set) or (abs(brick["y"] - max_y) < 0.1)
        
        split_bricks = _split_brick_to_1x1(brick, priority_color=(not is_anchor))
        if split_bricks:
            lines_to_delete.add(brick["line_idx"])
            start_idx = len(new_1x1_bricks)
            new_1x1_bricks.extend(split_bricks)
            
            if is_anchor:
                for i in range(len(split_bricks)):
                    anchor_indices.add(start_idx + i)
            
            if len(split_bricks) > 1:
                split_count += 1

    if not lines_to_delete and not new_1x1_bricks:
        return {"merged": 0, "split": 0, "rounds": 0}

    # 5. 재병합
    merged_new_lines, merged_indices, merge_count = merge_all_1x1(
        new_1x1_bricks, 
        group_by_color=False, 
        max_len=4, 
        anchor_indices=anchor_indices
    )
    
    # 6. 파일 업데이트
    final_lines = []
    
    for i, line in enumerate(lines):
        if i not in lines_to_delete:
            final_lines.append(line)
            
    final_lines.extend(merged_new_lines)

    if final_lines and not final_lines[-1].endswith("\n"):
        final_lines[-1] += "\n"

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(final_lines)

    logger.info(f"구조적 병합(1회) 완료: 분해 {split_count}개(안정 포함), 병합 {merge_count}개 그룹")
    return {"merged": merge_count, "split": split_count, "rounds": 1}
