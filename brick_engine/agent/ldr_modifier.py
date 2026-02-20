# ============================================================================
# LDR 파일 수정 모듈 (개선판)
# LLM의 결정에 따라 LDR 파일에서 브릭을 이동하거나 삭제
# 인덱스 밀림 및 ID 불일치 문제를 해결함 (LdrLoader의 글로벌 인덱스 방식 채택)
# ============================================================================

import logging
from typing import Tuple, Optional, List
from pathlib import Path
from collections import defaultdict, Counter

# ============================================================================
# 로깅 설정
# ============================================================================
logger = logging.getLogger(__name__)

# LDU 상수 (브릭 병합용)
STUD_SPACING = 20.0  # X/Z 그리드 간격
BRICK_HEIGHT = 24.0  # 일반 브릭 높이
PLATE_HEIGHT = 8.0   # 플레이트 높이

# 1x1 브릭/플레이트 부품 번호
SMALL_BRICK_PARTS = {"3005.dat", "3024.dat"}

# 병합 대상 브릭 매핑 (최대 1x4로 제한하여 모델 디테일 보존)
MERGE_TARGET_BRICKS = {
    2: "3004.dat",   # 1x2
    3: "3622.dat",   # 1x3
    4: "3010.dat",   # 1x4
}

# 플레이트 병합 대상 (최대 1x4)
PLATE_MERGE_TARGETS = {
    2: "3023.dat",   # 1x2 Plate
    3: "3623.dat",   # 1x3 Plate
    4: "3710.dat",   # 1x4 Plate
    6: "3666.dat",   # 1x6 Plate
    8: "3460.dat",   # 1x8 Plate
}

# 브릭 크기별 스터드 수 (역매핑용)
BRICK_STUD_COUNT = {
    # Bricks
    "3005.dat": 1, "3004.dat": 2, "3622.dat": 3, "3010.dat": 4, 
    "3009.dat": 6, "3008.dat": 8, "3007.dat": 10, "3006.dat": 12,
    "3003.dat": 4,   # 2x2 Brick (Special case for splitting)
    "3002.dat": 6,   # 2x3 Brick
    "3001.dat": 8,   # 2x4 Brick
    
    # Plates
    "3024.dat": 1, "3023.dat": 2, "3623.dat": 3, "3710.dat": 4,
    "3666.dat": 6, "3460.dat": 8, "3022.dat": 4, # 2x2 Plate
    "3021.dat": 6, # 2x3 Plate
    "3020.dat": 8, # 2x4 Plate
}

def _is_plate(part_name: str) -> bool:
    """부품 번호를 기준으로 플레이트 여부 판별"""
    # 전형적인 플레이트 번호대: 302, 3623, 3710, 3666, 3460 등
    # 또는 BRICK_STUD_COUNT에 등록된 플레이트들
    plates = {"3024.dat", "3023.dat", "3623.dat", "3710.dat", "3666.dat", "3460.dat", "3022.dat", "3021.dat", "3020.dat"}
    if part_name in plates:
        return True
    if part_name.startswith("302"): # 302x 계열은 대부분 플레이트
        return True
    return False

# 2x2 병합 매핑 (입력 1x2 -> 출력 2x2)
BRICK_2X2_MAPPING = {
    "3004.dat": "3003.dat", # 1x2 Brick -> 2x2 Brick
    "3023.dat": "3022.dat"  # 1x2 Plate -> 2x2 Plate
}

def _merge_2x2_bricks(bricks: list, group_by_color: bool = True) -> tuple:
    """
    1xN 병합 후 남은 1x2 브릭들을 2x2(정사각)로 2차 병합합니다.
    오직 1x2 + 1x2 = 2x2 케이스만 처리합니다.
    
    Returns:
        (merged_bricks_list, merge_count)
    """
    # 1. 1x2 브릭만 필터링 + 그룹화 (Y, Part, [Color], Orientation)
    # Orientation: X축 방향인지 Z축 방향인지 구분 필요
    # X축 방향(3004): matrix[0] > 0.5 (또는 -0.5)
    # Z축 방향(3004 + rotY): matrix[6] > 0.5

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

    # 그룹화 키: (y, part, color(opt), orientation_axis)
    groups = defaultdict(list)

    for i, b in enumerate(candidates):
        # 방향 판별
        a = b["matrix"][0]
        g = b["matrix"][6]
        axis = 'x' if abs(a) > 0.5 else 'z'

        if group_by_color:
            key = (b["y"], b["part"], b["color"], axis)
        else:
            key = (b["y"], b["part"], axis) # 색상 무관

        groups[key].append((i, b))

    for key, items in groups.items():
        if len(items) < 2:
            continue

        # 매칭 로직
        # axis='x'인 경우: 두 브릭은 나란히 Z축으로 2칸(STUD_SPACING*2) 떨어져 있어야 함?
        # 아니, 1x2 브릭은 길이 2칸, 폭 1칸임.
        # X축 방향 1x2 브릭 두 개가 2x2가 되려면, Z축으로 1칸(STUD_SPACING) 옆에 붙어 있어야 함.
        # 반대로 Z축 방향 1x2 브릭 두 개는, X축으로 1칸 옆에 붙어 있어야 함.

        axis = key[-1]

        if axis == 'x':
            items.sort(key=lambda x: (x[1]["x"], x[1]["z"]))
        else:
            items.sort(key=lambda x: (x[1]["z"], x[1]["x"]))

        used = set()

        for i in range(len(items)):
            if i in used: continue

            idx_i, b_i = items[i]

            # 인접한 브릭 찾기
            best_mate_idx = -1

            for j in range(i + 1, len(items)):
                if j in used: continue

                idx_j, b_j = items[j]

                if axis == 'x':
                    # 길이가 X축인 1x2 브릭들
                    # X좌표는 같아야 하고 (오차 허용)
                    # Z좌표 차이가 정확히 1칸 (STUD_SPACING = 20)
                    if abs(b_i["x"] - b_j["x"]) < 1.0:
                        dist = abs(b_i["z"] - b_j["z"])
                        if abs(dist - STUD_SPACING) < 1.0:
                            best_mate_idx = j
                            break
                else:
                    # 길이가 Z축인 1x2 브릭들
                    # Z좌표는 같아야 하고
                    # X좌표 차이가 정확히 1칸
                    if abs(b_i["z"] - b_j["z"]) < 1.0:
                        dist = abs(b_i["x"] - b_j["x"])
                        if abs(dist - STUD_SPACING) < 1.0:
                            best_mate_idx = j
                            break

            if best_mate_idx != -1:
                # 병합 수행
                idx_j, b_j = items[best_mate_idx]
                used.add(i)
                used.add(best_mate_idx)
                merged_indices.add(idx_i) # candidates 리스트 내 인덱스
                merged_indices.add(idx_j)

                # 새로운 2x2 브릭 생성
                # 위치: 두 1x2의 정중앙
                # [COORD FIX] 좌표 정수화 (Integer Snapping)
                new_x = int(round((b_i["x"] + b_j["x"]) / 2))
                new_y = int(round(b_i["y"])) 
                new_z = int(round((b_i["z"] + b_j["z"]) / 2))

                new_part = BRICK_2X2_MAPPING[b_i["part"]]
                
                # [COLOR VOLUME PRIORITY]
                # 사용자 요구: 2x1 통브릭(안정)이 1x1 낱개(불안정)보다 색상 우선권을 가져야 함.
                # 1순위: _orig_vol (원본 크기)이 큰 쪽
                # 2순위: _priority_color (불안정) 여부
                # 3순위: 다수결 (여기선 둘 뿐이라 의미 없음)
                
                vol_i = b_i.get("_orig_vol", 1)
                vol_j = b_j.get("_orig_vol", 1)
                
                if vol_i > vol_j:
                    final_color = b_i["color"]
                    final_vol = vol_i
                elif vol_j > vol_i:
                    final_color = b_j["color"]
                    final_vol = vol_j
                else:
                    # 크기가 같으면 불안정 브릭 우선
                    p_i = b_i.get("_priority_color", False)
                    p_j = b_j.get("_priority_color", False)
                    if p_i and not p_j:
                        final_color = b_i["color"]
                    elif p_j and not p_i:
                        final_color = b_j["color"]
                    else:
                        # 둘 다 같으면 i 우선
                        final_color = b_i["color"]
                    final_vol = vol_i 
                
                # 병합된 결과물 속성 전파
                has_priority = b_i.get("_priority_color") or b_j.get("_priority_color")

                new_brick = {
                    "part": new_part,
                    "color": final_color,
                    # [COORD FIX] 좌표 정수화 (Integer Snapping)
                    # LDraw Unit은 정수여야 함. 소수점 오차 완전 제거.
                    "x": int(round(new_x)), 
                    "y": int(round(new_y)), 
                    "z": int(round(new_z)),
                    "matrix": b_i["matrix"], # 방향 유지
                    "_orig_vol": final_vol   # 부피 정보 유지 (더 큰 병합 위해)
                }
                if has_priority:
                    new_brick["_priority_color"] = True
                    
                final_bricks.append(new_brick)
                merge_count += 1
            else:
                # 짝을 못 찾음 -> 그대로 유지
                final_bricks.append(b_i)
                used.add(i)

    return final_bricks, merge_count

def _get_brick_stud_positions(brick: dict) -> list:
    """
    브릭의 모든 스터드 위치를 반환합니다.
    회전 행렬을 분석하여 길이 방향(X/Z)을 판별합니다.
    """
    part = brick["part"]
    stud_count = BRICK_STUD_COUNT.get(part, 1)

    if stud_count == 1:
        return [(brick["x"], brick["y"], brick["z"])]

    matrix = brick["matrix"]
    # 2x2, 2x4 등의 정방형/두꺼운 브릭 처리 (현재는 1xN 위주로 계산하되 중앙점들 반환)
    # TODO: 2xN 브릭의 정확한 모든 스터드 위치 계산 (현재는 길이 방향만 처리)

    # 회전 행렬 a(matrix[0])와 g(matrix[6])으로 길이 방향 판별
    a = matrix[0]  
    g = matrix[6]  

    # [FIX] LDraw 원점(중앙) 기준 스터드 오프셋 계산 (정중앙 정렬 보장)
    start_offset = -(stud_count - 1) * STUD_SPACING / 2.0
    positions = []
    
    for i in range(stud_count):
        step = start_offset + (i * STUD_SPACING)
        # 1xN 브릭의 경우 스터드 간격은 STUD_SPACING
        if abs(a) > 0.5:
            # X축 방향 정렬 (Identity 등)
            dx = step * (1 if a > 0 else -1)
            # [COORD FIX] 좌표 정수화 (Integer Snapping)
            positions.append((int(round(brick["x"] + dx)), int(round(brick["y"])), int(round(brick["z"]))))
        elif abs(g) > 0.5:
            # Z축 방향 정렬 (RotateY90 등)
            dz = step * (1 if g > 0 else -1)
            positions.append((int(round(brick["x"])), int(round(brick["y"])), int(round(brick["z"] + dz))))
        else:
            # 회전이 복잡한 경우 (일단 중앙점만 반환하여 안전성 유지)
            positions.append((int(round(brick["x"])), int(round(brick["y"])), int(round(brick["z"]))))
            if i == 0: break 

    return positions


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
    """LDR 라인 재구성 (좌표 정수화 포함)"""
    # [COORD FIX] 최종 출력 시에도 정수 좌표 강제 (float 오차 방지)
    ix, iy, iz = int(round(x)), int(round(y)), int(round(z))
    # 매트릭스도 정수일 경우 .0 제거 (깔끔한 출력)
    fmt_matrix = []
    for m in matrix:
        if isinstance(m, float) and m.is_integer():
            fmt_matrix.append(str(int(m)))
        else:
            fmt_matrix.append(str(m))
            
    matrix_str = " ".join(fmt_matrix)
    return f"1 {color} {ix} {iy} {iz} {matrix_str} {part}"


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
        logger.warning(f"파일 없음: {ldr_path}")
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
            logger.warning(f"찾을 수 없는 브릭 ID: {brick_id} (건너뛰)")
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
                logger.debug(f"지지 브릭 추가 완료: {part} at {position}")
            else:
                stats["failed"] += 1
                
        elif action == "delete":
            lines[line_idx] = None  # 삭제 표시
            stats["deleted"] += 1
            logger.debug(f"브릭 삭제 처리 완료: {brick_id}")
            
        elif action == "keep":
            stats["kept"] += 1
    
    # 4. 결과 저장 (None이 아닌 라인만 쓰기)
    new_lines = [line for line in lines if line is not None]
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
        
    logger.info(f"LDR 수정 완료: {ldr_path}")
    logger.info(f"결과: 추가 {stats['added']}, 이동 {stats['moved']}, 삭제 {stats['deleted']}, 유지 {stats['kept']}, 실패 {stats['failed']}")
    
    return stats


# 하위 호환성을 위한 래퍼 함수들
def modify_brick_position(ldr_path: str, brick_id: str, new_position: Tuple[float, float, float]) -> bool:
    res = apply_llm_decisions(ldr_path, [{"brick_id": brick_id, "action": "move", "position": list(new_position)}])
    return res["moved"] > 0

def remove_brick(ldr_path: str, brick_id: str) -> bool:
    res = apply_llm_decisions(ldr_path, [{"brick_id": brick_id, "action": "delete"}])
    return res["deleted"] > 0


# ============================================================================
# 브릭 병합 기능 (MergeBricks)
# 같은 색상의 인접 1x1 브릭들을 큰 브릭으로 통합하여 구조적 안정성 향상
# ============================================================================

# 병합 가능한 1x1 브릭 파트 번호 (플레이트 제외)
SMALL_BRICK_PARTS = {"3005.dat"}  # 1x1 브릭만 대상

# 큰 브릭으로 교체할 매핑 (길이 -> 파트 번호)
# 플레이트는 사용하지 않음 (1x5, 1x7 브릭은 레고에 존재하지 않아 제외)
MERGE_TARGET_BRICKS = {
    2: "3004.dat",   # 1x2 브릭
    3: "3622.dat",   # 1x3 브릭
    4: "3010.dat",   # 1x4 브릭
    6: "3009.dat",   # 1x6 브릭
    8: "3008.dat",   # 1x8 브릭
}


def merge_small_bricks(ldr_path: str, target_brick_ids: Optional[list] = None, min_merge_count: int = 2, max_len: Optional[int] = None, group_by_color: bool = True) -> dict:
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
    
    # 레거시 호환용 (사용자가 이 값을 출력에 사용 중이므로 신중히 유지)
    stats["original_count"] = len(bricks) 

    if len(bricks) < min_merge_count:
        stats["new_count"] = len(bricks)
        stats["total_new_count"] = total_brick_count
        return stats
    
    # 신규 통합 엔진 호출
    new_lines_content, merged_indices, merge_count = _merge_all_1x1(
        bricks, 
        min_merge_count, 
        group_by_color=group_by_color, # 파라미터 적용
        max_len=max_len
    )
    
    if merged_indices:
        final_lines = []
        merged_line_indices = {bricks[idx]["line_idx"] for idx in merged_indices}
        
        for i, line in enumerate(lines):
            if i not in merged_line_indices:
                final_lines.append(line)
        
        # 병합된 새 라인들 추가
        final_lines.extend(new_lines_content)
        
        # [FIX] 마지막 라인 개행 보장
        if final_lines and not final_lines[-1].endswith("\n"):
            final_lines[-1] += "\n"
        
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(final_lines)
            
        stats["merged"] = merge_count
        stats["small_brick_new_count"] = len(bricks) - len(merged_indices) + len(new_lines_content)
        stats["total_new_count"] = total_brick_count - len(merged_indices) + len(new_lines_content)
        
        # 레거시 호환용
        stats["new_count"] = stats["total_new_count"]
    else:
        stats["small_brick_new_count"] = len(bricks)
        stats["total_new_count"] = total_brick_count
        stats["new_count"] = total_brick_count
        
    return stats


# ============================================================================
# 구조적 병합 (Structural Merge)
# 불안정 브릭과 안정 브릭의 경계를 분해 후 재병합하여 구조적 연결 강화
# X+Z 양방향, 색상 무관 병합 지원
# ============================================================================

# 브릭/플레이트 파트 → 스터드 수 역매핑 (분해 및 판별용)
BRICK_STUD_COUNT = {
    # Bricks
    "3005.dat": 1, "3004.dat": 2, "3622.dat": 3, "3010.dat": 4, 
    "3009.dat": 6, "3008.dat": 8, "3007.dat": 10, "3006.dat": 12,
    "3003.dat": 4,   # 2x2 Brick (Special case for splitting)
    "3002.dat": 6,   # 2x3 Brick
    "3001.dat": 8,   # 2x4 Brick
    
    # Plates
    "3024.dat": 1, "3023.dat": 2, "3623.dat": 3, "3710.dat": 4,
    "3666.dat": 6, "3460.dat": 8, "3022.dat": 4, # 2x2 Plate
    "3021.dat": 6, # 2x3 Plate
    "3020.dat": 8, # 2x4 Plate
}

# 플레이트 병합 매핑
PLATE_MERGE_TARGETS = {
    2: "3023.dat",   # 1x2 Plate
    3: "3623.dat",   # 1x3 Plate
    4: "3710.dat",   # 1x4 Plate
    6: "3666.dat",   # 1x6 Plate
    8: "3460.dat",   # 1x8 Plate
}

def _is_plate(part_name: str) -> bool:
    """부품 번호를 기준으로 플레이트 여부 판별"""
    # 전형적인 플레이트 번호대: 302, 3623, 3710, 3666, 3460 등
    # 또는 BRICK_STUD_COUNT에 등록된 플레이트들
    plates = {"3024.dat", "3023.dat", "3623.dat", "3710.dat", "3666.dat", "3460.dat", "3022.dat", "3021.dat", "3020.dat"}
    if part_name in plates:
        return True
    if part_name.startswith("302"): # 302x 계열은 대부분 플레이트
        return True
    return False

# 2xN 병합 매핑 (Target Size -> Target Part)
TARGET_RECT_SIZES = {
    (2, 2): "3003.dat", # 2x2 Brick
    (2, 3): "3002.dat", # 2x3 Brick
    (2, 4): "3001.dat", # 2x4 Brick
    # (2, 6) 등 추가 가능
    # (2, 2) Plate는 별도? 아니면 포함?
    # 지금은 Brick 위주. Plate는 두께가 달라서 높이 조정 필요할수도.
}

PLATE_RECT_SIZES = {
    (2, 2): "3022.dat",
    (2, 3): "3021.dat",
    (2, 4): "3020.dat",
    (2, 6): "3795.dat"
}

def _merge_rect_bricks(bricks: list, group_by_color: bool = True) -> tuple:
    """
    1차 병합된 브릭들을 대상으로, 인접하여 더 큰 직사각형(2xN)을 형성하는 쌍을 찾아 병합합니다.
    지원 병합 예:
      - 1x2 + 1x2 -> 2x2
      - 1x3 + 1x3 -> 2x3
      - 2x2 + 1x2 -> 2x3 (순차적 병합으로 달성 가능: 먼저 1x2+1x2=2x2 되고, 그게 다시 1x2와 합쳐짐?)
        -> 아니, 2x2가 된 후에는 1x2와 합쳐지려면 (2,2) + (1,2) -> (3,2)?? No.
        -> (2,2) + (2,1) -> (2,3) !
        -> 즉, Edge 길이가 맞는 쪽으로 합쳐져야 함.
    
    Args:
        bricks: 병합 대상 브릭 리스트
        group_by_color: 색상 구분 여부
        
    Returns:
        (merged_bricks, count)
    """
    current_bricks = bricks
    total_merge_count = 0
    
    # 반복적으로 병합 시도 (2x2 만든 뒤 2x3 만들 수 있도록)
    # 하지만 무한 루프 방지 위해 최대 2~3회 패스
    for _ in range(3):
        merged_bricks, count = _merge_rect_pass(current_bricks, group_by_color)
        if count == 0:
            break
        current_bricks = merged_bricks
        total_merge_count += count
        
    return current_bricks, total_merge_count

def _merge_rect_pass(bricks: list, group_by_color: bool) -> tuple:
    """1회 패스 실행"""
    # 1. 대상 브릭 필터링 (이미 2xN이 된 브릭도 포함하여 더 크게 합칠 수 있음)
    # 단, 직사각형 형태인 것만. (L형 등은 불가)
    
    # 2. 그룹화 (Y, [Color], Orientation??)
    # Orientation이 문제. 1x2가 가로(X)로 놓였냐 세로(Z)로 놓였냐.
    # 2x2는 대칭이라 구분이 없나? 아님. Matrix로 확인.
    
    groups = defaultdict(list)
    
    # [FIX] 크기가 큰 것도 포함시켜야 함. (삭제 방지)
    # 단, 병합 후보에는 넣지 않더라도 final_bricks에는 무조건 들어가야 함.
    # 하지만 여기선 groups에 다 넣고, groups 순회 시에 처리하는 게 안전.
    
    final_bricks = []
    for i, b in enumerate(bricks):
        # part -> dimensions
        rows, cols = BRICK_DIMENSIONS.get(b["part"], (1, 1))
        
        # Matrix로 방향 확인
        # (1,0,0) -> X축이 Col 방향
        # (0,0,1) -> Z축이 Row 방향
        
        if rows > 2 or cols > 10: 
             # 병합 대상 아님 -> 별도 그룹으로 빼거나, 아니면 그냥 final_bricks에 바로 추가?
             # 바로 추가하면 인덱스 꼬임? 아니, i는 bricks 인덱스.
             final_bricks.append(b)
             continue
        
        # 로직 단순화:
        # 두 브릭 A, B가 합쳐지려면:
        #  1. Y가 같음
        #  2. 색상이 같음 (옵션)
        #  3. 회전매트릭스가 동일함 (방향 일치)
        #  4. 인접함 (Bounding Box가 딱 붙음)
        #  5. 합쳤을 때 유효한 Target Part가 됨
        
        # 따라서 Group Key: (y, color(opt), matrix_tuple)
        mat_tuple = tuple(b["matrix"])
        key = (b["y"], b["color"] if group_by_color else -1, mat_tuple)
        groups[key].append((i, b))
        
    used_indices = set()
    merge_count = 0
    
    # 그룹별 병합 시도
    for key, items in groups.items():
        if len(items) < 2:
            for _, b in items:
                if b not in final_bricks: final_bricks.append(b) # 중복 방지 로직 필요
            continue
            
        # items를 위치 순으로 정렬?
        # 문제는 2차원 평면이라...
        # X우선 정렬 후 Z인접 확인?
        # Z우선 정렬 후 X인접 확인?
        
        # 그래프 매칭 문제에 가까움. 그리디하게 접근.
        # 정렬: Z, then X
        items.sort(key=lambda x: (x[1]["z"], x[1]["x"]))
        
        # 리스트 내에서 짝 찾기
        local_used = set()
        
        for i in range(len(items)):
            if i in local_used: continue
            
            idx_i, b_i = items[i]
            r1, c1 = BRICK_DIMENSIONS.get(b_i["part"], (1, 1))
            
            best_mate = -1
            
            # 인접 후보 탐색
            for j in range(i + 1, len(items)):
                if j in local_used: continue
                
                idx_j, b_j = items[j]
                r2, c2 = BRICK_DIMENSIONS.get(b_j["part"], (1, 1))
                
                # 두 브릭이 합쳐질 수 있는가?
                # Case 1: 가로(X, Cols)로 붙음 -> Rows 불변, Cols 합침
                # 조건: Z 일치, X 차이 == (C1/2 + C2/2) * 20
                if abs(b_i["z"] - b_j["z"]) < 1.0:
                    dist_x = abs(b_i["x"] - b_j["x"])
                    target_dist = (c1 + c2) * 10.0 # 20/2 + 20/2
                    if abs(dist_x - target_dist) < 1.0:
                        # 합쳤을 때 유효한지 확인
                        new_rows = r1 # = r2 matches by Z check? No, Z check only means center aligned.
                        # If sizes differ (e.g. 1x2 and 2x2), r1 != r2.
                        if r1 != r2: continue # 높이(Row)가 다르면 가로병합 불가 (L형됨)
                        
                        target_key = (r1, c1 + c2)
                        target_part = _get_target_part(target_key, b_i["part"])
                        
                        if target_part:
                            # 병합 수행!
                            # 새 위치: X는 둘의 중간
                            new_x = (b_i["x"] + b_j["x"]) / 2.0
                            new_brick = b_i.copy()
                            new_brick["x"] = new_x
                            new_brick["part"] = target_part
                            # 인덱스는 제거 대상
                            local_used.add(i)
                            local_used.add(j)
                            final_bricks.append(new_brick)
                            merge_count += 1
                            best_mate = j
                            break
                            
                # Case 2: 세로(Z, Rows)로 붙음 -> Cols 불변, Rows 합침
                # 조건: X 일치, Z 차이 == (R1/2 + R2/2) * 20
                if abs(b_i["x"] - b_j["x"]) < 1.0:
                    dist_z = abs(b_i["z"] - b_j["z"])
                    target_dist = (r1 + r2) * 10.0
                    if abs(dist_z - target_dist) < 1.0:
                        if c1 != c2: continue # 너비(Col)가 다르면 세로병합 불가
                        
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
                # 짝을 못 찾음 -> 그대로 유지 (단, 나중에 추가)
                pass

        # 매칭 안된 것들 추가
        for i in range(len(items)):
            if i not in local_used:
                final_bricks.append(items[i][1])
                
    # 순서 보존 위해 인덱스 기반 정렬? (필요없을듯)
    return final_bricks, merge_count

def _get_target_part(size_key: tuple, original_part: str) -> str:
    """합쳐진 (Row, Col) 크기에 해당하는 파트명 반환"""
    # 원본이 Plate면 Plate 목록에서, Brick이면 Brick 목록에서
    is_plate = _is_plate(original_part)
    
    # (Rows, Cols) 순서 정규화? No, (2,3) != (3,2) in LDraw logic?
    # BRICK_DIMENSIONS defines (Rows=Z, Cols=X).
    # If we merge, we produce a new shape.
    # We must match standard parts. 3002 is 2x3. 
    # Does 3002 implies 2 rows 3 cols? Or 3 rows 2 cols?
    # Usually 3002 is 2x3 (2 studs deep, 3 studs wide).
    
    # Try direct lookup
    if is_plate:
        target = PLATE_RECT_SIZES.get(size_key)
        if not target:
            # Try searching swapped? No, rotation handles that.
            # But maybe we merged along Rows making it (3, 2).
            # Then we need a 3x2 part. 3002 is 2x3.
            # So if (3,2) comes in, we might return 3002 but ROTATED 90 degrees?
            # NO, changing rotation is complex.
            # Just look for direct match first.
            pass
        return target
    else:
        return TARGET_RECT_SIZES.get(size_key)

# 브릭 치수 정보 (Part -> (Rows, Cols))
# 앞뒤(Z)가 Rows, 좌우(X)가 Cols (LDraw 기본 방향 기준)
BRICK_DIMENSIONS = {
    # 1xN Bricks
    "3005.dat": (1, 1), "3004.dat": (1, 2), "3622.dat": (1, 3), "3010.dat": (1, 4), 
    "3009.dat": (1, 6), "3008.dat": (1, 8), "3007.dat": (1, 10), "3006.dat": (1, 12),
    
    # 2xN Bricks
    "3003.dat": (2, 2), "3002.dat": (2, 3), "3001.dat": (2, 4), "3000.dat": (2, 2), 
    
    # 1xN Plates
    "3024.dat": (1, 1), "3023.dat": (1, 2), "3623.dat": (1, 3), "3710.dat": (1, 4),
    "3666.dat": (1, 6), "3460.dat": (1, 8), 
    
    # 2xN Plates
    "3022.dat": (2, 2), "3021.dat": (2, 3), "3020.dat": (2, 4), "3795.dat": (2, 6)
}

def _get_brick_stud_positions(brick: dict) -> list:
    """
    브릭의 모든 스터드 위치를 반환합니다 (2xN 브릭 포함).
    회전 행렬을 분석하여 2차원 그리드(Row, Col)의 실제 월드 좌표를 계산합니다.
    """
    part = brick["part"]
    
    # 기본값은 1x1 (Undefined parts)
    rows, cols = BRICK_DIMENSIONS.get(part, (1, 1))
    
    # [EXCEPTION] 1x1은 바로 반환
    if rows == 1 and cols == 1:
        return [(brick["x"], brick["y"], brick["z"])]

    matrix = brick["matrix"]
    
    # 회전 행렬 분석 (LDraw Standard: X=Right, Y=Down, Z=Forward)
    # 1.0, 0.0, 0.0 -> X축 (Cols 방향)
    # 0.0, 0.0, 1.0 -> Z축 (Rows 방향)
    
    # 기본 방향 벡터 (Identity Matrix 기준)
    # Col 증가 -> X 증가 (+20)
    # Row 증가 -> Z 증가 (+20)
    
    # 현재 브릭의 로컬 X축(Col방향) 단위 벡터
    local_col_vec = (matrix[0], matrix[1], matrix[2])
    # 현재 브릭의 로컬 Z축(Row방향) 단위 벡터
    local_row_vec = (matrix[6], matrix[7], matrix[8])

    # 스터드 그리드의 시작점(Top-Left) 오프셋 계산
    # LDraw 원점은 브릭의 정중앙에 위치함.
    # 예: 2x4 브릭 (Rows=2, Cols=4)
    # Width(X) = 4 * 20 = 80, Half = 40. Start X = -30 (첫 스터드 중심 -1.5칸) 
    # Depth(Z) = 2 * 20 = 40, Half = 20. Start Z = -10 (첫 스터드 중심 -0.5칸)
    
    # 공식: StartOffset = -((Count - 1) * 20) / 2
    col_start_offset = -((cols - 1) * STUD_SPACING) / 2.0
    row_start_offset = -((rows - 1) * STUD_SPACING) / 2.0
    
    positions = []
    
    # 2차원 그리드 순회
    for r in range(rows):
        for c in range(cols):
            # 로컬 좌표계에서의 오프셋
            local_x = col_start_offset + (c * STUD_SPACING)
            local_z = row_start_offset + (r * STUD_SPACING)
            
            # 월드 좌표 변환: Center + (LocalX * ColVec) + (LocalZ * RowVec)
            
            # X 성분
            wx = brick["x"] + (local_x * local_col_vec[0]) + (local_z * local_row_vec[0])
            # Y 성분 (보통 0이지만 회전 시 변경 가능)
            wy = brick["y"] + (local_x * local_col_vec[1]) + (local_z * local_row_vec[1])
            # Z 성분
            wz = brick["z"] + (local_x * local_col_vec[2]) + (local_z * local_row_vec[2])
            
            # [COORD FIX] 정수화 적용
            positions.append((int(round(wx)), int(round(wy)), int(round(wz))))

    return positions


def _split_brick_to_1x1(brick: dict, priority_color: bool = False) -> list:
    """
    큰 브릭/플레이트를 1x1 단위로 분해합니다.
    [중요] 원본이 플레이트면 1x1 플레이트로, 브릭이면 1x1 브릭으로 분해하여 높이 충돌 방지.
    
    Args:
        brick: 분해 대상 브릭
        priority_color: True이면 분해된 1x1 브릭에 '_priority_color' 속성 부여 (색상 유지용)
    """
    positions = _get_brick_stud_positions(brick)
    if len(positions) <= 1:
        # 단일 브릭이라도 속성 부여를 위해 리스트로 반환 (복사본 생성)
        b = brick.copy()
        if priority_color:
            b["_priority_color"] = True
        return [b]

    is_plate = _is_plate(brick["part"])
    target_part = "3024.dat" if is_plate else "3005.dat" # 1x1 Plate vs 1x1 Brick

    # 원본 브릭의 부피(스터드 개수) 정보 획득
    # 병합 시 "더 큰 덩어리" 였던 브릭의 색상을 우선하기 위함.
    # [FIX] 정적 매핑 대신 실제 계산된 스터드 수 사용 (2xN 등 다양한 브릭 대응)
    orig_vol = len(positions)

    result_bricks = []
    for x, y, z in positions:
        new_brick = {
            "type": 1,
            "color": brick["color"],
            # [COORD FIX] 좌표 정수화
            "x": int(round(x)),
            "y": int(round(y)),
            "z": int(round(z)),
            "matrix": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            "part": target_part,
            "_orig_vol": orig_vol # 원본 부피 메타데이터 추가
        }
        if priority_color:
            new_brick["_priority_color"] = True
        result_bricks.append(new_brick)

    return result_bricks


def _merge_all_1x1(bricks: list, min_merge_count: int = 2, group_by_color: bool = False, max_len: Optional[int] = None, anchor_indices: Optional[set] = None) -> tuple:
    """
    1x1 브릭/플레이트들을 병합합니다.
    1단계: 1xN (선형) 병합
    2단계: 2x2 (면) 병합 (1x2 + 1x2)
    
    Args:
        bricks: 1x1 브릭 객체 리스트
        min_merge_count: 최소 병합 수
        group_by_color: True이면 같은 색상끼리만 병합 (기본 병합용), False이면 색상 무관 병합 (구조적 병합용)
        max_len: 최대 병합 길이 제한 (None이면 4 적용)
        anchor_indices: 병합 시 반드시 포함되어야 하는 '안정 브릭'의 인덱스 집합 (group_by_color=False일 때 권장)
    """
    generated_bricks = [] # 1차 병합 결과(객체) 수집
    merged_indices = set()
    merge_count = 0

    merged_indices = set()
    generated_bricks = []
    
    # [FIX] 그룹별 병합이지만, 전체 병합된 인덱스는 공유해야 함 (마지막에 중복 추가 방지)
    all_merged_indices = set()

    # 그룹화 (Y좌표별, 색상별 or 전체) - Plate 여부도 포함
    groups = defaultdict(list)
    for i, b in enumerate(bricks):
        is_p = _is_plate(b["part"])
        if group_by_color:
            groups[(b["y"], b["color"], is_p)].append((i, b))
        else:
            groups[(b["y"], "all", is_p)].append((i, b))

    for key, group in groups.items():
        if len(group) < min_merge_count:
            continue
            
        # 해당 그룹 내에서 병합된 것 추적
        already_merged = set()
        
        # Plate/Brick에 따른 병합 테이블 결정
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

            i = 0
            while i < len(z_items):
                idx_i, brick_i = z_items[i]
                if idx_i in already_merged:
                    i += 1
                    continue

                sequence = [(idx_i, brick_i)]
                j = i + 1
                while j < len(z_items):
                    idx_j, brick_j = z_items[j]
                    if idx_j in already_merged:
                        j += 1
                        continue
                    prev_x = sequence[-1][1]["x"]
                    if abs(brick_j["x"] - prev_x - STUD_SPACING) < 0.1:
                        sequence.append((idx_j, brick_j))
                        j += 1
                    else:
                        break

                seq_len = len(sequence)
                # [FIX] 보수적 병합: 최대 1x4로 제한
                eff_max = int(max_len) if max_len is not None else 4
                seq_len = min(seq_len, eff_max)
                    
                merged_any = False
                while seq_len >= min_merge_count:
                    # [ANCHOR CHECK] 색상 무관 병합 시, 결과물에 최소 하나 이상의 '안정 브릭'이 포함되어야 함
                    if not group_by_color and anchor_indices:
                        has_anchor = any(orig_idx in anchor_indices for orig_idx, _ in sequence[:seq_len])
                        if not has_anchor:
                            seq_len -= 1
                            continue

                    if seq_len in target_mapping:
                        first_brick = sequence[0][1]
                        last_brick = sequence[seq_len - 1][1]
                        
                        center_x = (first_brick["x"] + last_brick["x"]) / 2
                        center_y = first_brick["y"]
                        center_z = (first_brick["z"] + last_brick["z"]) / 2
                        
                        # [FIX] Matrix 정규화: X축 병합은 무조건 Identity 행렬 강제
                        identity_mat = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

                        # [COLOR VOLUME PRIORITY]
                        # 1순위: _orig_vol 최대값인 브릭들의 색상
                        # 2순위: 그 중에서 _priority_color 있는 브릭
                        # 3순위: 다수결
                        
                        max_vol = 0
                        for _, rb in sequence[:seq_len]:
                            v = rb.get("_orig_vol", 1)
                            if v > max_vol:
                                max_vol = v
                                
                        # 최대 부피를 가진 후보군 추출
                        candidates = [rb for _, rb in sequence[:seq_len] if rb.get("_orig_vol", 1) == max_vol]
                        
                        # 후보군 중에서 불안정 브릭 필터링
                        priority_candidates = [rb for rb in candidates if rb.get("_priority_color")]
                        
                        target_group = priority_candidates if priority_candidates else candidates
                        
                        # 다수결
                        target_colors = [rb["color"] for rb in target_group]
                        final_color = Counter(target_colors).most_common(1)[0][0]
                        
                        has_priority = any(rb.get("_priority_color") for _, rb in sequence[:seq_len])

                        # 객체 생성 (문자열 아님)
                        new_brick = {
                            "part": target_mapping[seq_len],
                            "color": final_color,
                            # [COORD FIX] 좌표 정수화
                            "x": int(round(center_x)), 
                            "y": int(round(center_y)), 
                            "z": int(round(center_z)),
                            "matrix": identity_mat,
                            "_orig_vol": max_vol # 병합된 브릭은 구성원 중 최대 부피를 상속
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
                    # [FIX] 병합된 경우, already_merged에 포함된 인덱스를 건너뛰어야 함
                    # i를 다음 처리할 인덱스로 이동
                    while i < len(z_items):
                        idx_next, _ = z_items[i]
                        if idx_next not in already_merged:
                             break
                        i += 1

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

            i = 0
            while i < len(x_items):
                idx_i, brick_i = x_items[i]
                if idx_i in already_merged:
                    i += 1
                    continue

                sequence = [(idx_i, brick_i)]
                j = i + 1
                while j < len(x_items):
                    idx_j, brick_j = x_items[j]
                    if idx_j in already_merged:
                        j += 1
                        continue
                    prev_z = sequence[-1][1]["z"]
                    if abs(brick_j["z"] - prev_z - STUD_SPACING) < 0.1:
                        sequence.append((idx_j, brick_j))
                        j += 1
                    else:
                        break

                seq_len = len(sequence)
                eff_max = int(max_len) if max_len is not None else 4
                seq_len = min(seq_len, eff_max)
                    
                merged_any = False
                while seq_len >= min_merge_count:
                    # [ANCHOR CHECK] Z방향
                    if not group_by_color and anchor_indices:
                        has_anchor = any(orig_idx in anchor_indices for orig_idx, _ in sequence[:seq_len])
                        if not has_anchor:
                            seq_len -= 1
                            continue

                    if seq_len in target_mapping:
                        first_brick = sequence[0][1]
                        last_brick = sequence[seq_len - 1][1]
                        
                        center_x = (first_brick["x"] + last_brick["x"]) / 2
                        center_y = first_brick["y"]
                        center_z = (first_brick["z"] + last_brick["z"]) / 2
                        
                        # [FIX] Matrix 정규화: Z축 병합은 무조건 RotateY(90) 행렬 강제
                        rotate_y_90 = [0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
                        
                        # [COLOR VOLUME PRIORITY] Z축
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
                            # [COORD FIX] 좌표 정수화
                            "x": int(round(center_x)), 
                            "y": int(round(center_y)), 
                            "z": int(round(center_z)),
                            "matrix": rotate_y_90,
                            "_orig_vol": max_vol
                        }
                        if has_priority:
                            new_brick["_priority_color"] = True

                        generated_bricks.append(new_brick)
                        
                        for idx_s, _ in sequence[:seq_len]:
                            already_merged.add(idx_s)
                            merged_indices.add(idx_s)
                            all_merged_indices.add(idx_s) # [FIX] 전체 병합 집합에도 추가
                        merge_count += 1
                        merged_any = True
                        break
                    seq_len -= 1

                if not merged_any:
                    i = j if j > i + 1 else i + 1
                else:
                    # [FIX] 병합된 경우, already_merged에 포함된 인덱스를 건너뛰어야 함
                    # i를 다음 처리할 인덱스로 이동
                    while i < len(x_items):
                        idx_next, _ = x_items[i]
                        if idx_next not in already_merged:
                             break
                        i += 1
                        
        # 병합 안 된 1x1 브릭들도 generated_bricks에 추가

                
        if not group_by_color:
             print(f"      [Z-Merge] {len(z_groups)} rows processed.")

    # [FIX] 병합되지 않은 나머지 브릭들도 결과에 포함해야 함
    # all_merged_indices에 없는 인덱스의 브릭들을 generated_bricks에 추가
    for idx, brick in enumerate(bricks):
        if idx not in all_merged_indices:
            # 원본 속성 그대로 유지
            generated_bricks.append(brick)

    # 2단계: 2xN 범용 병합 (Rectangular Merge)
    # 1차 병합 결과를 다시 입력으로 사용
    final_bricks, count_rect = _merge_rect_bricks(generated_bricks, group_by_color)
    merge_count += count_rect
    
    if count_rect > 0:
        print(f"   [Rectangular Merge] {count_rect} pairs of bricks merged into larger rectangles.")

    # 3단계: 최종 LDR 라인 생성
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

    return new_brick_lines, merged_indices, merge_count


def _extend_into_neighbors(all_bricks: list, unstable_set: set) -> tuple:
    """
    불안정 1x1 브릭이 인접한 안정 브릭에 흡수될 수 있는지 확인하고 병합(확장)합니다.
    (예: 1x4 + 1x1 -> 1x5가 아니라 1x6 같은 유효한 브릭으로 변환되는 경우만)

    Returns:
        (new_lines: list, deleted_indices: set)
    """
    # 빠른 검색을 위한 매핑
    pos_to_brick = {}
    for b in all_bricks:
        # 1x1 브릭만 좌표 매핑 (확장 대상이 될 수 있는 큰 브릭은 시작점만 있어도 됨)
        # 하지만 여기선 인접 "큰" 브릭을 찾아야 하므로, 큰 브릭의 모든 스터드 좌표를 매핑해야 함
        positions = _get_brick_stud_positions(b)
        for x, y, z in positions:
            key = (round(x, 1), round(y, 1), round(z, 1))
            pos_to_brick[key] = b

    new_lines = []
    deleted_line_indices = set()
    merged_count = 0

    # 불안정 1x1 브릭 순회
    for brick in all_bricks:
        if brick["brick_idx"] not in unstable_set:
            continue
        if brick["part"] not in SMALL_BRICK_PARTS:
            continue
        if brick["line_idx"] in deleted_line_indices:
            continue

        bx, by, bz = brick["x"], brick["y"], brick["z"]

        # 1. X축 방향 탐색 (좌우)
        for dx in [-STUD_SPACING, STUD_SPACING]:
            target_pos = (round(bx + dx, 1), round(by, 1), round(bz, 1))
            neighbor = pos_to_brick.get(target_pos)

            if neighbor and neighbor["line_idx"] not in deleted_line_indices:
                # 같은 색상, 같은 Y, 같은 Z인지 확인 (X축 확장이므로)
                # 그리고 neighbor가 "안정" 브릭이어야 함? -> 굳이? 불안정끼리라도 합치면 좋음.
                # 하지만 여기선 "안정 브릭에 흡수"가 목표.
                if neighbor["color"] != brick["color"]:
                    continue
                
                # neighbor의 원래 길이 확인
                n_part = neighbor["part"]
                n_len = BRICK_STUD_COUNT.get(n_part)
                if not n_len: continue

                # 합쳤을 때 유효한 길이인가? (n_len + 1)
                new_len = n_len + 1
                if new_len not in MERGE_TARGET_BRICKS:
                    continue

                # neighbor가 X축 정렬인지 확인
                # a(matrix[0])가 1 또는 -1 이어야 함
                nms = neighbor["matrix"]
                if abs(nms[0]) < 0.9: # X축 정렬 아님
                    continue

                # 병합 수행!
                # 새 브릭은 neighbor의 원점 기준이 아니라, 
                # neighbor와 brick을 포함하는 새로운 범위의 중심...이 아니라
                # LDR은 "중심" 기준이 아님. 보통 "첫 번째 스터드" 기준이거나 중심 기준임.
                # LDraw 표준: 브릭 원점은 중앙.
                # 따라서 위치를 재계산해야 함.
                
                # 하지만 _merge_all_1x1 처럼 단순히 1x1로 쪼개고 다시 합치는 게 좌표 계산이 편함.
                # 여기서 좌표 계산하려니 복잡함 (회전 고려 등).
                
                # 따라서 "확장"은 포기하고, "경계면 분해"로 위임하는 게 낫겠음.
                # Why? 1x4 + 1x1 -> 1x5 (X) -> 1x4 + 1x1 (유지)
                # 만약 1x3 + 1x1 -> 1x4 (O) 라면?
                # 이걸 하려면 좌표 계산이 정확해야 함.
                pass

    return [], set(), 0


def structural_merge(ldr_path: str, unstable_ids: list) -> dict:
    """
    구조적 병합 (개선된 1회 병합 로직)
    
    1. 불안정 브릭 식별
    2. "경계면"에 있는 안정 브릭(수평 인접) 식별 및 분해 대상 포함
    3. 대상 브릭들 1x1로 모두 분해
    4. X/Z 양방향, 색상 무관 재병합
    5. 1회 실행 후 종료 (반복 없음)
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
    indices_to_split = set()
    
    # 공간 해싱 (좌표 -> 브릭) 및 인덱스 캐시
    pos_to_brick_idx = {}
    idx_to_brick = {b["brick_idx"]: b for b in all_bricks}
    for b in all_bricks:
        positions = _get_brick_stud_positions(b)
        for x, y, z in positions:
            key = (round(x, 1), round(y, 1), round(z, 1))
            pos_to_brick_idx[key] = b["brick_idx"]

    # 안정 브릭 중 경계면에 있는 것 찾기
    stable_boundary_indices = set()
    
    for b in all_bricks:
        if b["brick_idx"] not in unstable_set:
            continue # 불안정 브릭을 기준으로 주변 탐색
            
        # 불안정 브릭의 주변(수평) 탐색
        positions = _get_brick_stud_positions(b)
        for bx, by, bz in positions:
            # 4방향 (X+, X-, Z+, Z-)
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
                        # [FIX] 오직 1x1 브릭/플레이트만 주변 병합에 참여하도록 제한했으나,
                        # -> [IMPROVED] 1xN 브릭(1x2, 1x3 등)도 참여하도록 확장하여 결합력 강화
                        neighbor_brick = idx_to_brick.get(n_idx)
                        if neighbor_brick:
                            # BRICK_DIMENSIONS를 확인하여 1xN 모양이면 병합 후보로 인정
                            rows, cols = BRICK_DIMENSIONS.get(neighbor_brick["part"], (0, 0))
                            if rows == 1: 
                                stable_boundary_indices.add(n_idx)

    # 3. 분해 대상 선정
    # 3. 분해 대상 선정 (불안정 브릭 + 1x1 인접 안정 브릭)
    # [FIX] 안정 브릭 중에서는 오직 1x1만 포함되도록 필터링 완료
    indices_to_split = unstable_set | stable_boundary_indices

    # 3. 분해 실행 (1xN → N × 1x1)
    lines_to_delete = set()
    new_1x1_bricks = []
    anchor_indices = set() # new_1x1_bricks 리스트 내에서의 인덱스
    split_count = 0

    # 최하단 Y 좌표 (지면) 찾기
    max_y = max(b["y"] for b in all_bricks) if all_bricks else 0

    for brick in all_bricks:
        if brick["brick_idx"] not in indices_to_split:
            continue
        
        # [FIX] 원래 안정적이거나, 지면(Ground)에 닿아있는 브릭은 Anchor로 취급
        is_anchor = (brick["brick_idx"] not in unstable_set) or (abs(brick["y"] - max_y) < 0.1)
        
        # 분해
        # [COLOR PRIORITY] 사용자 요청: 불안정 브릭(외부)의 색상을 절대 사수하라
        # 따라서 불안정 브릭에게 _priority_color 플래그를 부여한다.
        split_bricks = _split_brick_to_1x1(brick, priority_color=(not is_anchor))
        if split_bricks:
            lines_to_delete.add(brick["line_idx"])
            start_idx = len(new_1x1_bricks)
            new_1x1_bricks.extend(split_bricks)
            
            # 안정 브릭에서 분해된 1x1들은 모두 anchor로 등록
            if is_anchor:
                for i in range(len(split_bricks)):
                    anchor_indices.add(start_idx + i)
            
            if len(split_bricks) > 1:
                split_count += 1

    if not lines_to_delete and not new_1x1_bricks:
        return {"merged": 0, "split": 0, "rounds": 0}

    # 4. 재병합 (X+Z 양방향, 색상 무관, 최대길이 4로 확장, Anchor 필수 포함)
    merged_new_lines, merged_indices, merge_count = _merge_all_1x1(
        new_1x1_bricks, 
        group_by_color=False, 
        max_len=4, 
        anchor_indices=anchor_indices
    )
    
    # 5. 파일 업데이트
    # 기존 라인 중 삭제되지 않은 것 + 병합된 새 라인 + (병합 안된 1x1들)
    # _merge_all_1x1은 병합된 결과 라인과, 병합에 사용된 인덱스를 반환함.
    # 병합 안 된 1x1은 new_1x1_bricks에 그대로 남아있음(인덱스로 구분).
    
    final_lines = []
    
    # (1) 보존된 기존 브릭 (분해 안 된 안정 브릭들)
    for i, line in enumerate(lines):
        if i not in lines_to_delete:
            final_lines.append(line)
            
    # (2) 병합 결과 추가 (이미 병합된 브릭 + 병합 안 된 1x1 모두 포함됨)
    final_lines.extend(merged_new_lines)


    # [FIX] 마지막 라인이 개행문자로 끝나지 않는 경우 대비
    if final_lines and not final_lines[-1].endswith("\n"):
        final_lines[-1] += "\n"

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(final_lines)

    logger.info(f"구조적 병합(1회) 완료: 분해 {split_count}개(안정 포함), 병합 {merge_count}개 그룹")
    return {"merged": merge_count, "split": split_count, "rounds": 1}

