# ============================================================================
# LDR 파일 수정 모듈 (개선판)
# LLM의 결정에 따라 LDR 파일에서 브릭을 이동하거나 삭제
# 인덱스 밀림 및 ID 불일치 문제를 해결함 (LdrLoader의 글로벌 인덱스 방식 채택)
# ============================================================================

import logging
from typing import Tuple, Optional, List
from pathlib import Path
from collections import defaultdict

# ============================================================================
# 로깅 설정
# ============================================================================
logger = logging.getLogger(__name__)

# LDU 상수 (브릭 병합용)
STUD_SPACING = 20.0  # X/Z 그리드 간격
BRICK_HEIGHT = 24.0  # 일반 브릭 높이
PLATE_HEIGHT = 8.0   # 플레이트 높이

# 1x1 브릭 부품 번호
SMALL_BRICK_PARTS = {"3005.dat", "3024.dat"}

# 병합 대상 브릭 매핑 (길이 -> 부품 번호)
# 레고 표준 브릭 (Brick 1 x N)
MERGE_TARGET_BRICKS = {
    2: "3004.dat",   # 1x2
    3: "3622.dat",   # 1x3
    4: "3010.dat",   # 1x4
    6: "3009.dat",   # 1x6
    8: "3008.dat",   # 1x8
    10: "3006.dat",  # 1x10
    12: "6112.dat",  # 1x12
    16: "2465.dat",  # 1x16
}

# 브릭 크기별 스터드 수 (역매핑용)
BRICK_STUD_COUNT = {
    "3005.dat": 1, "3024.dat": 1,   # 1x1
    "3004.dat": 2,                   # 1x2
    "3622.dat": 3,                   # 1x3
    "3010.dat": 4,                   # 1x4
    "3009.dat": 6,                   # 1x6
    "3008.dat": 8,                   # 1x8
    "3006.dat": 10,                  # 1x10
    "6112.dat": 12,                  # 1x12
    "2465.dat": 16,                  # 1x16
}


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

# 병합 가능한 1x1 브릭 파트 번호
SMALL_BRICK_PARTS = {"3005.dat", "3024.dat"}  # 1x1 브릭, 1x1 플레이트

# 큰 브릭으로 교체할 매핑 (길이 -> 파트 번호)
# 플레이트는 사용하지 않음 (1x5, 1x7 브릭은 레고에 존재하지 않아 제외)
MERGE_TARGET_BRICKS = {
    2: "3004.dat",   # 1x2 브릭
    3: "3622.dat",   # 1x3 브릭
    4: "3010.dat",   # 1x4 브릭
    6: "3009.dat",   # 1x6 브릭
    8: "3008.dat",   # 1x8 브릭
}


def merge_small_bricks(
    ldr_path: str,
    target_brick_ids: Optional[List[str]] = None,
    min_merge_count: int = 2
) -> dict:
    """
    같은 색상의 인접 1x1 브릭들을 큰 브릭으로 병합합니다.
    색상이 다른 브릭은 병합하지 않습니다.
    
    Args:
        ldr_path: LDR 파일 경로
        target_brick_ids: 병합 대상 브릭 ID 리스트 (None이면 모든 1x1 브릭 대상)
        min_merge_count: 최소 병합 개수 (기본값: 2)
        
    Returns:
        {"merged": 병합된 그룹 수, "original_count": 원본 브릭 수, "new_count": 병합 후 브릭 수}
    """
    stats = {"merged": 0, "original_count": 0, "new_count": 0}
    path = Path(ldr_path)
    
    if not path.exists():
        logger.warning(f"파일 없음: {ldr_path}")
        return stats
    
    # 1. 파일 읽기 및 브릭 파싱
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # 브릭 정보 수집 (ID 포함)
    bricks = []
    brick_counter = 0
    for i, line in enumerate(lines):
        parsed = parse_ldr_line(line)
        if parsed is None:
            continue
        
        brick_id = f"{parsed['part']}_{brick_counter}"
        parsed["brick_id"] = brick_id
        parsed["line_idx"] = i
        
        # 대상 브릭 필터링
        if target_brick_ids is None or brick_id in target_brick_ids:
            # 1x1 브릭만 병합 대상
            if parsed["part"] in SMALL_BRICK_PARTS:
                bricks.append(parsed)
        
        brick_counter += 1
    
    stats["original_count"] = len(bricks)
    logger.info(f"병합 대상 1x1 브릭 수: {len(bricks)}")
    
    if len(bricks) < min_merge_count:
        stats["new_count"] = len(bricks)
        return stats
    
    # 2. 동일 레이어(Y)와 색상 기준으로 그룹화
    # key = (y좌표, 색상) -> 브릭 리스트
    layer_color_groups = defaultdict(list)
    for brick in bricks:
        key = (brick["y"], brick["color"])
        layer_color_groups[key].append(brick)
    
    # 3. 각 그룹에서 X 또는 Z 방향으로 인접한 브릭 찾기
    merged_line_indices = set()  # 삭제할 라인 인덱스
    new_bricks = []  # 새로 생성할 브릭 라인
    
    for (y, color), group_bricks in layer_color_groups.items():
        if len(group_bricks) < min_merge_count:
            continue
        
        # X 방향 병합 시도 (같은 Z에서)
        z_groups = defaultdict(list)
        for b in group_bricks:
            z_groups[b["z"]].append(b)
        
        for z, z_bricks in z_groups.items():
            if len(z_bricks) < min_merge_count:
                continue
            
            # X 좌표 기준 정렬
            z_bricks.sort(key=lambda b: b["x"])
            
            # 연속된 브릭 그룹 찾기
            i = 0
            while i < len(z_bricks):
                # 연속 시퀀스 시작
                sequence = [z_bricks[i]]
                
                j = i + 1
                while j < len(z_bricks):
                    # 인접 여부 확인 (X 방향으로 STUD_SPACING 간격)
                    prev_x = sequence[-1]["x"]
                    curr_x = z_bricks[j]["x"]
                    
                    if abs(curr_x - prev_x - STUD_SPACING) < 0.1:
                        sequence.append(z_bricks[j])
                        j += 1
                    else:
                        break
                
                # 병합 가능한 시퀀스 처리
                seq_len = len(sequence)
                if seq_len >= min_merge_count and seq_len in MERGE_TARGET_BRICKS:
                    # 시퀀스의 첫 번째 브릭 위치를 기준으로 새 브릭 생성
                    first_brick = sequence[0]
                    new_part = MERGE_TARGET_BRICKS[seq_len]
                    
                    # 새 브릭 라인 생성 (첫 번째 브릭 위치 사용)
                    new_line = build_ldr_line(
                        color,
                        first_brick["x"],
                        first_brick["y"],
                        first_brick["z"],
                        first_brick["matrix"],
                        new_part
                    )
                    new_bricks.append(new_line + "\n")
                    
                    # 원본 브릭들 삭제 표시
                    for b in sequence:
                        merged_line_indices.add(b["line_idx"])
                    
                    stats["merged"] += 1
                    logger.debug(f"병합 완료: {seq_len}개 1x1 -> 1x{seq_len} (색상: {color})")
                
                i = j
    
    # 4. 파일 업데이트
    if merged_line_indices:
        # 삭제할 라인 제외하고 새 라인 추가
        new_lines = []
        for i, line in enumerate(lines):
            if i not in merged_line_indices:
                new_lines.append(line)
        
        # 새 브릭 추가
        new_lines.extend(new_bricks)
        
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        
        # 최종 브릭 수 계산
        stats["new_count"] = stats["original_count"] - len(merged_line_indices) + len(new_bricks)
        
        logger.info(f"브릭 병합 완료: {stats['merged']}개 그룹 병합됨")
        logger.info(f"원본 1x1 브릭: {len(merged_line_indices)}개 -> 새 브릭: {len(new_bricks)}개")
    else:
        stats["new_count"] = stats["original_count"]
        logger.info("병합 가능한 브릭 그룹이 없습니다.")
    
    return stats


# ============================================================================
# 구조적 병합 (Structural Merge)
# 불안정 브릭과 안정 브릭의 경계를 분해 후 재병합하여 구조적 연결 강화
# X+Z 양방향, 색상 무관 병합 지원
# ============================================================================

# 브릭 파트 → 스터드 수 역매핑 (분해용)
BRICK_STUD_COUNT = {
    "3005.dat": 1, "3024.dat": 1,   # 1x1
    "3004.dat": 2,                   # 1x2
    "3622.dat": 3,                   # 1x3
    "3010.dat": 4,                   # 1x4
    "3009.dat": 6,                   # 1x6
    "3008.dat": 8,                   # 1x8
}


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
    # 회전 행렬 a(matrix[0])와 g(matrix[6])으로 길이 방향 판별
    # a ≈ ±1 → 길이 방향 X, g ≈ ±1 → 길이 방향 Z
    a = matrix[0]  # 첫 번째 행의 X 요소
    g = matrix[6]  # 세 번째 행의 X 요소

    positions = []
    for i in range(stud_count):
        if abs(a) > 0.5:
            # 길이 방향이 X
            dx = i * STUD_SPACING * (1 if a > 0 else -1)
            positions.append((brick["x"] + dx, brick["y"], brick["z"]))
        elif abs(g) > 0.5:
            # 길이 방향이 Z
            dz = i * STUD_SPACING * (1 if g > 0 else -1)
            positions.append((brick["x"], brick["y"], brick["z"] + dz))
        else:
            # 판별 불가 → 1x1로 간주
            positions.append((brick["x"], brick["y"], brick["z"]))
            break

    return positions


def _split_brick_to_1x1(brick: dict) -> list:
    """
    큰 브릭(1x2~1x8)을 1x1 브릭 여러 개로 분해합니다.
    원본 브릭의 색상과 Y좌표를 유지합니다.
    """
    positions = _get_brick_stud_positions(brick)
    if len(positions) <= 1:
        return [brick]  # 이미 1x1이면 그대로 반환

    result_bricks = []
    for x, y, z in positions:
        new_brick = {
            "type": 1,
            "color": brick["color"],
            "x": x,
            "y": y,
            "z": z,
            "matrix": [1, 0, 0, 0, 1, 0, 0, 0, 1],  # 1x1은 회전 불필요
            "part": "3005.dat",  # 1x1 브릭
        }
        result_bricks.append(new_brick)

    return result_bricks


def _merge_all_1x1(bricks: list, min_merge_count: int = 2) -> tuple:
    """
    1x1 브릭들을 X방향과 Z방향 양쪽으로 병합합니다.
    색상에 상관없이 병합하며, 첫 번째 브릭의 색상을 사용합니다.

    Returns:
        (new_brick_lines: list, merged_indices: set)
        - new_brick_lines: 병합된 큰 브릭의 LDR 라인 문자열 리스트
        - merged_indices: 병합되어 삭제될 브릭의 인덱스 set
    """
    new_brick_lines = []
    merged_indices = set()
    merge_count = 0

    # Y 레이어별 그룹화 (색상 무관)
    layer_groups = defaultdict(list)
    for idx, brick in enumerate(bricks):
        if brick["part"] not in SMALL_BRICK_PARTS:
            continue
        layer_groups[brick["y"]].append((idx, brick))

    for y, group in layer_groups.items():
        if len(group) < min_merge_count:
            continue

        already_merged = set()

        # --- X 방향 병합 (같은 Z에서) ---
        z_groups = defaultdict(list)
        for idx, brick in group:
            z_groups[brick["z"]].append((idx, brick))

        for z, z_items in z_groups.items():
            if len(z_items) < min_merge_count:
                continue

            # X 좌표 정렬
            z_items.sort(key=lambda item: item[1]["x"])

            i = 0
            while i < len(z_items):
                idx_i, brick_i = z_items[i]
                if idx_i in already_merged:
                    i += 1
                    continue

                # 연속 시퀀스 탐색
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
                # 가장 큰 병합부터 시도 (8→6→4→3→2)
                merged_any = False
                while seq_len >= min_merge_count:
                    if seq_len in MERGE_TARGET_BRICKS:
                        first_brick = sequence[0][1]
                        new_line = build_ldr_line(
                            first_brick["color"],
                            first_brick["x"], first_brick["y"], first_brick["z"],
                            [1, 0, 0, 0, 1, 0, 0, 0, 1],  # X 방향 기본 행렬
                            MERGE_TARGET_BRICKS[seq_len]
                        )
                        new_brick_lines.append(new_line + "\n")
                        for idx_s, _ in sequence[:seq_len]:
                            already_merged.add(idx_s)
                            merged_indices.add(idx_s)
                        merge_count += 1
                        merged_any = True
                        break
                    seq_len -= 1

                if not merged_any:
                    i = j if j > i + 1 else i + 1
                else:
                    # 병합 후 남은 브릭에서 계속
                    remaining = sequence[seq_len:]
                    if remaining:
                        i = z_items.index(remaining[0]) if remaining[0] in z_items else j
                    else:
                        i = j

        # --- Z 방향 병합 (같은 X에서, 아직 병합 안 된 것만) ---
        x_groups = defaultdict(list)
        for idx, brick in group:
            if idx in already_merged:
                continue
            x_groups[brick["x"]].append((idx, brick))

        for x, x_items in x_groups.items():
            if len(x_items) < min_merge_count:
                continue

            # Z 좌표 정렬
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
                merged_any = False
                while seq_len >= min_merge_count:
                    if seq_len in MERGE_TARGET_BRICKS:
                        first_brick = sequence[0][1]
                        # Z 방향 → 90도 회전 행렬 적용
                        new_line = build_ldr_line(
                            first_brick["color"],
                            first_brick["x"], first_brick["y"], first_brick["z"],
                            [0, 0, -1, 0, 1, 0, 1, 0, 0],  # Z 방향 90도 회전
                            MERGE_TARGET_BRICKS[seq_len]
                        )
                        new_brick_lines.append(new_line + "\n")
                        for idx_s, _ in sequence[:seq_len]:
                            already_merged.add(idx_s)
                            merged_indices.add(idx_s)
                        merge_count += 1
                        merged_any = True
                        break
                    seq_len -= 1

                if not merged_any:
                    i = j if j > i + 1 else i + 1
                else:
                    remaining = sequence[seq_len:]
                    if remaining:
                        i = x_items.index(remaining[0]) if remaining[0] in x_items else j
                    else:
                        i = j

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
    
    # 공간 해싱 (좌표 -> 브릭)
    pos_to_brick_idx = {}
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
                        # 수평으로 인접한 안정 브릭 발견!
                        stable_boundary_indices.add(n_idx)

    # 분해 대상 최종 확정
    for b in all_bricks:
        bid = b["brick_idx"]
        if bid in unstable_set or bid in stable_boundary_indices:
            indices_to_split.add(bid)

    # 3. 분해 실행 (1xN → N × 1x1)
    lines_to_delete = set()
    new_1x1_bricks = []
    split_count = 0

    for brick in all_bricks:
        if brick["brick_idx"] not in indices_to_split:
            continue
        
        # 이미 1x1이면 분해 불필요 (하지만 재병합 대상에는 포함시켜야 함)
        if brick["part"] in SMALL_BRICK_PARTS:
            # 1x1도 삭제 리스트에 넣어서, 나중에 일괄 재병합되게 함
            # 단, _split_brick_to_1x1은 1x1을 그대로 반환함.
            pass
        
        # 분해
        split_bricks = _split_brick_to_1x1(brick)
        if split_bricks:
            lines_to_delete.add(brick["line_idx"])
            new_1x1_bricks.extend(split_bricks)
            if len(split_bricks) > 1:
                split_count += 1

    if not lines_to_delete and not new_1x1_bricks:
        return {"merged": 0, "split": 0, "rounds": 0}

    # 4. 재병합 (X+Z 양방향, 색상 무관)
    merged_new_lines, merged_indices, merge_count = _merge_all_1x1(new_1x1_bricks)
    
    # 5. 파일 업데이트
    # 기존 라인 중 삭제되지 않은 것 + 병합된 새 라인 + (병합 안된 1x1들)
    # _merge_all_1x1은 병합된 결과 라인과, 병합에 사용된 인덱스를 반환함.
    # 병합 안 된 1x1은 new_1x1_bricks에 그대로 남아있음(인덱스로 구분).
    
    final_lines = []
    
    # (1) 보존된 기존 브릭 (분해 안 된 안정 브릭들)
    for i, line in enumerate(lines):
        if i not in lines_to_delete:
            final_lines.append(line)
            
    # (2) 병합된 큰 브릭들
    final_lines.extend(merged_new_lines)
    
    # (3) 병합되지 못한 1x1 찌꺼기들
    for idx, b in enumerate(new_1x1_bricks):
        if idx not in merged_indices:
            # 다시 LDR 라인으로 변환
            line = build_ldr_line(
                b["color"], b["x"], b["y"], b["z"], b["matrix"], b["part"]
            )
            final_lines.append(line + "\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(final_lines)

    logger.info(f"구조적 병합(1회) 완료: 분해 {split_count}개(안정 포함), 병합 {merge_count}개 그룹")
    return {"merged": merge_count, "split": split_count, "rounds": 1}

