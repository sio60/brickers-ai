# ============================================================================
# LDR 파일 수정 모듈 (개선판)
# LLM의 결정에 따라 LDR 파일에서 브릭을 이동하거나 삭제
# 인덱스 밀림 및 ID 불일치 문제를 해결함 (LdrLoader의 글로벌 인덱스 방식 채택)
# ============================================================================

import re
import os
import logging
from typing import Tuple, Optional, List, Dict, Any
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
    new_lines = [l for l in lines if l is not None]
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
        final_brick_count = sum(1 for l in new_lines if l.strip().startswith("1 "))
        stats["new_count"] = stats["original_count"] - len(merged_line_indices) + len(new_bricks)
        
        logger.info(f"브릭 병합 완료: {stats['merged']}개 그룹 병합됨")
        logger.info(f"원본 1x1 브릭: {len(merged_line_indices)}개 -> 새 브릭: {len(new_bricks)}개")
    else:
        stats["new_count"] = stats["original_count"]
        logger.info("병합 가능한 브릭 그룹이 없습니다.")
    
    return stats

