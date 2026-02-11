"""
LDR Converter - LDR 파일 파싱

LDR → BrickModel 변환, 색상 변경
"""

from typing import List, Dict, Optional

from .models import Vector3, PlacedBrick, BrickModel


def matrix_to_rotation(matrix: List[float]) -> int:
    """
    회전 행렬을 각도(0, 90, 180, 270)로 변환

    가장 가까운 표준 회전 각도로 근사
    """
    # 표준 회전 행렬들
    standard_matrices = {
        0:   [1, 0, 0, 0, 1, 0, 0, 0, 1],
        90:  [0, 0, -1, 0, 1, 0, 1, 0, 0],
        180: [-1, 0, 0, 0, 1, 0, 0, 0, -1],
        270: [0, 0, 1, 0, 1, 0, -1, 0, 0],
    }

    # 가장 가까운 행렬 찾기
    best_match = 0
    best_diff = float('inf')

    for angle, std_matrix in standard_matrices.items():
        diff = sum(abs(a - b) for a, b in zip(matrix, std_matrix))
        if diff < best_diff:
            best_diff = diff
            best_match = angle

    return best_match


def parse_ldr_line(line: str) -> Optional[dict]:
    """
    LDR Line Type 1 파싱

    형식: 1 <색상> <x> <y> <z> <a> <b> <c> <d> <e> <f> <g> <h> <i> <파일명>

    Returns:
        파싱된 정보 딕셔너리 또는 None
    """
    line = line.strip()
    if not line or line.startswith('0'):
        return None

    parts = line.split()
    if len(parts) < 15 or parts[0] != '1':
        return None

    try:
        color = int(parts[1])
        x = float(parts[2])
        y = float(parts[3])
        z = float(parts[4])

        # 회전 행렬 (9개 값)
        matrix = [float(parts[i]) for i in range(5, 14)]

        # 파츠 파일명
        part_file = parts[14]

        # part_id 추출 (확장자 제거)
        part_id = part_file.replace('.dat', '').replace('.DAT', '')

        return {
            'color': color,
            'x': x,
            'y': y,
            'z': z,
            'matrix': matrix,
            'rotation': matrix_to_rotation(matrix),
            'part_file': part_file,
            'part_id': part_id,
        }
    except (ValueError, IndexError):
        return None


def parse_ldr_file(filepath: str) -> dict:
    """
    LDR 파일을 파싱하여 딕셔너리로 반환

    Returns:
        {
            'name': 모델 이름,
            'bricks': [파싱된 브릭 정보 리스트],
            'comments': [주석 리스트],
        }
    """
    result = {
        'name': '',
        'bricks': [],
        'comments': [],
    }

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()

            # 주석 (Line Type 0)
            if line.startswith('0 '):
                comment = line[2:]
                result['comments'].append(comment)
                # 첫 번째 주석을 이름으로
                if not result['name'] and not comment.startswith('!'):
                    result['name'] = comment

            # 파츠 (Line Type 1)
            elif line.startswith('1 '):
                parsed = parse_ldr_line(line)
                if parsed:
                    result['bricks'].append(parsed)

    return result


def ldr_to_brick_model(
    filepath: str,
    model_id: Optional[str] = None,
    mode: str = 'pro'
) -> BrickModel:
    """
    LDR 파일을 BrickModel로 변환

    Args:
        filepath: LDR 파일 경로
        model_id: 모델 ID (없으면 파일명 사용)
        mode: 'pro' 또는 'kids'

    Returns:
        BrickModel 객체
    """
    from pathlib import Path

    parsed = parse_ldr_file(filepath)

    if not model_id:
        model_id = Path(filepath).stem

    bricks = []
    for i, b in enumerate(parsed['bricks']):
        brick = PlacedBrick(
            id=f"b{i+1:03d}",
            part_id=b['part_id'].lower(),
            position=Vector3(x=b['x'], y=b['y'], z=b['z']),
            rotation=b['rotation'],
            color_code=b['color'],
            layer=0  # LDR에서는 레이어 정보 없음, 나중에 Y좌표로 계산 가능
        )
        bricks.append(brick)

    # Y좌표로 레이어 자동 계산
    if bricks:
        # Y좌표 기준 정렬 (LDraw: 큰 Y가 바닥/하단)
        # bottom-up 조립을 위해 Y가 큰 값(바닥)부터 작은 값(상단) 순서로 레이어 번호 부여
        y_values = sorted(set(b.position.y for b in bricks), reverse=True)
        y_to_layer = {y: i for i, y in enumerate(y_values)}
        for brick in bricks:
            brick.layer = y_to_layer[brick.position.y]

    return BrickModel(
        model_id=model_id,
        name=parsed['name'] or model_id,
        mode=mode,
        bricks=bricks
    )


def change_colors(model: BrickModel, color_map: Dict[int, int]) -> BrickModel:
    """
    BrickModel의 색상 변경

    Args:
        model: 원본 모델
        color_map: {원본색상: 새색상} 매핑

    Returns:
        색상이 변경된 새 BrickModel
    """
    new_bricks = []
    for brick in model.bricks:
        new_color = color_map.get(brick.color_code, brick.color_code)
        new_brick = PlacedBrick(
            id=brick.id,
            part_id=brick.part_id,
            position=brick.position,
            rotation=brick.rotation,
            color_code=new_color,
            layer=brick.layer
        )
        new_bricks.append(new_brick)

    return BrickModel(
        model_id=model.model_id,
        name=model.name,
        mode=model.mode,
        bricks=new_bricks,
        target_age=model.target_age,
        created_at=model.created_at
    )
