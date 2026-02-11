"""
LDR Converter - LDR 파일 출력

BrickModel → LDR 포맷 변환
"""

from datetime import datetime
from typing import Dict

from .models import PlacedBrick, BrickModel
from .rotation import get_rotation_matrix
from .validation import ValidationError, validate_model, validate_physics


# STEP 모드 옵션
STEP_MODE_NONE = 'none'    # STEP 없음
STEP_MODE_LAYER = 'layer'  # 레이어마다 STEP
STEP_MODE_BRICK = 'brick'  # 브릭마다 STEP
VALID_STEP_MODES = [STEP_MODE_NONE, STEP_MODE_LAYER, STEP_MODE_BRICK]


def brick_to_ldr_line(brick: PlacedBrick, parts_db: Dict) -> str:
    """
    PlacedBrick을 LDR Line Type 1 형식으로 변환

    형식: 1 <색상> <x> <y> <z> <a> <b> <c> <d> <e> <f> <g> <h> <i> <파일명>
    """
    # 파츠 정보 조회
    part_info = parts_db.get(brick.part_id)
    if not part_info:
        raise ValueError(f"Unknown part ID: {brick.part_id}")

    ldraw_file = part_info['ldrawFile']

    # 회전 행렬
    matrix = get_rotation_matrix(brick.rotation)

    # LDR 라인 생성
    # 좌표는 이미 LDU 단위로 들어온다고 가정
    line = f"1 {brick.color_code} {brick.position.x:.0f} {brick.position.y:.0f} {brick.position.z:.0f} "
    line += " ".join(str(m) for m in matrix)
    line += f" {ldraw_file}"

    return line


def model_to_ldr(
    model: BrickModel,
    parts_db: Dict,
    skip_validation: bool = False,
    skip_physics: bool = False,
    step_mode: str = STEP_MODE_NONE
) -> str:
    """
    BrickModel 전체를 LDR 파일 내용으로 변환

    Args:
        model: 변환할 브릭 모델
        parts_db: 파츠 데이터베이스 (bbox 정보 포함 권장)
        skip_validation: True면 L1 검증 스킵 (디버깅용, 기본값 False)
        skip_physics: True면 L2 물리 검증 스킵 (기본값 False)
        step_mode: STEP 출력 모드
            - 'none': STEP 없음 (기본값)
            - 'layer': 레이어마다 0 STEP 추가
            - 'brick': 브릭마다 0 STEP 추가 (조립 설명서용)

    Raises:
        ValidationError: 검증 실패 시 (skip_validation=False일 때)
        ValueError: 잘못된 step_mode
    """
    # step_mode 검증
    if step_mode not in VALID_STEP_MODES:
        raise ValueError(f"Invalid step_mode: '{step_mode}', must be one of {VALID_STEP_MODES}")

    # L1 검증 (스킵 옵션 없으면 항상 실행)
    if not skip_validation:
        errors = validate_model(model, parts_db)
        if errors:
            raise ValidationError(errors)

    # L2 물리 검증 (충돌, 부유) - 경고만 출력, 에러 아님
    l2_warnings = []
    if not skip_physics:
        physics_result = validate_physics(model, parts_db)
        if physics_result.collisions:
            l2_warnings.append(f"[L2 Warning] {len(physics_result.collisions)}개 충돌 발견")
            for c in physics_result.collisions[:3]:
                l2_warnings.append(f"  - {c[0]} <-> {c[1]}")
            if len(physics_result.collisions) > 3:
                l2_warnings.append(f"  ... 외 {len(physics_result.collisions) - 3}개")
        if physics_result.floating_bricks:
            l2_warnings.append(f"[L2 Warning] {len(physics_result.floating_bricks)}개 부유 브릭 발견")
            for f in physics_result.floating_bricks[:3]:
                l2_warnings.append(f"  - {f}")
            if len(physics_result.floating_bricks) > 3:
                l2_warnings.append(f"  ... 외 {len(physics_result.floating_bricks) - 3}개")
        # 경고 출력 (에러 아님)
        for w in l2_warnings:
            print(w)

    lines = []

    # 헤더 주석
    lines.append(f"0 {model.name}")
    lines.append(f"0 Name: {model.model_id}.ldr")
    lines.append(f"0 Author: Brick CoScientist")
    lines.append(f"0 Mode: {model.mode}")
    if model.target_age:
        lines.append(f"0 TargetAge: {model.target_age}")
    lines.append(f"0 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")  # 빈 줄

    # 레이어별로 정렬 (아래부터 위로)
    sorted_bricks = sorted(model.bricks, key=lambda b: b.layer)

    current_layer = -1
    for i, brick in enumerate(sorted_bricks):
        # 레이어 변경 시
        if brick.layer != current_layer:
            # 이전 레이어 끝에 STEP 추가 (layer 모드, 첫 레이어 제외)
            if step_mode == STEP_MODE_LAYER and current_layer != -1:
                lines.append("")
                lines.append("0 STEP")
                lines.append("")

            current_layer = brick.layer
            lines.append(f"0 // Layer {current_layer}")

        # 브릭 라인 추가
        try:
            ldr_line = brick_to_ldr_line(brick, parts_db)
            lines.append(ldr_line)

            # brick 모드: 매 브릭마다 STEP (마지막 브릭 제외)
            if step_mode == STEP_MODE_BRICK and i < len(sorted_bricks) - 1:
                lines.append("")
                lines.append("0 STEP")
                lines.append("")
        except ValueError as e:
            l2_warnings.append(f"[WARNING] Skipped brick: {e}")
            continue

    # 마지막 STEP 추가 (layer 모드)
    if step_mode == STEP_MODE_LAYER and len(sorted_bricks) > 0:
        lines.append("")
        lines.append("0 STEP")

    return "\n".join(lines)


def model_to_ldr_unsafe(model: BrickModel, parts_db: Dict) -> str:
    """
    검증 없이 LDR 변환 (디버깅/테스트용)

    주의: 프로덕션에서 사용 금지. 잘못된 데이터도 그대로 변환됨.
    """
    return model_to_ldr(model, parts_db, skip_validation=True)


def save_ldr_file(content: str, filepath: str):
    """LDR 파일 저장"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"LDR 파일 저장 완료: {filepath}")
