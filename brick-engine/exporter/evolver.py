"""
Evolver - 취약 구간 자동 보강 모듈 v3

고도화:
1. 지지대 다양화 - 브릭 크기에 맞는 지지대 선택
2. 브릭 위주 사용 (파츠 수 최소화)
3. 다중 지지점 - 모서리/양끝에 분산 배치 (안정성 ↑)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from copy import deepcopy


@dataclass
class EvolveResult:
    """Evolver 결과"""
    original_model: any
    evolved_model: any
    changes: List[str]
    added_bricks: int
    removed_bricks: int
    moved_bricks: int


# 지지대용 파츠 정의
SUPPORT_PARTS = {
    # part_id: (width_studs, depth_studs, height_ldu, name)
    "3001": (4, 2, 24, "Brick 2x4"),
    "3003": (2, 2, 24, "Brick 2x2"),
    "3004": (4, 1, 24, "Brick 1x4"),
    "3005": (1, 1, 24, "Brick 1x1"),
    "3010": (4, 1, 24, "Brick 1x4"),
    "3020": (4, 2, 8, "Plate 2x4"),
    "3022": (2, 2, 8, "Plate 2x2"),
    "3023": (2, 1, 8, "Plate 1x2"),
    "3024": (1, 1, 8, "Plate 1x1"),
}

# 브릭 크기별 추천 지지대 (큰 것부터)
SUPPORT_PRIORITY_BRICK = ["3001", "3003", "3004", "3005"]  # 브릭류 (높이 24)
SUPPORT_PRIORITY_PLATE = ["3020", "3022", "3023", "3024"]  # 플레이트류 (높이 8)


class Evolver:
    """취약 구간 자동 보강 v2"""

    BRICK_HEIGHT = 24
    PLATE_HEIGHT = 8
    SUPPORT_COLOR = 71  # Light Bluish Gray (더 자연스러움)

    def __init__(self, parts_db: Dict):
        self.parts_db = parts_db
        self.changes = []
        self.brick_counter = 1000

    def evolve(self, model, validation_result) -> EvolveResult:
        """모델 자동 보강"""
        self.changes = []
        evolved = deepcopy(model)

        added = 0
        removed = 0
        moved = 0

        # 1. 충돌 먼저 해결
        if validation_result.collisions:
            m, r = self._fix_collisions(evolved, validation_result.collisions)
            moved += m
            removed += r

        # 2. 부유 브릭 처리
        if validation_result.floating_bricks:
            added += self._fix_floating(evolved, validation_result.floating_bricks)

        return EvolveResult(
            original_model=model,
            evolved_model=evolved,
            changes=self.changes,
            added_bricks=added,
            removed_bricks=removed,
            moved_bricks=moved
        )

    def _get_brick_size(self, brick) -> Tuple[int, int]:
        """브릭의 스터드 크기 반환 (width, depth)"""
        part_id = brick.part_id

        # 알려진 파츠 크기
        PART_SIZES = {
            "3001": (4, 2), "3002": (3, 2), "3003": (2, 2), "3004": (4, 1),
            "3005": (1, 1), "3010": (4, 1), "3009": (3, 1), "3008": (2, 1),
            "3020": (4, 2), "3021": (3, 2), "3022": (2, 2), "3023": (2, 1),
            "3024": (1, 1), "3710": (4, 1), "3666": (6, 1),
        }

        return PART_SIZES.get(part_id, (2, 2))  # 기본값 2x2

    def _choose_support_part(self, brick_width: int, brick_depth: int,
                             remaining_height: int) -> Tuple[str, int]:
        """
        브릭 크기와 남은 높이에 맞는 지지대 파츠 선택

        Returns:
            (part_id, height)
        """
        # 남은 높이가 브릭 높이(24) 이상이면 브릭 사용
        if remaining_height >= self.BRICK_HEIGHT:
            # 브릭 크기에 맞는 지지대 선택
            if brick_width >= 4 and brick_depth >= 2:
                return ("3001", 24)  # 2x4 브릭
            elif brick_width >= 2 and brick_depth >= 2:
                return ("3003", 24)  # 2x2 브릭
            elif brick_width >= 4:
                return ("3004", 24)  # 1x4 브릭
            else:
                return ("3005", 24)  # 1x1 브릭

        # 남은 높이가 플레이트 높이(8) 이상이면 플레이트 사용
        elif remaining_height >= self.PLATE_HEIGHT:
            if brick_width >= 4 and brick_depth >= 2:
                return ("3020", 8)  # 2x4 플레이트
            elif brick_width >= 2 and brick_depth >= 2:
                return ("3022", 8)  # 2x2 플레이트
            elif brick_width >= 2:
                return ("3023", 8)  # 1x2 플레이트
            else:
                return ("3024", 8)  # 1x1 플레이트

        # 아주 작은 높이면 1x1 플레이트
        return ("3024", 8)

    def _get_support_positions(self, brick_x: float, brick_z: float,
                                brick_width: int, brick_depth: int) -> List[Tuple[float, float]]:
        """
        브릭 크기에 따른 지지점 위치 계산 (다중 지지점)

        Returns:
            [(x, z), ...] 지지대 배치 위치 목록
        """
        LDU_PER_STUD = 20

        # 브릭 크기 (LDU)
        width_ldu = brick_width * LDU_PER_STUD
        depth_ldu = brick_depth * LDU_PER_STUD

        positions = []

        if brick_width >= 4 and brick_depth >= 2:
            # 2x4 이상: 양쪽 끝에 2개
            offset_x = (width_ldu / 2) - LDU_PER_STUD  # 끝에서 1스터드 안쪽
            positions = [
                (brick_x - offset_x, brick_z),
                (brick_x + offset_x, brick_z),
            ]
        elif brick_width >= 3 and brick_depth >= 2:
            # 2x3: 양쪽 끝에 2개
            offset_x = LDU_PER_STUD
            positions = [
                (brick_x - offset_x, brick_z),
                (brick_x + offset_x, brick_z),
            ]
        elif brick_width >= 2 and brick_depth >= 2:
            # 2x2: 중앙 1개로 충분
            positions = [(brick_x, brick_z)]
        elif brick_width >= 4:
            # 1x4: 양쪽 끝에 2개
            offset_x = (width_ldu / 2) - LDU_PER_STUD
            positions = [
                (brick_x - offset_x, brick_z),
                (brick_x + offset_x, brick_z),
            ]
        elif brick_width >= 2:
            # 1x2: 중앙 1개
            positions = [(brick_x, brick_z)]
        else:
            # 1x1: 중앙 1개
            positions = [(brick_x, brick_z)]

        return positions

    def _fix_floating(self, model, floating_brick_ids: List[str]) -> int:
        """부유 브릭 아래에 다중 지지대 추가"""
        from ldr_converter import PlacedBrick, Vector3

        added = 0
        bricks_by_id = {b.id: b for b in model.bricks}

        for brick_id in floating_brick_ids:
            if brick_id not in bricks_by_id:
                continue

            brick = bricks_by_id[brick_id]
            brick_width, brick_depth = self._get_brick_size(brick)

            # 브릭 종류에 따른 실제 높이
            if brick.part_id in ["3024", "3023", "3022", "3020", "3021", "3710", "3666"]:
                actual_height = 8
            else:
                actual_height = 24

            # 브릭 바닥 Y
            brick_bottom_y = brick.position.y + actual_height

            # 이미 바닥에 닿아있으면 스킵
            if brick_bottom_y >= 0:
                continue

            # 다중 지지점 계산
            support_positions = self._get_support_positions(
                brick.position.x, brick.position.z,
                brick_width, brick_depth
            )

            # 각 지지점에 기둥 추가
            total_supports = 0
            for sx, sz in support_positions:
                supports_added = self._add_smart_support_column(
                    model, sx, sz, brick_bottom_y,
                    1, 1  # 지지대는 1x1 기준으로 선택 (여러 개 분산이므로)
                )
                total_supports += supports_added

            if total_supports > 0:
                self.changes.append(
                    f"브릭 {brick_id}: {len(support_positions)}곳에 지지대 총 {total_supports}개"
                )
                added += total_supports

        return added

    def _add_smart_support_column(self, model, x: float, z: float,
                                   brick_bottom_y: float,
                                   brick_width: int, brick_depth: int) -> int:
        """스마트 지지대 기둥 추가 - 브릭 크기에 맞춰 파츠 선택"""
        from ldr_converter import PlacedBrick, Vector3

        added = 0
        current_y = int(brick_bottom_y)
        max_supports = 20

        while current_y < 0 and added < max_supports:
            remaining_height = 0 - current_y  # 지면까지 남은 높이

            # 적절한 지지대 파츠 선택
            part_id, part_height = self._choose_support_part(
                brick_width, brick_depth, remaining_height
            )

            self.brick_counter += 1
            new_support = PlacedBrick(
                id=f"support_{self.brick_counter}",
                part_id=part_id,
                position=Vector3(x=x, y=current_y, z=z),
                rotation=0,
                color_code=self.SUPPORT_COLOR,
                layer=999
            )
            model.bricks.append(new_support)
            added += 1

            current_y += part_height

        return added

    def _fix_collisions(self, model, collisions: List[Tuple[str, str]]) -> Tuple[int, int]:
        """충돌 브릭 해결 - X축 이동"""
        from ldr_converter import get_brick_bbox

        moved = 0
        removed = 0
        bricks_by_id = {b.id: b for b in model.bricks}
        moved_ids = set()

        for brick1_id, brick2_id in collisions:
            if brick1_id in moved_ids or brick2_id in moved_ids:
                continue

            if brick1_id not in bricks_by_id or brick2_id not in bricks_by_id:
                continue

            brick1 = bricks_by_id[brick1_id]
            brick2 = bricks_by_id[brick2_id]

            bbox1 = get_brick_bbox(brick1, self.parts_db)
            bbox2 = get_brick_bbox(brick2, self.parts_db)

            if not bbox1 or not bbox2:
                continue

            # 후순위 브릭 이동
            if brick1_id > brick2_id:
                target, target_id, target_bbox = brick1, brick1_id, bbox1
                other_bbox = bbox2
            else:
                target, target_id, target_bbox = brick2, brick2_id, bbox2
                other_bbox = bbox1

            target_width = target_bbox.max_x - target_bbox.min_x
            old_x = target.position.x

            if target.position.x >= other_bbox.min_x:
                new_x = other_bbox.max_x + (target_width / 2) + 5
            else:
                new_x = other_bbox.min_x - (target_width / 2) - 5

            target.position.x = new_x
            moved += 1
            moved_ids.add(target_id)

            self.changes.append(f"브릭 {target_id} X이동: {old_x:.0f} -> {new_x:.0f}")

        return moved, removed


def is_valid(validation_result) -> bool:
    """L2ValidationResult가 유효한지 확인"""
    return (len(validation_result.collisions) == 0 and
            len(validation_result.floating_bricks) == 0)


def evolve_model(model, validation_result, parts_db: Dict) -> EvolveResult:
    """모델 자동 보강"""
    evolver = Evolver(parts_db)
    return evolver.evolve(model, validation_result)


def print_evolve_result(result: EvolveResult):
    """결과 출력"""
    print("=" * 50)
    print("Evolver 결과")
    print("=" * 50)
    print(f"추가: {result.added_bricks}개 / 삭제: {result.removed_bricks}개 / 이동: {result.moved_bricks}개")
    print("[변경 내역]")
    for change in result.changes:
        print(f"  - {change}")
    print("=" * 50)
