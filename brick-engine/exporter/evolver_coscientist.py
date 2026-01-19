"""
CoScientist Evolver v2 - LLM 기반 지능형 취약 구간 보강

CoScientist 철학:
- AI가 모형의 의도를 이해
- 모형 유형에 맞는 보강 전략 결정
- "왜 이렇게 바꿨는지" 설명 제공

v2 개선사항:
- evolver.py v3 로직 통합 (다중 지지점, 지지대 다양화)
- 모형 유형별 지지대 스타일/색상 커스터마이징
- 정확한 좌표 계산 (actual height 사용)
"""

import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from copy import deepcopy

# Ollama 선택적 import
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Warning: ollama not installed. LLM analysis will use fallback.")


@dataclass
class SupportStrategy:
    """지지대 전략"""
    model_type: str          # spaceship, car, building, animal, robot, other
    support_style: str       # landing_gear, wheels, pillars, legs, generic
    support_color: int       # LDraw 색상 코드
    preferred_parts: List[str]  # 우선 사용할 파츠 ID
    explanation: str         # 왜 이 전략인지


@dataclass
class CoScientistEvolveResult:
    """CoScientist Evolver 결과"""
    original_model: any
    evolved_model: any
    strategy: SupportStrategy
    changes: List[str]
    added_bricks: int
    removed_bricks: int
    moved_bricks: int
    explanation: str  # 사용자에게 보여줄 설명


# 모형 유형별 기본 전략
DEFAULT_STRATEGIES = {
    'spaceship': SupportStrategy(
        model_type='spaceship',
        support_style='landing_gear',
        support_color=71,  # Light Bluish Gray
        preferred_parts=['3005', '3004', '3003'],  # 1x1, 1x4, 2x2 브릭
        explanation='우주선의 착륙기어처럼 날개/동체 아래에 지지대 배치'
    ),
    'car': SupportStrategy(
        model_type='car',
        support_style='wheels',
        support_color=0,  # Black
        preferred_parts=['3005', '3024', '3023'],  # 1x1 브릭/플레이트 위주 (낮은 높이)
        explanation='자동차 바퀴 위치에 맞춰 낮은 지지대 배치'
    ),
    'building': SupportStrategy(
        model_type='building',
        support_style='pillars',
        support_color=7,  # Light Gray
        preferred_parts=['3001', '3003', '3005'],  # 큰 브릭 위주
        explanation='건물 기초처럼 견고한 기둥 지지대 배치'
    ),
    'animal': SupportStrategy(
        model_type='animal',
        support_style='legs',
        support_color=6,  # Brown
        preferred_parts=['3005', '3004'],  # 얇은 브릭
        explanation='동물의 다리 위치에 맞춰 자연스러운 지지대 배치'
    ),
    'robot': SupportStrategy(
        model_type='robot',
        support_style='legs',
        support_color=71,  # Light Bluish Gray
        preferred_parts=['3005', '3003'],  # 1x1, 2x2
        explanation='로봇 다리/발 위치에 기계적인 지지대 배치'
    ),
    'other': SupportStrategy(
        model_type='other',
        support_style='generic',
        support_color=71,  # Light Bluish Gray
        preferred_parts=['3005', '3003', '3001'],  # 일반 브릭
        explanation='일반적인 안정성 확보를 위한 지지대 배치'
    )
}


# 지지대용 파츠 정의 (evolver.py v3에서 가져옴)
SUPPORT_PARTS = {
    # part_id: (width_studs, depth_studs, height_ldu, name)
    "3001": (4, 2, 24, "Brick 2x4"),
    "3003": (2, 2, 24, "Brick 2x2"),
    "3004": (4, 1, 24, "Brick 1x4"),
    "3005": (1, 1, 24, "Brick 1x1"),
    "3020": (4, 2, 8, "Plate 2x4"),
    "3022": (2, 2, 8, "Plate 2x2"),
    "3023": (2, 1, 8, "Plate 1x2"),
    "3024": (1, 1, 8, "Plate 1x1"),
}

BRICK_HEIGHT = 24
PLATE_HEIGHT = 8
LDU_PER_STUD = 20


class CoScientistEvolver:
    """LLM 기반 지능형 Evolver"""

    def __init__(self, parts_db: Dict, model_name: str = "mistral"):
        self.parts_db = parts_db
        self.model_name = model_name
        self.brick_counter = 2000
        self.changes = []

    def analyze_model(self, model) -> SupportStrategy:
        """
        LLM으로 모형 분석 -> 지지 전략 결정
        """
        # 모형 정보 수집
        model_info = self._extract_model_info(model)

        # LLM 사용 가능하면 분석 시도
        if OLLAMA_AVAILABLE:
            try:
                return self._llm_analyze(model, model_info)
            except Exception as e:
                print(f"LLM 분석 실패: {e}")

        # Fallback: 규칙 기반 분석
        return self._rule_based_analyze(model, model_info)

    def _llm_analyze(self, model, model_info: Dict) -> SupportStrategy:
        """LLM 기반 모형 분석"""
        prompt = f"""당신은 레고 모형 분석 전문가입니다.

다음 레고 모형을 분석하고 적절한 지지대 전략을 JSON으로 답해주세요:

모형 이름: {model.name}
브릭 수: {len(model.bricks)}
크기 (LDU): {model_info['dimensions']}
특징: {model_info['features']}
부유 브릭 위치: {model_info.get('floating_positions', '정보 없음')}

다음 JSON 형식으로만 답하세요:
{{
    "model_type": "spaceship|car|building|animal|robot|other",
    "reasoning": "이 유형으로 판단한 이유"
}}

JSON만 출력하세요, 다른 텍스트 없이."""

        response = ollama.chat(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}]
        )

        content = response['message']['content']

        # JSON 추출 (```json ... ``` 형식 처리)
        if '```' in content:
            content = content.split('```')[1]
            if content.startswith('json'):
                content = content[4:]

        result = json.loads(content.strip())
        model_type = result.get('model_type', 'other')

        # 기본 전략에서 가져와서 LLM 설명 추가
        strategy = DEFAULT_STRATEGIES.get(model_type, DEFAULT_STRATEGIES['other'])

        return SupportStrategy(
            model_type=strategy.model_type,
            support_style=strategy.support_style,
            support_color=strategy.support_color,
            preferred_parts=strategy.preferred_parts,
            explanation=f"{strategy.explanation} (AI 분석: {result.get('reasoning', '')})"
        )

    def _rule_based_analyze(self, model, model_info: Dict) -> SupportStrategy:
        """규칙 기반 모형 분석 (LLM 없을 때 fallback)"""
        name_lower = model.name.lower() if model.name else ""

        # 이름 기반 판단
        if any(kw in name_lower for kw in ['ship', 'space', 'rocket', 'fighter', 'shuttle']):
            return DEFAULT_STRATEGIES['spaceship']
        elif any(kw in name_lower for kw in ['car', 'truck', 'vehicle', 'bus', 'taxi']):
            return DEFAULT_STRATEGIES['car']
        elif any(kw in name_lower for kw in ['house', 'building', 'tower', 'castle']):
            return DEFAULT_STRATEGIES['building']
        elif any(kw in name_lower for kw in ['animal', 'dog', 'cat', 'bird', 'dinosaur']):
            return DEFAULT_STRATEGIES['animal']
        elif any(kw in name_lower for kw in ['robot', 'mech', 'droid']):
            return DEFAULT_STRATEGIES['robot']

        # 형태 기반 판단
        features = model_info.get('features', '')
        if '가로로 긴 형태' in features and '좌우 대칭' in features:
            return DEFAULT_STRATEGIES['spaceship']
        elif '높은 구조물' in features:
            return DEFAULT_STRATEGIES['building']

        return DEFAULT_STRATEGIES['other']

    def _extract_model_info(self, model) -> Dict:
        """모형에서 분석용 정보 추출"""
        if not model.bricks:
            return {'dimensions': '없음', 'features': '없음'}

        # 크기 계산
        xs = [b.position.x for b in model.bricks]
        ys = [b.position.y for b in model.bricks]
        zs = [b.position.z for b in model.bricks]

        width = max(xs) - min(xs) if xs else 0
        height = max(ys) - min(ys) if ys else 0
        depth = max(zs) - min(zs) if zs else 0

        # 특징 추출
        features = []

        if width > depth * 1.5:
            features.append("가로로 긴 형태")
        elif depth > width * 1.5:
            features.append("세로로 긴 형태")

        if abs(height) > 100:
            features.append("높은 구조물")
        elif abs(height) < 50:
            features.append("낮은 구조물")

        # 대칭성
        left_count = sum(1 for b in model.bricks if b.position.x < 0)
        right_count = sum(1 for b in model.bricks if b.position.x > 0)
        if abs(left_count - right_count) < len(model.bricks) * 0.2:
            features.append("좌우 대칭")

        return {
            'dimensions': f"가로 {width}, 높이 {abs(height)}, 깊이 {depth}",
            'features': ", ".join(features) if features else "특이사항 없음"
        }

    def evolve(self, model, validation_result) -> CoScientistEvolveResult:
        """
        CoScientist 방식으로 모형 보강
        """
        self.changes = []
        evolved = deepcopy(model)

        added = 0
        removed = 0
        moved = 0

        # 1. 모형 분석
        print("=" * 50)
        print("CoScientist 모형 분석 중...")
        strategy = self.analyze_model(model)
        print(f"  모형 유형: {strategy.model_type}")
        print(f"  지지대 스타일: {strategy.support_style}")
        print(f"  전략: {strategy.explanation}")
        print("=" * 50)

        # 2. 충돌 먼저 해결
        if validation_result.collisions:
            m, r = self._fix_collisions(evolved, validation_result.collisions)
            moved += m
            removed += r

        # 3. 부유 브릭 처리 (전략 적용)
        if validation_result.floating_bricks:
            added += self._fix_floating(evolved, validation_result.floating_bricks, strategy)

        # 4. 설명 생성
        explanation = self._generate_explanation(model, strategy, self.changes)

        return CoScientistEvolveResult(
            original_model=model,
            evolved_model=evolved,
            strategy=strategy,
            changes=self.changes,
            added_bricks=added,
            removed_bricks=removed,
            moved_bricks=moved,
            explanation=explanation
        )

    def _get_brick_size(self, brick) -> Tuple[int, int]:
        """브릭의 스터드 크기 반환 (width, depth)"""
        part_id = brick.part_id

        PART_SIZES = {
            "3001": (4, 2), "3002": (3, 2), "3003": (2, 2), "3004": (4, 1),
            "3005": (1, 1), "3010": (4, 1), "3009": (3, 1), "3008": (2, 1),
            "3020": (4, 2), "3021": (3, 2), "3022": (2, 2), "3023": (2, 1),
            "3024": (1, 1), "3710": (4, 1), "3666": (6, 1),
        }

        return PART_SIZES.get(part_id, (2, 2))

    def _choose_support_part(self, strategy: SupportStrategy, remaining_height: int) -> Tuple[str, int]:
        """
        전략에 맞는 지지대 파츠 선택

        Returns:
            (part_id, height)
        """
        # 전략의 우선 파츠에서 선택
        for part_id in strategy.preferred_parts:
            if part_id not in SUPPORT_PARTS:
                continue

            _, _, part_height, _ = SUPPORT_PARTS[part_id]

            # 남은 높이에 맞는지 확인
            if remaining_height >= part_height:
                return (part_id, part_height)

        # Fallback: 플레이트
        if remaining_height >= PLATE_HEIGHT:
            return ("3024", PLATE_HEIGHT)

        return ("3024", PLATE_HEIGHT)

    def _get_support_positions(self, brick_x: float, brick_z: float,
                                brick_width: int, brick_depth: int,
                                strategy: SupportStrategy) -> List[Tuple[float, float]]:
        """
        브릭 크기와 전략에 따른 지지점 위치 계산
        지지대는 부유 브릭 **바로 아래**에 배치 (실제로 지지하도록)

        Returns:
            [(x, z), ...] 지지대 배치 위치 목록
        """
        # 부유 브릭 바로 아래에 지지대 배치 (같은 X, Z)
        # 브릭 크기에 따라 여러 지점에 분산
        width_ldu = brick_width * LDU_PER_STUD

        positions = []

        if brick_width >= 4:
            # 2x4 이상: 양 끝에 지지대 (브릭 내부)
            offset_x = (width_ldu / 2) - LDU_PER_STUD  # 끝에서 1스터드 안쪽
            positions = [
                (brick_x - offset_x, brick_z),
                (brick_x + offset_x, brick_z),
            ]
        elif brick_width >= 2:
            # 2x2: 중앙 1개
            positions = [(brick_x, brick_z)]
        else:
            # 1x1: 중앙 1개
            positions = [(brick_x, brick_z)]

        return positions

    def _fix_floating(self, model, floating_brick_ids: List[str], strategy: SupportStrategy) -> int:
        """부유 브릭 아래에 전략 기반 지지대 추가 (충돌 체크 + 중복 제거)"""
        from ldr_converter import PlacedBrick, Vector3

        added = 0
        bricks_by_id = {b.id: b for b in model.bricks}

        # 이미 지지대가 배치된 위치 추적 (중복 방지)
        used_positions = set()

        for brick_id in floating_brick_ids:
            if brick_id not in bricks_by_id:
                continue

            brick = bricks_by_id[brick_id]
            brick_width, brick_depth = self._get_brick_size(brick)

            # 브릭 종류에 따른 실제 높이
            if brick.part_id in ["3024", "3023", "3022", "3020", "3021", "3710", "3666"]:
                actual_height = PLATE_HEIGHT
            else:
                actual_height = BRICK_HEIGHT

            # 브릭 바닥 Y
            brick_bottom_y = brick.position.y + actual_height

            # 이미 바닥에 닿아있으면 스킵
            if brick_bottom_y >= 0:
                continue

            # 전략 기반 지지점 계산
            support_positions = self._get_support_positions(
                brick.position.x, brick.position.z,
                brick_width, brick_depth,
                strategy
            )

            # 중복 위치 제거
            unique_positions = []
            for sx, sz in support_positions:
                pos_key = (round(sx, 1), round(sz, 1))
                if pos_key not in used_positions:
                    unique_positions.append((sx, sz))
                    used_positions.add(pos_key)

            # 각 지지점에 기둥 추가
            total_supports = 0
            for sx, sz in unique_positions:
                supports_added = self._add_support_column(
                    model, sx, sz, brick_bottom_y, strategy, brick_id
                )
                total_supports += supports_added

            if total_supports > 0:
                style_name = {
                    'landing_gear': '착륙기어',
                    'wheels': '바퀴받침',
                    'pillars': '기둥',
                    'legs': '다리',
                    'generic': '일반'
                }.get(strategy.support_style, '일반')

                self.changes.append(
                    f"브릭 {brick_id}: {style_name} 스타일 지지대 {total_supports}개 "
                    f"({len(unique_positions)}곳)"
                )
                added += total_supports

        return added

    def _check_collision(self, model, x: float, y: float, z: float,
                         part_id: str, exclude_ids: set = None) -> bool:
        """
        해당 위치에 지지대를 놓으면 기존 브릭과 충돌하는지 확인

        Returns:
            True if collision exists, False if safe
        """
        exclude_ids = exclude_ids or set()

        # 지지대 크기 계산
        support_info = SUPPORT_PARTS.get(part_id, (1, 1, 24, ""))
        sw, sd, sh, _ = support_info
        half_w = (sw * LDU_PER_STUD) / 2
        half_d = (sd * LDU_PER_STUD) / 2

        # 지지대 bbox (position.y는 브릭 상단)
        s_min_x = x - half_w
        s_max_x = x + half_w
        s_min_y = y  # 상단
        s_max_y = y + sh  # 하단
        s_min_z = z - half_d
        s_max_z = z + half_d

        for brick in model.bricks:
            if brick.id in exclude_ids:
                continue
            if brick.id.startswith("cs_support_"):
                continue  # 다른 지지대는 스킵

            # 기존 브릭 크기
            brick_size = self._get_brick_size(brick)
            bw, bd = brick_size
            half_bw = (bw * LDU_PER_STUD) / 2
            half_bd = (bd * LDU_PER_STUD) / 2

            # 브릭 높이
            if brick.part_id in ["3024", "3023", "3022", "3020", "3021", "3710", "3666"]:
                bh = PLATE_HEIGHT
            else:
                bh = BRICK_HEIGHT

            # 브릭 bbox
            b_min_x = brick.position.x - half_bw
            b_max_x = brick.position.x + half_bw
            b_min_y = brick.position.y  # 상단
            b_max_y = brick.position.y + bh  # 하단
            b_min_z = brick.position.z - half_bd
            b_max_z = brick.position.z + half_bd

            # 충돌 체크 (AABB)
            if (s_min_x < b_max_x and s_max_x > b_min_x and
                s_min_y < b_max_y and s_max_y > b_min_y and
                s_min_z < b_max_z and s_max_z > b_min_z):
                return True  # 충돌!

        return False  # 안전

    def _add_support_column(self, model, x: float, z: float,
                            brick_bottom_y: float, strategy: SupportStrategy,
                            target_brick_id: str = None) -> int:
        """전략 기반 지지대 기둥 추가 - 바닥(Y=0)까지 반드시 연결"""
        from ldr_converter import PlacedBrick, Vector3

        added = 0
        current_y = int(brick_bottom_y)
        max_supports = 20

        # 바닥까지 지지대 쌓기 (충돌 체크 제거 - 위치 자체가 외부이므로)
        while current_y < 0 and added < max_supports:
            remaining_height = 0 - current_y

            # 전략에 맞는 파츠 선택
            part_id, part_height = self._choose_support_part(strategy, remaining_height)

            self.brick_counter += 1
            new_support = PlacedBrick(
                id=f"cs_support_{self.brick_counter}",
                part_id=part_id,
                position=Vector3(x=x, y=current_y, z=z),
                rotation=0,
                color_code=strategy.support_color,
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

    def _generate_explanation(self, model, strategy: SupportStrategy, changes: List[str]) -> str:
        """사용자에게 보여줄 설명 생성"""
        type_names = {
            'spaceship': '우주선',
            'car': '자동차',
            'building': '건물',
            'animal': '동물',
            'robot': '로봇',
            'other': '모형'
        }
        type_name = type_names.get(strategy.model_type, '모형')

        style_desc = {
            'landing_gear': '착륙기어처럼 날개/동체 아래에',
            'wheels': '바퀴 위치에 맞춰',
            'pillars': '건물 기둥처럼',
            'legs': '다리 형태로',
            'generic': '안정적으로'
        }
        style_text = style_desc.get(strategy.support_style, '안정적으로')

        explanation = f"[CoScientist 분석 결과]\n"
        explanation += f"=" * 40 + "\n"
        explanation += f"모형 유형: {type_name}\n"
        explanation += f"보강 전략: {style_text} 지지대 배치\n\n"

        if not changes:
            explanation += "변경사항 없음 - 이미 안정적입니다!\n"
        else:
            explanation += f"[변경사항] ({len(changes)}건):\n"
            for change in changes:
                explanation += f"  - {change}\n"

        explanation += f"\n[이유] {strategy.explanation}\n"

        return explanation


def evolve_with_coscientist(model, validation_result, parts_db: Dict) -> CoScientistEvolveResult:
    """
    CoScientist 방식 Evolver 실행

    Usage:
        from ldr_converter import validate_physics
        from evolver_coscientist import evolve_with_coscientist

        validation = validate_physics(model, parts_db)
        if not validation.is_valid:
            result = evolve_with_coscientist(model, validation, parts_db)
            print(result.explanation)
    """
    evolver = CoScientistEvolver(parts_db)
    return evolver.evolve(model, validation_result)


def print_coscientist_result(result: CoScientistEvolveResult):
    """결과 출력"""
    print(result.explanation)
    print("=" * 40)
    print(f"추가: {result.added_bricks}개 / 삭제: {result.removed_bricks}개 / 이동: {result.moved_bricks}개")
