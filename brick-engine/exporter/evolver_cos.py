"""
CoScientist Evolver v6 - LLM 기반 부유 브릭 판단 + 디자인 분석

CoScientist 철학:
- AI가 모형의 의도를 이해
- 모형 유형에 맞는 보강 전략 결정
- "왜 이렇게 바꿨는지" 설명 제공

v6 개선사항 (2026-01-22):
- [NEW] LLM 부유 브릭 판단: 삭제 vs 연결 결정
- [NEW] 연결 시 위치 제안 + 충돌 시 ±10, ±20 자동 조정
- get_supported_network() 함수로 연결성 기반 부유 검사

v5 개선사항 (2026-01-21):
- 대칭성 분석: 좌우 비대칭 브릭 자동 탐지 및 보완
- LLM 디자인 분석: 브릭 좌표를 LLM에 전달해 빠진 부분 탐지

v4 개선사항:
- 롤백 로직: 보강 후 재검증해서 나빠지면 원본 반환
- 충돌 회피, 점유 셀 체크
"""

import json
import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from copy import deepcopy
from collections import deque

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
    rolled_back: bool = False  # 롤백 여부
    rollback_reason: str = ""  # 롤백 사유


# =============================================================================
# 상수 정의
# =============================================================================

# LDraw 색상 코드
class LDrawColor:
    BLACK = 0
    BLUE = 1
    GREEN = 2
    RED = 4
    BROWN = 6
    LIGHT_GRAY = 7
    DARK_GRAY = 8
    YELLOW = 14
    WHITE = 15
    LIGHT_BLUISH_GRAY = 71
    DARK_BLUISH_GRAY = 72


# 지지대용 파츠 ID
class SupportParts:
    # 브릭 (높이 24 LDU)
    BRICK_1X1 = '3005'
    BRICK_1X2 = '3004'
    BRICK_1X4 = '3010'
    BRICK_2X2 = '3003'
    BRICK_2X4 = '3001'

    # 플레이트 (높이 8 LDU)
    PLATE_1X1 = '3024'
    PLATE_1X2 = '3023'
    PLATE_2X2 = '3022'
    PLATE_2X4 = '3020'


# 모형 분류 키워드
MODEL_KEYWORDS = {
    'plant': [
        'rose', 'flower', 'plant', 'tree', 'leaf', 'garden', 'bonsai',
        'bouquet', 'tulip', 'sunflower', 'orchid', 'cactus', '장미', '꽃', '나무'
    ],
    'figure': [
        'figure', 'character', 'minifig', 'person', 'statue', 'bust',
        'helmet', 'head', '피규어', '캐릭터'
    ],
    'spaceship': [
        'spaceship', 'starship', 'spacecraft', 'space ship',
        'rocket', 'starfighter', 'x-wing', 'tie fighter', 'shuttle',
        '우주선', '로켓'
    ],
    'aircraft': [
        'plane', 'airplane', 'aircraft', 'jet', 'helicopter', 'chopper',
        'bomber', 'fighter jet', '비행기', '헬리콥터'
    ],
    'ship': [
        'ship', 'boat', 'yacht', 'vessel', 'pirate', 'titanic',
        'sailboat', 'submarine', '배', '요트', '보트'
    ],
    'train': [
        'train', 'locomotive', 'railway', 'rail', 'steam engine',
        'tram', 'subway', 'metro', '기차', '열차'
    ],
    'car': [
        'car', 'truck', 'vehicle', 'bus', 'taxi', 'jeep', 'van',
        'motorcycle', 'bike', 'racing', 'racer', '자동차', '트럭'
    ],
    'building': [
        'house', 'building', 'tower', 'castle', 'temple', 'church',
        'shop', 'store', 'bridge', 'wall', 'gate', '집', '건물', '성'
    ],
    'animal': [
        'animal', 'dog', 'cat', 'bird', 'dinosaur', 'dragon', 'horse',
        'fish', 'shark', 'whale', 'lion', 'tiger', 'bear', 'frog',
        'chicken', 'duck', 'pig', 'cow', 'rabbit', 'elephant', 'monkey',
        'snake', 'turtle', 'crab', 'spider', 'bee', 'butterfly',
        '동물', '공룡', '용', '치킨', '닭', '오리', '돼지', '소', '토끼'
    ],
    'robot': [
        'robot', 'mech', 'droid', 'android', 'gundam', 'transformer',
        '로봇', '메카'
    ],
}

# 모형 유형별 기본 전략
DEFAULT_STRATEGIES = {
    'spaceship': SupportStrategy(
        model_type='spaceship',
        support_style='landing_gear',
        support_color=LDrawColor.LIGHT_BLUISH_GRAY,
        preferred_parts=[SupportParts.BRICK_1X1, SupportParts.BRICK_1X2, SupportParts.BRICK_2X2],
        explanation='우주선의 착륙기어처럼 날개/동체 아래에 지지대 배치'
    ),
    'aircraft': SupportStrategy(
        model_type='aircraft',
        support_style='landing_gear',
        support_color=LDrawColor.BLACK,
        preferred_parts=[SupportParts.BRICK_1X1, SupportParts.PLATE_1X1, SupportParts.BRICK_1X2],
        explanation='비행기 착륙기어 스타일로 동체 아래 지지대 배치'
    ),
    'car': SupportStrategy(
        model_type='car',
        support_style='wheels',
        support_color=LDrawColor.BLACK,
        preferred_parts=[SupportParts.BRICK_1X1, SupportParts.PLATE_1X1, SupportParts.PLATE_1X2],
        explanation='자동차 바퀴 위치에 맞춰 낮은 지지대 배치'
    ),
    'train': SupportStrategy(
        model_type='train',
        support_style='wheels',
        support_color=LDrawColor.BLACK,
        preferred_parts=[SupportParts.BRICK_1X1, SupportParts.PLATE_1X1, SupportParts.PLATE_1X2],
        explanation='기차 바퀴/레일 위치에 맞춰 하단 지지대 배치'
    ),
    'ship': SupportStrategy(
        model_type='ship',
        support_style='hull_support',
        support_color=LDrawColor.BROWN,
        preferred_parts=[SupportParts.BRICK_1X1, SupportParts.BRICK_1X2, SupportParts.BRICK_2X2],
        explanation='배 선체 아래에 거치대 스타일 지지대 배치'
    ),
    'building': SupportStrategy(
        model_type='building',
        support_style='pillars',
        support_color=LDrawColor.LIGHT_GRAY,
        preferred_parts=[SupportParts.BRICK_2X4, SupportParts.BRICK_2X2, SupportParts.BRICK_1X1],
        explanation='건물 기초처럼 견고한 기둥 지지대 배치'
    ),
    'plant': SupportStrategy(
        model_type='plant',
        support_style='minimal',
        support_color=LDrawColor.GREEN,
        preferred_parts=[SupportParts.PLATE_1X1, SupportParts.PLATE_1X2, SupportParts.BRICK_1X1],
        explanation='식물/꽃의 자연스러운 형태 유지, 최소 지지대만 배치'
    ),
    'animal': SupportStrategy(
        model_type='animal',
        support_style='legs',
        support_color=LDrawColor.BROWN,
        preferred_parts=[SupportParts.BRICK_1X1, SupportParts.BRICK_1X2],
        explanation='동물의 다리 위치에 맞춰 자연스러운 지지대 배치'
    ),
    'robot': SupportStrategy(
        model_type='robot',
        support_style='legs',
        support_color=LDrawColor.LIGHT_BLUISH_GRAY,
        preferred_parts=[SupportParts.BRICK_1X1, SupportParts.BRICK_2X2],
        explanation='로봇 다리/발 위치에 기계적인 지지대 배치'
    ),
    'figure': SupportStrategy(
        model_type='figure',
        support_style='base',
        support_color=LDrawColor.BLACK,
        preferred_parts=[SupportParts.PLATE_2X4, SupportParts.PLATE_2X2, SupportParts.PLATE_1X1],
        explanation='피규어/캐릭터 아래에 받침대 스타일 지지대 배치'
    ),
    'other': SupportStrategy(
        model_type='other',
        support_style='generic',
        support_color=LDrawColor.LIGHT_BLUISH_GRAY,
        preferred_parts=[SupportParts.BRICK_1X1, SupportParts.BRICK_2X2, SupportParts.BRICK_2X4],
        explanation='일반적인 안정성 확보를 위한 지지대 배치'
    )
}


BRICK_HEIGHT = 24
PLATE_HEIGHT = 8
LDU_PER_STUD = 20

# 플레이트 파츠 ID (높이 8 LDU)
PLATE_PART_IDS = frozenset([
    "3024", "3023", "3022", "3020", "3021", "3710", "3666"
])


def get_supported_network(bricks, parts_db):
    """
    BFS로 바닥에서 연결된 브릭 네트워크 찾기
    Returns: 연결된 브릭 ID set
    """
    from ldr_converter import get_brick_bbox

    if not bricks:
        return set()

    bboxes = {}
    for brick in bricks:
        bbox = get_brick_bbox(brick, parts_db)
        if bbox:
            bboxes[brick.id] = bbox

    if not bboxes:
        return set()

    # 가장 아래 (Y가 가장 큰)
    ground_y = max(bbox.max_y for bbox in bboxes.values())

    grounded = set()
    for brick_id, bbox in bboxes.items():
        if bbox.max_y >= ground_y - 2:
            grounded.add(brick_id)

    if not grounded:
        return set()

    # 인접 관계 구축
    adjacency = {brick.id: [] for brick in bricks}
    id_list = list(bboxes.keys())

    for i in range(len(id_list)):
        for j in range(i + 1, len(id_list)):
            id1, id2 = id_list[i], id_list[j]
            bbox1, bbox2 = bboxes[id1], bboxes[id2]
            x_overlap = bbox1.min_x < bbox2.max_x and bbox1.max_x > bbox2.min_x
            z_overlap = bbox1.min_z < bbox2.max_z and bbox1.max_z > bbox2.min_z
            y_touch = abs(bbox1.min_y - bbox2.max_y) < 2 or abs(bbox1.max_y - bbox2.min_y) < 2
            if y_touch and x_overlap and z_overlap:
                adjacency[id1].append(id2)
                adjacency[id2].append(id1)

    # BFS
    supported = set(grounded)
    queue = deque(grounded)

    while queue:
        current = queue.popleft()
        for neighbor in adjacency.get(current, []):
            if neighbor not in supported:
                supported.add(neighbor)
                queue.append(neighbor)

    return supported


class CoScientistEvolver:
    """LLM 기반 지능형 Evolver v6 - 부유 브릭 LLM 판단 + 충돌 회피 + 롤백"""

    # 공간 관련
    CELL_SIZE = 20  # 1 스터드 = 20 LDU

    # 대칭성 분석
    SYMMETRY_TOLERANCE = 5  # 위치 매칭 허용 오차 (LDU)
    SYMMETRY_CENTER_MARGIN = 5  # 중앙 영역 판정 마진

    # 충돌 검사
    COLLISION_TOLERANCE = 4.0  # bbox 겹침 허용 오차 (LDU)

    # 대칭 분석 스킵할 모형 유형 (비대칭이 자연스러운 것들)
    SKIP_SYMMETRY_TYPES = frozenset(['animal', 'plant'])

    # 지지대 관련
    MAX_SUPPORTS_PER_COLUMN = 20
    ADDED_BRICK_LAYER = 999  # 추가된 브릭의 레이어 번호

    # 모형 분석 기준
    HEIGHT_TALL_THRESHOLD = 100  # 높은 구조물 판정 (LDU)
    HEIGHT_SHORT_THRESHOLD = 50  # 낮은 구조물 판정 (LDU)

    # LLM 관련
    LLM_MAX_BRICKS = 30  # LLM에 전달할 최대 브릭 수

    # 파츠 관련
    PLATE_HEIGHT_THRESHOLD = 16  # 플레이트 판정 높이 기준
    DEFAULT_BBOX_SIZE = (40, 24, 40)  # bbox 없을 때 기본값

    # 회전 관련
    ROTATION_SWAP_ANGLES = frozenset([90, 270, -90, -270])  # width/depth 스왑 각도

    # 초기값
    BRICK_COUNTER_START = 2000

    def __init__(self, parts_db: Dict, model_name: str = "qwen2.5:7b"):
        self.parts_db = parts_db
        self.model_name = model_name
        self.brick_counter = self.BRICK_COUNTER_START
        self.changes = []
        self.occupied_cells: Set[Tuple[int, int, int]] = set()
        self.skipped_supports = 0  # 충돌로 스킵된 지지대
        self.bbox_warnings = 0     # bbox 없는 파츠 경고

    def _extract_json(self, content: str) -> Dict:
        """LLM 응답에서 JSON만 안전하게 추출"""
        try:
            # 1. 마크다운 코드 블록 제거 (```json ... ```)
            if '```' in content:
                pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    return json.loads(match.group(1))

            # 2. 코드 블록 없으면 중괄호 {} 로 감싸진 가장 큰 영역 찾기
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group(0))

            # 3. 그래도 안되면 그냥 시도
            return json.loads(content)
        except Exception as e:
            print(f"  [JSON 파싱 실패] {e}")
            return {}

    def _pos_to_cell(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """좌표를 셀 좌표로 변환"""
        return (
            int(round(x / self.CELL_SIZE)),
            int(round(y / self.CELL_SIZE)),
            int(round(z / self.CELL_SIZE))
        )

    def _build_occupancy_map(self, model):
        """기존 브릭들의 점유 셀 계산"""
        self.occupied_cells.clear()

        for brick in model.bricks:
            width, depth = self._get_brick_size(brick)
            height = PLATE_HEIGHT if brick.part_id.lower() in PLATE_PART_IDS else BRICK_HEIGHT

            cx, cy, cz = self._pos_to_cell(brick.position.x, brick.position.y, brick.position.z)

            # 정확한 셀 수 계산 (w=2 -> 2셀, w=4 -> 4셀)
            for dx in range(-width // 2, -width // 2 + width):
                for dz in range(-depth // 2, -depth // 2 + depth):
                    for dy in range(0, (height // self.CELL_SIZE) + 1):
                        self.occupied_cells.add((cx + dx, cy + dy, cz + dz))

    def _is_cell_occupied(self, x: float, y: float, z: float, part_height: int = 24) -> bool:
        """해당 위치에 브릭이 있는지 확인"""
        cx, cy, cz = self._pos_to_cell(x, y, z)

        for dy in range(0, (part_height // self.CELL_SIZE) + 1):
            if (cx, cy + dy, cz) in self.occupied_cells:
                return True
        return False

    def _mark_cell_occupied(self, x: float, y: float, z: float, part_height: int = 24):
        """셀을 점유 상태로 마킹"""
        cx, cy, cz = self._pos_to_cell(x, y, z)
        for dy in range(0, (part_height // self.CELL_SIZE) + 1):
            self.occupied_cells.add((cx, cy + dy, cz))

    def analyze_model(self, model) -> SupportStrategy:
        """
        모형 분석 -> 지지 전략 결정
        우선순위: 1. 키워드 기반 (명확한 경우) 2. LLM 3. 형태 기반
        """
        # 모형 정보 수집
        model_info = self._extract_model_info(model)

        # 1. 먼저 규칙 기반 분석 시도 (키워드가 명확하면 바로 결정)
        rule_result = self._rule_based_analyze(model, model_info)
        if rule_result.model_type != 'other':
            # 키워드로 명확히 판단됨
            return rule_result

        # 2. 키워드로 판단 안 되면 LLM 시도
        if OLLAMA_AVAILABLE:
            try:
                return self._llm_analyze(model, model_info)
            except Exception as e:
                print(f"LLM 분석 실패: {e}")

        # 3. Fallback: other 반환
        return rule_result

    def _llm_analyze(self, model, model_info: Dict) -> SupportStrategy:
        """LLM 기반 모형 분석 - v4 카테고리 확장"""
        prompt = f"""당신은 레고 모형 분석 전문가입니다.

다음 레고 모형을 분석하고 적절한 지지대 전략을 JSON으로 답해주세요:

모형 이름: {model.name}
브릭 수: {len(model.bricks)}
크기 (LDU): {model_info['dimensions']}
특징: {model_info['features']}
부유 브릭 위치: {model_info.get('floating_positions', '정보 없음')}

사용 가능한 모형 유형:
- spaceship: 우주선, 로켓, 스타파이터
- aircraft: 비행기, 헬리콥터, 제트기
- car: 자동차, 트럭, 오토바이
- train: 기차, 열차, 지하철
- ship: 배, 보트, 요트 (우주선 제외)
- building: 건물, 집, 성, 탑
- plant: 꽃, 나무, 식물, 장미, 선인장
- animal: 동물, 공룡, 용
- robot: 로봇, 메카, 드로이드
- figure: 피규어, 캐릭터, 미니피그, 조각상
- other: 기타

다음 JSON 형식으로만 답하세요:
{{
    "model_type": "spaceship|aircraft|car|train|ship|building|plant|animal|robot|figure|other",
    "reasoning": "이 유형으로 판단한 이유"
}}

JSON만 출력하세요, 다른 텍스트 없이."""

        response = ollama.chat(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}]
        )

        content = response['message']['content']
        result = self._extract_json(content)
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

    # 키워드 매칭 우선순위 (순서 중요: 더 구체적인 것 먼저)
    KEYWORD_MATCH_ORDER = [
        'plant', 'figure', 'spaceship', 'aircraft', 'ship',
        'train', 'car', 'building', 'animal', 'robot'
    ]

    def _rule_based_analyze(self, model, model_info: Dict) -> SupportStrategy:
        """규칙 기반 모형 분석 (LLM 없을 때 fallback)"""
        name_lower = model.name.lower() if model.name else ""

        # 키워드 기반 판단 (순서대로)
        for model_type in self.KEYWORD_MATCH_ORDER:
            keywords = MODEL_KEYWORDS.get(model_type, [])
            if any(kw in name_lower for kw in keywords):
                # ship은 spaceship 제외 처리
                if model_type == 'ship' and 'space' in name_lower:
                    continue
                return DEFAULT_STRATEGIES[model_type]

        # 형태 기반 판단 (이름으로 못 찾았을 때)
        features = model_info.get('features', '')
        if '가로로 긴 형태' in features and '좌우 대칭' in features:
            return DEFAULT_STRATEGIES['spaceship']
        elif '높은 구조물' in features:
            return DEFAULT_STRATEGIES['building']
        elif '세로로 긴 형태' in features and '좌우 대칭' not in features:
            return DEFAULT_STRATEGIES['plant']  # 세로로 긴 비대칭 = 식물 가능성

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

        if abs(height) > self.HEIGHT_TALL_THRESHOLD:
            features.append("높은 구조물")
        elif abs(height) < self.HEIGHT_SHORT_THRESHOLD:
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

    def _analyze_symmetry(self, model) -> List[Dict]:
        """
        좌우 대칭 분석 - 한쪽에만 있는 브릭 찾기

        Returns:
            [{'missing_side': 'left'|'right', 'mirror_brick': brick, 'suggested_pos': (x,y,z)}, ...]
        """
        if not model.bricks:
            return []

        margin = self.SYMMETRY_CENTER_MARGIN
        tolerance = self.SYMMETRY_TOLERANCE

        # Y 레이어별 브릭 수 카운트 (희소 레이어 판단용)
        SPARSE_LAYER_THRESHOLD = 2  # 이 개수 이하면 악센트/장식으로 판단
        y_layer_count = {}
        for b in model.bricks:
            # Y 좌표를 tolerance 기준으로 그룹화
            y_key = round(b.position.y / tolerance) * tolerance
            y_layer_count[y_key] = y_layer_count.get(y_key, 0) + 1

        def is_sparse_layer(brick):
            """해당 브릭이 희소 레이어(악센트/장식)에 있는지 확인"""
            y_key = round(brick.position.y / tolerance) * tolerance
            return y_layer_count.get(y_key, 0) <= SPARSE_LAYER_THRESHOLD

        # X=0 기준 좌우 분류 (희소 레이어 브릭은 제외)
        left_bricks = [b for b in model.bricks
                       if b.position.x < -margin and not is_sparse_layer(b)]
        right_bricks = [b for b in model.bricks
                        if b.position.x > margin and not is_sparse_layer(b)]

        missing = []

        # 오른쪽 브릭에 대해 왼쪽 대칭 브릭이 있는지 확인
        for rb in right_bricks:
            mirror_x = -rb.position.x
            found = False
            for lb in left_bricks:
                if (abs(lb.position.x - mirror_x) < tolerance and
                    abs(lb.position.y - rb.position.y) < tolerance and
                    abs(lb.position.z - rb.position.z) < tolerance and
                    lb.part_id.lower() == rb.part_id.lower()):
                    found = True
                    break

            if not found:
                missing.append({
                    'missing_side': 'left',
                    'mirror_brick': rb,
                    'suggested_pos': (mirror_x, rb.position.y, rb.position.z),
                    'part_id': rb.part_id,
                    'color': rb.color_code
                })

        # 왼쪽 브릭에 대해 오른쪽 대칭 브릭이 있는지 확인
        for lb in left_bricks:
            mirror_x = -lb.position.x
            found = False
            for rb in right_bricks:
                if (abs(rb.position.x - mirror_x) < tolerance and
                    abs(rb.position.y - lb.position.y) < tolerance and
                    abs(rb.position.z - lb.position.z) < tolerance and
                    rb.part_id.lower() == lb.part_id.lower()):
                    found = True
                    break

            if not found:
                missing.append({
                    'missing_side': 'right',
                    'mirror_brick': lb,
                    'suggested_pos': (mirror_x, lb.position.y, lb.position.z),
                    'part_id': lb.part_id,
                    'color': lb.color_code
                })

        return missing

    def _llm_design_analysis(self, model, strategy: SupportStrategy) -> List[Dict]:
        """
        LLM에게 브릭 배치를 분석시켜 디자인 개선점 찾기

        Returns:
            [{'type': 'symmetry'|'pattern'|'structure', 'suggestion': {...}}, ...]
        """
        if not OLLAMA_AVAILABLE:
            return []

        # 브릭 목록을 텍스트로 변환
        brick_list = []
        for b in model.bricks:
            brick_list.append(f"  - {b.part_id} at ({b.position.x:.0f}, {b.position.y:.0f}, {b.position.z:.0f})")

        max_bricks = self.LLM_MAX_BRICKS
        brick_text = "\n".join(brick_list[:max_bricks])
        if len(model.bricks) > max_bricks:
            brick_text += f"\n  ... 외 {len(model.bricks) - max_bricks}개"

        prompt = f"""당신은 레고 모형 디자인 전문가입니다.

다음 {strategy.model_type} 모형의 브릭 배치를 분석하고 빠진 브릭이나 개선점을 찾아주세요.

모형 이름: {model.name}
브릭 목록:
{brick_text}

분석 기준:
1. 좌우 대칭: X=0 기준으로 한쪽에만 브릭이 있으면 반대쪽에 추가 필요
2. 패턴 반복: 같은 패턴이 반복되다 끊긴 곳이 있으면 보완 필요
3. 구조 완성도: {strategy.model_type}으로서 빠진 부분이 있는지

빠진 브릭이 있다면 다음 JSON 형식으로 답하세요:
{{
    "missing_bricks": [
        {{"part_id": "3004", "x": -30, "y": -72, "z": 60, "reason": "좌우 대칭을 위해 필요"}}
    ],
    "analysis": "분석 내용"
}}

빠진 브릭이 없으면:
{{
    "missing_bricks": [],
    "analysis": "구조가 완전합니다"
}}

JSON만 출력하세요."""

        try:
            print(f"  [LLM] Ollama {self.model_name} 호출 중... (첫 실행 시 모델 로드로 오래 걸릴 수 있음)")
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}]
            )
            print("  [LLM] 응답 수신 완료")

            content = response['message']['content']
            result = self._extract_json(content)

            suggestions = []
            for mb in result.get('missing_bricks', []):
                suggestions.append({
                    'type': 'llm_suggestion',
                    'part_id': mb.get('part_id', SupportParts.BRICK_1X1),
                    'position': (mb.get('x', 0), mb.get('y', 0), mb.get('z', 0)),
                    'reason': mb.get('reason', '')
                })

            if suggestions:
                print(f"  [LLM 분석] {result.get('analysis', '')}")

            return suggestions

        except Exception as e:
            print(f"  LLM 디자인 분석 실패: {e}")
            return []

    def _fix_symmetry(self, model, strategy: SupportStrategy) -> int:
        """대칭성 분석 결과로 빠진 브릭 추가"""
        from ldr_converter import PlacedBrick, Vector3

        missing = self._analyze_symmetry(model)
        if not missing:
            print("  대칭 분석: 문제 없음")
            return 0

        print(f"  대칭 분석: {len(missing)}개 빠진 브릭 발견")

        added = 0
        for m in missing:
            x, y, z = m['suggested_pos']
            part_id = m['part_id']

            # 정밀 충돌 체크 (bbox 기반) - occupancy cell보다 정확
            if self._check_collision(model, x, y, z, part_id):
                print(f"    스킵 (충돌): ({x}, {y}, {z})")
                continue

            # 파츠 높이 결정
            part_height = PLATE_HEIGHT if part_id.lower() in PLATE_PART_IDS else BRICK_HEIGHT

            self.brick_counter += 1
            new_brick = PlacedBrick(
                id=f"sym_{self.brick_counter}",
                part_id=part_id,
                position=Vector3(x=x, y=y, z=z),
                rotation=m['mirror_brick'].rotation,
                color_code=m['color'],
                layer=m['mirror_brick'].layer
            )
            model.bricks.append(new_brick)
            self._mark_cell_occupied(x, y, z, part_height)
            added += 1

            self.changes.append(
                f"대칭 보완: {part_id} at ({x:.0f}, {y:.0f}, {z:.0f}) "
                f"- {m['missing_side']}쪽에 추가"
            )

        return added

    def _fix_from_llm_suggestions(self, model, strategy: SupportStrategy) -> int:
        """LLM 제안 기반 브릭 추가"""
        from ldr_converter import PlacedBrick, Vector3

        suggestions = self._llm_design_analysis(model, strategy)
        if not suggestions:
            return 0

        added = 0
        for s in suggestions:
            x, y, z = s['position']
            part_id = s['part_id']

            # 충돌 체크
            if self._is_cell_occupied(x, y, z, BRICK_HEIGHT):
                print(f"    LLM 제안 스킵 (충돌): ({x}, {y}, {z})")
                continue

            # 주변 브릭 색상 가져오기
            color = LDrawColor.LIGHT_BLUISH_GRAY
            min_dist = float('inf')
            for brick in model.bricks:
                dist = abs(brick.position.x - x) + abs(brick.position.y - y) + abs(brick.position.z - z)
                if dist < min_dist:
                    min_dist = dist
                    color = brick.color_code

            self.brick_counter += 1
            new_brick = PlacedBrick(
                id=f"llm_{self.brick_counter}",
                part_id=part_id,
                position=Vector3(x=x, y=y, z=z),
                rotation=0,
                color_code=color,
                layer=self.ADDED_BRICK_LAYER
            )
            model.bricks.append(new_brick)
            self._mark_cell_occupied(x, y, z, BRICK_HEIGHT)
            added += 1

            self.changes.append(
                f"LLM 제안: {part_id} at ({x:.0f}, {y:.0f}, {z:.0f}) - {s.get('reason', '')}"
            )

        return added

    def _ask_llm_for_floating_fix(self, floating_info: List[Dict], nearby_supported: List[Dict]) -> Dict:
        """
        LLM에게 부유 브릭 수정 방법만 질문 (CoScientist: 판단은 LLM, 실행은 알고리즘)

        Returns:
            {
                "action": "delete" | "connect",
                "reasoning": "설명"
            }
        """
        if not OLLAMA_AVAILABLE:
            return {"action": "delete", "reasoning": "Ollama 없음"}

        floating_text = "\n".join([
            f"  - {f['id']}: ({f['x']:.0f}, {f['y']:.0f}, {f['z']:.0f}) 파츠 {f['part_id']}"
            for f in floating_info
        ])

        supported_text = "\n".join([
            f"  - {s['id']}: ({s['x']:.0f}, {s['y']:.0f}, {s['z']:.0f}) 파츠 {s['part_id']}"
            for s in nearby_supported[:10]
        ])

        prompt = f"""당신은 레고 브릭 모형 전문가입니다.

다음 레고 모형에서 공중에 떠 있는 "부유 브릭"이 발견되었습니다.

[부유 브릭] (연결 안 됨, {len(floating_info)}개)
{floating_text}

[근처 연결된 브릭] ({len(nearby_supported)}개)
{supported_text}

선택지:
1. "delete": 부유 브릭 삭제 (형태 일부 손실, 단순화)
2. "connect": 새 브릭 추가해서 연결 (형태 유지, 브릭 추가)

판단 기준:
- 부유 브릭이 모형의 중요한 부분(날개, 꼬리 등)이면 → connect
- 부유 브릭이 장식적이거나 사소하면 → delete
- 근처에 연결 가능한 브릭이 있으면 → connect
- 연결이 어려워 보이면 → delete

JSON으로 답하세요:
```json
{{
    "action": "delete" 또는 "connect",
    "reasoning": "판단 이유"
}}
```

JSON만 출력하세요."""

        try:
            print(f"  [LLM] 부유 브릭 수정 판단 중...")
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}]
            )
            content = response['message']['content']
            result = self._extract_json(content)
            print(f"  [LLM] 판단: {result.get('action', 'unknown')}")
            return result
        except Exception as e:
            print(f"  [LLM] 오류: {e}")
            return {"action": "delete", "bricks_to_add": [], "reasoning": f"LLM 오류: {e}"}

    def _fix_floating_with_llm(self, model, floating_brick_ids: List[str], strategy: SupportStrategy) -> int:
        """
        LLM 기반 부유 브릭 수정
        - LLM이 삭제/연결 판단
        - 연결 시 supported bbox 경계에 스냅 + 충돌 조정
        """
        from ldr_converter import PlacedBrick, Vector3, get_brick_bbox

        if not floating_brick_ids:
            return 0

        # 부유/연결 브릭 분류
        supported_ids = get_supported_network(model.bricks, self.parts_db)
        floating_bricks = [b for b in model.bricks if b.id in floating_brick_ids]
        supported_bricks = [b for b in model.bricks if b.id in supported_ids]

        # 부유 브릭 정보
        floating_info = []
        for b in floating_bricks:
            floating_info.append({
                'id': b.id,
                'x': b.position.x,
                'y': b.position.y,
                'z': b.position.z,
                'part_id': b.part_id,
                'color': b.color_code
            })

        # 부유 브릭 중심 X 계산 (양수 X 기준)
        main_floating = [f for f in floating_info if f['x'] > 0]
        if main_floating:
            ref_x = sum(f['x'] for f in main_floating) / len(main_floating)
            ref_y_min = min(f['y'] for f in main_floating)
            ref_y_max = max(f['y'] for f in main_floating)
        else:
            ref_x = sum(f['x'] for f in floating_info) / len(floating_info)
            ref_y_min = min(f['y'] for f in floating_info)
            ref_y_max = max(f['y'] for f in floating_info)

        # 근처 supported 필터링 (X, Y 범위) + bbox 계산
        nearby_supported = []
        for b in supported_bricks:
            x_close = abs(b.position.x - ref_x) < 50
            y_close = (ref_y_min - 48) <= b.position.y <= (ref_y_max + 48)
            if x_close and y_close:
                bbox = get_brick_bbox(b, self.parts_db)
                nearby_supported.append({
                    'id': b.id,
                    'x': b.position.x,
                    'y': b.position.y,
                    'z': b.position.z,
                    'part_id': b.part_id,
                    'bbox': bbox
                })

        nearby_supported.sort(key=lambda s: abs(s['x'] - ref_x))

        print(f"  부유 브릭: {len(floating_info)}개, 근처 supported: {len(nearby_supported)}개")

        # LLM에게 질문
        llm_result = self._ask_llm_for_floating_fix(floating_info, nearby_supported)
        action = llm_result.get('action', 'delete')

        added_count = 0

        if action == 'delete':
            # 부유 브릭 삭제
            reasoning = llm_result.get('reasoning', '')
            print(f"  [LLM 이유] {reasoning}")
            model.bricks = [b for b in model.bricks if b.id not in floating_brick_ids]
            self.changes.append(f"부유 브릭 {len(floating_brick_ids)}개 삭제 (LLM: {reasoning})")
            print(f"  부유 브릭 삭제: {len(floating_brick_ids)}개")

        else:  # connect
            # [CoScientist] LLM이 connect 판단 → 알고리즘으로 위치 계산
            reasoning = llm_result.get('reasoning', '')
            print(f"  [LLM 이유] {reasoning}")

            # 부유 브릭 삭제
            model.bricks = [b for b in model.bricks if b.id not in floating_brick_ids]
            self.changes.append(f"부유 브릭 {len(floating_brick_ids)}개 삭제 후 연결 (LLM: {reasoning})")

            # [알고리즘] 정확한 위치 계산
            added_positions = set()

            # 1. Y 계산: 부유 브릭 가장 위층 - 24 (한 층 위에 새 브릭)
            floating_highest_y = min(f['y'] for f in floating_info)
            new_y = floating_highest_y - BRICK_HEIGHT

            # 2. X 계산: supported max_x 경계 (스터드 겹침 보장)
            supported_max_x = max((s['bbox'].max_x for s in nearby_supported if s['bbox']), default=0)
            new_x = round(supported_max_x / 20) * 20  # 1스터드 단위 정렬

            print(f"  [알고리즘] Y={new_y:.0f}, X={new_x:.0f} (supported max_x={supported_max_x:.0f})")

            # 3. Z 계산: 부유 브릭의 각 Z 값에 대해 브릭 추가
            floating_z_values = sorted(set(f['z'] for f in floating_info))

            for z in floating_z_values:
                # 중복 체크
                pos_key = (round(new_x), round(new_y), round(z))
                if pos_key in added_positions:
                    continue

                # 충돌 체크 및 X 조정
                final_x = new_x
                if self._check_collision(model, final_x, new_y, z, '3004'):
                    print(f"    충돌 감지: ({final_x:.0f}, {new_y:.0f}, {z:.0f})")
                    adjusted = False
                    for dx in [10, -10, 20, -20]:
                        if not self._check_collision(model, new_x + dx, new_y, z, '3004'):
                            final_x = new_x + dx
                            print(f"    X 조정: {new_x:.0f} → {final_x:.0f}")
                            adjusted = True
                            break
                    if not adjusted:
                        print(f"    조정 실패, 스킵")
                        continue

                # 색상: 해당 Z의 부유 브릭 색상
                color = strategy.support_color
                closest = min(floating_info, key=lambda f: abs(f['z'] - z))
                color = closest.get('color', color)

                self.brick_counter += 1
                new_brick = PlacedBrick(
                    id=f"connect_{self.brick_counter}",
                    part_id='3004',
                    position=Vector3(x=final_x, y=new_y, z=z),
                    rotation=0,
                    color_code=color,
                    layer=self.ADDED_BRICK_LAYER
                )
                model.bricks.append(new_brick)
                added_count += 1
                added_positions.add((round(final_x), round(new_y), round(z)))
                self._mark_cell_occupied(final_x, new_y, z, BRICK_HEIGHT)
                self.changes.append(f"연결 브릭: 3004 at ({final_x:.0f}, {new_y:.0f}, {z:.0f})")
                print(f"    추가: 3004 at ({final_x:.0f}, {new_y:.0f}, {z:.0f})")

            print(f"  연결 브릭 추가: {added_count}개")

        return added_count

    def evolve(self, model, validation_result, analyze_design: bool = True) -> CoScientistEvolveResult:
        """
        CoScientist 방식으로 모형 보강 (v5 - 대칭성/LLM 분석 추가)

        - 충돌/부유 없어도 대칭성/디자인 분석 수행 (analyze_design=True일 때)
        - 보강 후 재검증해서 나빠졌으면 원본 반환

        Args:
            model: BrickModel
            validation_result: validate_physics() 결과
            analyze_design: True면 대칭성/LLM 디자인 분석도 수행 (기본 True)
        """
        from ldr_converter import validate_physics

        # 0. 구조적으로 안정적이고 디자인 분석도 안 하면 스킵
        is_structurally_stable = not validation_result.collisions and not validation_result.floating_bricks

        if is_structurally_stable and not analyze_design:
            print("[OK] 이미 안정적인 모델 - Evolver 스킵")
            return CoScientistEvolveResult(
                original_model=model,
                evolved_model=model,  # 원본 그대로
                strategy=DEFAULT_STRATEGIES['other'],
                changes=[],
                added_bricks=0,
                removed_bricks=0,
                moved_bricks=0,
                explanation="[CoScientist] 이미 안정적인 모델입니다. 변경 없이 원본 유지.",
                rolled_back=False,
                rollback_reason=""
            )

        self.changes = []
        self.skipped_supports = 0
        self.bbox_warnings = 0
        evolved = deepcopy(model)

        added = 0
        removed = 0
        moved = 0

        # 보강 전 문제 수 저장
        before_collisions = len(validation_result.collisions)
        before_floating = len(validation_result.floating_bricks)

        # 1. 기존 브릭 점유 맵 생성
        self._build_occupancy_map(evolved)

        # 2. 모형 분석
        print("=" * 50)
        print("CoScientist 모형 분석 중...")
        strategy = self.analyze_model(model)
        print(f"  모형 유형: {strategy.model_type}")
        print(f"  지지대 스타일: {strategy.support_style}")
        print(f"  전략: {strategy.explanation}")
        print("=" * 50)

        # 3. [CoScientist v5] 대칭성 분석 - 좌우 비대칭 브릭 찾아서 추가
        if analyze_design:
            print("\n[디자인 분석]")
            if strategy.model_type in self.SKIP_SYMMETRY_TYPES:
                print(f"  대칭 분석 스킵: {strategy.model_type}은(는) 비대칭이 자연스러움")
                symmetry_added = 0
            else:
                symmetry_added = self._fix_symmetry(evolved, strategy)
                added += symmetry_added

            # 4. [CoScientist v5] LLM 기반 디자인 분석 (대칭성으로 못 찾은 것 보완)
            if OLLAMA_AVAILABLE and symmetry_added == 0 and strategy.model_type not in self.SKIP_SYMMETRY_TYPES:
                llm_added = self._fix_from_llm_suggestions(evolved, strategy)
                added += llm_added

        # 5. 부유 브릭 처리 (LLM 기반 또는 전략 기반)
        if validation_result.floating_bricks:
            if OLLAMA_AVAILABLE:
                # LLM이 삭제/연결 판단
                added += self._fix_floating_with_llm(evolved, validation_result.floating_bricks, strategy)
            else:
                # Fallback: 기존 전략 기반 지지대 추가
                added += self._fix_floating(evolved, validation_result.floating_bricks, strategy)

        # 6. 보강 후 재검증 (롤백 판단)
        after_validation = validate_physics(evolved, self.parts_db)
        after_collisions = len(after_validation.collisions)
        after_floating = len(after_validation.floating_bricks)

        print(f"\n[재검증] 충돌: {before_collisions} → {after_collisions}, 부유: {before_floating} → {after_floating}")

        rolled_back = False
        rollback_reason = ""

        if after_collisions > before_collisions:
            rolled_back = True
            rollback_reason = f"충돌 증가 ({before_collisions} → {after_collisions})"
        elif after_floating > before_floating:
            rolled_back = True
            rollback_reason = f"부유 브릭 증가 ({before_floating} → {after_floating})"

        if rolled_back:
            print(f"[ROLLBACK] {rollback_reason}")
            evolved = model  # 원본으로 롤백
            self.changes = [f"[롤백] {rollback_reason} - 원본 유지"]
            added = 0
            removed = 0
            moved = 0

        # 7. 설명 생성
        explanation = self._generate_explanation(model, strategy, self.changes)

        if self.skipped_supports > 0:
            explanation += f"\n(충돌로 스킵된 지지대: {self.skipped_supports}개)\n"

        if rolled_back:
            explanation += f"\n[ROLLBACK] {rollback_reason}\n보강이 오히려 상태를 악화시켜 원본을 유지합니다.\n"

        return CoScientistEvolveResult(
            original_model=model,
            evolved_model=evolved,
            strategy=strategy,
            changes=self.changes,
            added_bricks=added,
            removed_bricks=removed,
            moved_bricks=moved,
            explanation=explanation,
            rolled_back=rolled_back,
            rollback_reason=rollback_reason
        )

    def _get_brick_size(self, brick) -> Tuple[int, int]:
        """브릭의 스터드 크기 반환 (width, depth) - bbox + 회전 고려"""
        part_id = brick.part_id.lower()
        part = self.parts_db.get(part_id)

        if part and 'bbox' in part and 'size' in part['bbox']:
            size = part['bbox']['size']
            width = max(1, int(size[0] / LDU_PER_STUD))
            depth = max(1, int(size[2] / LDU_PER_STUD))

            # 90° 또는 270° 회전 시 width/depth 스왑
            rotation = getattr(brick, 'rotation', 0)
            if rotation in self.ROTATION_SWAP_ANGLES:
                width, depth = depth, width

            return (width, depth)

        # DB에 없으면 기본값 + 경고
        if self.bbox_warnings < 5:  # 경고 5개까지만
            print(f"  [경고] {part_id} bbox 없음, 기본값(2,2) 사용")
        self.bbox_warnings += 1
        return (2, 2)

    def _get_brick_height(self, brick) -> int:
        """브릭의 실제 높이 반환 (LDU) - bbox에서 계산"""
        part_id = brick.part_id.lower()
        part = self.parts_db.get(part_id)

        if part and 'bbox' in part and 'size' in part['bbox']:
            height = part['bbox']['size'][1]
            # 너무 작으면 최소 플레이트 높이
            return max(PLATE_HEIGHT, int(height))

        # DB에 없으면 기본 브릭 높이
        return BRICK_HEIGHT

    def _choose_support_part(self, strategy: SupportStrategy, remaining_height: int,
                             brick_width: int = 2, brick_depth: int = 2) -> Tuple[str, int]:
        """
        전략 + 브릭 크기에 맞는 지지대 파츠 선택 (evolver.py v4 로직 통합)

        Args:
            strategy: 지지대 전략
            remaining_height: 지면까지 남은 높이
            brick_width: 지지할 브릭의 가로 스터드 수
            brick_depth: 지지할 브릭의 세로 스터드 수

        Returns:
            (part_id, height)
        """
        # 1단계: 전략의 preferred_parts에서 브릭 크기에 맞는 것 선택
        for part_id in strategy.preferred_parts:
            part = self.parts_db.get(part_id)
            if not part or 'bbox' not in part:
                continue

            size = part['bbox'].get('size', list(self.DEFAULT_BBOX_SIZE))
            pw = max(1, int(size[0] / LDU_PER_STUD))
            pd = max(1, int(size[2] / LDU_PER_STUD))
            part_height = int(size[1]) if size[1] > self.PLATE_HEIGHT_THRESHOLD else PLATE_HEIGHT

            # 남은 높이와 브릭 크기 모두 고려
            if remaining_height >= part_height:
                # 지지대가 브릭보다 크지 않아야 함
                if pw <= brick_width and pd <= brick_depth:
                    return (part_id, part_height)

        # 2단계: Fallback - 브릭 크기 기반 선택
        if remaining_height >= BRICK_HEIGHT:
            if brick_width >= 4 and brick_depth >= 2:
                return (SupportParts.BRICK_2X4, BRICK_HEIGHT)
            elif brick_width >= 2 and brick_depth >= 2:
                return (SupportParts.BRICK_2X2, BRICK_HEIGHT)
            elif brick_width >= 4:
                return (SupportParts.BRICK_1X4, BRICK_HEIGHT)
            else:
                return (SupportParts.BRICK_1X1, BRICK_HEIGHT)

        elif remaining_height >= PLATE_HEIGHT:
            if brick_width >= 4 and brick_depth >= 2:
                return (SupportParts.PLATE_2X4, PLATE_HEIGHT)
            elif brick_width >= 2 and brick_depth >= 2:
                return (SupportParts.PLATE_2X2, PLATE_HEIGHT)
            elif brick_width >= 2:
                return (SupportParts.PLATE_1X2, PLATE_HEIGHT)
            else:
                return (SupportParts.PLATE_1X1, PLATE_HEIGHT)

        return (SupportParts.PLATE_1X1, PLATE_HEIGHT)

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
            actual_height = PLATE_HEIGHT if brick.part_id.lower() in PLATE_PART_IDS else BRICK_HEIGHT

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
                    model, sx, sz, brick_bottom_y, strategy,
                    brick_width, brick_depth, brick_id
                )
                total_supports += supports_added

            if total_supports > 0:
                style_name = {
                    'landing_gear': '착륙기어',
                    'wheels': '바퀴받침',
                    'hull_support': '선체받침',
                    'pillars': '기둥',
                    'minimal': '최소',
                    'legs': '다리',
                    'base': '받침대',
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
        tolerance = self.COLLISION_TOLERANCE

        # 지지대 크기 계산 (bbox에서)
        part = self.parts_db.get(part_id.lower())
        if part and 'bbox' in part and 'size' in part['bbox']:
            size = part['bbox']['size']
            sw = max(1, int(size[0] / LDU_PER_STUD))
            sd = max(1, int(size[2] / LDU_PER_STUD))
            sh = int(size[1]) if size[1] > self.PLATE_HEIGHT_THRESHOLD else PLATE_HEIGHT
        else:
            sw, sd, sh = 1, 1, BRICK_HEIGHT
        half_w = (sw * LDU_PER_STUD) / 2
        half_d = (sd * LDU_PER_STUD) / 2

        # 지지대 bbox - tolerance 적용 (약간 축소)
        s_min_x = x - half_w + tolerance
        s_max_x = x + half_w - tolerance
        s_min_y = y + tolerance
        s_max_y = y + sh - tolerance
        s_min_z = z - half_d + tolerance
        s_max_z = z + half_d - tolerance

        for brick in model.bricks:
            if brick.id in exclude_ids:
                continue
            if brick.id.startswith("cs_support_") or brick.id.startswith("sym_") or brick.id.startswith("llm_"):
                continue  # 새로 추가한 지지대는 스킵

            # 기존 브릭 크기 (회전 고려됨)
            bw, bd = self._get_brick_size(brick)
            half_bw = (bw * LDU_PER_STUD) / 2
            half_bd = (bd * LDU_PER_STUD) / 2

            # 브릭 높이 (실제 bbox에서 가져옴)
            bh = self._get_brick_height(brick)

            # 브릭 bbox
            b_min_x = brick.position.x - half_bw
            b_max_x = brick.position.x + half_bw
            b_min_y = brick.position.y
            b_max_y = brick.position.y + bh
            b_min_z = brick.position.z - half_bd
            b_max_z = brick.position.z + half_bd

            # 충돌 체크 (AABB) - 축소된 bbox로 체크
            if (s_min_x < b_max_x and s_max_x > b_min_x and
                s_min_y < b_max_y and s_max_y > b_min_y and
                s_min_z < b_max_z and s_max_z > b_min_z):
                return True  # 충돌!

        return False  # 안전

    def _add_support_column(self, model, x: float, z: float,
                            brick_bottom_y: float, strategy: SupportStrategy,
                            brick_width: int = 2, brick_depth: int = 2,
                            target_brick_id: str = None) -> int:
        """전략 기반 지지대 기둥 추가 - 충돌 회피 + 브릭 크기 고려"""
        from ldr_converter import PlacedBrick, Vector3

        added = 0
        current_y = int(brick_bottom_y)

        while current_y < 0 and added < self.MAX_SUPPORTS_PER_COLUMN:
            remaining_height = 0 - current_y

            # 전략 + 브릭 크기에 맞는 파츠 선택
            part_id, part_height = self._choose_support_part(
                strategy, remaining_height, brick_width, brick_depth
            )

            # 충돌 체크 - 해당 위치에 이미 브릭이 있으면 스킵
            if self._is_cell_occupied(x, current_y, z, part_height):
                self.skipped_supports += 1
                current_y += part_height
                continue

            self.brick_counter += 1
            new_support = PlacedBrick(
                id=f"cs_support_{self.brick_counter}",
                part_id=part_id,
                position=Vector3(x=x, y=current_y, z=z),
                rotation=0,
                color_code=strategy.support_color,
                layer=self.ADDED_BRICK_LAYER
            )
            model.bricks.append(new_support)
            added += 1

            # 새로 추가한 지지대도 점유 맵에 등록
            self._mark_cell_occupied(x, current_y, z, part_height)

            current_y += part_height

        return added

    def _generate_explanation(self, model, strategy: SupportStrategy, changes: List[str]) -> str:
        """사용자에게 보여줄 설명 생성 - v4 카테고리 확장"""
        type_names = {
            'spaceship': '우주선',
            'aircraft': '비행기',
            'car': '자동차',
            'train': '기차',
            'ship': '배',
            'building': '건물',
            'plant': '식물/꽃',
            'animal': '동물',
            'robot': '로봇',
            'figure': '피규어',
            'other': '모형'
        }
        type_name = type_names.get(strategy.model_type, '모형')

        style_desc = {
            'landing_gear': '착륙기어처럼 동체 아래에',
            'wheels': '바퀴 위치에 맞춰',
            'hull_support': '선체 아래 거치대 스타일로',
            'pillars': '건물 기둥처럼',
            'minimal': '형태 유지하며 최소한으로',
            'legs': '다리 형태로',
            'base': '받침대 스타일로',
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


if __name__ == "__main__":
    """
    사용법: python evolver_cos.py 입력.ldr [출력.ldr]
    """
    import sys
    import os
    from pathlib import Path
    from dotenv import load_dotenv
    from pymongo import MongoClient
    from ldr_converter import ldr_to_brick_model, validate_physics, model_to_ldr, save_ldr_file

    # 버퍼링 없이 출력 (진행 상황 실시간 표시)
    import functools
    print = functools.partial(print, flush=True)

    # .env 로드
    ROOT = Path(__file__).parent.parent.parent
    load_dotenv(ROOT / ".env")

    if len(sys.argv) < 2:
        print("사용법: python evolver_coscientist.py 입력.ldr [출력.ldr]")
        sys.exit(1)

    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else inp.replace(".ldr", "_evolved.ldr")

    if not os.path.exists(inp):
        print(f"파일 없음: {inp}")
        sys.exit(1)

    # 파츠 DB 로드 (로컬 캐시 우선)
    CACHE_FILE = Path(__file__).parent / "parts_cache.json"

    if CACHE_FILE.exists():
        print("[1/5] 파츠 DB 로딩 중... (캐시)")
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            parts_db = json.load(f)
        print(f"[OK] 파츠 DB: {len(parts_db)}개 (캐시에서 로드)")
    else:
        print("[1/5] MongoDB 연결 중...")
        client = MongoClient(os.getenv('MONGODB_URI'))
        print("[2/5] 파츠 DB 로딩 중... (최초 실행, 캐시 생성)")
        parts_db = {}
        cursor = client['brickers']['ldraw_parts'].find(
            {},
            {'partId': 1, 'ldrawFile': 1, 'bbox': 1, 'name': 1, 'category': 1, '_id': 0}
        )
        for p in cursor:
            part_id = p['partId'].lower()
            parts_db[part_id] = {
                'partId': part_id,
                'ldrawFile': p.get('ldrawFile') or f"{part_id}.dat",
                'bbox': p.get('bbox', {}),
                'name': p.get('name', ''),
                'category': p.get('category', '')
            }
        client.close()
        # 캐시 저장
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(parts_db, f)
        print(f"[OK] 파츠 DB: {len(parts_db)}개 (캐시 저장 완료)")

    # 실행
    print("[3/5] LDR 파싱 중...")
    model = ldr_to_brick_model(inp)
    print(f"[OK] 브릭: {len(model.bricks)}개")

    print("[4/5] 물리 검증 중...")
    validation = validate_physics(model, parts_db)
    print(f"[OK] 검증: 충돌 {len(validation.collisions)}, 부유 {len(validation.floating_bricks)}")

    print("[5/5] CoScientist Evolver 실행 중...")
    result = CoScientistEvolver(parts_db).evolve(model, validation, analyze_design=True)

    if not result.rolled_back and result.added_bricks > 0:
        ldr = model_to_ldr(result.evolved_model, parts_db, skip_validation=True, skip_physics=True)
        save_ldr_file(ldr, out)
        print(f"\n저장: {out}")
    else:
        print("\n변경 없음 또는 롤백됨")
