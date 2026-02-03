"""
brick-judge/parser.py
LDR 파일 파싱 → JSON 구조 변환

좌표 변환:
- LDR: X(좌우), Y(-가 위), Z(앞뒤)
- 일반: X(좌우), Y(앞뒤), Z(+가 위)
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from pathlib import Path


# ============================================
# 브릭 사이즈 카탈로그 (LDU 단위)
# 1 LDU = 0.4mm, 1x1 브릭 = 20x20x24 LDU
# ============================================

BRICK_CATALOG: Dict[str, Dict] = {
    # Brick (높이 24 LDU)
    "3001.dat": {"name": "Brick 2x4", "w": 80, "d": 40, "h": 24, "studs": 8},
    "3002.dat": {"name": "Brick 2x3", "w": 60, "d": 40, "h": 24, "studs": 6},
    "3003.dat": {"name": "Brick 2x2", "w": 40, "d": 40, "h": 24, "studs": 4},
    "3004.dat": {"name": "Brick 1x2", "w": 40, "d": 20, "h": 24, "studs": 2},
    "3005.dat": {"name": "Brick 1x1", "w": 20, "d": 20, "h": 24, "studs": 1},
    "3006.dat": {"name": "Brick 2x10", "w": 200, "d": 40, "h": 24, "studs": 20},
    "3007.dat": {"name": "Brick 2x8", "w": 160, "d": 40, "h": 24, "studs": 16},
    "3008.dat": {"name": "Brick 1x8", "w": 160, "d": 20, "h": 24, "studs": 8},
    "3009.dat": {"name": "Brick 1x6", "w": 120, "d": 20, "h": 24, "studs": 6},
    "3010.dat": {"name": "Brick 1x4", "w": 80, "d": 20, "h": 24, "studs": 4},
    "3622.dat": {"name": "Brick 1x3", "w": 60, "d": 20, "h": 24, "studs": 3},
    "2456.dat": {"name": "Brick 2x6", "w": 120, "d": 40, "h": 24, "studs": 12},

    # Plate (높이 8 LDU)
    "3020.dat": {"name": "Plate 2x4", "w": 80, "d": 40, "h": 8, "studs": 8},
    "3021.dat": {"name": "Plate 2x3", "w": 60, "d": 40, "h": 8, "studs": 6},
    "3022.dat": {"name": "Plate 2x2", "w": 40, "d": 40, "h": 8, "studs": 4},
    "3023.dat": {"name": "Plate 1x2", "w": 40, "d": 20, "h": 8, "studs": 2},
    "3024.dat": {"name": "Plate 1x1", "w": 20, "d": 20, "h": 8, "studs": 1},
    "3034.dat": {"name": "Plate 2x8", "w": 160, "d": 40, "h": 8, "studs": 16},
    "3035.dat": {"name": "Plate 4x8", "w": 160, "d": 80, "h": 8, "studs": 32},
    "3036.dat": {"name": "Plate 6x8", "w": 160, "d": 120, "h": 8, "studs": 48},
    "3460.dat": {"name": "Plate 1x8", "w": 160, "d": 20, "h": 8, "studs": 8},
    "3666.dat": {"name": "Plate 1x6", "w": 120, "d": 20, "h": 8, "studs": 6},
    "3710.dat": {"name": "Plate 1x4", "w": 80, "d": 20, "h": 8, "studs": 4},
    "3623.dat": {"name": "Plate 1x3", "w": 60, "d": 20, "h": 8, "studs": 3},

    # Slope
    "3040.dat": {"name": "Slope 2x1 45", "w": 20, "d": 40, "h": 24, "studs": 2},
    "3039.dat": {"name": "Slope 2x2 45", "w": 40, "d": 40, "h": 24, "studs": 4},
}

DEFAULT_BRICK_SIZE = {"name": "Unknown", "w": 20, "d": 20, "h": 24, "studs": 1}


# ============================================
# 데이터 클래스
# ============================================

@dataclass
class Point3D:
    x: float
    y: float
    z: float

    def to_dict(self) -> Dict:
        return {"x": self.x, "y": self.y, "z": self.z}


@dataclass
class BoundingBox:
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def depth(self) -> float:
        return self.max_y - self.min_y

    @property
    def height(self) -> float:
        return self.max_z - self.min_z

    def to_dict(self) -> Dict:
        return {
            "min": {"x": self.min_x, "y": self.min_y, "z": self.min_z},
            "max": {"x": self.max_x, "y": self.max_y, "z": self.max_z},
            "size": {"w": self.width, "d": self.depth, "h": self.height}
        }


@dataclass
class Brick:
    id: int
    part: str
    name: str
    color: int

    # 원본 LDR 좌표
    ldr_x: float
    ldr_y: float
    ldr_z: float

    # 변환된 좌표 (Z=높이, +가 위)
    x: float
    y: float
    z: float

    # 크기 (LDU)
    width: float
    depth: float
    height: float

    # 메타
    studs: int
    volume: float = field(init=False)

    # 원본 라인 (수정용)
    raw_line: str
    line_number: int

    # 회전 행렬 (3x3)
    rotation: List[float] = field(default_factory=lambda: [1, 0, 0, 0, 1, 0, 0, 0, 1])

    def __post_init__(self):
        self.volume = self.width * self.depth * self.height

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "part": self.part,
            "name": self.name,
            "color": self.color,
            "position": {"x": self.x, "y": self.y, "z": self.z},
            "size": {"w": self.width, "d": self.depth, "h": self.height},
            "studs": self.studs,
            "volume": self.volume,
            "line_number": self.line_number
        }

    def get_bounds(self) -> Tuple[Point3D, Point3D]:
        """브릭의 경계 박스 반환 (min, max)"""
        half_w = self.width / 2
        half_d = self.depth / 2

        return (
            Point3D(self.x - half_w, self.y - half_d, self.z),
            Point3D(self.x + half_w, self.y + half_d, self.z + self.height)
        )


@dataclass
class ParsedModel:
    model_name: str
    total_bricks: int
    bricks: List[Brick]
    bounds: BoundingBox
    center_of_mass: Point3D
    raw_lines: List[str]

    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "total_bricks": self.total_bricks,
            "bricks": [b.to_dict() for b in self.bricks],
            "bounds": self.bounds.to_dict(),
            "center_of_mass": self.center_of_mass.to_dict()
        }

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


# ============================================
# 좌표 변환 함수
# ============================================

def ldr_to_standard(ldr_x: float, ldr_y: float, ldr_z: float) -> Tuple[float, float, float]:
    """
    LDR 좌표 → 일반 좌표 변환

    LDR: X(좌우), Y(-가 위), Z(앞뒤)
    일반: X(좌우), Y(앞뒤), Z(+가 위)
    """
    return (
        ldr_x,      # X 그대로
        ldr_z,      # LDR Z → Y (앞뒤)
        -ldr_y      # LDR -Y → Z (높이)
    )


def standard_to_ldr(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    일반 좌표 → LDR 좌표 변환 (역변환)
    """
    return (
        x,          # X 그대로
        -z,         # Z → -Y
        y           # Y → Z
    )


# ============================================
# 파싱 함수
# ============================================

def get_brick_info(part: str) -> Dict:
    """파트 번호로 브릭 정보 조회"""
    # 소문자 변환 및 경로 제거
    part_clean = part.lower().split('/')[-1].split('\\')[-1]

    if part_clean in BRICK_CATALOG:
        return BRICK_CATALOG[part_clean]

    # 대소문자 무시 검색
    for key, value in BRICK_CATALOG.items():
        if key.lower() == part_clean:
            return value

    return DEFAULT_BRICK_SIZE.copy()


def parse_brick_line(line: str, line_num: int, brick_id: int) -> Optional[Brick]:
    """
    LDR 브릭 라인 파싱

    포맷: 1 색상 X Y Z a b c d e f g h i 파트.dat
    """
    parts = line.split()

    if len(parts) < 15:
        return None

    try:
        color = int(parts[1])
        ldr_x = float(parts[2])
        ldr_y = float(parts[3])
        ldr_z = float(parts[4])

        # 회전 행렬 (9개 값)
        rotation = [float(parts[i]) for i in range(5, 14)]

        # 파트 파일명
        part_file = parts[14]

        # 좌표 변환
        x, y, z = ldr_to_standard(ldr_x, ldr_y, ldr_z)

        # 브릭 정보 조회
        info = get_brick_info(part_file)

        return Brick(
            id=brick_id,
            part=part_file,
            name=info["name"],
            color=color,
            ldr_x=ldr_x,
            ldr_y=ldr_y,
            ldr_z=ldr_z,
            x=x,
            y=y,
            z=z,
            width=info["w"],
            depth=info["d"],
            height=info["h"],
            studs=info["studs"],
            raw_line=line,
            line_number=line_num,
            rotation=rotation
        )

    except (ValueError, IndexError) as e:
        print(f"[Parser] 라인 {line_num} 파싱 실패: {e}")
        return None


def calculate_bounds(bricks: List[Brick]) -> BoundingBox:
    """전체 브릭의 경계 박스 계산"""
    if not bricks:
        return BoundingBox(0, 0, 0, 0, 0, 0)

    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')

    for brick in bricks:
        b_min, b_max = brick.get_bounds()

        min_x = min(min_x, b_min.x)
        min_y = min(min_y, b_min.y)
        min_z = min(min_z, b_min.z)
        max_x = max(max_x, b_max.x)
        max_y = max(max_y, b_max.y)
        max_z = max(max_z, b_max.z)

    return BoundingBox(min_x, max_x, min_y, max_y, min_z, max_z)


def calculate_center_of_mass(bricks: List[Brick]) -> Point3D:
    """무게중심 계산 (부피 기준)"""
    if not bricks:
        return Point3D(0, 0, 0)

    total_volume = 0
    weighted_x = weighted_y = weighted_z = 0

    for brick in bricks:
        vol = brick.volume
        # 브릭 중심점 (높이는 바닥에서 절반)
        cx = brick.x
        cy = brick.y
        cz = brick.z + brick.height / 2

        weighted_x += cx * vol
        weighted_y += cy * vol
        weighted_z += cz * vol
        total_volume += vol

    if total_volume == 0:
        return Point3D(0, 0, 0)

    return Point3D(
        weighted_x / total_volume,
        weighted_y / total_volume,
        weighted_z / total_volume
    )


def parse_ldr(file_path: str) -> ParsedModel:
    """
    LDR 파일 파싱 → ParsedModel

    Args:
        file_path: LDR 파일 경로

    Returns:
        ParsedModel: 파싱된 모델 데이터
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"LDR 파일 없음: {file_path}")

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    bricks: List[Brick] = []
    model_name = path.stem  # 파일명을 기본 이름으로

    brick_id = 0

    for line_num, line in enumerate(lines):
        line = line.strip()

        if not line:
            continue

        # 주석/메타 라인 (0으로 시작)
        if line.startswith("0 "):
            content = line[2:].strip()
            # 첫 번째 의미있는 주석을 모델 이름으로
            if content and not content.startswith("//") and model_name == path.stem:
                if not content.startswith(("FILE", "Name:", "Author:", "!LDRAW")):
                    model_name = content
            continue

        # 브릭 라인 (1로 시작)
        if line.startswith("1 "):
            brick = parse_brick_line(line, line_num, brick_id)
            if brick:
                bricks.append(brick)
                brick_id += 1

    # 통계 계산
    bounds = calculate_bounds(bricks)
    center_of_mass = calculate_center_of_mass(bricks)

    return ParsedModel(
        model_name=model_name,
        total_bricks=len(bricks),
        bricks=bricks,
        bounds=bounds,
        center_of_mass=center_of_mass,
        raw_lines=lines
    )


def parse_ldr_string(ldr_content: str, model_name: str = "Unnamed") -> ParsedModel:
    """
    LDR 문자열 파싱 (파일 없이)

    Args:
        ldr_content: LDR 파일 내용 문자열
        model_name: 모델 이름

    Returns:
        ParsedModel: 파싱된 모델 데이터
    """
    lines = ldr_content.strip().split('\n')

    bricks: List[Brick] = []
    brick_id = 0

    for line_num, line in enumerate(lines):
        line = line.strip()

        if not line:
            continue

        if line.startswith("0 "):
            content = line[2:].strip()
            if content and not content.startswith("//") and model_name == "Unnamed":
                if not content.startswith(("FILE", "Name:", "Author:", "!LDRAW")):
                    model_name = content
            continue

        if line.startswith("1 "):
            brick = parse_brick_line(line, line_num, brick_id)
            if brick:
                bricks.append(brick)
                brick_id += 1

    bounds = calculate_bounds(bricks)
    center_of_mass = calculate_center_of_mass(bricks)

    return ParsedModel(
        model_name=model_name,
        total_bricks=len(bricks),
        bricks=bricks,
        bounds=bounds,
        center_of_mass=center_of_mass,
        raw_lines=[line + '\n' for line in lines]
    )


# ============================================
# 테스트
# ============================================

if __name__ == "__main__":
    # 테스트용 LDR
    test_ldr = """0 Test Tower
1 4 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat
1 1 0 -24 0 1 0 0 0 1 0 0 0 1 3003.dat
1 14 0 -48 0 1 0 0 0 1 0 0 0 1 3005.dat
"""

    model = parse_ldr_string(test_ldr)

    print(f"모델명: {model.model_name}")
    print(f"브릭 수: {model.total_bricks}")
    print(f"무게중심: {model.center_of_mass}")
    print(f"경계: {model.bounds.to_dict()}")
    print()

    for brick in model.bricks:
        print(f"  #{brick.id} {brick.name} @ ({brick.x}, {brick.y}, {brick.z})")

    print()
    print("JSON:")
    print(model.to_json())
