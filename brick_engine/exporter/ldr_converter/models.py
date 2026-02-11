"""
LDR Converter - 데이터 타입 정의

BrickModel, PlacedBrick, Vector3 등 핵심 데이터 클래스
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Vector3:
    x: float
    y: float
    z: float


@dataclass
class PlacedBrick:
    id: str
    part_id: str
    position: Vector3
    rotation: int  # 0, 90, 180, 270
    color_code: int
    layer: int


@dataclass
class BrickModel:
    model_id: str
    name: str
    mode: str  # 'pro' or 'kids'
    bricks: List[PlacedBrick]
    target_age: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class BBox:
    """3D 바운딩 박스"""
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float

    def intersects(self, other: 'BBox', tolerance: float = 10.0) -> bool:
        """
        다른 바운딩 박스와 겹치는지 확인

        tolerance: 허용 오차 (LDU 단위)
        - 레고 브릭은 스터드 그리드(20 LDU)에 맞춰 나란히 배치됨
        - 슬로프, 바퀴 등 특수 파츠는 bbox가 실제보다 크게 계산됨
        - 기본값 10.0 (0.5 스터드)으로 정상 배치 허용
        """
        return (
            self.min_x < other.max_x - tolerance and
            self.max_x > other.min_x + tolerance and
            self.min_y < other.max_y - tolerance and
            self.max_y > other.min_y + tolerance and
            self.min_z < other.max_z - tolerance and
            self.max_z > other.min_z + tolerance
        )

    def is_supported_by(self, other: 'BBox', tolerance: float = 2.0) -> bool:
        """다른 박스 위에 올라가 있는지 확인 (Y축 기준)"""
        y_contact = abs(self.max_y - other.min_y) < tolerance
        x_overlap = self.min_x < other.max_x and self.max_x > other.min_x
        z_overlap = self.min_z < other.max_z and self.max_z > other.min_z
        return y_contact and x_overlap and z_overlap
