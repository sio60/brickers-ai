# 이 파일은 검증 시스템에서 사용되는 핵심 데이터 구조(Brick, VerificationResult 등)를 정의하는 모델 파일입니다.
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any
import numpy as np

@dataclass
class Brick:
    id: str
    x: float
    y: float
    z: float
    width: float
    depth: float
    height: float
    # 원본 LDraw 데이터 (메쉬 검증용)
    matrix: Optional[np.ndarray] = None # 회전 행렬 (3x3)
    origin: Optional[np.ndarray] = None # 원본 LDraw 위치 (x,y,z)
    part_file: Optional[str] = None     # 원본 파일명 (예: "3001.dat")
    mass: float = 1.0
    
    @property
    def center_of_mass(self) -> np.ndarray:
        return np.array([
            self.x + self.width / 2.0,
            self.y + self.depth / 2.0,
            self.z + self.height / 2.0
        ])
    
    @property
    def top_z(self) -> float:
        return self.z + self.height
        
    @property
    def volume(self) -> float:
        return self.width * self.depth * self.height

    @property
    def footprint_area(self) -> float:
        return self.width * self.depth

    @property
    def footprint_poly(self) -> List[Tuple[float, float]]:
        # 기하학적 계산을 위한 (x, y) 모서리 좌표
        return [
            (self.x, self.y),
            (self.x + self.width, self.y),
            (self.x + self.width, self.y + self.depth),
            (self.x, self.y + self.depth)
        ]

@dataclass
class Evidence:
    type: str        # 'FLOATING' (공중부양), 'UNSTABLE' (불안정), 'WEAK_CONNECTION' (연결 약함)
    severity: str    # 'CRITICAL' (치명적), 'WARNING' (경고)
    brick_ids: List[str]
    message: str
    layer: Optional[int] = None

@dataclass
class VerificationResult:
    is_valid: bool = True
    score: float = 100.0
    evidence: List[Evidence] = field(default_factory=list)
    
    def add_hard_fail(self, evidence: Evidence):
        self.is_valid = False
        self.score = 0.0
        self.evidence.append(evidence)
        
    def add_penalty(self, evidence: Evidence, deduction: float):
        self.score = max(0.0, self.score - deduction)
        self.evidence.append(evidence)

class BrickPlan:
    def __init__(self, bricks: List[Brick]):
        self.bricks = {b.id: b for b in bricks}
        
    def get_all_bricks(self) -> List[Brick]:
        return list(self.bricks.values())
