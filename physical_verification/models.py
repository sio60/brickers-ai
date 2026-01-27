# ============================================================================
# 데이터 모델 정의 모듈
# 이 파일은 물리 검증 시스템에서 사용하는 핵심 데이터 클래스들을 정의합니다.
# - Brick: 레고 브릭의 위치, 크기, 질량 등을 담는 클래스
# - Evidence: 검증 결과의 증거(문제점)를 담는 클래스
# - VerificationResult: 전체 검증 결과를 담는 클래스
# - BrickPlan: 브릭 컬렉션을 관리하는 클래스
# ============================================================================

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any
import numpy as np

@dataclass
class Brick:
    """
    레고 브릭을 나타내는 데이터 클래스
    
    Attributes:
        id: 브릭의 고유 식별자
        x, y, z: 브릭의 최소 좌표 (왼쪽 아래 모서리)
        width: X축 방향 크기
        depth: Y축 방향 크기 (깊이)
        height: Z축 방향 크기 (높이)
        mass: 브릭의 질량 (기본값 1.0)
    """
    id: str
    x: float
    y: float
    z: float
    width: float
    depth: float
    height: float
    mass: float = 1.0
    
    # LDraw data for PyBullet
    origin: Optional[Tuple[float, float, float]] = None
    part_file: Optional[str] = None
    matrix: Optional[Any] = None
    
    @property
    def center_of_mass(self) -> np.ndarray:
        """브릭의 질량 중심 좌표를 반환합니다."""
        return np.array([
            self.x + self.width / 2.0,
            self.y + self.depth / 2.0,
            self.z + self.height / 2.0
        ])
    
    @property
    def top_z(self) -> float:
        """브릭의 상단 Z 좌표를 반환합니다."""
        return self.z + self.height
        
    @property
    def volume(self) -> float:
        """브릭의 부피를 반환합니다."""
        return self.width * self.depth * self.height

    @property
    def footprint_area(self) -> float:
        """브릭의 바닥 면적(발자국)을 반환합니다."""
        return self.width * self.depth

    @property
    def footprint_poly(self) -> List[Tuple[float, float]]:
        """기하학적 계산을 위한 (x, y) 코너 좌표 목록을 반환합니다."""
        return [
            (self.x, self.y),
            (self.x + self.width, self.y),
            (self.x + self.width, self.y + self.depth),
            (self.x, self.y + self.depth)
        ]

@dataclass
class Evidence:
    """
    검증 결과의 증거(문제점)를 나타내는 데이터 클래스
    
    Attributes:
        type: 문제 유형 ('FLOATING', 'UNSTABLE', 'WEAK_CONNECTION' 등)
        severity: 심각도 ('CRITICAL', 'WARNING')
        brick_ids: 관련된 브릭 ID 목록
        message: 상세 메시지
        layer: 문제가 발생한 레이어(높이) (선택적)
    """
    type: str        # 'FLOATING', 'UNSTABLE', 'WEAK_CONNECTION'
    severity: str    # 'CRITICAL', 'WARNING'
    brick_ids: List[str]
    message: str
    layer: Optional[int] = None

@dataclass
class VerificationResult:
    """
    전체 검증 결과를 담는 데이터 클래스
    
    Attributes:
        is_valid: 검증 통과 여부
        score: 점수 (0-100)
        evidence: 발견된 문제점 목록
    """
    is_valid: bool = True
    score: float = 100.0
    evidence: List[Evidence] = field(default_factory=list)
    
    def add_hard_fail(self, evidence: Evidence):
        """치명적인 실패를 추가합니다. 점수가 0이 되고 유효하지 않음으로 표시됩니다."""
        self.is_valid = False
        self.score = 0.0
        self.evidence.append(evidence)
        
    def add_penalty(self, evidence: Evidence, deduction: float):
        """감점을 추가합니다. 점수에서 지정된 만큼 차감됩니다."""
        self.score = max(0.0, self.score - deduction)
        self.evidence.append(evidence)

class BrickPlan:
    """
    브릭 컬렉션을 관리하는 클래스
    
    Attributes:
        bricks: 브릭 ID를 키로 하는 브릭 딕셔너리
    """
    def __init__(self, bricks: List[Brick]):
        """브릭 리스트로 BrickPlan을 초기화합니다."""
        self.bricks = {b.id: b for b in bricks}
        
    def get_all_bricks(self) -> List[Brick]:
        """모든 브릭을 리스트로 반환합니다."""
        return list(self.bricks.values())
