from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class TuneParameters(BaseModel):
    """
    구조물 안정성을 개선하기 위해 GLB-to-LDR 변환 파라미터를 조정합니다.
    이전 시도 결과를 바탕으로 새로운 파라미터 조합을 제안해야 합니다.
    """
    target: int = Field(..., description="목표 스터드 크기 (기본값: 25). 크기가 클수록 디테일이 살지만 불안정할 수 있음.")
    budget: int = Field(..., description="최대 브릭 사용 개수 (기본값: 150).")
    interlock: bool = Field(..., description="인터락(엇갈려 쌓기) 활성화 여부. 안정성을 위해 필수적임.")
    fill: bool = Field(..., description="내부 채움 활성화 여부. 끄면 속이 비어 가벼워지지만 약해짐.")
    smart_fix: bool = Field(..., description="스마트 보정 활성화 여부.")
    plates_per_voxel: int = Field(..., description="복셀당 플레이트 수 (1~3). 3이면 정밀하지만 브릭 수가 늘어남.")
    reasoning: str = Field(..., description="이 파라미터를 선택한 이유에 대한 간략한 설명.")

class FixFloatingBricks(BaseModel):
    """
    물리 검증 결과 공중에 떠 있거나(Floating) 불안정한 브릭들을 삭제합니다.
    삭제 목록에 없는 브릭은 그대로 유지됩니다.
    """
    bricks_to_delete: List[str] = Field(..., description="삭제할 브릭의 ID 목록.")
    reasoning: str = Field(..., description="삭제 대상 선정 이유.")
