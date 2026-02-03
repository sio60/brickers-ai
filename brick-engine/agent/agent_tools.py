from typing import List, Optional, Literal
from pydantic import BaseModel, Field


# ============================================================================
# 에이전트 도구 스키마 정의
# LLM이 Function Calling으로 사용할 수 있는 도구들의 입출력 형식 정의
# ============================================================================

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
    auto_remove_1x1: bool = Field(..., description="True면 1x1 브릭을 자동 삭제하여 안정성을 확보합니다. 디테일이 중요하다면 False로 설정하세요.")
    reasoning: str = Field(..., description="이 파라미터를 선택한 이유에 대한 간략한 설명.")


class FixFloatingBricks(BaseModel):
    """
    물리 검증 결과 공중에 떠 있거나(Floating) 불안정한 브릭들을 삭제합니다.
    삭제 목록에 없는 브릭은 그대로 유지됩니다.
    """
    bricks_to_delete: List[str] = Field(..., description="삭제할 브릭의 ID 목록.")
    reasoning: str = Field(..., description="삭제 대상 선정 이유.")


class MergeBricks(BaseModel):
    """
    같은 색상의 인접 1x1 브릭들을 큰 브릭(1x2, 1x3, 1x4)으로 병합합니다.
    병합하면 연결이 강화되어 구조적 안정성이 향상됩니다.
    색상이 다른 브릭은 병합되지 않습니다.
    """
    target_brick_ids: Optional[List[str]] = Field(
        default=None, 
        description="병합할 대상 브릭 ID 목록. None이면 모든 1x1 브릭을 대상으로 합니다."
    )
    min_merge_count: int = Field(
        default=2, 
        description="최소 병합 개수 (2~4). 2면 1x2, 3이면 1x3, 4면 1x4 브릭으로 병합."
    )
    reasoning: str = Field(..., description="병합이 필요한 이유에 대한 설명.")

