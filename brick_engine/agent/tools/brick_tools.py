# ============================================================================
# CoScientist 에이전트 도구 스키마
# 브릭 재생성 에이전트(CoScientist)가 사용하는 도구들의 Pydantic 스키마 정의
# ============================================================================

from typing import List
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
    auto_remove_1x1: bool = Field(..., description="True면 1x1 브릭을 자동 삭제하여 안정성을 확보합니다. 디테일이 중요하다면 False로 설정하세요.")
    support_ratio: float = Field(..., description="지지 비율(0~1). 높을수록 안정적이지만 브릭 수가 증가합니다.")
    small_side_contact: bool = Field(..., description="작은 브릭의 사이드 접촉 허용 여부.")
    reasoning: str = Field(..., description="이 파라미터를 선택한 이유에 대한 간략한 설명.")


class RemoveBricks(BaseModel):
    """
    특정 브릭들을 삭제하여 안정성을 확보합니다.
    주의: 점수가 90점 이상이고, 소수의 공중부양(Floating) 브릭만 문제일 때 사용하세요.
    """
    brick_ids: List[str] = Field(..., description="삭제할 브릭 ID 목록 (예: ['3005.dat_0', '3024.dat_5'])")
    reasoning: str = Field(..., description="삭제 이유 (예: '점수 92점이나 공중부양 브릭 2개 발생하여 제거')")


class MergeBricks(BaseModel):
    """
    불안정한 브릭들을 구조적으로 보강하기 위해 병합 작업을 수행합니다.
    
    [동작 방식]
    1. 검증 결과에서 '불안정(Floating/Isolated)'으로 판명된 브릭을 식별합니다.
    2. 해당 브릭과 인접한 안정적인 브릭들을 1x1 단위로 분해합니다.
    3. 분해된 브릭들을 색상과 관계없이 가장 튼튼한 방향(X/Z축)으로 재조립(Merge)합니다.
    
    [주의사항]
    - 1x1 플레이트(3024)는 불안정하므로 생성하지 않습니다. 오직 브릭(Brick)으로만 병합합니다.
    - 1x1 브릭 비율이 높거나 구조적 결함이 있을 때 사용하세요.
    """
    strategy: str = Field("structural", description="[Deprecated] 병합 전략 (현재는 항상 'structural'로 고정되어 무시됨).")
    reasoning: str = Field(..., description="병합을 선택한 이유 (예: '1x1 브릭 과다 및 Floating 브릭 발생으로 구조 보강 필요')")
