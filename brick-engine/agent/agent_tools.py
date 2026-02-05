from typing import List, Optional, Literal
from pydantic import BaseModel, Field


# ============================================================================
# 에이전트 도구 스키마 정의
# LLM이 Function Calling으로 사용할 수 있는 도구들의 입출력 형식 정의
#
# NOTE: LLM은 검증/분석만 수행하고, LDR 직접 수정은 하지 않음.
#       개선은 알고리즘 재실행(TuneParameters)을 통해서만 이루어짐.
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
    support_ratio: float = Field(..., description="지지 비율(0~1). 높을수록 안정적이지만 브릭 수가 증가합니다.")
    small_side_contact: bool = Field(..., description="작은 브릭의 사이드 접촉 허용 여부.")
    reasoning: str = Field(..., description="이 파라미터를 선택한 이유에 대한 간략한 설명.")
