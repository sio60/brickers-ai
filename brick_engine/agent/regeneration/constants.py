# ============================================================================
# 브릭 변환 기본 파라미터
# ============================================================================

DEFAULT_PARAMS = {
    "target": 25,              # 목표 스터드 크기
    "min_target": 5,           # 최소 스터드 크기 (25 -> 5로 완화)
    "budget": 200,             # 최대 브릭 수 (Kids 기본)
    "shrink": 0.6,             # 축소 비율 (0.8 -> 0.6)
    "search_iters": 10,        # 이진 탐색 반복 횟수
    "flipx180": False,         # X축 180도 회전
    "flipy180": False,         # Y축 180도 회전
    "flipz180": False,         # Z축 180도 회전
    "kind": "brick",           # 브릭 종류 (brick/plate)
    "plates_per_voxel": 3,     # 복셀당 플레이트 수
    "interlock": True,         # 인터락 활성화
    "max_area": 20,            # 최대 영역
    "solid_color": 4,          # 단색 색상 ID
    "use_mesh_color": True,    # 메시 색상 사용
    "invert_y": False,         # Y축 반전
    "smart_fix": True,         # 스마트 보정 활성화
    # 추가 파라미터 (Legacy Match)
    "span": 4,
    "max_new_voxels": 12000,
    "refine_iters": 8,
    "ensure_connected": True,
    "min_embed": 2,
    "erosion_iters": 1,        # 노이즈 제거
    "fast_search": True,
    "extend_catalog": True,
    "max_len": 8,
    "fill": True,              # 내부 채움 활성화 (안정적 모델 생성)
    "step_order": "bottomup",  # 조립 순서
    "auto_remove_1x1": True,   # 기본값: 안전하게 1x1 삭제
    "support_ratio": 0.3,      # 기본 지지 비율 복구
    "small_side_contact": True, # 작은 브릭 사이드 접촉 허용
}
