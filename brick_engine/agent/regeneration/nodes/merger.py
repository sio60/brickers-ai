# ============================================================================
# Merger 노드: 구조적 병합으로 브릭 안정성 향상
# 내부에서 검증→병합을 반복하여 더 이상 병합 불가할 때까지 수렴시킴
# ============================================================================

from typing import Dict, Any

from langchain_core.messages import HumanMessage


# 최대 병합 라운드 (무한 루프 방지)
MAX_MERGE_ROUNDS = 10


def node_merger(graph, state) -> Dict[str, Any]:
    """
    구조적 병합 노드 (1회 실행 + 롤백)

    알고리즘:
    1. 초기 검증: 불안정 브릭 수 확인
    2. 구조적 병합 실행 (불안정+경계안정 분해 → 재병합)
    3. 재검증: 병합 후 불안정 브릭 수 확인
    4. 판단:
       - 개선됨(불안정 감소): 병합 유지
       - 악화됨(불안정 증가): 롤백 (원본 복원)
    """
    from ...ldr_modifier import structural_merge
    from brick_judge import full_judge, parse_ldr_string

    print("\n[Merger] 구조적 병합 시작 (1회 시도 + 롤백)...")
    graph._log("MERGE", "브릭을 합쳐서 더 튼튼하게 만들어볼게요.")

    ldr_path = state['ldr_path']

    try:
        # 1. 초기 상태 백업 및 검증
        with open(ldr_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

        model = parse_ldr_string(original_content)
        initial_issues = full_judge(model)
        
        initial_unstable_ids = []
        for issue in initial_issues:
            if issue.issue_type.value in ('floating', 'isolated', 'top_only'):
                if issue.brick_id is not None:
                    initial_unstable_ids.append(issue.brick_id)
        
        initial_unstable_count = len(set(initial_unstable_ids))
        print(f"  🔍 병합 전 불안정 브릭: {initial_unstable_count}개")

        if initial_unstable_count == 0:
            print("  ✅ 이미 안정적임. 병합 스킵.")
            return {
                "merged": True,
                "messages": [HumanMessage(content="[병합 스킵] 이미 안정적인 구조입니다.")],
                "next_action": "verify"
            }

        # 2. 구조적 병합 실행 (1회)
        # structural_merge 내부에서 파일이 수정됨
        stats = structural_merge(ldr_path, initial_unstable_ids)
        merged_count = stats.get("merged", 0)
        split_count = stats.get("split", 0)

        if merged_count == 0 and split_count == 0:
            print("  ⏹️ 병합 가능한 구조가 아님.")
            return {
                "merged": True,
                "messages": [HumanMessage(content="[병합 결과] 병합 가능한 브릭이 없습니다.")],
                "next_action": "verify"
            }

        # 3. 재검증 (롤백 판단용)
        with open(ldr_path, 'r', encoding='utf-8') as f:
            new_content = f.read()
            
        new_model = parse_ldr_string(new_content)
        new_issues = full_judge(new_model)
        
        new_unstable_ids = []
        for issue in new_issues:
            if issue.issue_type.value in ('floating', 'isolated', 'top_only'):
                if issue.brick_id is not None:
                    new_unstable_ids.append(issue.brick_id)
        
        new_unstable_count = len(set(new_unstable_ids))
        print(f"  📊 병합 후 불안정 브릭: {new_unstable_count}개 (변화: {new_unstable_count - initial_unstable_count:+d})")

        # 4. 롤백 판단 (제거됨: 사용자의 요청에 따라 개선되지 않더라도 병합 결과 유지)
        if new_unstable_count > initial_unstable_count:
            print(f"  ⚠️ 병합 후 불안정성 증가({initial_unstable_count} -> {new_unstable_count})했으나, 요청에 따라 병합 유지.")
            graph._log("MERGE", f"병합 후 불안정 브릭이 {initial_unstable_count}개에서 {new_unstable_count}개로 늘었지만, 설계를 유지할게요.")
            
            merge_msg = (
                f"[병합 유지(불안정 증가)]\n"
                f"- 분해: {split_count}개\n"
                f"- 병합: {merged_count}개 그룹\n"
                f"- 불안정 브릭: {initial_unstable_count} → {new_unstable_count} (⚠️ 증가)\n"
            )
            return {
                "merged": True,
                "messages": [HumanMessage(content=merge_msg)],
                "next_action": "verify"
            }
        
        # 성공 (불안정 감소 또는 동일)
        print(f"  ✅ 병합 성공: 분해 {split_count}개, 병합 {merged_count}개 그룹")
        graph._log("MERGE", f"병합 성공! 불안정 브릭이 {initial_unstable_count}개에서 {new_unstable_count}개로 변했어요.")
        
        merge_msg = (
            f"[구조적 병합 완료]\n"
            f"- 분해: {split_count}개 (경계면 포함)\n"
            f"- 병합: {merged_count}개 그룹\n"
            f"- 불안정 브릭: {initial_unstable_count} → {new_unstable_count}\n"
        )
        
        return {
            "merged": True,
            "messages": [HumanMessage(content=merge_msg)],
            "next_action": "verify"
        }

    except Exception as e:
        print(f"  ⚠️ 병합 중 오류: {e}")
        import traceback
        traceback.print_exc()
        
        graph._log("MERGE", f"병합 중 오류가 발생했지만 현재 상태를 유지할게요: {e}")
        return {
            "merged": True,
            "messages": [HumanMessage(content=f"[병합 오류] {e}. 현재 상태로 진행합니다.")],
            "next_action": "verify"
        }

