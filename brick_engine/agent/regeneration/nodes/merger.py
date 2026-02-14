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
    구조적 병합 노드 (내부 루프 방식)

    알고리즘:
    1. brick_judge로 불안정 브릭 탐지
    2. 불안정 경계 분해 + X/Z 양방향 색상무관 재병합
    3. 병합 결과가 0이면 수렴 → 최종 검증으로 이동
    4. 병합 있으면 다시 검증 → 반복
    """
    from ...ldr_modifier import structural_merge

    print("\n[Merger] 구조적 병합 시작 (내부 루프)...")
    graph._log("MERGE", "작은 브릭들을 큰 브릭으로 병합하여 구조를 강화하고 있어요.")

    ldr_path = state['ldr_path']
    total_merged = 0
    total_split = 0
    rounds = 0

    try:
        for round_num in range(1, MAX_MERGE_ROUNDS + 1):
            rounds = round_num
            print(f"\n  📐 [Merge Round {round_num}/{MAX_MERGE_ROUNDS}]")

            # 1단계: 미니 검증 → 불안정 브릭 탐지
            try:
                from brick_judge import full_judge, parse_ldr_string

                with open(ldr_path, 'r', encoding='utf-8') as f:
                    ldr_content = f.read()

                model = parse_ldr_string(ldr_content)
                issues = full_judge(model)

                # 불안정 브릭 ID 수집 (floating + isolated + top_only)
                unstable_ids = []
                for issue in issues:
                    if issue.issue_type.value in ('floating', 'isolated', 'top_only'):
                        if issue.brick_id is not None:
                            unstable_ids.append(issue.brick_id)

                unstable_ids = list(set(unstable_ids))  # 중복 제거

                if not unstable_ids:
                    print(f"    ✅ 불안정 브릭 없음 → 수렴 완료!")
                    break

                print(f"    🔍 불안정 브릭 {len(unstable_ids)}개 발견")

            except Exception as e:
                print(f"    ⚠️ 미니 검증 실패: {e} → 병합 종료")
                break

            # 2단계: 구조적 병합 실행
            stats = structural_merge(ldr_path, unstable_ids)
            merged = stats.get("merged", 0)
            split = stats.get("split", 0)

            total_merged += merged
            total_split += split

            print(f"    📊 결과: 분해 {split}개, 병합 {merged}개 그룹")

            if merged == 0 and split == 0:
                print(f"    ⏹️ 더 이상 병합 불가 → 수렴 완료!")
                break

            graph._log("MERGE", f"병합 라운드 {round_num}: {merged}개 그룹 병합 완료")

        # 결과 메시지 생성
        if total_merged > 0:
            print(f"\n  ✅ 전체 병합 완료: {rounds}라운드, 분해 {total_split}개, 병합 {total_merged}개 그룹")
            graph._log("MERGE", f"병합 완료! {total_merged}개 그룹을 큰 브릭으로 통합했어요.")
            merge_msg = (
                f"[구조적 병합 완료] {rounds}라운드 수행.\n"
                f"- 분해: {total_split}개 큰 브릭 → 1x1로 분해\n"
                f"- 병합: {total_merged}개 그룹을 큰 브릭으로 재병합\n"
                f"최종 검증을 수행합니다."
            )
        else:
            print(f"\n  ℹ️ 병합 가능한 브릭 없음 (스킵)")
            graph._log("MERGE", "병합할 브릭이 없어서 바로 다음 단계로 넘어갈게요.")
            merge_msg = "[병합 결과] 병합 가능한 브릭이 없습니다. 최종 검증을 수행합니다."

        return {
            "merged": True,
            "messages": [HumanMessage(content=merge_msg)],
            "next_action": "verify"
        }

    except Exception as e:
        print(f"  ⚠️ 병합 중 오류: {e}")
        import traceback
        traceback.print_exc()
        graph._log("MERGE", "병합 중 문제가 생겼지만 다음 단계로 넘어갈게요.")
        return {
            "merged": True,
            "messages": [HumanMessage(content=f"[병합 오류] {e}. 병합 없이 최종 검증합니다.")],
            "next_action": "verify"
        }
