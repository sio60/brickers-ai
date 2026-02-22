# ============================================================================
# Verifier 노드: 물리 검증 (brick_judge 기반)
# ============================================================================

import os
import time
import traceback
from typing import Dict, Any

from langchain_core.messages import HumanMessage

from ...llm_state import VerificationFeedback
from ..feedback import format_feedback


def node_verifier(graph, state) -> Dict[str, Any]:
    """물리 검증 노드 (brick_judge 기반 - Rust 네이티브)"""
    from brick_judge import full_judge, calc_score_from_issues, parse_ldr_string

    print("\n[Verifier] 물리 검증 수행 중 (brick_judge)...")
    graph._log("VERIFY", "내구성과 조립 가능성을 확인 중이에요.")

    if not os.path.exists(state['ldr_path']):
        return {"messages": [HumanMessage(content="LDR 파일이 생성되지 않았습니다.")], "next_action": "model"}

    try:
        with open(state['ldr_path'], 'r', encoding='utf-8') as f:
            ldr_content = f.read()

        model = parse_ldr_string(ldr_content)
        total_bricks = len(model.bricks)

        # 1x1 브릭 비율 계산
        small_brick_parts = {"3005.dat", "3024.dat"}
        small_brick_count = sum(1 for b in model.bricks if b.name in small_brick_parts)
        small_brick_ratio = small_brick_count / total_bricks if total_bricks > 0 else 0.0

        # brick_judge로 물리 검증 수행
        issues = full_judge(model)

        # -- 부피 기반 점수 계산 로직 도입 --
        total_volume = 0.0
        problem_brick_ids = set()
        
        # 모델의 모든 브릭 부피의 합 계산
        for b in model.bricks:
            # 부피 = width * depth * height (각각 고유 ldu 단위)
            vol = b.width * b.depth * b.height
            total_volume += vol
            
        # unstable_base 제외한 이슈 브릭 선별
        for i in issues:
            if i.issue_type.value != 'unstable_base' and i.brick_id is not None:
                # top_only는 문제긴 하지만 무게에 큰 영향을 안주는 경우도 있음 (하지만 기존 설계상 문제로 간주)
                if i.issue_type.value != 'top_only':
                    problem_brick_ids.add(i.brick_id)

        problem_volume = 0.0
        # id 기준으로 다시 순회
        if problem_brick_ids:
            for b in model.bricks:
                if b.id in problem_brick_ids:
                    problem_volume += (b.width * b.depth * b.height)

        # 안정성 점수
        if total_volume > 0:
            score = int(100.0 * (1.0 - (problem_volume / total_volume)))
        else:
            score = 100
            
        score = max(0, min(100, score))

        # 이슈 분석 및 ID 수집
        floating_bricks = [i for i in issues if i.issue_type.value == 'floating']
        isolated_bricks = [i for i in issues if i.issue_type.value == 'isolated']
        top_only_bricks = [i for i in issues if i.issue_type.value == 'top_only']
        
        floating_count = len(floating_bricks)
        isolated_count = len(isolated_bricks)
        top_only_count = len(top_only_bricks)
        has_unstable_base = any(i.issue_type.value == 'unstable_base' for i in issues)

        # ID 목록 추출 (문자열 ID 포맷팅)
        floating_brick_ids = []
        for i in floating_bricks:
            if i.brick_id is not None and 0 <= i.brick_id < len(model.bricks):
                b = model.bricks[i.brick_id]
                floating_brick_ids.append(f"{b.name}_{i.brick_id}")

        # fallen_brick_ids는 현재 구조상 unstable_base가 났을 때 발생하는 것으로 가정하거나 빈 목록 처리
        fallen_brick_ids = [] 

        stable = (
            not has_unstable_base
            and floating_count == 0
            and isolated_count == 0
            # top_only는 안정성 판정에서 제외 (점수 계산과 일관성 유지)
        )

        # 피드백 생성
        feedback = VerificationFeedback(
            stable=stable,
            total_bricks=total_bricks,
            fallen_bricks=0,
            floating_bricks=floating_count,
            floating_brick_ids=floating_brick_ids,
            fallen_brick_ids=fallen_brick_ids,
            failure_ratio=(floating_count + isolated_count) / total_bricks if total_bricks > 0 else 0.0,  # top_only 제외
            stability_score=score,
            stability_grade="STABLE" if stable else ("MEDIUM" if score >= 50 else "UNSTABLE"),
            small_brick_count=small_brick_count,
            small_brick_ratio=small_brick_ratio
        )

        feedback_text = format_feedback(feedback)

        # 상태 메시지 결정
        if stable:
            short_status = "✅ 안정"
        elif floating_count > 0:
            short_status = f"❌ 불안정 (floating {floating_count}개)"
        elif top_only_count > 0:
            short_status = f"❌ 아래 지지 없음 (top_only {top_only_count}개)"
        elif has_unstable_base:
            short_status = "❌ 무게중심 불안정"
        else:
            short_status = "❌ 불안정"

        print(f"  결과: {short_status} (점수: {score}점)")

        if not stable:
            summary_text = feedback_text.replace('\n', ', ').replace('\r', '')
            if len(summary_text) > 200:
                summary_text = summary_text[:200] + "..."
            print(f"  요약: {summary_text}")

        # 현재 메트릭 저장
        budget = state['params'].get('budget', 200)
        current_metrics = {
            "failure_ratio": feedback.failure_ratio,
            "small_brick_ratio": small_brick_ratio,
            "small_brick_count": small_brick_count,
            "total_bricks": total_bricks,
            "floating_count": floating_count,
            "fallen_count": 0,
            "isolated_count": isolated_count,
            "top_only_count": top_only_count,
            "floating_brick_ids": floating_brick_ids,
            "fallen_brick_ids": fallen_brick_ids,
            "has_unstable_base": has_unstable_base,
            "stability_score": score,
            "budget_exceeded": total_bricks > budget,
            "target_budget": budget,
            "subject_name": state.get("subject_name", "Unknown Object"),
            "backend": "brick_judge_rs"
        }

        # 성공 여부 판단: top_only는 성공 판정에서 제외 (점수 계산과 일관성 유지)
        is_success = floating_count == 0 and isolated_count == 0
        is_over_budget = total_bricks > budget

        if is_success:
            base_warning = ""
            if has_unstable_base:
                base_warning = " (⚠️ 무게중심 불안정 경고 있음)"
            print(f"🎉 목표 달성! 프로세스를 종료합니다.{base_warning}")
            final_report = {
                "success": True,
                "total_attempts": state['attempts'],
                "tool_usage": state.get('tool_usage_count', {}),
                "final_metrics": current_metrics,
                "message": f"안정적인 구조물 생성 완료{base_warning}"
            }
            return {"next_action": "end", "final_report": final_report}

        # 1차 구조적 병합(Merger) 노드를 단계적으로 삭제했으므로 바로 reflect로 넘어감
        if not state.get('merged', False):
            # 병합 플래그만 변경해줌 (에이전트 판단 지원용)
            pass

        if state['attempts'] > state['max_retries']:
            print("💥 최대 시도 횟수 초과. 마지막 결과를 기록한 뒤 종료합니다.")
            final_report = {
                "success": False,
                "total_attempts": state['attempts'],
                "tool_usage": state.get('tool_usage_count', {}),
                "final_metrics": current_metrics,
                "message": "최대 시도 횟수 초과로 종료"
            }
            return {"next_action": "reflect", "final_report": final_report, "force_end": True}

        # LLM에게 전달할 메시지 보강
        custom_feedback = feedback_text

        if is_over_budget:
            custom_feedback += f"\n\n🚨 **중요: 예산 초과! 현재 {total_bricks}개 브릭입니다. 목표 예산은 {budget}개입니다. `TuneParameters` 도구를 사용하여 `target` 값을 줄여야 합니다.**"
        elif floating_count > 0:
            if score >= 90:
                custom_feedback += f"\n\n✅ **점수 {score}점으로 높음! 공중부양 브릭 {floating_count}개만 `RemoveBricks`로 삭제하면 성공입니다.**"
            else:
                custom_feedback += f"\n\n⚠️ **점수 {score}점으로 낮음. `TuneParameters`로 파라미터를 조정하여 구조를 개선하세요.**"
        elif top_only_count > 0:
            custom_feedback += (
                f"\n\n⚠️ **아래 지지 없는 브릭(top_only)이 {top_only_count}개 있습니다. "
                "위/아래 결합 기준을 만족하도록 `MergeBricks` 또는 `TuneParameters`로 아래 지지를 추가하세요.**"
            )

        if has_unstable_base:
            custom_feedback += "\n\n⚠️ **참고: 무게중심이 지지면을 벗어났습니다. 가능하면 구조를 더 안정적으로 만드세요.**"

        # [FIX] brick_judge_rs는 brick_id를 integer (index)로 반환함.
        # MergeBricks 등 다른 도구들은 "{part}_{index}" 형태의 문자열 ID를 기대함.
        formatted_issues = []
        for i in issues:
            if i.brick_id is not None and 0 <= i.brick_id < len(model.bricks):
                b = model.bricks[i.brick_id]
                string_id = f"{b.name}_{i.brick_id}"
                formatted_issues.append({"type": i.issue_type.value, "brick_id": string_id})
            else:
                formatted_issues.append({"type": i.issue_type.value, "brick_id": None})

        return {
            "verification_raw_result": {"issues": formatted_issues},
            "messages": [HumanMessage(content=custom_feedback)],
            "current_metrics": current_metrics,
            "next_action": "reflect"
        }

    except Exception as e:
        print(f"  ❌ 검증 중 에러: {e}")
        traceback.print_exc()

        verification_errors = state.get('verification_errors', 0) + 1
        if verification_errors >= 3:
            print(f"  ⚠️ 검증 에러 {verification_errors}회 - 재생성으로 전환합니다.")
            return {
                "messages": [HumanMessage(content=f"검증 시스템 에러가 반복됨: {e}")],
                "verification_errors": 0,
                "next_action": "model"
            }
        else:
            print(f"  🔄 검증 재시도 ({verification_errors}/3)...")
            time.sleep(1)
            return {"verification_errors": verification_errors, "next_action": "verifier"}
