# ============================================================================
# Model 노드: LLM이 상황 분석 + 도구 선택
# ============================================================================

import re
import time
from typing import Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from ..prompts import STRATEGY_GUIDE, build_stability_hint
from ..rag_ranker import rerank_and_filter_cases


def node_model(graph, state) -> Dict[str, Any]:
    """LLM이 상황을 분석하고 도구를 선택하는 노드"""
    from ...agent_tools import TuneParameters, RemoveBricks, MergeBricks
    from ...memory_utils import memory_manager

    print("\n[Co-Scientist] 상황 분석 중...")
    graph._log("ANALYZE", "불필요한 복잡성이 있는지 검토하고 있어요.")

    tools = [RemoveBricks, MergeBricks] # TuneParameters 일시 비활성화


    messages_to_send = state['messages'][:]
    
    # --- [New] 1x1 브릭 비율 분석 및 MergeBricks 권장 로직 ---
    ldr_path = state.get('ldr_path')
    merged_flag = state.get('merged', False)
    
    if ldr_path and not merged_flag:
        try:
            total_bricks = 0
            small_bricks = 0
            with open(ldr_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip().startswith('1 '): # 브릭 정의 라인
                        total_bricks += 1
                        if "3005.dat" in line:
                            small_bricks += 1
            
            if total_bricks > 0:
                ratio = small_bricks / total_bricks
                if ratio > 0.2: # 20% 이상이면 경고
                    warning_msg = (
                        f"⚠️ **1x1 브릭 비율 경고 ({ratio*100:.1f}%)**\n"
                        f"현재 구조물에 1x1 브릭이 너무 많아 구조적 안정성이 떨어질 위험이 큽니다.\n"
                        f"👉 **`MergeBricks` 도구를 사용하여 불안정 부위를 구조적으로 보강하세요.**"
                    )
                    messages_to_send.append(SystemMessage(content=warning_msg))
                    print(f"  [Hint] 1x1 ratio {ratio:.2f} > 0.2 -> MergeBricks 추천")
        except Exception as e:
            print(f"  [Error] LDR 분석 중 오류: {e}")

    # --- 전략 가이드 주입 ---
    messages_to_send.append(SystemMessage(content=STRATEGY_GUIDE))

    # --- Memory 정보 주입 (RAG) ---
    last_human_msg = next((m for m in reversed(messages_to_send) if isinstance(m, HumanMessage)), None)
    subject_prefix = f"[{state.get('subject_name', 'Object')}] "
    current_observation = subject_prefix + (last_human_msg.content if last_human_msg else "")

    if memory_manager:
        verification_metrics = state.get("verification_result")
        raw_cases = memory_manager.search_similar_cases(
            current_observation,
            limit=10,
            min_score=0.4,
            verification_metrics=verification_metrics,
            subject_name=state.get("subject_name", "Object")
        )
        similar_cases = rerank_and_filter_cases(graph.default_client, current_observation, raw_cases)

        if similar_cases:
            memory_info = "\n**📚 유사한 과거 실험 사례 (RAG):**\n"
            for i, case in enumerate(similar_cases, 1):
                exp = case.get('experiment', {})
                ver = case.get('verification', {})
                imp = case.get('improvement', {})

                metrics = ver.get('metrics_after', ver)
                vol = metrics.get('total_volume', 0)
                dims = metrics.get('dimensions', {})
                dim_str = f"{dims.get('width', 0):.0f}x{dims.get('height', 0):.0f}x{dims.get('depth', 0):.0f}" if dims else "N/A"

                tool = exp.get('tool', 'Unknown')
                result = ver.get('numerical_analysis', 'N/A')
                lesson = imp.get('lesson_learned', 'No lesson')
                outcome = "성공" if case.get('result_success') else "실패"
                score = case.get('similarity_score', 0)
                rel = case.get('reliability_grade', 'Low')

                memory_info += f"[{i}] {outcome} 사례 (신뢰도: {rel}, 유사도: {score:.2f})\n"
                memory_info += f"    - 물리 특성: 부피 {vol:.1f}, 크기 {dim_str}, 브릭 {metrics.get('total_bricks', 0)}개\n"
                memory_info += f"    - 도구: {tool} -> 결과: {result}\n"
                memory_info += f"    - 교훈: {lesson}\n"

            memory_info += "\n위 부피와 형태적 유사성을 고려하여 최적의 파라미터를 결정하세요.\n"
            messages_to_send.append(SystemMessage(content=memory_info))
            print(f"  📚 RAG 검색 결과 {len(similar_cases)}건 주입됨")

    # Legacy Memory (Fallback)
    memory = state.get('memory', {})
    lessons = memory.get('lessons', [])
    failed_approaches = memory.get('failed_approaches', [])

    if lessons or failed_approaches:
        memory_info = "\n**📚 이전 경험 (Memory):**\n"
        if lessons:
            memory_info += "- 최근 교훈: " + "; ".join(lessons[-3:]) + "\n"
        if failed_approaches:
            memory_info += "- 피해야 할 접근법: " + "; ".join(failed_approaches[-3:]) + "\n"

        messages_to_send.append(SystemMessage(content=memory_info))
        print(f"  📚 Memory 정보 {len(lessons)}개 교훈 전달됨")

    # 직전 검증 결과에서 안정성 등급 파싱 → 힌트 주입
    target_msg = None
    for msg in reversed(messages_to_send):
        if isinstance(msg, HumanMessage) and "검증 결과" in str(msg.content):
            target_msg = msg
            break

    if target_msg:
        content = str(target_msg.content)
        grade_match = re.search(r"안정성 등급: \S+ \((\w+)\)", content)
        score_match = re.search(r"점수:\s*(\d+)", content)

        grade = grade_match.group(1) if grade_match else "UNKNOWN"
        score = int(score_match.group(1)) if score_match else 0

        hint = build_stability_hint(grade, score)
        if hint:
            if score >= 90:
                print(f"  💡 [Strategy Hint] 🌟 안정 (점수: {score}) -> 잔존물 삭제 모드")
            elif grade == "UNSTABLE":
                print(f"  💡 [Strategy Hint] 불안정 (점수: {score}) -> 파라미터 대폭 변경 필요")
            elif grade == "MEDIUM":
                print(f"  💡 [Strategy Hint] 중간 (점수: {score}) -> 파라미터 소폭 조정 필요")
            messages_to_send.append(SystemMessage(content=hint))

    # 모델 호출
    try:
        client_to_use = graph.gemini_client
        print(f"  🤖 Active Model: Gemini-2.5-Flash (Fixed)")

        model_with_tools = client_to_use.bind_tools(tools)
        response = model_with_tools.invoke(messages_to_send)

        if response.tool_calls:
            tc = response.tool_calls[0]
            tool_name = tc['name']
            print(f"  🔨 도구 선택: {[tc['name'] for tc in response.tool_calls]}")

            if tool_name == "RemoveBricks":
                graph._log("MODEL", "구조가 거의 완성되었습니다! 불안정한 브릭들만 핀셋으로 도려낼게요.")
            # elif tool_name == "TuneParameters":
            #     graph._log("MODEL", "현재 파라미터로는 한계가 있네요. 새로운 관점에서 설계를 다시 시도해 보겠습니다.")
            elif tool_name == "MergeBricks":
                graph._log("MODEL", "브릭이 너무 조각나 있네요. 튼튼한 구조로 합병 작업을 진행합니다.")

            return {"messages": [response], "next_action": "tool"}
        else:
            print(f"  💭 LLM 의견: {response.content}")

            current_metrics = state.get('current_metrics', {})
            floating_count = current_metrics.get('floating_count', 0)
            failure_ratio = current_metrics.get('failure_ratio', 0)

            if floating_count == 0 and failure_ratio <= state['acceptable_failure_ratio']:
                print("🎉 모든 조건 충족. 종료합니다.")
                return {"messages": [response], "next_action": "end"}
            else:
                print(f"⚠️ 경고: 문제가 남았는데({floating_count}개 공중부양) 종료 시도함. 재지시 중...")
                error_feedback = f"아직 완료되지 않았습니다. {floating_count}개의 공중부양 브릭이 남아있습니다. MergeBricks 또는 RemoveBricks를 사용하여 구조를 수정하세요."
                hint = HumanMessage(content=error_feedback)
                return {"messages": [response, hint], "next_action": "model"}

    except Exception as e:
        print(f"  ⚠️ LLM 호출 에러: {e}")
        if "429" in str(e):
            print("  💤 API 할당량 초과. 잠시 대기 후 재시도합니다...")
            time.sleep(10)
            return {"next_action": "model"}
        return {"next_action": "end"}
