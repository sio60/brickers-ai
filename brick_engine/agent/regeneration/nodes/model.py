# ============================================================================
# Model 노드: LLM이 상황 분석 + 도구 선택
# 리팩토링: 단일 거대 함수를 역할별 헬퍼 함수로 분리
# ============================================================================

import re
import time
import logging
from typing import Dict, Any, Optional, List

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from ..prompts import STRATEGY_GUIDE, build_stability_hint
from ..rag_ranker import rerank_and_filter_cases
from ..constants import MAX_TOOL_USES, MAX_REMOVE_FALLBACK

logger = logging.getLogger("agent.regeneration.model")


# ---------------------------------------------------------------------------
# 헬퍼 함수: 점수 기반 도구 강제 선택 전략
# ---------------------------------------------------------------------------
def _select_forced_tool(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    점수 기반 도구 강제 선택 전략.
    
    Returns:
        {"forced_tool": str, "tools": list, "end": bool} 또는 None
    """
    from ...agent_tools import TuneParameters, RemoveBricks, MergeBricks

    tool_usage_count = state.get('tool_usage_count', {})
    current_metrics = state.get('current_metrics', {})
    score = current_metrics.get('score', 0)

    tune_used = tool_usage_count.get('TuneParameters', 0)
    merge_used = tool_usage_count.get('MergeBricks', 0)
    remove_used = tool_usage_count.get('RemoveBricks', 0)

    if score >= 100:
        # 완벽한 점수는 도구 불필요
        return {"forced_tool": None, "tools": [TuneParameters, RemoveBricks, MergeBricks], "end": False}

    if score < 95:
        candidates = [
            ('TuneParameters', tune_used, MAX_TOOL_USES, [TuneParameters]),
            ('MergeBricks', merge_used, MAX_TOOL_USES, [MergeBricks]),
            ('RemoveBricks', remove_used, MAX_REMOVE_FALLBACK, [RemoveBricks]),
        ]
    else:  # 95 <= score < 100
        candidates = [
            ('MergeBricks', merge_used, MAX_TOOL_USES, [MergeBricks]),
            ('TuneParameters', tune_used, MAX_TOOL_USES, [TuneParameters]),
            ('RemoveBricks', remove_used, MAX_REMOVE_FALLBACK, [RemoveBricks]),
        ]

    for name, used, max_uses, tools in candidates:
        if used < max_uses:
            logger.info(f"🎯 [전략] 점수 {score}점 → {name} 강제 ({used+1}/{max_uses})")
            return {"forced_tool": name, "tools": tools, "end": False}

    logger.warning("⛔ [전략] 모든 도구 사용 한도 초과. 현재 상태로 종료합니다.")
    return {"forced_tool": None, "tools": [], "end": True}


# ---------------------------------------------------------------------------
# 헬퍼 함수: 1x1 브릭 비율 분석 → MergeBricks 권장 메시지 생성
# ---------------------------------------------------------------------------
def _analyze_brick_ratio(state: Dict[str, Any]) -> Optional[str]:
    """1x1 브릭 비율이 20% 이상이면 경고 메시지를 반환."""
    ldr_path = state.get('ldr_path')
    merged_flag = state.get('merged', False)

    if not ldr_path or merged_flag:
        return None

    try:
        total_bricks = 0
        small_bricks = 0
        with open(ldr_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip().startswith('1 '):  # 브릭 정의 라인
                    total_bricks += 1
                    if "3005.dat" in line:
                        small_bricks += 1

        if total_bricks > 0:
            ratio = small_bricks / total_bricks
            if ratio > 0.2:  # 20% 이상이면 경고
                logger.info(f"[Hint] 1x1 ratio {ratio:.2f} > 0.2 -> MergeBricks 추천")
                return (
                    f"⚠️ **1x1 브릭 비율 경고 ({ratio*100:.1f}%)**\n"
                    f"현재 구조물에 1x1 브릭이 너무 많아 구조적 안정성이 떨어질 위험이 큽니다.\n"
                    f"👉 **`MergeBricks` 도구를 사용하여 불안정 부위를 구조적으로 보강하세요.**"
                )
    except Exception as e:
        logger.error(f"LDR 분석 중 오류: {e}")

    return None


# ---------------------------------------------------------------------------
# 헬퍼 함수: RAG 메모리 검색 + 컨텍스트 주입
# ---------------------------------------------------------------------------
def _inject_rag_context(
    graph, state: Dict[str, Any], messages: List, current_observation: str
) -> None:
    """RAG 기반 유사 사례 검색 후 메시지에 주입."""
    from ...memory_utils import memory_manager

    if not memory_manager:
        return

    verification_metrics = state.get("verification_result")
    raw_cases = memory_manager.search_similar_cases(
        current_observation,
        limit=10,
        min_score=0.4,
        verification_metrics=verification_metrics,
        subject_name=state.get("subject_name", "Object"),
    )
    similar_cases = rerank_and_filter_cases(
        graph.default_client, current_observation, raw_cases
    )

    if not similar_cases:
        return

    memory_info = "\n**📚 유사한 과거 실험 사례 (RAG):**\n"
    for i, case in enumerate(similar_cases, 1):
        exp = case.get('experiment', {})
        ver = case.get('verification', {})
        imp = case.get('improvement', {})

        metrics = ver.get('metrics_after', ver)
        vol = metrics.get('total_volume', 0)
        dims = metrics.get('dimensions', {})
        dim_str = (
            f"{dims.get('width', 0):.0f}x{dims.get('height', 0):.0f}x{dims.get('depth', 0):.0f}"
            if dims else "N/A"
        )

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
    messages.append(SystemMessage(content=memory_info))
    logger.info(f"📚 RAG 검색 결과 {len(similar_cases)}건 주입됨")


# ---------------------------------------------------------------------------
# 헬퍼 함수: Legacy Memory 주입
# ---------------------------------------------------------------------------
def _inject_legacy_memory(state: Dict[str, Any], messages: List) -> None:
    """레거시 메모리(lessons, failed_approaches)를 메시지에 주입."""
    if memory_manager:
        verification_metrics = state.get("verification_result")
        
        # Native vector search is already sorted by similarity score. 
        # Skipping LLM reranking to save ~10s of latency.
        raw_cases = memory_manager.search_similar_cases(
            current_observation,
            limit=3, # limit to 3 directly
            min_score=0.4,
            verification_metrics=verification_metrics,
            subject_name=state.get("subject_name", "Object")
        )
        similar_cases = raw_cases

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

    if not lessons and not failed_approaches:
        return

    memory_info = "\n**📚 이전 경험 (Memory):**\n"
    if lessons:
        memory_info += "- 최근 교훈: " + "; ".join(lessons[-3:]) + "\n"
    if failed_approaches:
        memory_info += "- 피해야 할 접근법: " + "; ".join(failed_approaches[-3:]) + "\n"

    messages.append(SystemMessage(content=memory_info))
    logger.info(f"📚 Memory 정보 {len(lessons)}개 교훈 전달됨")


# ---------------------------------------------------------------------------
# 헬퍼 함수: 안정성 등급 힌트 주입
# ---------------------------------------------------------------------------
def _inject_stability_hint(messages: List) -> None:
    """직전 검증 결과에서 안정성 등급을 파싱하여 힌트를 주입."""
    target_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and "검증 결과" in str(msg.content):
            target_msg = msg
            break

    if not target_msg:
        return

    content = str(target_msg.content)
    grade_match = re.search(r"안정성 등급: \S+ \((\w+)\)", content)
    score_match = re.search(r"점수:\s*(\d+)", content)

    grade = grade_match.group(1) if grade_match else "UNKNOWN"
    score = int(score_match.group(1)) if score_match else 0

    hint = build_stability_hint(grade, score)
    if hint:
        if score >= 90:
            logger.info(f"💡 [Strategy Hint] 🌟 안정 (점수: {score}) -> 잔존물 삭제 모드")
        elif grade == "UNSTABLE":
            logger.info(f"💡 [Strategy Hint] 불안정 (점수: {score}) -> 파라미터 대폭 변경 필요")
        elif grade == "MEDIUM":
            logger.info(f"💡 [Strategy Hint] 중간 (점수: {score}) -> 파라미터 소폭 조정 필요")
        messages.append(SystemMessage(content=hint))


# ---------------------------------------------------------------------------
# 헬퍼 함수: LLM 응답 후 도구 미선택 시 분기 처리
# ---------------------------------------------------------------------------
def _handle_no_tool_response(response, state: Dict[str, Any]) -> Dict[str, Any]:
    """LLM이 도구를 선택하지 않았을 때의 분기 처리."""
    logger.info(f"💭 LLM 의견: {response.content}")

    current_metrics = state.get('current_metrics', {})

    # 검증 메트릭이 없으면 아직 검증이 안 된 상태 → 종료하면 안 됨
    if not current_metrics:
        logger.warning("⚠️ 검증 메트릭이 없습니다. 도구 선택을 재지시합니다.")
        hint = HumanMessage(
            content="아직 검증 결과가 없습니다. 반드시 RemoveBricks 또는 MergeBricks 도구를 사용하여 구조를 최적화하세요."
        )
        return {"messages": [response, hint], "next_action": "model"}

    floating_count = current_metrics.get('floating_count', 0)
    failure_ratio = current_metrics.get('failure_ratio', 0)

    if floating_count == 0 and failure_ratio <= state['acceptable_failure_ratio']:
        logger.info("🎉 모든 조건 충족. 종료합니다.")
        return {"messages": [response], "next_action": "end"}

    logger.warning(
        f"⚠️ 문제가 남았는데({floating_count}개 공중부양) 종료 시도함. 도구 선택을 재지시합니다."
    )
    error_feedback = (
        f"아직 완료되지 않았습니다. {floating_count}개의 공중부양 브릭이 남아있습니다. "
        f"반드시 RemoveBricks 또는 MergeBricks 도구를 사용하여 구조를 수정하세요. "
        f"도구 없이 종료할 수 없습니다."
    )
    hint = HumanMessage(content=error_feedback)
    return {"messages": [response, hint], "next_action": "model"}


# ---------------------------------------------------------------------------
# 메인 노드 함수
# ---------------------------------------------------------------------------
def node_model(graph, state) -> Dict[str, Any]:
    """LLM이 상황을 분석하고 도구를 선택하는 노드"""
    from ...agent_tools import TuneParameters, RemoveBricks, MergeBricks

    logger.info("\n[Co-Scientist] 상황 분석 중...")
    graph._log("ANALYZE", "불필요한 복잡성이 있는지 검토하고 있어요.")

    # ── 1단계: 점수 기반 도구 강제 선택 전략 ──
    selection = _select_forced_tool(state)
    if selection["end"]:
        return {"messages": state['messages'], "next_action": "end"}

    forced_tool = selection["forced_tool"]
    tools = selection["tools"] if forced_tool else [TuneParameters, RemoveBricks, MergeBricks]

    # ── 2단계: 메시지 준비 ──
    messages_to_send = state['messages'][:]

    # 강제 도구 힌트 주입
    if forced_tool:
        tool_usage_count = state.get('tool_usage_count', {})
        force_hint = (
            f"\n🔧 **필수 지시사항**: 현재 점수({state.get('current_metrics', {}).get('score', 0)}점) "
            f"기준으로 반드시 `{forced_tool}` 도구를 사용하세요. "
            f"다른 도구를 선택하지 마세요. 도구 사용 현황: "
            f"TuneParameters={tool_usage_count.get('TuneParameters', 0)}/{MAX_TOOL_USES}, "
            f"MergeBricks={tool_usage_count.get('MergeBricks', 0)}/{MAX_TOOL_USES}, "
            f"RemoveBricks={tool_usage_count.get('RemoveBricks', 0)}/{MAX_REMOVE_FALLBACK}"
        )
        messages_to_send.append(SystemMessage(content=force_hint))

    # ── 3단계: 1x1 브릭 비율 분석 ──
    brick_warning = _analyze_brick_ratio(state)
    if brick_warning:
        messages_to_send.append(SystemMessage(content=brick_warning))

    # ── 4단계: 전략 가이드 주입 ──
    messages_to_send.append(SystemMessage(content=STRATEGY_GUIDE))

    # ── 5단계: RAG 메모리 주입 ──
    last_human_msg = next(
        (m for m in reversed(messages_to_send) if isinstance(m, HumanMessage)), None
    )
    subject_prefix = f"[{state.get('subject_name', 'Object')}] "
    current_observation = subject_prefix + (last_human_msg.content if last_human_msg else "")

    _inject_rag_context(graph, state, messages_to_send, current_observation)

    # ── 6단계: Legacy Memory 주입 ──
    _inject_legacy_memory(state, messages_to_send)

    # ── 7단계: 안정성 힌트 주입 ──
    _inject_stability_hint(messages_to_send)

    # ── 8단계: LLM 호출 ──
    try:
        client_to_use = graph.gemini_client
        logger.info("🤖 Active Model: Gemini-2.5-Flash (Fixed)")
    if target_msg:
        content = str(target_msg.content)
        grade_match = re.search(r"안정성 등급: \S+ \((\w+)\)", content)
        score_match = re.search(r"점수:\s*(\d+)", content)

        grade = grade_match.group(1) if grade_match else "UNKNOWN"
        score = int(score_match.group(1)) if score_match else 0

    # --- [Loop Control] 시도 횟수 제한 확인 ---
    mod_attempts = state.get('modification_attempts', 0)
    hal_count = state.get('hallucination_count', 0)

    if mod_attempts >= 3:
        print(f"  🛑 [Stop] 최대 수정 시도 횟수(3회) 도달. 최적 결과 복원 단계로 이동합니다.")
        return {"next_action": "end"}

    if hal_count >= 3:
        print(f"  🛑 [Stop] 존재하지 않는 도구 반복 호출(3회) 감지. 강제 종료합니다.")
        return {"next_action": "end"}

    # 힌트 주입
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
        # [FIX] 하드코딩된 이름 대신 실제 모델명을 표시
        print(f"  🤖 Active Model: {client_to_use.model_name}")

        try:
            model_with_tools = client_to_use.bind_tools(tools)
            response = model_with_tools.invoke(messages_to_send)
        except Exception as e:
            if "400" in str(e) and "thought_signature" in str(e):
                print(f"  ⚠️ Gemini 400 (thought_signature) 에러 발생. 수동 파싱 폴백을 시도합니다.")
                # 도구 호출을 JSON 형식으로 요청하는 시스템 메시지 추가
                fallback_prompt = (
                    "\n\n[FALLBACK] 현재 API 오류로 인해 도구 호출 기능을 직접 사용할 수 없습니다. "
                    "대신 다음 JSON 형식으로 사용할 도구를 답변의 처음에 명시해주세요.\n"
                    "예: {\"tool\": \"MergeBricks\", \"args\": {\"reasoning\": \"...\"}}\n"
                    "또는 {\"tool\": \"RemoveBricks\", \"args\": {\"brick_ids\": [\"...\"], \"reasoning\": \"...\"}}"
                )
                messages_to_send.append(SystemMessage(content=fallback_prompt))
                
                # 도구 없이 일반 텍스트로 생성 요청
                response = client_to_use.invoke(messages_to_send)
                
                # 응답에서 JSON 추출 시도
                import json
                try:
                    # JSON 블록(```json ... ```) 또는 중괄호 내용 추출
                    json_match = re.search(r'(\{.*?"tool".*?\})', response.content, re.DOTALL)
                    if json_match:
                        tool_data = json.loads(json_match.group(1))
                        tool_name = tool_data.get("tool")
                        args = tool_data.get("args", {})
                        
                        # AIMessage의 tool_calls 구조 모사
                        from langchain_core.messages import ToolCall
                        response.tool_calls = [{
                            "name": tool_name,
                            "args": args,
                            "id": "manual_fallback_" + str(int(time.time())),
                            "type": "tool_call"
                        }]
                        print(f"  🛠️ 수동 도구 추출 성공: {tool_name}")
                except Exception as ex:
                    print(f"  ❌ 수동 도구 파싱 실패: {ex}")
            else:
                raise e

        if response.tool_calls:
            tc = response.tool_calls[0]
            tool_name = tc['name']

            # 강제 도구가 지정되었는데 LLM이 다른 도구를 선택한 경우 → 강제 교정
            if forced_tool and tool_name != forced_tool:
                logger.warning(f"⚠️ LLM이 {tool_name}을 선택했으나 전략상 {forced_tool}로 교정합니다.")
                tc['name'] = forced_tool

            logger.info(f"🔨 도구 선택: {[tc['name'] for tc in response.tool_calls]}")

            # SSE 로그 메시지
            tool_log_msgs = {
                "RemoveBricks": "구조가 거의 완성되었습니다! 불안정한 브릭들만 핀셋으로 도려낼게요.",
                "TuneParameters": "현재 파라미터로는 한계가 있네요. 새로운 관점에서 설계를 다시 시도해 보겠습니다.",
                "MergeBricks": "브릭이 너무 조각나 있네요. 튼튼한 구조로 합병 작업을 진행합니다.",
            }
            log_msg = tool_log_msgs.get(tc['name'], "도구를 실행합니다.")
            graph._log("MODEL", log_msg)

            return {"messages": [response], "next_action": "tool"}
        else:
            return _handle_no_tool_response(response, state)
            print(f"  💭 LLM 의견: {response.content}")

            current_metrics = state.get('current_metrics', {})
            # current_metrics가 비어있으면 아직 검증이 안 된 상태이므로 종료하면 안 됨
            if not current_metrics:
                print("⚠️ 검증 메트릭이 없습니다. 도구 선택을 재지시합니다.")
                hint = HumanMessage(content="아직 검증 결과가 없습니다. 반드시 RemoveBricks 또는 MergeBricks 도구를 사용하여 구조를 최적화하세요.")
                return {"messages": [response, hint], "next_action": "model"}

            floating_count = current_metrics.get('floating_count', 0)
            failure_ratio = current_metrics.get('failure_ratio', 0)

            if floating_count == 0 and failure_ratio <= state['acceptable_failure_ratio']:
                print("🎉 모든 조건 충족. 종료합니다.")
                return {"messages": [response], "next_action": "end"}
            else:
                # [FIX] 무한 루프 방지: 도구 선택 재지시 횟수 제한
                retry_count = sum(1 for m in messages_to_send if "아직 완료되지 않았습니다" in str(m.content))
                if retry_count >= 2:
                    print("⚠️ 무한 루프 감지: 도구 선택을 2회 이상 거부함. 강제 종료합니다.")
                    return {"messages": [response], "next_action": "end"}

                print(f"⚠️ 경고: 문제가 남았는데({floating_count}개 공중부양) 종료 시도함. 도구 선택을 재지시합니다.")
                error_feedback = f"아직 완료되지 않았습니다. {floating_count}개의 공중부양 브릭이 남아있습니다. 반드시 RemoveBricks 또는 MergeBricks 도구를 사용하여 구조를 수정하세요. 도구 없이 종료할 수 없습니다. (현재 {retry_count+1}차 경고)"
                hint = HumanMessage(content=error_feedback)
                return {"messages": [response, hint], "next_action": "model"}

    except Exception as e:
        error_str = str(e)
        logger.error(f"⚠️ LLM 호출 에러: {e}")

        # 연속 에러 카운트 추적
        llm_errors = state.get('verification_errors', 0) + 1

        if "429" in error_str:
            logger.warning("💤 API 할당량 초과. 잠시 대기 후 재시도합니다...")
            time.sleep(10)
            return {"verification_errors": llm_errors, "next_action": "model"}
        
        if "400" in error_str and "thought_signature" in error_str:
            # Gemini 3.x thought_signature 호환성 에러 → 재시도 1회
            logger.warning("🔄 thought_signature 에러 감지. 메시지를 정리하고 재시도합니다...")
            if llm_errors < 2:
                return {"verification_errors": llm_errors, "next_action": "model"}
            else:
                logger.error("❌ thought_signature 에러 반복. 종료합니다.")
                return {"next_action": "end"}
        
        if llm_errors < 2:
            logger.warning(f"🔄 일반 에러. 재시도합니다... ({llm_errors}/2)")
            time.sleep(2)
            return {"verification_errors": llm_errors, "next_action": "model"}
        else:
            logger.error("❌ 반복 에러로 종료합니다.")
            return {"next_action": "end"}
