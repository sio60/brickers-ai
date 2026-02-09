import json
import asyncio
import logging
from typing import Dict, Any, List, Optional
from .state import HypothesisState
from .prompts import get_draft_prompt, get_critic_prompt, get_refine_prompt

logger = logging.getLogger("HypothesisMakerNodes")

# 1. RAG 기반 성공/실패 사례 검색 노드 (대량 검색)
async def node_search_cases(state: HypothesisState) -> Dict[str, Any]:
    """과거의 성공 및 실패 사례를 메모리에서 대량으로 검색합니다."""
    print("\n" + "🔍" * 20)
    print(" [Hypothesis Search] 토론을 위한 맞춤형 사례 데이터베이스 검색 중...")
    observation = state.get("observation", "")
    subject_name = state.get("subject_name", "Unknown Object")
    verification = state.get("verification_raw_result", {})
    hypothesis_maker = state.get("hypothesis_maker")
    
    if not hypothesis_maker:
        return {"success_cases": [], "failure_cases": [], "round_count": 0, "debate_history": []}
        
    # 메모리 유틸을 통해 대량 케이스 검색 (Subject 태그 반영)
    rag_results = await asyncio.to_thread(
        hypothesis_maker.memory.search_success_and_failure,
        observation=observation,
        limit=15, 
        min_score=0.3,
        verification_metrics=verification,
        subject_name=subject_name
    )
    
    success_cases = rag_results.get("success", [])[:5]
    failure_cases = rag_results.get("failure", [])[:15]
    
    print(f"  📚 검색 완료: 성공 사례 {len(success_cases)}건, 실패 사례 {len(failure_cases)}건 확보")
    
    return {
        "success_cases": success_cases,
        "failure_cases": failure_cases,
        "round_count": 0,
        "debate_history": [],
        "internal_score": 0
    }

# 2. Gemini를 활용한 가설 초안/수정 생성 노드
async def node_draft_creator(state: HypothesisState) -> Dict[str, Any]:
    """성공 사례를 바탕으로 Gemini가 가설을 수립하거나 수정합니다."""
    round_num = state.get("round_count", 0) + 1
    print("\n" + "━" * 60)
    print(f" 🚩 [Round {round_num}] Gemini 전문가의 가설 수립 및 정교화 단계")
    print("━" * 60)
    
    observation = state.get("observation", "")
    success_cases = state.get("success_cases", [])
    verification = state.get("verification_raw_result", {})
    hypothesis_maker = state.get("hypothesis_maker")
    critique = state.get("critique_feedback", "")
    draft = state.get("draft_hypothesis")
    
    # 해당 라운드에 사용할 성공 사례 1개 선택
    current_success = success_cases[round_num-1] if len(success_cases) >= round_num else (success_cases[0] if success_cases else None)
    success_text = f"- 성공사례: Algo={current_success.get('algorithm')}, Params={current_success.get('experiment', {}).get('parameters')}" if current_success else "직접적인 성공 사례 없음."

    previous_score = state.get("internal_score", 0)
    
    if round_num == 1:
        # 초안 생성
        prompt = get_draft_prompt(observation, success_text, json.dumps(verification))
        result = await asyncio.to_thread(hypothesis_maker.gemini_client.generate_json, prompt)
    else:
        # 비평 반영하여 수정
        prompt = get_refine_prompt(json.dumps(draft), critique, round_num)
        result = await asyncio.to_thread(hypothesis_maker.gemini_client.generate_json, prompt)
    
    internal_score = result.get("internal_score", 70)
    hypothesis = result.get("hypothesis", "가설 생성 실패")
    
    print(f"  💭 Gemini: \"{hypothesis[:100]}...\"")
    if round_num > 1:
        score_diff = internal_score - previous_score
        sign = "+" if score_diff >= 0 else ""
        print(f"  ✨ 개선 포인트: {result.get('improvement_points', 'N/A')}")
        print(f"  📈 점수 변화: {previous_score} -> {internal_score} ({sign}{score_diff}점)")
        print(f"  ℹ️ 점수 산정 이유: {result.get('score_rationale', 'N/A')}")
    else:
        print(f"  🎯 가설 점수: {internal_score}점")
    
    history = state.get("debate_history", [])
    history.append(f"Round {round_num} Gemini: {hypothesis}")
    
    return {
        "draft_hypothesis": result,
        "internal_score": internal_score,
        "round_count": round_num,
        "debate_history": history
    }

# 3. GPT를 활용한 실패 사례 기반 비평 노드 (1:3 비율)
async def node_critic(state: HypothesisState) -> Dict[str, Any]:
    """실패 사례 3개를 바탕으로 GPT가 가설의 취약점을 비평합니다."""
    round_num = state.get("round_count", 0)
    print(f"\n 🛡️  [Round {round_num}] GPT 비평가의 리스크 심층 분석 (1:3 티키타카)")
    
    failure_cases = state.get("failure_cases", [])
    draft = state.get("draft_hypothesis", {})
    observation = state.get("observation", "")
    hypothesis_maker = state.get("hypothesis_maker")
    
    # 해당 라운드에 사용할 실패 사례 3개 선택
    start_idx = (round_num - 1) * 3
    current_failures = failure_cases[start_idx : start_idx + 3]
    
    # 실패 사례가 부족하면 순환하여 사용
    if not current_failures and failure_cases:
        current_failures = failure_cases[:3]
    
    failure_text = "\n".join([f"- 실측 실패 패턴: {c.get('verification', {}).get('numerical_analysis', 'Unknown failure')}" for c in current_failures]) or "알려진 실패 사례 없음."
    
    prompt = get_critic_prompt(failure_text, json.dumps(draft), observation)
    
    # GPT(Critic) 호출
    if hypothesis_maker.gpt_client:
        feedback = await asyncio.to_thread(hypothesis_maker.gpt_client.generate, prompt)
    else:
        # GPT 클라이언트가 없는 경우 Gemini로 대체하여 토론 유지
        feedback = await asyncio.to_thread(hypothesis_maker.gemini_client.generate, "[GPT 비평가 대역] " + prompt)
        
    print(f"  🔥 GPT 비평: \"{feedback[:100]}...\"")
    
    history = state.get("debate_history", [])
    history.append(f"Round {round_num} GPT Critic: {feedback}")
    
    return {
        "critique_feedback": feedback,
        "debate_history": history
    }

# 4. 최종 확정 노드
async def node_refiner(state: HypothesisState) -> Dict[str, Any]:
    """모든 토론 결과를 종합하여 최종 실행 가설을 확정합니다."""
    print("\n[Finalize] 토론을 마치고 최종 실행 계획을 확정합니다.")
    final_draft = state.get("draft_hypothesis", {})
    
    return {
        "current_hypothesis": final_draft,
        "next_action": "strategy" 
    }
