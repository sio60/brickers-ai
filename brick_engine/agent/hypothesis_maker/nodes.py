import json
import asyncio
import logging
from typing import Dict, Any, List, Optional
from .state import HypothesisState
from .prompts import get_draft_prompt, get_critic_prompt, get_refine_prompt
from service.backend_client import send_agent_log

logger = logging.getLogger("HypothesisMakerNodes")

# 1. RAG 기반 성공/실패 사례 검색 노드 (대량 검색)
async def node_search_cases(state: HypothesisState) -> Dict[str, Any]:
    """과거의 성공 및 실패 사례를 메모리에서 대량으로 검색합니다."""
    print("\n" + "🔍" * 20)
    print(" [Hypothesis Search] 토론을 위한 맞춤형 사례 데이터베이스 검색 중...")
    
    # [LOG]
    job_id = state.get("job_id", "offline")
    await send_agent_log(job_id, "search_cases", "토론을 위한 성공/실패 사례를 데이터베이스에서 검색하고 있어요.")

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
    
    await send_agent_log(job_id, "search_cases", f"검색 완료: 성공 사례 {len(success_cases)}건, 실패 사례 {len(failure_cases)}건을 찾았습니다.")

    return {
        "success_cases": success_cases,
        "failure_cases": failure_cases,
        "round_count": 0,
        "debate_history": [],
        "internal_score": 0,
        "job_id": job_id # Pass through
    }

# 2. Gemini를 활용한 가설 초안/수정 생성 노드
async def node_draft_creator(state: HypothesisState) -> Dict[str, Any]:
    """성공 사례를 바탕으로 Gemini가 가설을 수립하거나 수정합니다."""
    round_num = state.get("round_count", 0) + 1
    job_id = state.get("job_id", "offline")
    
    print("\n" + "━" * 60)
    print(f" 🚩 [Round {round_num}] Gemini 전문가의 가설 수립 및 정교화 단계")
    print("━" * 60)
    
    if round_num == 1:
        await send_agent_log(job_id, "draft_hypothesis", "Gemini 전문가가 초기 가설을 수립하고 있습니다...")
    else:
        await send_agent_log(job_id, "refine_hypothesis", f"Gemini 전문가가 비평을 반영하여 가설을 수정하고 있습니다... (Round {round_num})")

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
        
        await send_agent_log(job_id, "refine_hypothesis", f"가설 수정 완료 (점수: {previous_score} -> {internal_score})")
    else:
        print(f"  🎯 가설 점수: {internal_score}점")
        await send_agent_log(job_id, "draft_hypothesis", f"초기 가설 수립 완료 (점수: {internal_score}점)")
    
    history = state.get("debate_history", [])
    history.append(f"Round {round_num} Gemini: {hypothesis}")
    
    return {
        "draft_hypothesis": result,
        "internal_score": internal_score,
        "round_count": round_num,
        "debate_history": history
    }
    
# 4. 최종 확정 노드
async def node_refiner(state: HypothesisState) -> Dict[str, Any]:
    """모든 토론 결과를 종합하여 최종 실행 가설을 확정합니다."""
    print("\n[Finalize] 토론을 마치고 최종 실행 계획을 확정합니다.")
    
    job_id = state.get("job_id", "offline")
    await send_agent_log(job_id, "finalize_hypothesis", "토론을 마치고 최종 가설을 확정하여 실행 준비 중입니다.")

    final_draft = state.get("draft_hypothesis", {})
    
    return {
        "current_hypothesis": final_draft,
        "next_action": "strategy" 
    }
