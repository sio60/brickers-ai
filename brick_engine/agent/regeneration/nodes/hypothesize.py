# ============================================================================
# Hypothesize 노드: 가설 생성 + RAG 검색
# ============================================================================

from typing import Dict, Any

from langchain_core.messages import HumanMessage

from ..rag_ranker import rerank_and_filter_cases


async def node_hypothesize(graph, state) -> Dict[str, Any]:
    """가설 생성 노드: RAG 검색 및 Dual-Model 협업 가설 수립"""
    from ...memory_utils import memory_manager

    print("\n[Hypothesize] 가설 수립 및 RAG 검색 중 (Dual-Model)...")
    graph._log("HYPOTHESIZE", "유사한 브릭 구조를 참고해서 가능한 형태를 가정하고 있어요.")

    # 1. RAG 검색
    current_observation = ""
    last_msg = state['messages'][-1]
    if isinstance(last_msg, HumanMessage):
        current_observation = str(last_msg.content)[:500]

    similar_cases = []
    if memory_manager:
        verification_metrics = state.get("verification_result")
        raw_cases = memory_manager.search_similar_cases(
            current_observation,
            limit=10,
            min_score=0.5,
            verification_metrics=verification_metrics
        )
        similar_cases = rerank_and_filter_cases(graph.default_client, current_observation, raw_cases)
        print(f"  📚 유사 실패 사례 {len(similar_cases)}건 선정 (Re-ranked)")

    # 2. 가설 생성 (HypothesisMaker 사용)
    try:
        hypothesis_result = await graph.hypothesis_maker.make_hypothesis(state)

        print(f"  💭 최종 가설: {hypothesis_result.get('hypothesis')}")
        print(f"  📝 근거: {hypothesis_result.get('reasoning')}")
        print(f"  📊 난이도: {hypothesis_result.get('difficulty')}")

        obs = hypothesis_result.get("observation", "")
        hypo = hypothesis_result.get("hypothesis", "")
        if obs or hypo:
            graph._log("HYPOTHESIZE", f"분석 결과: {obs} {hypo}")

        return {
            "current_hypothesis": hypothesis_result,
            "next_action": "strategy"
        }
    except Exception as e:
        print(f"  ⚠️ 가설 생성 실패: {e}")
        return {
            "current_hypothesis": {
                "hypothesis": "기본 물리 법칙에 따른 안정화 시도",
                "reasoning": "AI 분석 실패로 인한 기본 전략 사용",
                "difficulty": "Medium"
            },
            "next_action": "strategy"
        }
