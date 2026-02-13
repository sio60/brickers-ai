# ============================================================================
# LLM 기반 RAG Re-ranking
# ============================================================================

from typing import Dict, List

from .prompts import build_rerank_prompt


def rerank_and_filter_cases(llm_client, observation: str, cases: List[Dict]) -> List[Dict]:
    """LLM 기반 RAG Re-ranking (Semantic Scoring)

    Args:
        llm_client: BaseLLMClient 인스턴스 (generate_json 메서드 필요)
        observation: 현재 상황 텍스트
        cases: RAG 검색 결과 후보군

    Returns:
        Re-ranked Top 3 사례 리스트
    """
    if not cases:
        return []

    print(f"  🔍 Re-ranking: {len(cases)}개 후보 분석 중...")

    # 후보군 텍스트 변환
    candidates_text = ""
    for i, case in enumerate(cases):
        candidates_text += f"""
[Case {i}]
- Observation: {case['hypothesis'].get('observation', '')[:200]}...
- Action: {case['experiment'].get('tool', '')}
- Result: {case['verification'].get('numerical_analysis', '')}
--------------------------------------------------"""

    prompt = build_rerank_prompt(observation, candidates_text)

    try:
        response = llm_client.generate_json(prompt)
        top_cases = response.get("top_cases", [])

        reranked_results = []
        for item in sorted(top_cases, key=lambda x: x.get('relevance_score', 0), reverse=True):
            idx = item.get("case_index")
            if 0 <= idx < len(cases):
                case = cases[idx]
                case['_rerank_score'] = item.get('relevance_score')
                case['_rerank_reason'] = item.get('reason')
                reranked_results.append(case)

        print(f"  ✨ Re-ranking 완료: Top {len(reranked_results)} 선정 (Max Score: {reranked_results[0]['_rerank_score'] if reranked_results else 0})")
        return reranked_results

    except Exception as e:
        print(f"  ⚠️ Re-ranking 실패 (Fallback to raw vector rank): {e}")
        return cases[:3]
