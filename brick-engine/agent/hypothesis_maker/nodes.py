import json
from typing import Dict, Any
from .state import HypothesisState

# LLM Client Import (Environment setup assumed)
try:
    from ..llm_clients import GeminiClient
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
    from llm_clients import GeminiClient

class HypothesisNodes:
    def __init__(self):
        self.llm = GeminiClient(model="gemini-2.5-flash") # Fast model

    def generate_hypothesis(self, state: HypothesisState) -> Dict[str, Any]:
        """
        가설 생성 노드 (Basic)
        """
        observation = state['observation']
        similar_cases = state.get('similar_cases', [])
        
        # 1. RAG Context Formatting
        rag_context = ""
        for case in similar_cases:
            rerank_reason = case.get('_rerank_reason', 'N/A')
            lesson = case.get('improvement', {}).get('lesson_learned', 'No lesson')
            tool = case['experiment'].get('tool')
            result = case['verification'].get('numerical_analysis', 'N/A')
            rag_context += f"- [Case: {tool}]\n  Relevance: {rerank_reason}\n  Lesson: {lesson}\n  Result: {result}\n"
            
        if not rag_context:
            rag_context = "(유사 사례 없음)"

        # 2. Prompt Construction (Basic + RAG aware)
        prompt = f"""
당신은 레고 구조 공학 전문가입니다. 현재 발생한 물리적 검증 실패의 원인을 분석하고 해결 가설을 수립하세요.

[현재 상황 (Observation)]
{observation}
- Metrics: {state.get('verification_result', {})}

[유사 과거 사례 (RAG Context)]
{rag_context}

[지침]
1. 실패의 '근본 원인(Root Cause)'을 분석하세요.
2. 과거 사례의 교훈(Lesson)을 참고하여 가장 성공 확률이 높은 접근법을 도출하세요.
3. 구체적인 가설을 수립하세요.

[응답 포맷 (JSON)]
{{
    "observation": "현재 문제 상황 요약 (1문장)",
    "root_cause": "물리적 실패 원인 분석",
    "hypothesis": "구체적인 해결 가설 (If-Then-Because)",
    "reasoning": "가설의 근거",
    "difficulty": "Easy|Medium|Hard"
}}
"""
        try:
            response = self.llm.generate_json(prompt)
            print(f"  💭 [HypothesisMaker] 가설 생성 완료: {response.get('hypothesis')}")
            return {"final_hypothesis": response}
            
        except Exception as e:
            print(f"  ⚠️ [HypothesisMaker] 가설 생성 실패: {e}")
            fallback = {
                "observation": "분석 실패",
                "hypothesis": "기본 전략 유지 (생성 실패)",
                "difficulty": "Medium"
            }
            return {"final_hypothesis": fallback}
