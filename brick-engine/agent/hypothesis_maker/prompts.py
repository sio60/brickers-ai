# 전문가 수준의 가설 수립을 위한 한국어 프롬프트 정의 파일

# Gemini Draft Creator를 위한 프롬프트 생성 함수
def get_draft_prompt(observation: str, success_text: str, current_metrics_json: str) -> str:
    return f"""
당신은 레고 구조 공학 전문가입니다. 과거의 성공 사례를 분석하여 현재 문제를 해결할 초안 가설을 세워주세요.

[현재 상황]
관찰 내용: {observation}
현재 상태 데이터: {current_metrics_json}

[과거 성공 패턴]
{success_text}

[튜닝 가능한 파라미터 가이드]
- target (int): 전체 크기. 안정성을 위해 조절 가능.
- shrink (float: 0.1-1.0): 모델 스케일. 낮을수록 가볍고 촘촘해짐.
- plates_per_voxel (int: 1-3): 수직 밀도. 높을수록 튼튼하지만 무거워짐.
- support_ratio (float: 0.0-2.0): 지지대 밀도. 공중부양 문제 해결의 핵심.
- fill (bool): 내부 채움 여부. True일 때 구조적 강도 급증.
- interlock (bool): 브릭 겹침 허용. True일 때 붕괴 방지 효과 극대화.
- smart_fix (bool): 알고리즘 보정 활성화.

위 가이드를 바탕으로, 오로지 '성공 사례'와 '물리 원칙'에 기반한 1차 초안을 JSON으로 성실히 답변하세요.
모든 설명(hypothesis, reasoning)은 반드시 한국어로 작성하세요.

JSON 출력 형식:
{{
    "hypothesis": "가설 내용 (한국어)",
    "reasoning": "수립 근거 (한국어)",
    "proposed_params": {{
        "target": 60,
        "support_ratio": 1.2,
        ...
    }}
}}
"""

# GPT Critic을 위한 비평 프롬프트 생성 함수
def get_critic_prompt(failure_text: str, draft_summary: str, observation: str) -> str:
    return f"""
당신은 구조적 결함을 찾아내는 엄격한 비평가입니다. 제안된 초안이 과거의 실패를 반복하지 않는지 검토하세요.

[검토 대상 초안]
{draft_summary}

[유사 상황에서의 과거 실패 사례]
{failure_text}

[분석 지침]
1. 제안된 초안이 과거의 어떤 실패와 유사한 리스크를 가지고 있습니까?
2. 구체적인 물리적 붕괴 위험 요소는 무엇입니까?
3. 실패를 피하기 위한 '단 하나의 구체적인 수정안'을 제시하세요.

모든 답변은 반드시 한국어로, 간결하고 날카롭게 작성하세요.
"""

# Gemini Refiner를 위한 최종 확정 프롬프트 생성 함수
def get_refine_prompt(draft: str, critique: str, round_num: int) -> str:
    return f"""
당신은 가설을 최종 확정하는 수석 엔지니어입니다. 초안과 비평을 종합하여 완벽한 실행 계획을 수립하세요.
현재는 토론 {round_num}라운드입니다.

[기초 초안]
{draft}

[GPT의 날카로운 비평 및 리스크 분석]
{critique}

[최종 임무]
1. 비평가가 지적한 리스크를 완벽히 해결하도록 초안을 수정하세요. 
2. 수정된 가설의 완성도를 스스로 평가하여 0~100점 사이의 점수를 매기세요.
   - **95~100점**: 모든 물리적 리스크가 완벽히 해결되었으며 즉시 실행 가능함.
   - **90~94점**: 대부분의 리스크가 해결되었으나 아직 미세한 보완 여지가 있음.
   - **90점 미만**: 여전히 리스크가 존재하거나 GPT의 비평이 충분히 반영되지 않음.
3. **이전 가설 대비 점수가 변경된 구체적인 이유**와 **비평을 통해 개선된 포인트**를 설명하세요.
4. 모든 텍스트 필드는 한국어로 작성하며, 파라미터는 즉시 실행 가능한 형태여야 합니다.

JSON 출력 형식:
{{
    "hypothesis": "최종 확정 가설 (한국어)",
    "improvement_points": "비평을 듣고 구체적으로 수정한 부분 (한국어)",
    "reasoning": "비평을 수용한 논리적 근거 (한국어)",
    "internal_score": 90,
    "score_rationale": "이전 라운드 대비 점수가 오르거나 내린 이유 (한국어)",
    "proposed_params": {{
        "target": 25,
        "shrink": 0.8,
        "support_ratio": 1.2,
        "fill": true,
        "interlock": true
    }}
}}
"""
