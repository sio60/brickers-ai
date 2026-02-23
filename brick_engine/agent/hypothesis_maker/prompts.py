# brick_engine/agent/hypothesis_maker/prompts.py
"""
전문가 수준의 가설 수립을 위한 한국어 프롬프트 정의 파일.
HypothesisMaker(core.py)에서 사용되는 모든 프롬프트를 관리합니다.
"""
import json
from typing import List, Dict, Any

# 1. Gemini Draft Creator를 위한 프롬프트
def get_draft_creator_prompt(observation: str, success_text: str, verification: Dict[str, Any]) -> str:
    # current_metrics_json으로 이름 유지 및 내용 구성
    metrics_data = verification.get('metrics_after', {})
    current_metrics_json = json.dumps(metrics_data, indent=2, ensure_ascii=False)
    
    return f"""
당신은 레고 구조 공학 전문가입니다. 과거의 성공 사례를 분석하여 현재 문제를 해결할 초안 가설을 세워주세요.

[현재 상황]
관찰 내용: {observation}
현재 상태 데이터 (공중부양 브릭 ID 등 포함): {current_metrics_json}

💡 **특별 지시사항:** 만약 `current_metrics_json` 또는 관찰 내용에 `floating_brick_ids` (공중부양 브릭 ID 목록)가 포함되어 있다면, 해당 브릭들이 구조적으로 지지받지 못하고 있는 **정확한 원인과 위치**를 분석하고 이를 해결하기 위한 맞춤형 접근법을 가설에 반드시 포함하세요.

[튜닝 가능한 파라미터 가이드]
- target (int): 전체 크기. 안정성을 위해 조절 가능.
- shrink (float: 0.1-1.0): 모델 스케일. 낮을수록 가볍고 촘촘해짐.
- plates_per_voxel (int: 1-3): 수집 밀도. 높을수록 튼튼하지만 무거워짐.
- support_ratio (float: 0.0-2.0): 지지대 밀도. 공중부양 문제 해결의 핵심.
- fill (bool): 내부 채움 여부. True일 때 구조적 강도 급증.
- interlock (bool): 브릭 겹침 허용. True일 때 붕괴 방지 효과 극대화.
- erosion_iters (int: 0-3): 얇은 부분 제거. 높을수록 깔끔하지만 디테일을 잃을 수 있음.
- auto_remove_1x1 (bool): 약한 1x1 브릭 제거. True인 경우 더 안전함.
- smart_fix (bool): 알고리즘 보정 활성화.

[과거 성공 패턴]
{success_text}

위 가이드를 바탕으로, 오로지 '성공 사례'와 '물리 원칙'에 기반한 1차 초안을 JSON으로 성실히 답변하세요.
당신의 가설이 얼마나 효과적일지 스스로 평가하여 **internal_score (0~100점)**를 매기세요.
모든 설명(hypothesis, reasoning)은 반드시 한국어로 작성하세요.

JSON 출력 형식:
{{ 
    "hypothesis": "가설 내용 (한국어)", 
    "reasoning": "수립 근거 (한국어)", 
    "internal_score": 75,
    "proposed_params": {{
        "target": 60,
        "support_ratio": 1.2,
        ...
    }} 
}}
"""

# 2. GPT Critic을 위한 비평 프롬프트
def get_critic_prompt(failures_text: str, draft_summary: str, current_observation: str) -> str:
    return f"""
당신은 구조적 결함을 찾아내는 엄격한 비평가입니다. 제안된 초안이 과거의 실패를 반복하지 않는지 검토하세요.

[검토 대상 초안]
{draft_summary}

[유사 상황에서의 과거 실패 사례 ("{current_observation}")]
{failures_text}

분석 지침:
1. 제안된 계획이 과거의 어떤 실패와 유사한 리스크를 가지고 있습니까?
2. 구체적인 물리적 붕괴 위험 요소는 무엇입니까?
3. 실패를 피하기 위한 '단 하나의 구체적인 수정안'을 제시하세요 (반드시 한국어로 작성!).

모든 답변은 반드시 한국어로, 간결하고 날카롭게 작성하세요.
"""

# 3. Gemini Refiner를 위한 최종 확정 프롬프트
def get_final_refine_prompt(draft: Dict[str, Any], critique: str) -> str:
    hypothesis = draft.get('hypothesis')
    reasoning = draft.get('reasoning')
    
    return f"""
당신은 가설을 최종 확정하는 수석 엔지니어입니다. 초안과 비평을 종합하여 완벽한 구조적 실행 계획을 수립하세요.

[기초 초안]
가설: {hypothesis} (근거: {reasoning})

[GPT의 날카로운 비평 및 리스크 분석]
{critique}

최종 임무:
1. 비평가가 지적한 리스크를 완벽히 해결하도록 초안을 수정 및 정교화하세요. 
2. 수정된 가설의 완성도를 스스로 평가하여 **internal_score (0~100점)**를 매기세요.
   - **95~100점**: 모든 물리적 리스크가 완벽히 해결되었으며 즉시 실행 가능함.
   - **90~94점**: 대부분의 리스크가 해결되었으나 아직 미세한 보완 여지가 있음.
   - **90점 미만**: 여전히 리스크가 존재하거나 GPT의 비평이 충분히 반영되지 않음.
3. **비평을 통해 개선된 포인트**와 **이전 가설 대비 점수가 변경된 이유**를 설명하세요.
4. 모든 텍스트 필드는 반드시 한국어로 작성해야 합니다.

JSON 출력 형식:
{{
    "hypothesis": "최종 확정 가설 (한국어)",
    "improvement_points": "비평을 듣고 구체적으로 수정한 부분 (한국어)",
    "reasoning": "비평을 수용한 논리적 근거 (한국어)", 
    "internal_score": 95,
    "score_rationale": "점수 부여 근거 (한국어)",
    "prediction": "예상되는 결과 (한국어)",
    "proposed_params": {{
        "target": 60,
        "shrink": 0.8,
        "plates_per_voxel": 3,
        "support_ratio": 1.2,
        "fill": true,
        "interlock": true,
        "erosion_iters": 1,
        "auto_remove_1x1": true,
        "smart_fix": true,
        "step_order": "bottomup"
    }}
}}
"""

# --- 하위 호환성을 위한 기존 함수들 ---

def get_draft_prompt(observation: str, success_text: str, current_metrics_json: str) -> str:
    """기존 파이프라인 호환용"""
    try:
        metrics = json.loads(current_metrics_json) if current_metrics_json else {}
    except:
        metrics = {}
    return get_draft_creator_prompt(observation, success_text, {"metrics_after": metrics})

def get_refine_prompt(draft: str, critique: str, round_num: int) -> str:
    """기존 파이프라인 호환용"""
    return get_final_refine_prompt({"hypothesis": draft, "reasoning": "N/A"}, critique)
