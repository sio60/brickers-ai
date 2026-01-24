# ============================================================================
# LLM 재생성 에이전트
# GLB → LDR 변환 후 물리 검증 실패 시 LLM을 활용해 파라미터를 조정하고 재생성하는 시스템
# 
# 핵심 흐름:
# 1. GLB → LDR 변환 (glb_to_ldr_embedded)
# 2. 물리 검증 (pybullet_verifier)
# 3. 실패 시 → LLM에게 피드백 전달 → 파라미터 조정 → 재생성
# 4. 성공 또는 최대 재시도 도달까지 반복
# ============================================================================

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field

# 모듈 경로 설정 (어디서든 실행 가능하도록)
_THIS_DIR = Path(__file__).resolve().parent
_BRICK_ENGINE_DIR = _THIS_DIR.parent
_PROJECT_ROOT = _BRICK_ENGINE_DIR.parent
_PHYSICAL_VERIFICATION_DIR = _PROJECT_ROOT / "physical_verification"

# 필요한 모든 경로를 sys.path에 추가
for p in (_THIS_DIR, _BRICK_ENGINE_DIR, _PROJECT_ROOT, _PHYSICAL_VERIFICATION_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# LLM 클라이언트 임포트 (패키지/직접 실행 모두 지원)
try:
    from .llm_clients import BaseLLMClient, GroqClient
except ImportError:
    from llm_clients import BaseLLMClient, GroqClient


# ============================================================================
# 기본 파라미터 정의
# ============================================================================

DEFAULT_PARAMS = {
    "target": 60,              # 목표 스터드 크기
    "min_target": 5,           # 최소 스터드 크기
    "budget": 100,             # 최대 브릭 수
    "shrink": 0.85,            # 축소 비율
    "search_iters": 6,         # 이진 탐색 반복 횟수
    "flipx180": False,         # X축 180도 회전
    "flipy180": False,         # Y축 180도 회전
    "flipz180": False,         # Z축 180도 회전
    "kind": "brick",           # 브릭 종류 (brick/plate)
    "plates_per_voxel": 3,     # 복셀당 플레이트 수
    "interlock": True,         # 인터락 활성화
    "max_area": 20,            # 최대 영역
    "solid_color": 4,          # 단색 색상 ID
    "use_mesh_color": True,    # 메시 색상 사용
    "invert_y": False,         # Y축 반전
    "smart_fix": True,         # 스마트 보정 활성화
    "step_order": "bottomup",  # 조립 순서
}


# ============================================================================
# 검증 결과 → LLM 피드백 변환
# ============================================================================

@dataclass
class VerificationFeedback:
    """PyBullet 검증 결과를 LLM에게 전달하기 위한 구조화된 피드백"""
    
    stable: bool = True                    # 안정성 통과 여부
    total_bricks: int = 0                  # 총 브릭 수
    fallen_bricks_count: int = 0           # 떨어진 브릭 수
    floating_bricks_count: int = 0         # 공중부양 브릭 수
    failure_ratio: float = 0.0             # 실패 비율 (0.0 ~ 1.0)
    first_failure_brick: Optional[str] = None  # 최초 실패 브릭 ID
    max_drift: float = 0.0                 # 최대 이동 거리
    collision_count: int = 0               # 충돌 횟수
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "stable": self.stable,
            "total_bricks": self.total_bricks,
            "fallen_bricks_count": self.fallen_bricks_count,
            "floating_bricks_count": self.floating_bricks_count,
            "failure_ratio": round(self.failure_ratio, 3),
            "first_failure_brick": self.first_failure_brick,
            "max_drift": round(self.max_drift, 2),
            "collision_count": self.collision_count,
        }


def extract_verification_feedback(result, total_bricks: int) -> VerificationFeedback:
    """
    PyBullet VerificationResult를 LLM 피드백 형식으로 변환
    
    Args:
        result: PyBulletVerifier.run_stability_check() 결과
        total_bricks: 총 브릭 수
        
    Returns:
        VerificationFeedback 객체
    """
    feedback = VerificationFeedback()
    feedback.total_bricks = total_bricks
    feedback.stable = result.is_valid
    
    # Evidence 분석
    fallen_bricks = set()
    floating_bricks = set()
    first_failure = None
    max_drift = 0.0
    collision_count = 0
    
    for ev in result.evidence:
        if ev.type == "FIRST_FAILURE":
            if ev.brick_ids:
                first_failure = ev.brick_ids[0]
                fallen_bricks.update(ev.brick_ids)
        elif ev.type == "COLLAPSE_AFTERMATH":
            if ev.brick_ids:
                fallen_bricks.update(ev.brick_ids)
        elif ev.type == "FLOATING_BRICK":
            if ev.brick_ids:
                floating_bricks.update(ev.brick_ids)
        elif ev.type == "COLLISION":
            collision_count += 1
    
    feedback.fallen_bricks_count = len(fallen_bricks)
    feedback.floating_bricks_count = len(floating_bricks)
    feedback.first_failure_brick = first_failure
    feedback.collision_count = collision_count
    
    # 실패 비율 계산
    if total_bricks > 0:
        feedback.failure_ratio = (len(fallen_bricks) + len(floating_bricks)) / total_bricks
    
    return feedback


# ============================================================================
# LLM 프롬프트 생성
# ============================================================================

SYSTEM_PROMPT = """당신은 레고 브릭 구조물 설계 전문가입니다.
GLB 3D 모델을 레고 브릭으로 변환하는 시스템의 파라미터를 최적화하는 역할을 합니다.

당신의 목표:
1. 물리 시뮬레이션에서 무너지지 않는 안정적인 구조물 생성
2. 원본 형상을 최대한 유지하면서 구조적 안정성 확보
3. 브릭 수를 적절히 유지 (너무 적으면 형상 손실, 너무 많으면 복잡해짐)

응답 형식:
반드시 JSON 형식으로만 응답하세요. 다른 설명 없이 JSON만 출력하세요."""


def build_adjustment_prompt(
    feedback: VerificationFeedback,
    current_params: Dict[str, Any],
    attempt: int,
    max_attempts: int,
) -> str:
    """
    LLM에게 파라미터 조정을 요청하는 프롬프트 생성
    
    Args:
        feedback: 검증 피드백
        current_params: 현재 파라미터
        attempt: 현재 시도 횟수
        max_attempts: 최대 시도 횟수
        
    Returns:
        LLM 프롬프트 문자열
    """
    prompt = f"""## 물리 검증 결과 (시도 {attempt}/{max_attempts})

{_format_feedback(feedback)}

## 현재 파라미터
{_format_params(current_params)}

## 조정 가능한 파라미터와 효과

| 파라미터 | 현재값 | 설명 | 안정성 향상 방법 |
|---------|--------|------|-----------------|
| target | {current_params.get('target', 60)} | 목표 스터드 크기 | 줄이면 단순해져서 안정 ↑ |
| budget | {current_params.get('budget', 100)} | 최대 브릭 수 | 늘리면 더 촘촘해져서 안정 ↑ |
| interlock | {current_params.get('interlock', True)} | 브릭 맞물림 | True면 연결 강화 |
| smart_fix | {current_params.get('smart_fix', True)} | 부동 브릭 보정 | True면 자동 보정 |

⚠️ 주의: kind와 plates_per_voxel은 변경하지 마세요! 변경 시 브릭 연결이 깨집니다.

## 요청
위 검증 결과를 바탕으로 구조적 안정성을 높이기 위한 새로운 파라미터를 JSON으로 제안하세요.

응답 형식:
```json
{{
    "reasoning": "변경 이유 설명 (한 문장)",
    "params": {{
        "target": 숫자,
        "budget": 숫자,
        "interlock": true/false,
        "smart_fix": true/false
    }},
    "confidence": 0.0~1.0
}}
```"""
    
    return prompt


def _format_feedback(feedback: VerificationFeedback) -> str:
    """피드백을 읽기 쉬운 문자열로 포맷"""
    status = "✅ 안정" if feedback.stable else "❌ 불안정"
    
    lines = [
        f"- 상태: {status}",
        f"- 총 브릭 수: {feedback.total_bricks}개",
    ]
    
    if not feedback.stable:
        lines.extend([
            f"- 떨어진 브릭: {feedback.fallen_bricks_count}개",
            f"- 공중부양 브릭: {feedback.floating_bricks_count}개",
            f"- 실패율: {feedback.failure_ratio * 100:.1f}%",
        ])
        if feedback.first_failure_brick:
            lines.append(f"- 최초 붕괴 브릭: {feedback.first_failure_brick}")
    
    if feedback.collision_count > 0:
        lines.append(f"- 충돌 감지: {feedback.collision_count}건")
    
    return "\n".join(lines)


def _format_params(params: Dict[str, Any]) -> str:
    """파라미터를 읽기 쉬운 문자열로 포맷"""
    lines = []
    for key, value in params.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


# ============================================================================
# LLM 응답 파싱
# ============================================================================

def parse_llm_response(response: Dict[str, Any], current_params: Dict[str, Any]) -> Tuple[Dict[str, Any], str, float]:
    """
    LLM 응답에서 새 파라미터 추출
    
    Args:
        response: LLM JSON 응답
        current_params: 현재 파라미터 (기본값으로 사용)
        
    Returns:
        (새 파라미터, 이유 설명, 신뢰도)
    """
    if "error" in response:
        print(f"[경고] LLM 응답 파싱 오류, 기본 전략 사용")
        return _fallback_strategy(current_params), "LLM 응답 파싱 실패로 기본 전략 적용", 0.0
    
    reasoning = response.get("reasoning", "이유 없음")
    confidence = response.get("confidence", 0.5)
    new_params_partial = response.get("params", {})
    
    # 기존 파라미터에 새 값 병합
    new_params = current_params.copy()
    
    # 유효한 파라미터만 업데이트
    # ⚠️ plates_per_voxel 제외: 이 값을 변경하면 브릭 연결이 완전히 깨짐!
    valid_keys = {"target", "budget", "interlock", "smart_fix", "kind",
                  "min_target", "shrink", "search_iters", "max_area"}
    
    for key, value in new_params_partial.items():
        if key in valid_keys:
            new_params[key] = value
    
    return new_params, reasoning, confidence


def _fallback_strategy(current_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    LLM 실패 시 사용할 기본 전략: target 축소
    """
    new_params = current_params.copy()
    current_target = new_params.get("target", 60)
    new_params["target"] = max(10, int(current_target * 0.8))  # 20% 축소
    new_params["smart_fix"] = True  # 스마트 보정 활성화
    return new_params


# ============================================================================
# 공중부양 브릭 수정 프롬프트 (Phase 2)
# ============================================================================

BRICK_FIX_SYSTEM_PROMPT = """당신은 레고 브릭 구조물 수리 전문가입니다.
공중에 떠 있는(연결되지 않은) 브릭을 분석하고 삭제 여부를 결정합니다.

당신의 목표:
1. 공중부양 브릭을 삭제하여 구조적 안정성 확보
2. 아이들이 모델을 들었을 때 브릭이 떨어지지 않도록 함
3. 형상 손실을 최소화하되, 안정성을 우선

⚠️ 중요: 이동(move)은 사용하지 마세요! 삭제(delete)만 사용하세요.
이동은 좌표 오류가 발생하기 쉽고 오히려 불안정해집니다.

각 브릭에 대해 다음 중 하나만 결정:
- "delete": 삭제 (권장 - 공중부양 브릭은 대부분 삭제)
- "keep": 유지 (정말 형상에 중요한 경우에만)

응답은 반드시 JSON 형식으로만 출력하세요."""


def build_brick_fix_prompt(
    floating_analysis: Dict[str, Any],
    total_bricks: int,
) -> str:
    """
    공중부양 브릭 수정을 위한 LLM 프롬프트 생성
    
    Args:
        floating_analysis: brick_fixer.analyze_floating_bricks() 결과
        total_bricks: 전체 브릭 수
        
    Returns:
        LLM 프롬프트 문자열
    """
    floating_count = len(floating_analysis)
    
    # 분석 결과 포맷
    analysis_text = []
    for brick_id, candidates in floating_analysis.items():
        analysis_text.append(f"\n### {brick_id}")
        if candidates:
            analysis_text.append("연결 가능 후보:")
            for i, c in enumerate(candidates, 1):
                analysis_text.append(f"  {i}. {c['target_brick']}의 {c['type']} (거리: {c['distance']} LDU)")
                analysis_text.append(f"     새 위치: ({c['new_position'][0]}, {c['new_position'][1]}, {c['new_position'][2]})")
        else:
            analysis_text.append("⚠️ 연결 가능한 후보 없음")
    
    prompt = f"""## 공중부양 브릭 분석

총 브릭 수: {total_bricks}개
공중부양 브릭: {floating_count}개

{''.join(analysis_text)}

## 요청

위 공중부양 브릭들을 어떻게 처리할지 결정해주세요.
형상 유지가 중요하면 이동(move), 중요하지 않으면 삭제(delete)를 선택하세요.

응답 형식:
```json
{{
    "reasoning": "전체적인 결정 이유 (한 문장)",
    "decisions": [
        {{"brick_id": "3005.dat_0", "action": "move", "position": [x, y, z]}},
        {{"brick_id": "3005.dat_1", "action": "delete"}},
        {{"brick_id": "3005.dat_2", "action": "keep"}}
    ]
}}
```

⚠️ 주의:
- "move" 선택 시 반드시 후보 목록의 position 중 하나를 사용하세요
- 후보가 없는 브릭은 "delete" 또는 "keep"만 선택 가능"""
    
    return prompt


def parse_brick_fix_response(response: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    """
    공중부양 브릭 수정 LLM 응답 파싱
    
    Returns:
        (결정 리스트, 이유 설명)
    """
    if "error" in response:
        print(f"[경고] LLM 응답 파싱 오류")
        return [], "LLM 응답 파싱 실패"
    
    reasoning = response.get("reasoning", "이유 없음")
    decisions = response.get("decisions", [])
    
    # 유효성 검사
    valid_decisions = []
    for d in decisions:
        if "brick_id" not in d:
            continue
        action = d.get("action", "keep")
        if action not in ("move", "delete", "keep"):
            action = "keep"
        
        decision = {"brick_id": d["brick_id"], "action": action}
        
        if action == "move":
            position = d.get("position")
            if position and len(position) == 3:
                decision["position"] = position
            else:
                # 위치 없으면 삭제로 변경
                decision["action"] = "delete"
        
        valid_decisions.append(decision)
    
    return valid_decisions, reasoning


# ============================================================================
# 메인 재생성 루프
# ============================================================================

@dataclass
class RegenerationResult:
    """재생성 결과"""
    success: bool = False                  # 최종 성공 여부
    ldr_path: str = ""                     # 생성된 LDR 파일 경로
    attempts: int = 0                      # 총 시도 횟수
    final_params: Dict[str, Any] = field(default_factory=dict)  # 최종 파라미터
    final_feedback: Optional[VerificationFeedback] = None  # 최종 검증 결과
    history: list = field(default_factory=list)  # 시도 이력


class RegenerationAgent:
    """
    LLM 기반 재생성 에이전트
    """
    
    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        max_retries: int = 5,
        acceptable_failure_ratio: float = 0.1,
        verification_duration: float = 2.0,
    ):
        """
        에이전트 초기화
        
        Args:
            llm_client: LLM 클라이언트 (없으면 Groq 사용)
            max_retries: 최대 재시도 횟수
            acceptable_failure_ratio: 허용 가능한 실패율 (기본 10%)
            verification_duration: 물리 시뮬레이션 시간 (초)
        """
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.acceptable_failure_ratio = acceptable_failure_ratio
        self.verification_duration = verification_duration
    
    def _get_llm_client(self) -> BaseLLMClient:
        """LLM 클라이언트 lazy 초기화"""
        if self.llm_client is None:
            self.llm_client = GroqClient()
        return self.llm_client
    
    def run(
        self,
        glb_path: str,
        output_ldr_path: str,
        initial_params: Optional[Dict[str, Any]] = None,
        gui: bool = False,
    ) -> RegenerationResult:
        """
        재생성 루프 실행
        
        Args:
            glb_path: 입력 GLB 파일 경로
            output_ldr_path: 출력 LDR 파일 경로
            initial_params: 초기 파라미터 (없으면 기본값 사용)
            gui: PyBullet GUI 표시 여부
            
        Returns:
            RegenerationResult 객체
        """
        # 모듈 임포트 (지연 로딩) - 복사본 사용 (원본 보존)
        from glb_to_ldr_embedded_copy import convert_glb_to_ldr
        from physical_verification.pybullet_verifier import PyBulletVerifier
        from physical_verification.ldr_loader import LdrLoader
        
        result = RegenerationResult()
        current_params = initial_params.copy() if initial_params else DEFAULT_PARAMS.copy()
        
        print("=" * 60)
        print("🤖 LLM 재생성 에이전트 시작")
        print("=" * 60)
        print(f"입력: {glb_path}")
        print(f"출력: {output_ldr_path}")
        print(f"최대 재시도: {self.max_retries}회")
        print(f"허용 실패율: {self.acceptable_failure_ratio * 100:.0f}%")
        print("=" * 60)
        
        for attempt in range(1, self.max_retries + 1):
            print(f"\n{'='*60}")
            print(f"📦 시도 {attempt}/{self.max_retries}")
            print(f"{'='*60}")
            
            # 1. GLB → LDR 변환
            print("\n[1/3] GLB → LDR 변환 중...")
            try:
                conv_result = convert_glb_to_ldr(
                    glb_path,
                    output_ldr_path,
                    **current_params
                )
                total_bricks = conv_result.get("parts", 0)
                print(f"    ✅ 변환 완료: {total_bricks}개 브릭 생성")
            except Exception as e:
                print(f"    ❌ 변환 실패: {e}")
                result.history.append({
                    "attempt": attempt,
                    "stage": "conversion",
                    "error": str(e),
                    "params": current_params.copy(),
                })
                # 변환 실패 시 파라미터 조정 후 재시도
                current_params = _fallback_strategy(current_params)
                continue
            
            # 2. 물리 검증
            print("\n[2/3] 물리 검증 중...")
            try:
                loader = LdrLoader()
                plan = loader.load_from_file(output_ldr_path)
                
                verifier = PyBulletVerifier(plan, gui=gui)
                stab_result = verifier.run_stability_check(duration=self.verification_duration)
                
                feedback = extract_verification_feedback(stab_result, total_bricks)
                print(f"    결과: {'✅ 안정' if feedback.stable else '❌ 불안정'}")
                if not feedback.stable:
                    print(f"    실패율: {feedback.failure_ratio * 100:.1f}%")
                
            except Exception as e:
                print(f"    ❌ 검증 실패: {e}")
                result.history.append({
                    "attempt": attempt,
                    "stage": "verification",
                    "error": str(e),
                    "params": current_params.copy(),
                })
                continue
            
            # 결과 기록
            result.history.append({
                "attempt": attempt,
                "stage": "completed",
                "feedback": feedback.to_dict(),
                "params": current_params.copy(),
            })
            
            # 3. 성공 여부 판단 (엄격한 기준)
            # - 완전 안정 OR (실패율 허용 범위 + 공중부양 없음 + 충돌 없음)
            is_acceptable = (
                feedback.stable or 
                (
                    feedback.failure_ratio <= self.acceptable_failure_ratio and
                    feedback.floating_bricks_count == 0 and
                    feedback.collision_count == 0
                )
            )
            
            if is_acceptable:
                print(f"\n🎉 성공! (시도 {attempt}회)")
                result.success = True
                result.ldr_path = output_ldr_path
                result.attempts = attempt
                result.final_params = current_params
                result.final_feedback = feedback
                return result
            
            # 공중부양 브릭이 있으면 LLM에게 수정 요청 (Phase 2)
            if feedback.floating_bricks_count > 0:
                print(f"\n[3/4] 공중부양 브릭 {feedback.floating_bricks_count}개 수정 중...")
                try:
                    # 브릭 분석 모듈 임포트
                    from brick_fixer import analyze_floating_bricks
                    from ldr_modifier import apply_llm_decisions
                    
                    # 공중부양 브릭 ID 추출
                    floating_ids = []
                    for ev in stab_result.evidence:
                        if ev.type == "FLOATING_BRICK" and ev.brick_ids:
                            floating_ids.extend(ev.brick_ids)
                    
                    if floating_ids:
                        # 반복 삭제: 공중부양이 0이 될 때까지 최대 3회 반복
                        max_fix_attempts = 3
                        for fix_attempt in range(1, max_fix_attempts + 1):
                            print(f"\n[브릭 수정 {fix_attempt}/{max_fix_attempts}]")
                            
                            # 현재 LDR에 실제 존재하는 브릭 ID만 필터링
                            existing_ids = set(plan.bricks.keys())
                            valid_floating_ids = [fid for fid in floating_ids if fid in existing_ids]
                            
                            if not valid_floating_ids:
                                print("    ℹ️ 삭제할 공중부양 브릭 없음 (모두 이미 처리됨)")
                                break
                            
                            print(f"    📍 대상: {len(valid_floating_ids)}개 (검증에서 감지: {len(floating_ids)}개)")
                            
                            # 연결 후보 분석 (plan.get_all_bricks()로 브릭 객체 리스트 전달)
                            floating_analysis = analyze_floating_bricks(valid_floating_ids, plan.get_all_bricks())
                            
                            # LLM에게 결정 요청
                            llm = self._get_llm_client()
                            fix_prompt = build_brick_fix_prompt(floating_analysis, len(plan.bricks))
                            fix_response = llm.generate_json(fix_prompt, BRICK_FIX_SYSTEM_PROMPT)
                            
                            decisions, fix_reasoning = parse_brick_fix_response(fix_response)
                            
                            print(f"    💡 LLM 결정: {fix_reasoning}")
                            
                            # LDR 파일 수정
                            if decisions:
                                stats = apply_llm_decisions(output_ldr_path, decisions)
                                print(f"    🔧 삭제: {stats['deleted']}개, 유지: {stats['kept']}개")
                                
                                # 수정 후 재검증
                                print("    📋 재검증 중...")
                                loader = LdrLoader()
                                plan = loader.load_from_file(output_ldr_path)
                                verifier = PyBulletVerifier(plan, gui=gui)
                                stab_result = verifier.run_stability_check(duration=self.verification_duration)
                                
                                new_total = len(plan.bricks)
                                new_feedback = extract_verification_feedback(stab_result, new_total)
                                
                                print(f"    결과: 공중부양 {new_feedback.floating_bricks_count}개")
                                
                                # 성공 여부 판단
                                if new_feedback.floating_bricks_count == 0 and new_feedback.collision_count == 0:
                                    print(f"\n🎉 수정 성공! (시도 {attempt}회)")
                                    result.success = True
                                    result.ldr_path = output_ldr_path
                                    result.attempts = attempt
                                    result.final_params = current_params
                                    result.final_feedback = new_feedback
                                    return result
                                
                                # 아직 공중부양이 남아있으면 새로운 ID 추출
                                floating_ids = []
                                for ev in stab_result.evidence:
                                    if ev.type == "FLOATING_BRICK" and ev.brick_ids:
                                        floating_ids.extend(ev.brick_ids)
                                
                                if not floating_ids:
                                    break  # 공중부양 없으면 반복 종료
                                
                                feedback = new_feedback
                            else:
                                break  # 결정 없으면 반복 종료
                            
                except Exception as e:
                    print(f"    ⚠️ 브릭 수정 실패: {e}")
            else:
                # 실패 사유 출력 (공중부양이 아닌 다른 사유)
                reasons = []
                if feedback.failure_ratio > self.acceptable_failure_ratio:
                    reasons.append(f"실패율 {feedback.failure_ratio*100:.1f}% > {self.acceptable_failure_ratio*100:.0f}%")
                if feedback.collision_count > 0:
                    reasons.append(f"충돌 {feedback.collision_count}건")
                if reasons:
                    print(f"    ⚠️ 불합격 사유: {', '.join(reasons)}")
            
            # 4. LLM에게 파라미터 조정 요청
            if attempt < self.max_retries:
                print("\n[3/3] LLM에게 파라미터 조정 요청 중...")
                try:
                    llm = self._get_llm_client()
                    prompt = build_adjustment_prompt(
                        feedback, current_params, attempt, self.max_retries
                    )
                    response = llm.generate_json(prompt, SYSTEM_PROMPT)
                    
                    new_params, reasoning, confidence = parse_llm_response(response, current_params)
                    
                    print(f"    💡 LLM 제안: {reasoning}")
                    print(f"    📊 신뢰도: {confidence * 100:.0f}%")
                    print(f"    🔧 변경된 파라미터:")
                    for key in new_params:
                        if new_params[key] != current_params.get(key):
                            print(f"       - {key}: {current_params.get(key)} → {new_params[key]}")
                    
                    current_params = new_params
                    
                except Exception as e:
                    print(f"    ⚠️ LLM 호출 실패: {e}")
                    print("    → 기본 전략(target 축소) 적용")
                    current_params = _fallback_strategy(current_params)
        
        # 최대 재시도 도달
        print(f"\n❌ 최대 재시도({self.max_retries}회) 도달")
        result.success = False
        result.ldr_path = output_ldr_path
        result.attempts = self.max_retries
        result.final_params = current_params
        result.final_feedback = feedback if 'feedback' in dir() else None
        
        return result


def regeneration_loop(
    glb_path: str,
    output_ldr_path: str,
    llm_client: Optional[BaseLLMClient] = None,
    max_retries: int = 5,
    acceptable_failure_ratio: float = 0.1,
    gui: bool = False,
) -> RegenerationResult:
    """
    간편 함수: LLM 기반 재생성 루프 실행
    
    Args:
        glb_path: 입력 GLB 파일 경로
        output_ldr_path: 출력 LDR 파일 경로
        llm_client: LLM 클라이언트 (없으면 Groq 사용)
        max_retries: 최대 재시도 횟수
        acceptable_failure_ratio: 허용 가능한 실패율
        gui: PyBullet GUI 표시 여부
        
    Returns:
        RegenerationResult 객체
    """
    agent = RegenerationAgent(
        llm_client=llm_client,
        max_retries=max_retries,
        acceptable_failure_ratio=acceptable_failure_ratio,
    )
    return agent.run(glb_path, output_ldr_path, gui=gui)


# ============================================================================
# CLI 인터페이스
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="LLM 기반 GLB → LDR 재생성 에이전트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python llm_regeneration_agent.py input.glb --out output.ldr
  python llm_regeneration_agent.py model.glb --out result.ldr --max-retries 3 --gui
        """
    )
    
    parser.add_argument("glb", help="입력 GLB 파일 경로")
    parser.add_argument("--out", default="output.ldr", help="출력 LDR 파일 경로 (기본값: output.ldr)")
    parser.add_argument("--max-retries", type=int, default=5, help="최대 재시도 횟수 (기본값: 5)")
    parser.add_argument("--failure-ratio", type=float, default=0.1, help="허용 실패율 (기본값: 0.1)")
    parser.add_argument("--gui", action="store_true", help="PyBullet GUI 표시")
    parser.add_argument("--api-key", help="Groq API 키 (환경변수 GROQ_API_KEY 대신 사용)")
    
    args = parser.parse_args()
    
    # 입력 파일 확인
    if not os.path.exists(args.glb):
        print(f"❌ 오류: 입력 파일을 찾을 수 없습니다: {args.glb}")
        sys.exit(1)
    
    # LLM 클라이언트 초기화
    llm_client = None
    if args.api_key:
        llm_client = GroqClient(api_key=args.api_key)
    
    # 재생성 루프 실행
    result = regeneration_loop(
        args.glb,
        args.out,
        llm_client=llm_client,
        max_retries=args.max_retries,
        acceptable_failure_ratio=args.failure_ratio,
        gui=args.gui,
    )
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("📋 최종 결과")
    print("=" * 60)
    print(f"성공 여부: {'✅ 성공' if result.success else '❌ 실패'}")
    print(f"시도 횟수: {result.attempts}회")
    print(f"출력 파일: {result.ldr_path}")
    
    if result.final_feedback:
        fb = result.final_feedback
        print(f"최종 안정성: {'안정' if fb.stable else '불안정'}")
        print(f"최종 실패율: {fb.failure_ratio * 100:.1f}%")
    
    print("=" * 60)
    
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
