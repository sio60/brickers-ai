# ============================================================================
# 물리 검증 실행 스크립트
# 이 파일은 커맨드 라인에서 순수 코드 기반 물리 검증을 실행하기 위한 진입점입니다.
# 사용법: python physical_verification/verify_pybullet.py <ldr_file_path>
# ============================================================================
import sys
import os
import argparse

# 프로젝트 루트 경로를 path에 추가 (physical_verification 패키지 인식을 위해)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from physical_verification.ldr_loader import LdrLoader
from physical_verification.verifier import PhysicalVerifier
from physical_verification.models import VerificationResult

def main():
    parser = argparse.ArgumentParser(description="Physical Verification Runner")
    parser.add_argument("file", help="Path to the LDR file to verify")
    args = parser.parse_args()

    target_file = args.file
    if not os.path.exists(target_file):
        # 상대 경로로 시도 (프로젝트 루트 기준)
        target_file = os.path.join(project_root, args.file)
        if not os.path.exists(target_file):
            print(f"에러: 파일을 찾을 수 없습니다: {args.file}")
            return

    print(f"물리 검증 시작: {target_file}")

    # 1. LDR 로드
    loader = LdrLoader()
    try:
        plan = loader.load_from_file(target_file)
        print(f"모델 로드 완료: 브릭 {len(plan.bricks)}개")
    except Exception as e:
        print(f"로드 실패: {e}")
        return

    # 2. PhysicalVerifier 초기화
    verifier = PhysicalVerifier(plan)

    # 3. 충돌 검사 (Collision Check)
    print("\n[1/2] 충돌 검사 실행 중...")
    col_result = verifier.run_collision_check()
    if not col_result.is_valid:
        print("충돌 감지됨!")
        for ev in col_result.evidence:
            print(f"  - {ev.message}")

    # 4. 안정성 검사 (Stability Check)
    print("\n[2/2] 구조적 안정성 검사 중...")
    stab_result = verifier.run_stability_check()

    print(f"\n안정성 등급: {stab_result.stability_grade} (점수: {stab_result.score:.0f}/100)")
    if stab_result.evidence:
        for ev in stab_result.evidence:
            print(f"  - [{ev.severity}] {ev.message}")

    print("\n" + "="*40)
    if col_result.is_valid and stab_result.is_valid:
        print("최종 결과: [PASS] 모든 검증 통과!")
    else:
        print("최종 결과: [FAIL] 검증 실패")
        if not col_result.is_valid: print(" - 사유: 부품 간 충돌 발생")
        if not stab_result.is_valid: print(" - 사유: 구조적 불안정")
    print("="*40)

if __name__ == "__main__":
    main()
