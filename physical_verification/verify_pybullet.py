# ============================================================================
# PyBullet 물리 검증 실행 스크립트
# 이 파일은 커맨드 라인에서 PyBullet 기반 물리 검증을 실행하기 위한 진입점입니다.
# 사용법: python physical_verification/verify_pybullet.py <ldr_file_path> [--gui]
# ============================================================================
import sys
import os
import argparse

# 프로젝트 루트 경로를 path에 추가 (physical_verification 패키지 인식을 위해)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from physical_verification.ldr_loader import LdrLoader
from physical_verification.pybullet_verifier import PyBulletVerifier
from physical_verification.models import VerificationResult

def main():
    parser = argparse.ArgumentParser(description="PyBullet Physical Verification Runner")
    parser.add_argument("file", help="Path to the LDR file to verify")
    parser.add_argument("--gui", action="store_true", help="Enable GUI visualization")
    parser.add_argument("--time", type=float, default=5.0, help="Simulation duration in seconds (default: 5.0)")
    args = parser.parse_args()

    target_file = args.file
    if not os.path.exists(target_file):
        # 상대 경로로 시도 (프로젝트 루트 기준)
        target_file = os.path.join(project_root, args.file)
        if not os.path.exists(target_file):
            print(f"❌ 에러: 파일을 찾을 수 없습니다: {args.file}")
            return

    print(f"🚀 PyBullet 물리 검증 시작: {target_file}")
    
    # 1. LDR 로드
    loader = LdrLoader()
    try:
        plan = loader.load_from_file(target_file)
        print(f"✅ 모델 로드 완료: 브릭 {len(plan.bricks)}개")
    except Exception as e:
        print(f"❌ 로드 실패: {e}")
        return

    # 2. PyBullet Verifier 초기화
    # GUI 모드일 때 시각화를 위해 gui=True 전달
    verifier = PyBulletVerifier(plan, gui=args.gui)
    
    # 3. 충돌 검사 (Collision Check)
    print("\n[1/2] 정밀 충돌 검사 실행 중...")
    col_result = verifier.run_collision_check()
    if not col_result.is_valid:
        print("⚠️ 충돌 감지됨!")
        # 상세 내용은 PyBulletVerifier 내부에서 출력

    # 4. 안정성 검사 (Stability Check)
    print(f"\n[2/2] 구조적 안정성(중력) 시뮬레이션 ({args.time}초)...")
    stab_result = verifier.run_stability_check(duration=args.time)
    
    print("\n" + "="*40)
    if col_result.is_valid and stab_result.is_valid:
        print("🎉 최종 결과: [PASS] 모든 검증 통과!")
    else:
        print("🚫 최종 결과: [FAIL] 검증 실패")
        if not col_result.is_valid: print(" - 사유: 부품 간 충돌 발생")
        if not stab_result.is_valid: print(" - 사유: 구조적 붕괴 발생")
    print("="*40)

if __name__ == "__main__":
    main()
