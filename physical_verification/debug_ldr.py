# ============================================================================
# LDR 파일 디버깅 모듈
# 이 파일은 LDR 파일을 로드하고 물리 검증을 실행하여 결과를 디버깅하는
# 스크립트입니다. 검증 결과와 상세 감점 내역을 콘솔에 출력합니다.
# ============================================================================

import sys
import os

# 현재 경로를 파이썬 경로에 추가
sys.path.append(os.getcwd())

from physical_verification.ldr_loader import LdrLoader
from physical_verification.verifier import PhysicalVerifier

def run_debug(target_file):
    """
    지정된 LDR 파일에 대해 물리 검증을 실행하고 결과를 출력합니다.
    
    Args:
        target_file: 검증할 LDR 파일 경로
    """
    if not os.path.exists(target_file):
        print(f"❌ 에러: 파일을 찾을 수 없습니다: {target_file}")
        return

    print(f"🚀 {target_file} 물리 검증 시작...")
    
    # 1. LDR 로드
    loader = LdrLoader()
    try:
        plan = loader.load_from_file(target_file)
        print(f"✅ 로드 완료: 브릭 {len(plan.bricks)}개")
    except Exception as e:
        print(f"❌ 로드 실패: {e}")
        return

    # 2. 검증 실행
    verifier = PhysicalVerifier(plan)
    result = verifier.run_all_checks()

    # 3. 결과 출력
    print("\n" + "="*30)
    print(f"📊 검증 결과: {'✅ PASS' if result.is_valid else '❌ FAIL'}")
    print(f"💯 최종 점수: {result.score} / 100")
    print("="*30)

    if result.evidence:
        print("\n🔍 상세 감점 내역:")
        for ev in result.evidence:
             print(f"  [{ev.type}] ({ev.severity}) - {ev.message}")
    else:
        print("\n✨ 특이사항 없음: 완벽한 구조입니다.")

if __name__ == "__main__":
    # 실행 인자가 없으면 기본으로 car.ldr 테스트
    target = sys.argv[1] if len(sys.argv) > 1 else "ldr/car.ldr"
    run_debug(target)
