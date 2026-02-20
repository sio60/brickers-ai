# ============================================================================
# 단독 LDR 병합 도구 (Standalone LDR Merger)
# 사용법: python merge_ldr.py <입력파일.ldr> [출력파일.ldr]
# ============================================================================

import sys
import os
import shutil
import argparse
from pathlib import Path

# 프로젝트 루트를 path에 추가하여 모듈 임포트 가능하게 함
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from brick_engine.agent.ldr_modifier import merge_small_bricks, structural_merge
    from brick_judge import full_judge, parse_ldr_string
except ImportError:
    print("오류: 필요한 모듈(ldr_modifier, brick_judge)을 찾을 수 없습니다.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="LDR 파일을 물리 검증하고 병합하여 최적화합니다.")
    parser.add_argument("input", help="원본 LDR 파일 경로")
    parser.add_argument("output", nargs="?", help="저장할 결과 LDR 파일 경로")
    parser.add_argument("--skip-structural", action="store_true", help="구조적 병합(Cross-color) 건너뛰기")
    parser.add_argument("--cleanup", action="store_true", help="안정적인 1x1 브릭도 포함하여 전체 병합(Cleanup) 수행")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"오류: 파일을 찾을 수 없습니다: {args.input}")
        return

    output_path = Path(args.output) if args.output else input_path.parent / f"{input_path.stem}_merged{input_path.suffix}"

    print(f"--- 작업 시작: {input_path.name} ---")
    shutil.copy(input_path, output_path)

    try:
        # 1. 물리 검증 (불안정 브릭 탐지)
        print("1. 물리적 안정성 검사 중 (Stability Check)...")
        with open(output_path, "r", encoding="utf-8") as f:
            ldr_content = f.read()
        
        model = parse_ldr_string(ldr_content)
        issues = full_judge(model)
        
        # 불안정(floating, isolated) 브릭 ID 수집 (top_only 제외)
        unstable_ids = [
            str(i.brick_id) for i in issues 
            if i.brick_id is not None and i.issue_type.value in ('floating', 'isolated')
        ]
        unstable_types = {i.issue_type.value for i in issues}
        
        print(f"   - 발견된 물리적 이슈 유형: {', '.join(unstable_types) if unstable_types else '없음'}")
        print(f"   - 탐지된 불안정(보강대상) 브릭 수: {len(unstable_ids)}개 (hanging 제외)")

        # 2. 구조적 병합 (불안정 브릭 보강)
        if not args.skip_structural and unstable_ids:
            print(f"2. 구조적 병합(Structural Merge) 수행 중 (Color-blind)... (대상: {len(unstable_ids)}개)")
            # 1x1 브릭 인덱스/ID 기반으로 병합 실행
            results = structural_merge(str(output_path), unstable_ids)
            print(f"   - 구조 보강 완료")
        
        # 3. 동일 색상 기본 병합 (Cleanup) - 옵션 선택 시에만 수행
        stats = {}
        if args.cleanup:
            print("3. 전체 병합 수행 중 (Cleanup, 색상 무관)...")
            stats = merge_small_bricks(str(output_path), max_len=2, group_by_color=False)
        else:
            print("3. 전체 병합(Cleanup) 건너뜀 (디테일 보존).")
            # structural_merge 결과만 사용하거나, stats 초기화 필요 시
            pass
            
        print("\n[최종 결과]")
        print(f"모델 규모 변화: {stats.get('total_original_count', 0)} -> {stats.get('total_new_count', 0)}")
        print(f"소형 브릭(1x1) 변화: {stats.get('small_brick_count', 0)} -> {stats.get('small_brick_new_count', 0)}")
        print(f"병합된 그룹 수: {stats.get('merged', 0)}개")
        print(f"결과 저장: {output_path.absolute()}")
        
    except Exception as e:
        print(f"\n작업 중 오류 발생: {e}")
        # import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
