# ============================================================================
# CoScientist 로컬 테스트 스크립트
# Docker 컨테이너 내에서 GLB 파일로 에이전트 파이프라인을 직접 실행합니다.
# Triposr 토큰을 사용하지 않고 기존 GLB 파일로 병합/검증 로직을 테스트합니다.
#
# 사용법:
#   docker exec -it <container_id> python test_agent.py
#   또는
#   docker exec -it <container_id> python test_agent.py --glb /path/to/file.glb --level 3
# ============================================================================

import argparse
import asyncio
import os
import sys

# brick_engine을 import하기 위해 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# 레벨별 파라미터 설정
LEVEL_CONFIGS = {
    1: {"budget": 400,  "target": 25, "max_new_voxels": 6000,  "label": "L1 (4-5세)"},
    2: {"budget": 800,  "target": 35, "max_new_voxels": 6000,  "label": "L2 (6-7세)"},
    3: {"budget": 1200, "target": 50, "max_new_voxels": 20000, "label": "L3 (8-10세)"},
    4: {"budget": 5000, "target": 100,"max_new_voxels": 50000, "label": "PRO"},
}


async def run_test(glb_path: str, level: int = 3, max_retries: int = 2):
    """에이전트 파이프라인 실행"""
    from brick_engine.agent.regeneration.pipeline import regeneration_loop
    from brick_engine.agent.llm_clients import GeminiClient

    config = LEVEL_CONFIGS.get(level, LEVEL_CONFIGS[3])
    print(f"\n{'='*60}")
    print(f"🧪 CoScientist 로컬 테스트")
    print(f"{'='*60}")
    print(f"  GLB: {glb_path}")
    print(f"  레벨: {config['label']}")
    print(f"  Budget: {config['budget']}")
    print(f"  Target: {config['target']}")
    print(f"  Max Retries: {max_retries}")
    print(f"{'='*60}\n")

    # 출력 경로 (GLB 파일 옆에 result.ldr 생성)
    from pathlib import Path
    glb_dir = Path(glb_path).parent
    output_ldr = str(glb_dir / "test_result.ldr")

    # 파라미터 구성 (kids_render.py와 동일)
    params = dict(
        target=config["target"],
        budget=config["budget"],
        min_target=5,
        shrink=0.6,
        search_iters=10,
        kind="brick",
        plates_per_voxel=3,
        interlock=True,
        max_area=20,
        solid_color=4,
        use_mesh_color=True,
        invert_y=False,
        smart_fix=True,
        span=4,
        max_new_voxels=config["max_new_voxels"],
        refine_iters=4,
        ensure_connected=True,
        min_embed=2,
        erosion_iters=1,
        fast_search=True,
        step_order="bottomup",
        extend_catalog=True,
        max_len=8,
        avoid_1x1=True,
    )

    # Gemini 클라이언트 (환경변수에서 API 키 로드)
    client = GeminiClient()

    # 파이프라인 실행
    final_state = await regeneration_loop(
        glb_path=glb_path,
        output_ldr_path=output_ldr,
        subject_name="Test Object",
        llm_client=client,
        max_retries=max_retries,
        acceptable_failure_ratio=0.1,
        params=params,
    )

    # 결과 출력
    print(f"\n{'='*60}")
    print(f"📋 테스트 결과")
    print(f"{'='*60}")

    report = final_state.get('final_report', {})
    if report:
        success = report.get('success', False)
        print(f"  상태: {'✅ 성공' if success else '❌ 실패'}")
        print(f"  총 시도: {report.get('total_attempts', 0)}회")
        print(f"  메시지: {report.get('message', '')}")

        metrics = report.get('final_metrics', {})
        if metrics:
            print(f"  총 브릭: {metrics.get('total_bricks', 0)}개")
            print(f"  실패율: {metrics.get('failure_ratio', 0)*100:.1f}%")
            print(f"  1x1 비율: {metrics.get('small_brick_ratio', 0)*100:.1f}%")

        tool_usage = report.get('tool_usage', {})
        if tool_usage:
            print(f"  도구 사용:")
            for tool, count in tool_usage.items():
                print(f"    - {tool}: {count}회")

    print(f"\n  출력 LDR: {output_ldr}")
    if Path(output_ldr).exists():
        ldr_text = Path(output_ldr).read_text(encoding='utf-8')
        brick_count = sum(1 for line in ldr_text.splitlines() if line.startswith('1 '))
        print(f"  LDR 브릭 수: {brick_count}개")
    else:
        print(f"  ⚠️ LDR 파일 생성 안 됨")

    print(f"{'='*60}")
    return final_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CoScientist 로컬 테스트")
    parser.add_argument(
        "--glb",
        default="./e7b6107d-e9df-41f8-b953-3d687d90fdb0_pbr.glb",
        help="입력 GLB 파일 경로"
    )
    parser.add_argument("--level", type=int, default=3, choices=[1,2,3,4], help="레벨 (1=L1, 2=L2, 3=L3, 4=PRO)")
    parser.add_argument("--max-retries", type=int, default=2, help="최대 재시도 횟수")

    args = parser.parse_args()

    if not os.path.exists(args.glb):
        print(f"❌ GLB 파일을 찾을 수 없습니다: {args.glb}")
        sys.exit(1)

    asyncio.run(run_test(args.glb, args.level, args.max_retries))
