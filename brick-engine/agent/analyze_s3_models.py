import os
import sys
import boto3
import tempfile
from pathlib import Path

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent))

from agent.memory_utils import memory_manager
from physical_verification.ldr_loader import LdrLoader
from physical_verification.pybullet_verifier import PyBulletVerifier

def analyze_ldr_file(ldr_path: Path):
    """LDR 파일을 분석하여 물리 메트릭 추출"""
    loader = LdrLoader()
    plan = loader.load_from_file(str(ldr_path))
    
    # 1. 기본 메트릭
    verifier = PyBulletVerifier(plan, gui=False)
    # 충돌 및 안정성 검사 (빠른 분석을 위해 짧게 진행)
    col_result = verifier.run_collision_check()
    stab_result = verifier.run_stability_check(duration=1.0)
    
    # 2. 상세 물리 메트릭 계산 (MemoryUtils의 헬퍼 사용)
    phys_metrics = memory_manager.calculate_model_metrics(plan, stab_result)
    
    return {
        "plan": plan,
        "col_result": col_result,
        "stab_result": stab_result,
        "phys_metrics": phys_metrics
    }

def process_s3_ldr(bucket: str, key: str, session_id: str = "batch_import"):
    """S3의 단일 LDR 파일을 로컬로 받아 분석 후 DB 저장"""
    s3 = boto3.client('s3')
    
    with tempfile.NamedTemporaryFile(suffix=".ldr", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            print(f"📥 Downloading s3://{bucket}/{key}...")
            s3.download_file(bucket, key, str(tmp_path))
            
            # 분석 실행
            analysis = analyze_ldr_file(tmp_path)
            metrics = analysis["phys_metrics"]
            
            # DB 저장 (RAG 사례로 등록)
            print("💾 Logging to RAG DB...")
            memory_manager.log_experiment(
                session_id=session_id,
                model_id=Path(key).name,
                agent_type="historical_data",
                iteration=0,
                hypothesis={"observation": f"Historical data from S3: {key}", "hypothesis": "Recovered from S3"},
                experiment={"tool": "Original Generation", "params": {}},
                verification={
                    "passed": analysis["stab_result"].is_valid,
                    "metrics_after": metrics,
                    "numerical_analysis": f"Volume={metrics['total_volume']:.1f}, Bricks={metrics['total_bricks']}, Stability={analysis['stab_result'].stability_grade}"
                },
                improvement={"lesson_learned": "Historical case imported from S3"},
                async_save=False
            )
            print(f"✅ Processed {key}")
            
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

def main():
    # 설정 (환경 변수 또는 직접 입력)
    BUCKET = os.getenv("AWS_S3_BUCKET", "bricker-uploads")
    PREFIX = os.getenv("S3_PREFIX", "uploads/ai-generated")
    
    print(f"🚀 Starting S3 LDR Analysis Batch | Bucket: {BUCKET}")
    
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('.ldr'):
                try:
                    process_s3_ldr(BUCKET, key)
                except Exception as e:
                    print(f"❌ Failed to process {key}: {e}")

if __name__ == "__main__":
    if not os.getenv("AWS_ACCESS_KEY_ID"):
        print("⚠️ AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.")
    else:
        main()
