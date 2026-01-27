# ============================================================================
# 물리 검증 API 서버 모듈
# 이 파일은 FastAPI를 사용하여 LDraw(.ldr) 파일의 물리적 검증을 위한
# API 서버를 구현합니다. 클라이언트로부터 LDR 파일을 업로드받아
# 로드하고, 해당 파일에 대한 부동(floating) 및 안정성(stability) 검사를
# 수행한 후, 그 결과를 JSON 형태로 반환하는 엔드포인트를 제공합니다.
# ============================================================================
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import shutil
from typing import List, Dict, Any

# 검증 로직 임포트
# server.py가 이 모듈들과 같은 디렉토리에 있다고 가정함
try:
    from ldr_loader import LdrLoader
    from models import VerificationResult, Evidence
    from verifier import PhysicalVerifier
except ImportError:
    # 필요한 경우 루트 또는 다른 컨텍실트에서 실행될 때를 위한 폴백
    from .ldr_loader import LdrLoader
    from .models import VerificationResult, Evidence
    from .verifier import PhysicalVerifier

app = FastAPI()

# CORS 설정 활성화
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/verify")
async def verify_ldr(file: UploadFile = File(...)):
    temp_file_path = ""
    try:
        # 업로드된 파일 저장
        suffix = os.path.splitext(file.filename)[1]
        if not suffix:
            suffix = ".ldr"
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_file_path = tmp.name

        print(f"파일 처리 중: {temp_file_path}")

        # LDR 로드
        loader = LdrLoader()
        try:
            plan = loader.load_from_file(temp_file_path)
            print(f"{len(plan.bricks)}개의 브릭을 포함한 플랜을 로드했습니다.")
        except Exception as e:
            print(f"로드 오류: {e}")
            return {"is_valid": False, "score": 0, "evidence": [], "error": f"LDR 로드 실패: {str(e)}"}

        # 검증 수행
        result = VerificationResult()
        verifier = PhysicalVerifier(plan)
        
        print("부동 브릭 검사 실행 중...")
        verifier.verify_floating(result)
        
        print("안정성 검사 실행 중...")
        verifier.verify_stability(result)

        # 결과 직렬화
        response_data = {
            "is_valid": result.is_valid,
            "score": result.score,
            "evidence": [
                {
                    "type": ev.type,
                    "severity": ev.severity,
                    "brick_ids": ev.brick_ids,
                    "message": ev.message
                } for ev in result.evidence
            ]
        }
        print("검증 완료.")
        return response_data
        
    except Exception as e:
        print(f"Server error: {e}")
        return {"is_valid": False, "score": 0, "evidence": [], "error": str(e)}
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
