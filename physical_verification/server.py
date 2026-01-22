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

# Import verification logic
# Assuming server.py is in the same directory as these modules
try:
    from ldr_loader import LdrLoader
    from models import VerificationResult, Evidence
    from verifier import PhysicalVerifier
except ImportError:
    # Fallback for when running from root or different context if needed
    from .ldr_loader import LdrLoader
    from .models import VerificationResult, Evidence
    from .verifier import PhysicalVerifier

app = FastAPI()

# Enable CORS
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
        # Save uploaded file
        suffix = os.path.splitext(file.filename)[1]
        if not suffix:
            suffix = ".ldr"
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_file_path = tmp.name

        print(f"Processing file: {temp_file_path}")

        # Load LDR
        loader = LdrLoader()
        try:
            plan = loader.load_from_file(temp_file_path)
            print(f"Loaded plan with {len(plan.bricks)} bricks.")
        except Exception as e:
            print(f"Load error: {e}")
            return {"is_valid": False, "score": 0, "evidence": [], "error": f"Failed to load LDR: {str(e)}"}

        # Verify
        result = VerificationResult()
        verifier = PhysicalVerifier(plan)
        
        print("Running floating check...")
        verifier.verify_floating(result)
        
        print("Running stability check...")
        verifier.verify_stability(result)

        # Serialize result
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
        print("Verification complete.")
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
