# ============================================================================
# Evolver Post-Processing (직접 호출 방식)
# ============================================================================

import os
import sys
import shutil
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_AGENT_DIR = _THIS_DIR.parent
_BRICK_ENGINE_DIR = _AGENT_DIR.parent
_EVOLVER_DIR = _BRICK_ENGINE_DIR / "exporter" / "evolver"


def run_evolver(ldr_path: str, glb_path: str = None) -> dict:
    """Evolver 에이전트를 직접 호출로 실행 (형태 개선)

    메인 에이전트 완료 후 후처리로 실행.
    실패해도 원본 LDR은 보존됨.
    """
    if not Path(ldr_path).exists():
        return {"success": False, "reason": "LDR file not found"}

    # evolver 디렉토리를 sys.path에 추가 (run_agent.py의 import 해결)
    evolver_str = str(_EVOLVER_DIR)
    if evolver_str not in sys.path:
        sys.path.insert(0, evolver_str)

    try:
        from run_agent import run_evolver_direct
        result = run_evolver_direct(ldr_path, glb_path)

        if not result.get("success"):
            return result

        # evolved 파일을 원본에 덮어쓰기 (기존 동작 유지)
        ldr_p = Path(ldr_path)
        evolved_path = ldr_p.parent / f"{ldr_p.stem}_evolved.ldr"

        if evolved_path.exists() and evolved_path.stat().st_size > 0:
            shutil.copy2(str(evolved_path), str(ldr_path))
            evolved_path.unlink()
            return {"success": True}
        else:
            return {"success": False, "reason": "No evolved file generated"}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "reason": str(e)}


# 하위 호환: 기존 이름으로도 호출 가능
run_evolver_subprocess = run_evolver
