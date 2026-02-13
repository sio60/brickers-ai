# ============================================================================
# Evolver Post-Processing (서브프로세스 방식)
# ============================================================================

import sys
import subprocess
import shutil
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_AGENT_DIR = _THIS_DIR.parent
_BRICK_ENGINE_DIR = _AGENT_DIR.parent


def run_evolver_subprocess(ldr_path: str, glb_path: str = None) -> dict:
    """Evolver 에이전트를 서브프로세스로 실행 (형태 개선)

    메인 에이전트 완료 후 후처리로 실행.
    agent 패키지 이름 충돌 방지를 위해 별도 프로세스로 격리.
    실패해도 원본 LDR은 보존됨.
    """
    evolver_script = _BRICK_ENGINE_DIR / "exporter" / "evolver" / "run_agent.py"

    if not evolver_script.exists():
        return {"success": False, "reason": "evolver run_agent.py not found"}

    if not Path(ldr_path).exists():
        return {"success": False, "reason": "LDR file not found"}

    cmd = [sys.executable, str(evolver_script), str(ldr_path)]
    if glb_path and Path(glb_path).exists():
        cmd.append(str(glb_path))

    try:
        result = subprocess.run(
            cmd,
            timeout=300,
            cwd=str(evolver_script.parent),
        )

        ldr_p = Path(ldr_path)
        evolved_path = ldr_p.parent / f"{ldr_p.stem}_evolved.ldr"

        if evolved_path.exists() and evolved_path.stat().st_size > 0:
            shutil.copy2(str(evolved_path), str(ldr_path))
            evolved_path.unlink()
            return {"success": True}
        else:
            return {"success": False, "reason": "No evolved file generated"}

    except subprocess.TimeoutExpired:
        return {"success": False, "reason": "Timeout (5min)"}
    except Exception as e:
        return {"success": False, "reason": str(e)}
