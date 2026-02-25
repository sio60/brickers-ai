# service/brickify_loader.py
"""brick_engine 동적 로드"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Dict

from service.kids_config import PROJECT_ROOT

_CONVERT_FN = None
_REGEN_LOOP_FN = None
_GEMINI_CLIENT_CLS = None


def load_engine_convert():
    """glb_to_ldr_embedded.convert_glb_to_ldr 함수 로드 (lazy singleton)"""
    global _CONVERT_FN
    if _CONVERT_FN is not None:
        return _CONVERT_FN

    import importlib.util

    engine_path = (PROJECT_ROOT / "brick_engine" / "exporter" / "ldr_converter" / "glb_to_ldr_embedded.py").resolve()
    if not engine_path.exists():
        raise RuntimeError(f"engine file missing: {engine_path}")

    spec = importlib.util.spec_from_file_location("glb_to_ldr_embedded", str(engine_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load spec for glb_to_ldr_embedded")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore
    if not hasattr(mod, "convert_glb_to_ldr"):
        raise RuntimeError("convert_glb_to_ldr not found in engine module")

    _CONVERT_FN = mod.convert_glb_to_ldr
    return _CONVERT_FN


def load_agent_modules():
    """brick_engine.agent 에서 regeneration_loop, GeminiClient 로드 (lazy singleton)

    NOTE: brick_engine/ 를 sys.path에 넣고 `from agent.*` 로 임포트하면
    agent 가 top-level 패키지가 되어, 내부 `from ....exporter` 같은
    4-level 상대 임포트가 "attempted relative import beyond top-level package" 에러 발생.
    반드시 `brick_engine.agent.*` 정규 패키지 경로로 임포트해야 함.
    """
    global _REGEN_LOOP_FN, _GEMINI_CLIENT_CLS
    if _REGEN_LOOP_FN is not None:
        return _REGEN_LOOP_FN, _GEMINI_CLIENT_CLS

    from brick_engine.agent.regeneration import regeneration_loop
    from brick_engine.agent.core.llm_clients import GeminiClient

    _REGEN_LOOP_FN = regeneration_loop
    _GEMINI_CLIENT_CLS = GeminiClient
    return _REGEN_LOOP_FN, _GEMINI_CLIENT_CLS


def find_glb_in_dir(out_dir: Path) -> Optional[Path]:
    glbs = [p for p in out_dir.rglob("*.glb") if p.is_file() and p.stat().st_size > 0]
    return glbs[0] if glbs else None


def pick_glb_from_downloaded(downloaded: Dict[str, str], out_dir: Path) -> Optional[Path]:
    for _, v in (downloaded or {}).items():
        p = Path(v)
        if p.suffix.lower() == ".glb" and p.exists() and p.stat().st_size > 0:
            return p
    return find_glb_in_dir(out_dir)
