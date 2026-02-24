from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Callable

@dataclass
class AgentConfig:
    parts_db: Dict[str, Any]
    exporter_dir: Path
    is_initialized: bool = False
    vision_model: str = "gpt-4o"
    log_callback: Optional[Callable] = None

_CONFIG: Optional[AgentConfig] = None

def init_config(parts_db: Dict[str, Any], exporter_dir: Path, log_callback: Optional[Callable] = None) -> None:
    global _CONFIG
    _CONFIG = AgentConfig(parts_db=parts_db, exporter_dir=exporter_dir, is_initialized=True, log_callback=log_callback)

def get_config() -> AgentConfig:
    if _CONFIG is None:
        return AgentConfig({}, Path("."), is_initialized=False)
    return _CONFIG

def log_sse(step: str, message: str) -> None:
    if _CONFIG and _CONFIG.log_callback:
        try:
            _CONFIG.log_callback(step, message)
        except Exception:
            pass
