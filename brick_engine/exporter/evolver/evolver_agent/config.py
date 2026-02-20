from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

@dataclass
class AgentConfig:
    parts_db: Dict[str, Any]
    exporter_dir: Path
    is_initialized: bool = False
    vision_model: str = "gpt-4o-mini"

_CONFIG: Optional[AgentConfig] = None

def init_config(parts_db: Dict[str, Any], exporter_dir: Path) -> None:
    global _CONFIG
    _CONFIG = AgentConfig(parts_db=parts_db, exporter_dir=exporter_dir, is_initialized=True)

def get_config() -> AgentConfig:
    if _CONFIG is None:
        # Default empty config if not initialized (safe fallback)
        return AgentConfig({}, Path("."), is_initialized=False)
    return _CONFIG
