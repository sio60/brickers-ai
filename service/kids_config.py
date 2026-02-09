# service/kids_config.py
"""Kids Mode 설정 / 상수 모음"""
from __future__ import annotations

import os
from pathlib import Path


def _is_truthy(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


DEBUG = _is_truthy(os.environ.get("DEBUG", "false"))


def _find_project_root(start: Path) -> Path:
    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent
    markers = ("pyproject.toml", "requirements.txt", ".git")
    for p in [cur] + list(cur.parents):
        for m in markers:
            if (p / m).exists():
                return p
    return cur


PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "")).expanduser()
PROJECT_ROOT = PROJECT_ROOT.resolve() if str(PROJECT_ROOT).strip() else _find_project_root(Path(__file__))

PUBLIC_DIR = Path(os.environ.get("PUBLIC_DIR", PROJECT_ROOT / "public")).resolve()
GENERATED_DIR = Path(os.environ.get("GENERATED_DIR", PUBLIC_DIR / "generated")).resolve()
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

# URL prefix: /api/generated
STATIC_PREFIX = os.environ.get("GENERATED_URL_PREFIX", "/api/generated").rstrip("/")

# Timeouts
KIDS_TOTAL_TIMEOUT_SEC = int(os.environ.get("KIDS_TOTAL_TIMEOUT_SEC", "1800"))
TRIPO_WAIT_TIMEOUT_SEC = int(os.environ.get("TRIPO_WAIT_TIMEOUT_SEC", "900"))
DOWNLOAD_TIMEOUT_SEC = float(os.environ.get("KIDS_DOWNLOAD_TIMEOUT_SEC", "180.0"))

# Concurrency
MAX_CONCURRENT_TASKS = int(os.environ.get("MAX_CONCURRENT_TASKS", "5"))

# Age → budget mapping
AGE_TO_BUDGET = {"4-5": 400, "6-7": 450, "8-10": 500, "PRO": 2000}


def budget_to_start_target(eff_budget: int) -> int:
    if eff_budget <= 400:
        return 35 # 45 -> 35
    if eff_budget <= 450:
        return 40 # 50 -> 40
    if eff_budget <= 500:
        return 45 # 55 -> 45
    if eff_budget <= 2000:
        return 120 # Pro target (Safe starting point)
    return 130
