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

# Age → budget mapping (L1: 400, L2: 800, L3: 1200)
AGE_TO_BUDGET = {"4-5": 400, "6-7": 800, "8-10": 1200, "PRO": 5000}


def budget_to_start_target(eff_budget: int) -> int:
    if eff_budget <= 400:
        return 25
    if eff_budget <= 800:
        return 35
    if eff_budget <= 1200:
        return 50
    if eff_budget >= 2000: # PRO 모드
        return 100
    return 60


# --- Model Pricing (Price per 1M tokens in USD) ---
MODEL_PRICING = {
    # Gemini Models (Google)
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 3.50, "output": 10.50},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    
    # OpenAI Models
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    
    # Anthropic Models
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
}

# Fixed generation costs
TRIPO_GEN_COST = 0.30  # Assumed fixed cost per successful generation


def calculate_token_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """토큰 수와 모델명을 기반으로 예상 비용(USD)을 정확히 계산합니다."""
    m_lower = model_name.lower()
    pricing = None
    
    # 1. OpenAI Matching
    if "gpt-4o-mini" in m_lower:
        pricing = MODEL_PRICING["gpt-4o-mini"]
    elif "gpt-4o" in m_lower:
        pricing = MODEL_PRICING["gpt-4o"]
        
    # 2. Anthropic Matching
    elif "claude-3-5-sonnet" in m_lower or "claude-3.5-sonnet" in m_lower:
        pricing = MODEL_PRICING["claude-3-5-sonnet-20241022"]
        
    # 3. Gemini Matching
    elif "flash" in m_lower:
        if "2.0" in m_lower or "25" in m_lower:
             pricing = MODEL_PRICING["gemini-2.0-flash"]
        else:
             pricing = MODEL_PRICING["gemini-1.5-flash"]
    elif "pro" in m_lower:
        pricing = MODEL_PRICING["gemini-1.5-pro"]
    
    # Fallback to a cheap default if completely unknown
    if not pricing:
        pricing = MODEL_PRICING["gemini-1.5-flash"]
        
    cost = (input_tokens * (pricing["input"] / 1_000_000)) + (output_tokens * (pricing["output"] / 1_000_000))
    return round(cost, 6)

