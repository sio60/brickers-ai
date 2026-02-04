from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable
from collections import Counter

@dataclass(frozen=True)
class BomKey:
    part: str
    color: int

def normalize_part(token: str) -> str:
    t = token.strip().replace("\\", "/")
    return t.split("/")[-1].lower()

def parse_ldr_steps(ldr_path: str | Path) -> List[List[Tuple[int, str]]]:
    """
    Returns steps as a list.
    Each step is a list of (color_code, part_file) tuples from Type-1 lines.
    STEP boundary: line == '0 STEP' (case-insensitive, leading/trailing spaces allowed)
    """
    path = Path(ldr_path)
    if not path.exists():
        raise FileNotFoundError(f"LDR not found: {path}")

    steps: List[List[Tuple[int, str]]] = []
    current: List[Tuple[int, str]] = []

    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue

        # STEP boundary
        if line.lower() == "0 step":
            steps.append(current)
            current = []
            continue

        # Type-1 line
        if not line.startswith("1 "):
            continue

        tokens = line.split()
        if len(tokens) < 15:
            continue

        try:
            color = int(tokens[1])
        except ValueError:
            continue

        part = normalize_part(tokens[-1])
        current.append((color, part))

    # last step (even if empty)
    if current or not steps:
        steps.append(current)
    return steps

def step_to_counter(step_rows: List[Tuple[int, str]]) -> Counter:
    c = Counter()
    for color, part in step_rows:
        c[BomKey(part=part, color=color)] += 1
    return c

def parse_ldr_step_boms(ldr_path: str | Path) -> List[Dict[BomKey, int]]:
    steps = parse_ldr_steps(ldr_path)
    return [dict(step_to_counter(rows)) for rows in steps]
