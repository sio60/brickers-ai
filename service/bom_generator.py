# service/bom_generator.py
"""LDR -> BOM (Bill of Materials) 변환"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict


def generate_bom_from_ldr(ldr_path: Path) -> Dict[str, Any]:
    """
    LDR 파일에서 BOM 생성
    Returns: {"total_parts": int, "parts": [{"part_id": str, "color": str, "quantity": int}, ...]}
    """
    if not ldr_path.exists():
        return {"total_parts": 0, "parts": []}

    content = ldr_path.read_text(encoding="utf-8", errors="ignore")
    lines = content.splitlines()

    parts_counter: Counter = Counter()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("0"):
            continue

        parts = line.split()
        if len(parts) >= 15 and parts[0] == "1":
            color = parts[1]
            part_id = parts[14] if len(parts) > 14 else "unknown"
            if part_id.endswith(".dat"):
                part_id = part_id[:-4]
            key = f"{part_id}_{color}"
            parts_counter[key] += 1

    bom_parts = []
    for key, qty in parts_counter.most_common():
        part_id, color = key.rsplit("_", 1)
        bom_parts.append({"part_id": part_id, "color": color, "quantity": qty})

    return {"total_parts": sum(parts_counter.values()), "parts": bom_parts}
