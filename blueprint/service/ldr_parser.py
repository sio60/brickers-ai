# blueprint/service/ldr_parser.py
"""LDR 파일 BOM 파싱 — 프론트엔드 parseBOM()과 동일한 형식"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class PartInfo:
    id: str
    color: int
    count: int


@dataclass
class StepBOM:
    stepIndex: int
    parts: List[PartInfo]


def normalize_part(token: str) -> str:
    """파일명 정규화: 경로 제거, 확장자 제거, 소문자"""
    t = token.strip().replace("\\", "/")
    name = t.split("/")[-1].lower()
    return name.replace(".dat", "").replace(".ldr", "")


def parse_ldr_step_boms(ldr_text: str) -> List[StepBOM]:
    """LDR 텍스트를 Step 단위로 파싱하여 BOM 리스트 반환"""
    lines = ldr_text.replace("\r\n", "\n").split("\n")
    steps: List[StepBOM] = []
    current_parts: Dict[str, PartInfo] = {}
    step_index = 1

    def flush_step():
        nonlocal step_index, current_parts
        steps.append(StepBOM(
            stepIndex=step_index,
            parts=list(current_parts.values()),
        ))
        current_parts = {}
        step_index += 1

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if re.match(r"^0\s+(STEP|ROTSTEP)\b", line, re.IGNORECASE):
            flush_step()
            continue

        if line.startswith("1 "):
            tokens = line.split()
            if len(tokens) >= 15:
                try:
                    color = int(tokens[1])
                except ValueError:
                    continue

                part_id = normalize_part(tokens[-1])
                key = f"{part_id}_{color}"

                if key in current_parts:
                    current_parts[key].count += 1
                else:
                    current_parts[key] = PartInfo(id=part_id, color=color, count=1)

    flush_step()
    return steps
