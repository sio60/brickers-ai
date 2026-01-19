# scripts/render_steps.py
from __future__ import annotations
import sys
import argparse
from pathlib import Path
from typing import List

# ✅ 프로젝트 루트(brickers-ai)를 import path에 추가
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ldr.step_bom import parse_ldr_steps  # 너가 만든 parse_ldr_steps 그대로 사용

def main(ldr: str, out: str):
    ldr_path = Path(ldr)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 원본에서 STEP 단위로 (color, part)만 뽑히는 게 아니라
    # 실제 렌더를 위해 "원본 type-1 라인 그대로"를 써야 함.
    # => 원본 파일을 라인 단위로 step별로 분해해서 누적 저장하는 방식이 가장 안전.

    lines = ldr_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    # step_blocks: 각 step에 들어가는 "type-1 라인 원문" 리스트
    step_blocks: List[List[str]] = []
    cur: List[str] = []

    for raw in lines:
        s = raw.strip()
        if s.lower() == "0 step":
            step_blocks.append(cur)
            cur = []
            continue
        if s.startswith("1 "):
            cur.append(raw)  # 원문 유지
    step_blocks.append(cur)  # 마지막 step

    # 누적 저장
    cumulative: List[str] = []
    stem = ldr_path.stem

    written = 0
    for i, block in enumerate(step_blocks, start=1):
        if not block:
            continue
        cumulative.extend(block)

        out_path = out_dir / f"{stem}_step_{i:02d}.ldr"
        out_lines = [
            f"0 Name: {out_path.name}",
            f"0 Generated from: {ldr_path.name}",
            "0",
            *cumulative,
            "0",
        ]
        out_path.write_text("\n".join(out_lines), encoding="utf-8")
        written += 1
        print(out_path)

    print(f"[OK] wrote {written} cumulative LDRs")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ldr", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.ldr, args.out)
