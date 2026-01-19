from __future__ import annotations

import sys
from pathlib import Path
from dotenv import load_dotenv
import os
from pymongo import MongoClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brick_engine.exporter.renderer_matplotlib import render_ldr_to_png_matplotlib


def _step_index(p: Path) -> int:
    """
    pyramid_step_01.ldr / pyramid_step_02.ldr ... 에서 01/02 파싱
    실패하면 파일명 정렬 fallback(큰 값)
    """
    stem = p.stem  # pyramid_step_01
    for token in reversed(stem.split("_")):
        if token.isdigit():
            return int(token)
    return 10**9


def main(ldr_dir: str, out_dir: str, dpi: int = 180):
    load_dotenv()
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI not set (.env 확인)")

    client = MongoClient(uri)
    db = client["brickers"]
    parts_col = db["ldraw_parts"]

    in_dir = Path(ldr_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ldr_files = list(in_dir.glob("*.ldr"))
    if not ldr_files:
        raise RuntimeError(f"No .ldr files in {in_dir}")

    # ✅ step 번호 기준으로 정렬 (문자열 정렬로 10이 2 앞에 오는 문제 방지)
    ldr_files = sorted(ldr_files, key=_step_index)

    prev = None
    for i, p in enumerate(ldr_files):
        png_path = out / f"{p.stem}.png"

        # ✅ step 강조: 두 번째 스텝부터 이전 누적 LDR을 prev로 넘김
        prev_ldr = prev if prev is not None else None

        render_ldr_to_png_matplotlib(
            ldr_path=p,
            output_path=png_path,
            parts_col=parts_col,
            prev_ldr_path=prev_ldr,  # ✅ 핵심
            dpi=dpi,
        )

        print("[OK]", png_path)

        # ✅ 다음 스텝을 위한 prev 갱신
        prev = p


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ldrDir", required=True, help="directory containing *_step_XX.ldr files")
    ap.add_argument("--outDir", default="out/steps_png")
    ap.add_argument("--dpi", type=int, default=180)
    args = ap.parse_args()

    main(args.ldrDir, args.outDir, dpi=args.dpi)
