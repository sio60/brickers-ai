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


def main(ldr_dir: str | None, ldr_file: str | None, out_dir: str, dpi: int = 180):
    load_dotenv()
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI not set (.env 확인)")

    client = MongoClient(uri)
    db = client["brickers"]
    parts_col = db["ldraw_parts"]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ✅ (1) 단일 파일 모드
    if ldr_file:
        p = Path(ldr_file)
        if not p.exists():
            raise FileNotFoundError(p)

        png_path = out / f"{p.stem}.png"
        render_ldr_to_png_matplotlib(
            ldr_path=p,
            output_path=png_path,
            parts_col=parts_col,
            prev_ldr_path=None,
            dpi=dpi,
        )
        print("[OK]", png_path)
        return

    # ✅ (2) 폴더 모드 (기존 그대로)
    in_dir = Path(ldr_dir)
    ldr_files = list(in_dir.glob("*.ldr"))
    if not ldr_files:
        raise RuntimeError(f"No .ldr files in {in_dir}")

    ldr_files = sorted(ldr_files, key=_step_index)

    prev = None
    for p in ldr_files:
        png_path = out / f"{p.stem}.png"
        render_ldr_to_png_matplotlib(
            ldr_path=p,
            output_path=png_path,
            parts_col=parts_col,
            prev_ldr_path=prev,
            dpi=dpi,
        )
        print("[OK]", png_path)
        prev = p


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ldrDir", help="directory containing *_step_XX.ldr files")
    ap.add_argument("--ldr", help="single .ldr file")
    ap.add_argument("--outDir", default="out/steps_png")
    ap.add_argument("--dpi", type=int, default=180)
    args = ap.parse_args()

    if not args.ldr and not args.ldrDir:
        raise SystemExit("Need --ldr or --ldrDir")

    main(args.ldrDir, args.ldr, args.outDir, dpi=args.dpi)