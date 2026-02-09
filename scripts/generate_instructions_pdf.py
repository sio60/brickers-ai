# scripts/generate_instructions_pdf.py
from __future__ import annotations

import sys
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]   # brickers-ai/
sys.path.insert(0, str(ROOT))               # ✅ 루트를 import 경로에 추가

from db import get_db                        # ✅ 이제 잡힘

INSTR_CANDIDATES = [
    ROOT / "brick_engine" / "exporter" / "instructions_pdf.py",   # ✅ 현재 네 구조 가능성 1
    ROOT / "brick_engine" / "exporter" / "instructions_pdf.py",   # 가능성 2 (하이픈)
]

def find_instructions_path() -> Path:
    for p in INSTR_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("instructions_pdf.py not found. tried:\n" + "\n".join(map(str, INSTR_CANDIDATES)))

INSTR_PATH = find_instructions_path()

def load_instructions_module():
    if not INSTR_PATH.exists():
        raise FileNotFoundError(f"instructions_pdf.py not found: {INSTR_PATH}")

    spec = importlib.util.spec_from_file_location("instructions_pdf", str(INSTR_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create module spec for: {INSTR_PATH}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

def main(job_id: str, out_dir: str = "out", mode: str = "pro", images_dir: str | None = None):
    db = get_db()
    doc = db["boms"].find_one({"jobId": job_id})
    if not doc:
        raise RuntimeError(f"[NOT FOUND] boms jobId={job_id}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pdf_path = out_path / f"instructions_{job_id}.pdf"

    instr = load_instructions_module()
    out_pdf = instr.generate_instructions_pdf_from_boms_doc(
        boms_doc=doc,
        output_path=str(pdf_path),
        mode=mode,
        include_overall_bom=True,
        step_images_dir=images_dir,
    )
    print("[OK] generated:", out_pdf)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobId", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--mode", default="pro", choices=["pro", "kids"])
    ap.add_argument("--imagesDir", default=None)
    args = ap.parse_args()

    main(args.jobId, args.out, args.mode, args.imagesDir)
