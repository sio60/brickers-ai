"""
validate_ldr_to_atlas_mapping.py

- LDR 파일의 type-1 part file(.dat)을 MongoDB Atlas(brickers.ldraw_parts)에서 매핑 가능한지 검증
- 우선순위:
  1) ldraw_parts: canonicalFile / partFile / name == part_file
  2) ldraw_aliases를 통해 canonicalFile 추정 후 ldraw_parts 재조회
- 실행:
  python scripts/validate_ldr_to_atlas_mapping.py
"""

from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
import os
from pymongo import MongoClient


# ----------------------------
# Setup
# ----------------------------
load_dotenv()

uri = os.getenv("MONGODB_URI")
if not uri:
    raise RuntimeError("MONGODB_URI is not set. Check .env and load_dotenv()")

client = MongoClient(uri, serverSelectionTimeoutMS=5000)
db = client["brickers"]
parts = db["ldraw_parts"]
aliases = db["ldraw_aliases"]


# ----------------------------
# LDR parsing
# ----------------------------
def extract_unique_part_files(ldr_path: Path) -> Counter:
    c = Counter()

    for line in ldr_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line.startswith("1 "):
            continue

        tokens = line.split()
        if len(tokens) < 15:
            continue

        # last token is part file: can be "parts/3001.dat" or "3001.dat"
        part_file = tokens[-1].lower().replace("\\", "/").split("/")[-1]
        c[part_file] += 1

    return c


# ----------------------------
# Atlas lookup
# ----------------------------
def atlas_lookup_part(part_file: str):
    """
    Return a part doc if found, else None.
    """
    # 1) direct match
    doc = parts.find_one(
        {"$or": [{"canonicalFile": part_file}, {"partFile": part_file}, {"name": part_file}]},
        {"_id": 1, "partId": 1, "name": 1, "canonicalFile": 1, "partFile": 1, "movedTo": 1},
    )
    if doc:
        return doc

    # 2) alias-based match (⚠️ alias 스키마가 다를 수 있으니 여러 키로 넓게 탐색)
    a = aliases.find_one(
        {"$or": [{"alias": part_file}, {"from": part_file}, {"partFile": part_file}, {"name": part_file}]}
    )
    if not a:
        return None

    canonical = (a.get("canonicalFile") or a.get("to") or a.get("canonical") or "").lower().strip()
    if not canonical:
        return None

    doc = parts.find_one(
        {"$or": [{"canonicalFile": canonical}, {"partFile": canonical}, {"name": canonical}]},
        {"_id": 1, "partId": 1, "name": 1, "canonicalFile": 1, "partFile": 1, "movedTo": 1},
    )
    return doc


# ----------------------------
# Main
# ----------------------------
def main():
    # ✅ scripts/validate_ldr_to_atlas_mapping.py 기준
    # project_root = scripts 폴더의 부모
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # ✅ car.ldr 위치 자동 탐색: (1) project_root/car.ldr, (2) project_root/ldr/car.ldr
    candidates = [
        project_root / "car.ldr",
        project_root / "ldr" / "car.ldr",
    ]

    ldr_path = None
    for p in candidates:
        if p.exists():
            ldr_path = p
            break

    if ldr_path is None:
        raise FileNotFoundError(
            "car.ldr not found. Tried:\n" + "\n".join(str(p) for p in candidates)
        )

    print("[INFO] Using LDR:", ldr_path)

    # Ping
    print("[INFO] Mongo ping:", client.admin.command("ping"))
    print("[INFO] DB:", db.name)
    print("[INFO] parts count:", parts.estimated_document_count())
    print("[INFO] aliases count:", aliases.estimated_document_count())

    cnt = extract_unique_part_files(ldr_path)
    total_unique = len(cnt)

    mapped = 0
    unknown = []
    mapped_examples = []

    for pf in cnt.keys():
        doc = atlas_lookup_part(pf)
        if doc:
            mapped += 1
            if len(mapped_examples) < 10:
                mapped_examples.append((pf, doc.get("partId"), doc.get("canonicalFile") or doc.get("partFile")))
        else:
            unknown.append(pf)

    print(f"\n[MAPPING] mapped {mapped}/{total_unique} unique parts ({(mapped/total_unique*100 if total_unique else 0):.1f}%)")

    print("\n[MAPPED sample]")
    for pf, part_id, canon in mapped_examples:
        print(f"  {pf} -> partId={part_id}, canonical={canon}")

    print("\n[UNKNOWN sample] (first 30)")
    print(unknown[:30])

    # 추가로: unknown이 primitive일 가능성 체크용(선택)
    prim_like = [u for u in unknown if u.startswith(("stud", "axle", "pin")) or u in ("box5.dat",)]
    if prim_like:
        print("\n[HINT] Some unknowns look like primitives/aux parts:", prim_like[:20])


if __name__ == "__main__":
    main()
