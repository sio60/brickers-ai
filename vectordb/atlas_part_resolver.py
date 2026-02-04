from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pymongo.collection import Collection

@dataclass
class ResolvedPart:
    part_file: str          # 입력: "3024.dat"
    part_id: str            # Atlas: "3024" (또는 "101" 같은 내부 id)
    canonical_file: str     # Atlas: "3024.dat"
    name: str               # Atlas: "Plate 1 x 1" 같은 이름이 있으면
    doc: Dict[str, Any]     # 원본 문서(필요하면)

def resolve_part_by_file(
    part_file: str,
    parts_col: Collection,
    alias_col: Optional[Collection] = None
) -> Optional[ResolvedPart]:
    pf = (part_file or "").lower().replace("\\", "/").split("/")[-1].strip()
    if not pf:
        return None

    # 1) direct
    doc = parts_col.find_one(
        {"$or": [{"canonicalFile": pf}, {"partFile": pf}, {"name": pf}]},
        {"_id": 1, "partId": 1, "name": 1, "canonicalFile": 1, "partFile": 1, "movedTo": 1},
    )
    if doc:
        return ResolvedPart(
            part_file=pf,
            part_id=str(doc.get("partId", "")),
            canonical_file=str(doc.get("canonicalFile") or doc.get("partFile") or pf),
            name=str(doc.get("name") or ""),
            doc=doc,
        )

    # 2) alias
    if alias_col is not None:
        a = alias_col.find_one(
            {"$or": [{"alias": pf}, {"from": pf}, {"partFile": pf}, {"name": pf}]},
        )
        if a:
            canonical = (a.get("canonicalFile") or a.get("to") or a.get("canonical") or "").lower().strip()
            if canonical:
                doc = parts_col.find_one(
                    {"$or": [{"canonicalFile": canonical}, {"partFile": canonical}, {"name": canonical}]},
                    {"_id": 1, "partId": 1, "name": 1, "canonicalFile": 1, "partFile": 1, "movedTo": 1},
                )
                if doc:
                    return ResolvedPart(
                        part_file=pf,
                        part_id=str(doc.get("partId", "")),
                        canonical_file=str(doc.get("canonicalFile") or doc.get("partFile") or canonical),
                        name=str(doc.get("name") or ""),
                        doc=doc,
                    )

    return None
