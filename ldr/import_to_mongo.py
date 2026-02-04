from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Dict

from pymongo.collection import Collection

from db import get_db
from ldr.bom import parse_ldr_bom

from vectordb.atlas_part_resolver import resolve_part_by_file
from ldr.step_bom import parse_ldr_step_boms, BomKey
from ldr.colors import get_color_name

def get_boms_collection() -> Collection:
    return get_db()["boms"]


def bom_to_items_with_atlas(bom, parts_col, aliases_col) -> List[dict]:
    items = []
    for k, qty in sorted(bom.items(), key=lambda x: (-x[1], x[0].part, x[0].color)):
        resolved = resolve_part_by_file(k.part, parts_col=parts_col, alias_col=aliases_col)

        # Atlas 문서에서 있을 법한 메타(없으면 None)
        doc = resolved.doc if resolved else {}
        category = doc.get("category") or doc.get("partCategory") or None
        bbox = doc.get("bbox") or doc.get("boundingBox") or None  # 스키마 다르면 None

        items.append({
            "partFile": k.part,
            "canonicalFile": resolved.canonical_file if resolved else k.part,
            "partId": resolved.part_id if resolved else None,
            "name": resolved.name if resolved else None,
            "category": category,
            "bbox": bbox,
            "color": k.color,
            "colorName": get_color_name(k.color),
            "qty": qty,
            "resolved": bool(resolved),
        })
    return items



def import_ldr_bom_with_steps(job_id: str, ldr_path: str, candidate_id: int | None = None) -> Dict:
    # 1) STEP별 bom
    step_boms = parse_ldr_step_boms(ldr_path)

    db = get_db()
    parts_col = db["ldraw_parts"]
    aliases_col = db["ldraw_aliases"]
    col = get_boms_collection()

    # 2) step 문서 만들기
    steps_doc = []
    for idx, step_bom in enumerate(step_boms, start=1):
        step_items = bom_to_items_with_atlas(step_bom, parts_col, aliases_col)
        steps_doc.append({
            "step": idx,
            "uniqueItems": len(step_items),
            "totalQty": sum(i["qty"] for i in step_items),
            "items": step_items,
            "stats": {
                "resolvedItems": sum(1 for i in step_items if i["resolved"]),
                "unresolvedItems": sum(1 for i in step_items if not i["resolved"]),
            }
        })

    # 3) 전체 BOM(전체 step 합산)
    merged = {}
    for step_bom in step_boms:
        for k, v in step_bom.items():
            merged[k] = merged.get(k, 0) + v

    all_items = bom_to_items_with_atlas(merged, parts_col, aliases_col)

    now = datetime.utcnow()
    doc = {
        "jobId": job_id,
        "candidateId": candidate_id,  # None이면 단일
        "source": {"type": "ldr", "path": str(Path(ldr_path).resolve())},
        "uniqueItems": len(all_items),
        "totalQty": sum(i["qty"] for i in all_items),
        "items": all_items,      # 전체 BOM
        "steps": steps_doc,      # 구조 매핑 핵심
        "stepCount": len(steps_doc),
        "stats": {
            "resolvedItems": sum(1 for i in all_items if i["resolved"]),
            "unresolvedItems": sum(1 for i in all_items if not i["resolved"]),
        },
        "updatedAt": now,
    }

    # candidate 구조 고려: candidateId 있으면 복합키로 upsert 추천
    filter_q = {"jobId": job_id}
    if candidate_id is not None:
        filter_q["candidateId"] = candidate_id

    col.update_one(
        filter_q,
        {"$set": doc, "$setOnInsert": {"createdAt": now}},
        upsert=True,
    )

    return {
        "jobId": job_id,
        "candidateId": candidate_id,
        "stepCount": doc["stepCount"],
        "uniqueItems": doc["uniqueItems"],
        "totalQty": doc["totalQty"],
        "resolvedItems": doc["stats"]["resolvedItems"],
        "unresolvedItems": doc["stats"]["unresolvedItems"],
    }