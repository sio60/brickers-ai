# vectordb/loader.py
from __future__ import annotations
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from pymongo import UpdateOne, ASCENDING
from pymongo.collection import Collection

import config
from db import get_db
from vectordb.utils import (
    MOVED_RE, ORG_RE, NAME_RE, AUTHOR_RE, CATEGORY_RE, KEYWORDS_RE,
    ensure_dat_ext, sha1_file, walk_files, classify_part, relpath_lower, basename_lower
)

logger = logging.getLogger("VectorDB.Loader")

LDRAW_BASE = config.LDRAW_BASE_DIR
PARTS_COLLECTION = config.PARTS_COLLECTION
ALIASES_COLLECTION = "ldraw_aliases"
MODELS_COLLECTION = "ldraw_models"
BULK_SIZE = 1000

def get_col(name: str) -> Collection:
    """MongoDB 컬렉션 객체를 가져옵니다."""
    return get_db()[name]

def ensure_indexes() -> None:
    """DB 성능 향상을 위해 필수 인덱스를 생성합니다."""
    col_parts = get_col(PARTS_COLLECTION)
    col_models = get_col(MODELS_COLLECTION)
    col_alias = get_col(ALIASES_COLLECTION)
    col_parts.create_index([("partPath", ASCENDING)], unique=True)
    col_models.create_index([("modelPath", ASCENDING)], unique=True)
    col_alias.create_index([("fromPath", ASCENDING)], unique=True)

def parse_part_dat(fp: Path) -> Dict:
    """LDraw .dat 파일을 파싱하여 메타데이터를 추출합니다."""
    org = name = author = category = None
    keywords, refs = [], []
    moved_to = None
    stats = {"lines": 0, "type0": 0, "type1": 0, "type2": 0, "type3": 0, "type4": 0, "other": 0}

    with fp.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line: continue
            stats["lines"] += 1
            if moved_to is None:
                m = MOVED_RE.match(line)
                if m: moved_to = ensure_dat_ext(m.group(1))
            if org is None:
                m = ORG_RE.match(line)
                if m: org = m.group(1).strip()
            if name is None:
                m = NAME_RE.match(line)
                if m: name = m.group(1).strip()
            if author is None:
                m = AUTHOR_RE.match(line)
                if m: author = m.group(1).strip()
            if category is None:
                m = CATEGORY_RE.match(line)
                if m: category = m.group(1).strip()
            m = KEYWORDS_RE.match(line)
            if m: keywords.extend([k.strip() for k in m.group(1).split(",") if k.strip()])

            if line.startswith("0 "): stats["type0"] += 1
            elif line.startswith("1 "):
                stats["type1"] += 1
                toks = line.split()
                if len(toks) >= 15: refs.append(ensure_dat_ext(toks[-1]))
            elif line.startswith("2 "): stats["type2"] += 1
            elif line.startswith("3 "): stats["type3"] += 1
            elif line.startswith("4 "): stats["type4"] += 1
            else: stats["other"] += 1

    return {
        "org": org, "name": name, "author": author, "category": category,
        "keywords": sorted(set([k for k in keywords if k])),
        "movedTo": moved_to, "refs": sorted(set(refs)), "stats": stats,
    }

def ingest_parts() -> Dict[str, int]:
    """LDraw 부품 파일(.dat)을 스캔하여 DB에 인제스트합니다."""
    now = datetime.now(timezone.utc)
    roots = [LDRAW_BASE / "parts", LDRAW_BASE / "p"]
    files = []
    for r in roots:
        for fp in walk_files(r, {".dat"}):
            files.append((fp, relpath_lower(fp, LDRAW_BASE)))
    
    if not files: raise RuntimeError("No LDraw parts found.")
    
    col_parts, col_alias = get_col(PARTS_COLLECTION), get_col(ALIASES_COLLECTION)
    ops_parts, ops_alias = [], []
    moved_count = 0

    for idx, (fp, part_path) in enumerate(files, 1):
        part_file = fp.name.lower()
        part_id = basename_lower(part_file)[:-4] if part_file.endswith(".dat") else basename_lower(part_file)
        kind = classify_part(part_path)
        parsed = parse_part_dat(fp)
        moved_to = parsed["movedTo"]

        doc = {
            "partPath": part_path, "partFile": part_file, "partId": part_id,
            "partType": kind.partType, "primitiveLevel": kind.primitiveLevel,
            "org": parsed["org"], "name": parsed["name"], "author": parsed["author"],
            "category": parsed["category"], "keywords": parsed["keywords"],
            "isRedirect": bool(moved_to), "movedTo": moved_to,
            "canonicalFile": moved_to or part_file, "refs": parsed["refs"],
            "stats": parsed["stats"], "sha1": sha1_file(fp),
            "source": {"base": str(LDRAW_BASE), "file": str(fp)}, "updatedAt": now,
        }
        ops_parts.append(UpdateOne({"partPath": part_path}, {"$set": doc, "$setOnInsert": {"createdAt": now}}, upsert=True))
        
        if moved_to:
            moved_count += 1
            ops_alias.append(UpdateOne({"fromPath": part_path}, {"$set": {"fromPath": part_path, "fromFile": part_file, "toFile": moved_to, "updatedAt": now}, "$setOnInsert": {"createdAt": now}}, upsert=True))

        if len(ops_parts) >= BULK_SIZE:
            col_parts.bulk_write(ops_parts, ordered=False); ops_parts.clear()
        if len(ops_alias) >= BULK_SIZE:
            col_alias.bulk_write(ops_alias, ordered=False); ops_alias.clear()
            
    if ops_parts: col_parts.bulk_write(ops_parts, ordered=False)
    if ops_alias: col_alias.bulk_write(ops_alias, ordered=False)
    return {"files": len(files), "moved": moved_count}

def ingest_models(store_text: bool = False) -> Dict[str, int]:
    """LDraw 모델 파일(.ldr)을 스캔하여 DB에 인제스트합니다."""
    now = datetime.now(timezone.utc)
    root = LDRAW_BASE / "models"
    files = [(fp, relpath_lower(fp, LDRAW_BASE)) for fp in walk_files(root, {".ldr"})]
    if not files: return {"files": 0}

    col_models = get_col(MODELS_COLLECTION)
    ops = []
    for fp, model_path in files:
        refs = []
        stats = {"lines": 0, "type0": 0, "type1": 0, "type2": 0, "type3": 0, "type4": 0, "other": 0}
        with fp.open("r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line: continue
                stats["lines"] += 1
                if line.startswith("1 "):
                    stats["type1"] += 1
                    toks = line.split()
                    if len(toks) >= 15: refs.append(toks[-1])
                elif line.startswith("0 "): stats["type0"] += 1
                elif line.startswith("2 "): stats["type2"] += 1
                elif line.startswith("3 "): stats["type3"] += 1
                elif line.startswith("4 "): stats["type4"] += 1
                else: stats["other"] += 1
        
        doc = {
            "modelPath": model_path, "modelFile": fp.name, "ext": fp.suffix.lower(),
            "refTokens": sorted(set(refs)), "stats": stats, "sha1": sha1_file(fp),
            "source": {"base": str(LDRAW_BASE), "file": str(fp)}, "updatedAt": now,
        }
        if store_text: doc["text"] = fp.read_text(encoding="utf-8", errors="ignore")
        ops.append(UpdateOne({"modelPath": model_path}, {"$set": doc, "$setOnInsert": {"createdAt": now}}, upsert=True))
        if len(ops) >= BULK_SIZE:
            col_models.bulk_write(ops, ordered=False); ops.clear()
            
    if ops: col_models.bulk_write(ops, ordered=False)
    return {"files": len(files)}
