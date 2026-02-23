# vectordb/processor.py
from __future__ import annotations
import logging
import torch
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from transformers import AutoTokenizer, AutoModel
from pymongo import UpdateOne
from pymongo.collection import Collection

import config
from db import get_db
from vectordb.utils import (
    BBox, XForm, norm_slash, ensure_dat_ext, sha1_file, sha1_text
)

logger = logging.getLogger("VectorDB.Processor")

LDRAW_BASE = Path(config.LDRAW_BASE_DIR)
PARTS_COLLECTION = config.PARTS_COLLECTION
ALIASES_COLLECTION = "ldraw_aliases"

# =========================
# BBox Calculation
# =========================
def _apply_xform_point(xf: XForm, x: float, y: float, z: float) -> Tuple[float, float, float]:
    """3x3 매트릭스 변환을 점에 적용합니다."""
    return (xf.a*x + xf.b*y + xf.c*z + xf.tx, xf.d*x + xf.e*y + xf.f*z + xf.ty, xf.g*x + xf.h*y + xf.i*z + xf.tz)

def _transform_bbox(xf: XForm, child: BBox) -> BBox:
    """변환 행렬을 적용하여 새로운 BBox를 계산합니다."""
    out = BBox()
    if not child.is_valid(): return out
    for (x, y, z) in child.corners():
        nx, ny, nz = _apply_xform_point(xf, x, y, z)
        out.include_point(nx, ny, nz)
    return out

def _parse_dat_for_bbox(fp: Path) -> Tuple[BBox, List[Tuple[XForm, str]]]:
    """BBox 계산을 위해 .dat 파일의 기하학적 정보를 파싱합니다."""
    bbox, refs = BBox(), []
    with fp.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line: continue
            if line.startswith("1 "):
                t = line.split()
                if len(t) >= 15: refs.append((XForm(float(t[2]),float(t[3]),float(t[4]), float(t[5]),float(t[6]),float(t[7]), float(t[8]),float(t[9]),float(t[10]), float(t[11]),float(t[12]),float(t[13])), t[14]))
            elif line.startswith("3 "):
                t = line.split()
                if len(t) >= 11:
                    for i in range(2, 11, 3): bbox.include_point(float(t[i]), float(t[i+1]), float(t[i+2]))
            elif line.startswith("4 "):
                t = line.split()
                if len(t) >= 14:
                    for i in range(2, 14, 3): bbox.include_point(float(t[i]), float(t[i+1]), float(t[i+2]))
    return bbox, refs

def compute_bbox_recursive(pp: str, file_to_paths: Dict, all_paths: Set, alias_map: Dict, cache: Dict, visiting: Set) -> BBox:
    """부품의 BBox를 하위 부품을 포함하여 재귀적으로 계산합니다."""
    pp = pp.lower()
    if pp in cache: return cache[pp]
    if pp in visiting: return BBox()
    visiting.add(pp)
    
    fp = LDRAW_BASE / pp
    if not fp.exists():
        visiting.remove(pp); return BBox()
        
    local_bbox, refs = _parse_dat_for_bbox(fp)
    for xf, token in refs:
        token = norm_slash(token).lower()
        child_pp = None
        for hint in [f"p/48/{ensure_dat_ext(token)}", f"p/8/{ensure_dat_ext(token)}", f"parts/s/{ensure_dat_ext(token)}"]:
            if hint in all_paths: child_pp = hint; break
        if not child_pp:
            base = alias_map.get(ensure_dat_ext(token), ensure_dat_ext(token))
            cand = file_to_paths.get(base, [])
            if cand: child_pp = sorted(cand, key=lambda p: 1 if p.startswith("parts/") and not p.startswith("parts/s/") else 2 if p.startswith("parts/s/") else 3 if "p/48/" in p else 4 if "p/8/" in p else 5)[0]
            
        if child_pp:
            local_bbox.union(_transform_bbox(xf, compute_bbox_recursive(child_pp, file_to_paths, all_paths, alias_map, cache, visiting)))
            
    visiting.remove(pp)
    cache[pp] = local_bbox
    return local_bbox

def update_all_bboxes(only_missing=True) -> Dict:
    """DB의 모든 부품에 대해 BBox 정보를 일괄 업데이트합니다."""
    db = get_db()
    col, alias_col = db[PARTS_COLLECTION], db[ALIASES_COLLECTION]
    
    file_to_paths, all_paths = {}, set()
    for d in col.find({}, {"partFile":1, "partPath": 1}):
        f, p = d.get("partFile","").lower(), d.get("partPath","").lower()
        if f and p: file_to_paths.setdefault(f, []).append(p); all_paths.add(p)
        
    alias_map = {ensure_dat_ext(d.get("fromFile","")): ensure_dat_ext(d.get("toFile","")) for d in alias_col.find({})}
    
    cache, ops = {}, []
    scanned, updated = 0, 0
    for d in col.find({}, {"partPath":1, "sha1": 1, "bboxSha1": 1, "bbox": 1}):
        scanned += 1
        pp, sha1 = d.get("partPath","").lower(), d.get("sha1")
        if only_missing and sha1 and d.get("bboxSha1") == sha1 and d.get("bbox"): continue
        if not pp: continue
        
        bbox = compute_bbox_recursive(pp, file_to_paths, all_paths, alias_map, cache, set())
        ops.append(UpdateOne({"partPath": pp}, {"$set": {
            "bbox": bbox.to_doc(), 
            "bboxVolume": bbox.volume(), 
            "bboxSha1": sha1, 
            "bboxUpdatedAt": datetime.now(timezone.utc),
            "bboxMode": "recursive"
        }}))
        updated += 1
        if len(ops) >= 1000:
            col.bulk_write(ops, ordered=False); ops.clear()
            
    if ops: col.bulk_write(ops, ordered=False)
    return {"scanned": scanned, "updated": updated}

# =========================
# Embedding
# =========================
class HFEmbedder:
    def __init__(self, model_name=None):
        self.model_name = model_name or getattr(config, "HF_EMBED_MODEL", "intfloat/multilingual-e5-small")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        with torch.no_grad():
            self.dims = self.model(**{k: v.to(self.device) for k, v in self.tokenizer("passage: test", return_tensors="pt").items()}).last_hidden_state.shape[-1]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """텍스트 리스트를 입력받아 벡터 임베딩을 생성합니다."""
        vecs = []
        for i in range(0, len(texts), 64):
            tok = {k: v.to(self.device) for k, v in self.tokenizer(texts[i:i+64], padding=True, truncation=True, max_length=256, return_tensors="pt").items()}
            with torch.no_grad():
                out = self.model(**tok)
                mask = tok["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
                emb = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            vecs.extend(emb.detach().cpu().tolist())
        return vecs

def update_all_embeddings(only_missing=True) -> Dict:
    """DB의 모든 부품에 대해 AI 벡터 임베딩을 생성 및 업데이트합니다."""
    col = get_db()[PARTS_COLLECTION]
    embedder = HFEmbedder()
    to_update = []
    scanned = 0
    for d in col.find({"partPath": {"$exists": True}}, {"partPath":1, "name":1, "category":1, "keywords":1, "partType":1, "primitiveLevel":1, "embedding": 1}):
        scanned += 1
        text = f"passage: name: {d.get('name','') or ''} | category: {d.get('category','') or ''} | type: {d.get('partType','') or ''}"
        text_hash = sha1_text(text)
        old_hash = (d.get("embedding") or {}).get("textHash")
        if only_missing and old_hash == text_hash and d.get("embedding",{}).get("vector"): continue
        to_update.append((d["partPath"], text, text_hash))
        
    for i in range(0, len(to_update), 64):
        chunk = to_update[i:i+64]
        vecs = embedder.embed_texts([x[1] for x in chunk])
        now = datetime.now(timezone.utc)
        ops = [UpdateOne({"partPath": p}, {"$set": {"embedding": {"model": embedder.model_name, "dims": embedder.dims, "vector": v, "textHash": h, "updatedAt": now}}}) for (p, t, h), v in zip(chunk, vecs)]
        col.bulk_write(ops, ordered=False)
        
    return {"scanned": scanned, "updated": len(to_update), "dims": embedder.dims}
