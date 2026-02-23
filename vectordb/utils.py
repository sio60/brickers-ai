# vectordb/utils.py
import re
import hashlib
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any

# =========================
# LDraw 파싱용 정규표현식
# =========================
MOVED_RE = re.compile(r"^0\s+~Moved\s+to\s+(\S+)", re.IGNORECASE)
ORG_RE = re.compile(r"^0\s+!LDRAW_ORG\s+(.+)$", re.IGNORECASE)
NAME_RE = re.compile(r"^0\s+Name:\s*(.+)$", re.IGNORECASE)
AUTHOR_RE = re.compile(r"^0\s+Author:\s*(.+)$", re.IGNORECASE)
CATEGORY_RE = re.compile(r"^0\s+!CATEGORY\s+(.+)$", re.IGNORECASE)
KEYWORDS_RE = re.compile(r"^0\s+!KEYWORDS\s+(.+)$", re.IGNORECASE)


@dataclass
class PartKind:
    partType: str
    primitiveLevel: Optional[int]


@dataclass
class BBox:
    minx: float = math.inf
    miny: float = math.inf
    minz: float = math.inf
    maxx: float = -math.inf
    maxy: float = -math.inf
    maxz: float = -math.inf

    def is_valid(self) -> bool:
        return self.minx <= self.maxx and self.miny <= self.maxy and self.minz <= self.maxz

    def include_point(self, x: float, y: float, z: float) -> None:
        self.minx = min(self.minx, x); self.miny = min(self.miny, y); self.minz = min(self.minz, z)
        self.maxx = max(self.maxx, x); self.maxy = max(self.maxy, y); self.maxz = max(self.maxz, z)

    def union(self, other: "BBox") -> None:
        if not other.is_valid(): return
        self.include_point(other.minx, other.miny, other.minz)
        self.include_point(other.maxx, other.maxy, other.maxz)

    def corners(self) -> List[Tuple[float, float, float]]:
        return [
            (self.minx, self.miny, self.minz), (self.minx, self.miny, self.maxz),
            (self.minx, self.maxy, self.minz), (self.minx, self.maxy, self.maxz),
            (self.maxx, self.miny, self.minz), (self.maxx, self.miny, self.maxz),
            (self.maxx, self.maxy, self.minz), (self.maxx, self.maxy, self.maxz),
        ]

    def to_doc(self) -> Dict:
        dx = (self.maxx - self.minx) if self.is_valid() else 0.0
        dy = (self.maxy - self.miny) if self.is_valid() else 0.0
        dz = (self.maxz - self.minz) if self.is_valid() else 0.0
        return {
            "min": [self.minx, self.miny, self.minz],
            "max": [self.maxx, self.maxy, self.maxz],
            "size": [dx, dy, dz],
        }

    def volume(self) -> float:
        if not self.is_valid(): return 0.0
        return (self.maxx - self.minx) * (self.maxy - self.miny) * (self.maxz - self.minz)


@dataclass
class XForm:
    tx: float; ty: float; tz: float
    a: float; b: float; c: float
    d: float; e: float; f: float
    g: float; h: float; i: float


@dataclass
class ResolvedPart:
    part_file: str
    part_id: str
    canonical_file: str
    name: str
    doc: Dict[str, Any]


def norm_slash(s: str) -> str:
    """경로 구분자를 슬래시(/)로 통일하고 공백을 제거합니다."""
    return s.strip().replace("\\", "/").strip()


def basename_lower(s: str) -> str:
    """파일의 이름 부분만 추출하여 소문자로 변환합니다."""
    return norm_slash(s).split("/")[-1].lower()


def ensure_dat_ext(name_or_path: str) -> str:
    """파일명에 .dat 확장자가 없으면 인위적으로 붙여줍니다."""
    x = basename_lower(name_or_path)
    if "." not in x:
        return x + ".dat"
    return x


def sha1_file(path: Path) -> str:
    """파일 내용의 SHA1 해시값을 계산합니다."""
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha1_text(s: str) -> str:
    """텍스트 문자열의 SHA1 해시값을 계산합니다."""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def walk_files(root: Path, exts: Set[str]) -> List[Path]:
    """특정 디렉토리에서 지정된 확장자 파일들을 재귀적으로 찾습니다."""
    if not root.exists():
        return []
    out: List[Path] = []
    for fp in root.rglob("*"):
        if fp.is_file() and fp.suffix.lower() in exts:
            out.append(fp)
    return out


def classify_part(part_path: str) -> PartKind:
    """경로 패턴을 분석하여 부품의 타입(subpart, primitive 등)을 판별합니다."""
    pp = part_path.lower()
    if pp.startswith("parts/s/"):
        return PartKind("subpart", None)
    if pp.startswith("p/48/"):
        return PartKind("primitive", 48)
    if pp.startswith("p/8/"):
        return PartKind("primitive", 8)
    if pp.startswith("p/"):
        return PartKind("primitive", None)
    return PartKind("part", None)


def relpath_lower(p: Path, base: Path) -> str:
    """기준 경로 대비 상대 경로를 소문자로 변환하여 반환합니다."""
    return p.relative_to(base).as_posix().lower()


def seed_dummy_parts(col, dims: int, vector_field: str, overwrite: bool = True) -> int:
    """더미 파츠 데이터를 DB에 생성합니다."""
    samples = [
        ("3001", "Brick", 4, 2), ("3002", "Brick", 3, 2),
        ("3022", "Plate", 2, 2), ("3040", "Slope", 2, 1),
        ("3710", "Plate", 4, 2),
    ]
    if overwrite:
        col.delete_many({"partId": {"$in": [p[0] for p in samples]}})
    
    docs = []
    for partId, category, x, z in samples:
        docs.append({
            "partId": partId, "category": category,
            "bbox": {"x": x, "y": 1.2, "z": z, "min": [0,0,0], "max": [x, 1.2, z], "size": [x, 1.2, z]},
            vector_field: [random.uniform(-1, 1) for _ in range(dims)],
        })
    res = col.insert_many(docs)
    return len(res.inserted_ids)
