"""LDraw Parts Bundler — LDR 파일에서 참조하는 모든 파트를 JSON 번들로 생성."""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Dict, Optional, Set

log = logging.getLogger(__name__)

LDRAW_DIR = Path(os.environ.get("LDRAWDIR", "/usr/share/ldraw"))

# LDR type-1 line: 1 color x y z a b c d e f g h i filename.dat
_TYPE1_RE = re.compile(r"^1\s+\S+(?:\s+\S+){12}\s+(.+)$")

# Subfile reference inside .dat files (same pattern)
_SUBFILE_RE = _TYPE1_RE


def _resolve_part_path(ref: str) -> Optional[Path]:
    """Resolve a part reference to a file path under LDRAWDIR."""
    ref_clean = ref.strip().replace("\\", "/")

    # Try exact path first
    candidates = [
        LDRAW_DIR / ref_clean,
        LDRAW_DIR / "parts" / ref_clean,
        LDRAW_DIR / "p" / ref_clean,
        LDRAW_DIR / "parts" / "s" / ref_clean,
        LDRAW_DIR / "p" / "48" / ref_clean,
        LDRAW_DIR / "p" / "8" / ref_clean,
    ]

    for c in candidates:
        if c.is_file():
            return c

    # Lowercase fallback
    ref_lower = ref_clean.lower()
    candidates_lower = [
        LDRAW_DIR / ref_lower,
        LDRAW_DIR / "parts" / ref_lower,
        LDRAW_DIR / "p" / ref_lower,
        LDRAW_DIR / "parts" / "s" / ref_lower,
        LDRAW_DIR / "p" / "48" / ref_lower,
        LDRAW_DIR / "p" / "8" / ref_lower,
    ]

    for c in candidates_lower:
        if c.is_file():
            return c

    return None


def _relative_key(abs_path: Path) -> str:
    """Convert absolute path to CDN-relative key (e.g. 'parts/3001.dat')."""
    try:
        return abs_path.relative_to(LDRAW_DIR).as_posix()
    except ValueError:
        return abs_path.name


def _collect_parts(
    ref: str,
    visited: Set[str],
    parts: Dict[str, str],
    depth: int = 0,
    max_depth: int = 10,
) -> None:
    """Recursively collect a part and all its sub-references."""
    if depth > max_depth:
        return

    ref_key = ref.strip().replace("\\", "/").lower()
    if ref_key in visited:
        return
    visited.add(ref_key)

    resolved = _resolve_part_path(ref)
    if resolved is None:
        log.warning(f"[LDraw Bundle] Part not found: {ref}")
        return

    try:
        content = resolved.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return

    rel_key = _relative_key(resolved)
    parts[rel_key] = content

    # Parse sub-references
    for line in content.splitlines():
        line = line.strip()
        m = _SUBFILE_RE.match(line)
        if m:
            sub_ref = m.group(1).strip()
            _collect_parts(sub_ref, visited, parts, depth + 1, max_depth)


def generate_parts_bundle(ldr_path: Path) -> dict:
    """Parse an LDR file and bundle all referenced parts into a JSON-serializable dict.

    Returns:
        {
            "version": 1,
            "ldconfig": "<LDConfig.ldr content>",
            "parts": {
                "parts/3001.dat": "<content>",
                "p/stud.dat": "<content>",
                "parts/s/3001s01.dat": "<content>",
                ...
            }
        }
    """
    ldr_text = ldr_path.read_text(encoding="utf-8", errors="replace")

    visited: Set[str] = set()
    parts: Dict[str, str] = {}

    # Extract all type-1 references from the LDR
    for line in ldr_text.splitlines():
        line = line.strip()
        m = _TYPE1_RE.match(line)
        if m:
            ref = m.group(1).strip()
            _collect_parts(ref, visited, parts, depth=0)

    # Load LDConfig.ldr
    ldconfig_content = ""
    ldconfig_path = LDRAW_DIR / "LDConfig.ldr"
    if ldconfig_path.is_file():
        try:
            ldconfig_content = ldconfig_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass

    return {
        "version": 1,
        "ldconfig": ldconfig_content,
        "parts": parts,
    }
