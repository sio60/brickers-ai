# 이 파일은 LDraw 부품의 형상 및 메타데이타를 관리하며 로컬 파일 또는 DB에서 정보를 불러오는 라이브러리입니다.
import os
import math
import re
import sys
from pathlib import Path

# from . import db  # DB Removed
import json
import os

# Dummy fallback for DB parts since we are using local files mostly
# or relying on default dimensions in ldr_loader.py

def get_part_dims(part_id: str):
    # For now, return None to let ldr_loader USE DEFAULTS
    # Or implement a simple lookup if needed.
    # In ldr_loader.py: if not dims: Use default 1x1x1
    return None

def get_part_geometry(part_id: str):
    # This was used to get mesh data.
    return None 

# Global cache for part dimensions
DIMENSION_CACHE = {}

# Path to LDraw Library (Adjusted to actual location)
LDRAW_LIB_PATH = r"C:\complete\ldraw"
PARTS_DIR = os.path.join(LDRAW_LIB_PATH, "parts")
P_DIR = os.path.join(LDRAW_LIB_PATH, "p")

def resolve_file_path(filename):
    """
    Finds the absolute path of a .dat file.
    Checks 'parts' first, then 'p'.
    Handles backslashes/forward slashes.
    """
    filename = filename.replace("\\", os.sep).replace("/", os.sep)
    
    # Check parts
    path = os.path.join(PARTS_DIR, filename)
    if os.path.exists(path):
        return path
        
    # Check p (primitives)
    path = os.path.join(P_DIR, filename)
    if os.path.exists(path):
        return path
        
    # Some subfiles might be relative to LDRAW root (rare, but possible)
    # usually they are just 's\file.dat' which is inside parts/s/file.dat
    # or '4-4cyli.dat' which is inside p/4-4cyli.dat
    
    # Try case-insensitive search
    clean_name = filename.lower()
    for root_dir in [PARTS_DIR, P_DIR]:
        try:
             path_lower = os.path.join(root_dir, clean_name)
             if os.path.exists(path_lower):
                 return path_lower
        except: pass
        
    # Valid Debug: Print what we tried if failed
    # print(f"[DEBUG] Failed to resolve: {filename}")
    return None

def parse_ldraw_part(part_id, depth=0, max_depth=3, accumulated_matrix=None):
    if part_id in DIMENSION_CACHE and depth == 0:
        pass

    file_path = resolve_file_path(part_id)
    if not file_path:
        # print(f"[DEBUG] Local parse failed - File not found: {part_id}")
        return None
    # ... (Rest of parse_ldraw_part logic is same, keep it concise)
    # Re-implementing simplified parse logic to ensure it works
    min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
    max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')
    found_geometry = False
    
    try:
        with open(file_path, 'r', encoding='latin-1') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('0'): continue
                
                parts = line.split()
                if not parts: continue
                line_type = parts[0]
                
                if line_type == '1': # Subfile
                    if depth >= max_depth: continue
                    try:
                        sub_x, sub_y, sub_z = float(parts[2]), float(parts[3]), float(parts[4])
                        a, b, c = float(parts[5]), float(parts[6]), float(parts[7])
                        d, e, f = float(parts[8]), float(parts[9]), float(parts[10])
                        g, h, i = float(parts[11]), float(parts[12]), float(parts[13])
                        sub_file = " ".join(parts[14:])
                        
                        sub_bbox = get_raw_bbox(sub_file, depth + 1, max_depth)
                        if sub_bbox:
                            s_min_x, s_min_y, s_min_z, s_max_x, s_max_y, s_max_z = sub_bbox
                            corners = [
                                (s_min_x, s_min_y, s_min_z), (s_min_x, s_min_y, s_max_z),
                                (s_min_x, s_max_y, s_min_z), (s_min_x, s_max_y, s_max_z),
                                (s_max_x, s_min_y, s_min_z), (s_max_x, s_min_y, s_max_z),
                                (s_max_x, s_max_y, s_min_z), (s_max_x, s_max_y, s_max_z)
                            ]
                            for cx, cy, cz in corners:
                                nx = a*cx + b*cy + c*cz + sub_x
                                ny = d*cx + e*cy + f*cz + sub_y
                                nz = g*cx + h*cy + i*cz + sub_z
                                min_x, max_x = min(min_x, nx), max(max_x, nx)
                                min_y, max_y = min(min_y, ny), max(max_y, ny)
                                min_z, max_z = min(min_z, nz), max(max_z, nz)
                                found_geometry = True
                    except: pass
                        
                elif line_type in ('3', '4'): # Tri/Quad
                    try:
                        coords = [float(x) for x in parts[2:]]
                        for k in range(0, len(coords), 3):
                            vx, vy, vz = coords[k], coords[k+1], coords[k+2]
                            min_x, max_x = min(min_x, vx), max(max_x, vx)
                            min_y, max_y = min(min_y, vy), max(max_y, vy)
                            min_z, max_z = min(min_z, vz), max(max_z, vz)
                            found_geometry = True
                    except: pass
    except: return None

    if not found_geometry: return None
    return (min_x, min_y, min_z, max_x, max_y, max_z)

RAW_BBOX_CACHE = {}
def get_raw_bbox(part_id, current_depth, max_depth):
    part_id = part_id.lower().strip()
    if part_id in RAW_BBOX_CACHE: return RAW_BBOX_CACHE[part_id]
    bbox = parse_ldraw_part(part_id, current_depth, max_depth)
    if bbox: RAW_BBOX_CACHE[part_id] = bbox
    return bbox

def get_part_dims(part_id: str):
    clean_id = part_id.lower().replace(".dat", "").strip()
    if clean_id in DIMENSION_CACHE:
        return DIMENSION_CACHE[clean_id]
    
    # 2. Query MongoDB
    # 2. Query MongoDB - REMOVED for standalone
    # try:
    #     coll = db.get_parts_collection()
    #     # ...
    # except Exception as e:
    #     print(f"[ERROR] DB Metadata Read Error: {e}")

    # 3. Fallback
    # print(f"[Fallback] Parsing local file for {clean_id}")
    clean_filename = f"{clean_id}.dat"
    bbox = get_raw_bbox(clean_filename, 0, 10)
    
    if bbox:
        DIMENSION_CACHE[clean_id] = bbox
        return bbox
    else:
        print(f"[ERROR] All methods failed for {clean_id}. DB failed, Local file failed.")

    return None

# Cache for part descriptions
# Cache for part descriptions and metadata
DESC_CACHE = {}
META_CACHE = {}

def get_part_metadata_from_db(part_id: str):
    """
    Retrieves part metadata (description, category, keywords) from MongoDB.
    Returns a dict or None.
    """
    # 1. Strip instance suffix
    base_id = re.sub(r'_\d+$', '', part_id)
    # 2. Normalize
    clean_id = base_id.lower().replace(".dat", "").strip()
    
    # Check cache, but ONLY if it has a real description (not just "*.dat")
    if clean_id in META_CACHE:
        cached = META_CACHE[clean_id]
        # If cached description is just a filename, ignore cache and re-fetch
        if not cached["description"].lower().endswith(".dat"):
            return cached
        
    try:
        coll = db.get_parts_collection()
        # Try ALL possible fields
        query = {
            "$or": [
                {"partFile": f"{clean_id}.dat"},
                {"filename": f"{clean_id}.dat"},
                {"partFile": clean_id}, 
                {"partPath": {"$regex": rf"/{clean_id}\.dat$", "$options": "i"}}
            ]
        }
        doc = coll.find_one(query)
            
        if doc:
            # Check what fields are actually in the doc.
            desc = doc.get("name") or doc.get("description") or "Unknown Part"
            meta = {
                "description": desc,
                "category": doc.get("category", ""),
                "keywords": doc.get("keywords", [])
            }
            META_CACHE[clean_id] = meta
            return meta
    except Exception as e:
        print(f"[ERROR] DB Metadata Read Error: {e}")
        pass
        
    return None

def get_part_description(part_id: str) -> str:
    """
    Gets part description using DB first, falling back to local file.
    Improved: If DB returns a filename (e.g. "3004.dat") instead of a real name,
    force fallback to local file to get the real English description.
    """
    # 0. Strip instance suffix first (e.g. "11477.dat_0" -> "11477.dat")
    base_id = re.sub(r'_\d+$', '', part_id)
    
    # 1. Try DB
    meta = get_part_metadata_from_db(base_id)
    if meta:
        desc = meta["description"]
        # If DB gave us a real name (doesn't end in .dat), use it.
        # Otherwise, fall through to local file to get the REAL name.
        if not desc.lower().endswith(".dat"):
            return desc

    # 2. Fallback to Local
    clean_id = base_id.lower().replace(".dat", "").strip()
    filename = f"{clean_id}.dat"
    
    if clean_id in DESC_CACHE:
        return DESC_CACHE[clean_id]
        
    file_path = resolve_file_path(filename)
    if not file_path:
        # If we have a DB result (even if it's just a filename), return that as a last resort
        if meta:
            return meta["description"]
        return "Unknown Part"
        
    try:
        with open(file_path, 'r', encoding='latin-1') as f:
            first_line = f.readline().strip()
            # Format: "0 Part Name" or just "0 Name"
            if first_line.startswith("0"):
                desc = first_line[1:].strip()
                # If the file also just says "0 3004.dat" (rare), we are stuck, but usually it's "0 Brick 1 x 2"
                DESC_CACHE[clean_id] = desc
                return desc
    except:
        pass
    
    # Last resort: if we had a DB result, return it
    if meta:
        return meta["description"]
        
    return "Unknown Part"

# Cache for geometries (list of triangles)
GEO_CACHE = {}

def get_part_geometry(part_id: str, depth=0, max_depth=3):
    """
    Returns a list of triangles in local LDU coordinates.
    Each triangle is a tuple of 3 points: ((x1,y1,z1), (x2,y2,z2), (x3,y3,z3))
    """
    clean_id = part_id.lower().replace(".dat", "").strip()
    filename = f"{clean_id}.dat"
    
    # Check cache (only top level calls use cache typically, but we can cache raw geometry)
    if depth == 0 and clean_id in GEO_CACHE:
        return GEO_CACHE[clean_id]
        
    file_path = resolve_file_path(filename)
    if not file_path:
        return []

    triangles = []
    
    try:
        with open(file_path, 'r', encoding='latin-1') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('0'): continue
                
                parts = line.split()
                if not parts: continue
                line_type = parts[0]
                
                # Type 1: Sub-file
                if line_type == '1' and depth < max_depth:
                     # 1 <colour> x y z a b c d e f g h <file>
                    try:
                        tx, ty, tz = float(parts[2]), float(parts[3]), float(parts[4])
                        a, b, c = float(parts[5]), float(parts[6]), float(parts[7])
                        d, e, f = float(parts[8]), float(parts[9]), float(parts[10])
                        g, h, i = float(parts[11]), float(parts[12]), float(parts[13])
                        sub_file = " ".join(parts[14:])
                        
                        sub_tris = get_part_geometry(sub_file, depth + 1, max_depth)
                        
                        # Transform sub-triangles
                        for t in sub_tris:
                            new_t = []
                            for p in t:
                                px, py, pz = p
                                nx = a*px + b*py + c*pz + tx
                                ny = d*px + e*py + f*pz + ty
                                nz = g*px + h*py + i*pz + tz
                                new_t.append((nx, ny, nz))
                            triangles.append(tuple(new_t))
                    except:
                        pass

                # Type 3: Triangle
                elif line_type == '3':
                    try:
                        coords = [float(x) for x in parts[2:11]]
                        p1 = (coords[0], coords[1], coords[2])
                        p2 = (coords[3], coords[4], coords[5])
                        p3 = (coords[6], coords[7], coords[8])
                        triangles.append((p1, p2, p3))
                    except:
                        pass
                        
                # Type 4: Quad (Split into 2 Triangles)
                elif line_type == '4':
                     try:
                        coords = [float(x) for x in parts[2:14]]
                        p1 = (coords[0], coords[1], coords[2])
                        p2 = (coords[3], coords[4], coords[5])
                        p3 = (coords[6], coords[7], coords[8])
                        p4 = (coords[9], coords[10], coords[11])
                        # Split: (p1, p2, p3) and (p3, p4, p1)
                        triangles.append((p1, p2, p3))
                        triangles.append((p3, p4, p1))
                     except:
                        pass
                        
    except:
        pass
        
    if depth == 0:
        GEO_CACHE[clean_id] = triangles
        
    return triangles
