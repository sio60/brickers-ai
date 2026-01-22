# 이 파일은 LDraw 부품의 형상 및 메타데이타를 관리하며 로컬 파일 또는 DB에서 정보를 불러오는 라이브러리입니다.
import os
import math
import re
import sys
from pathlib import Path

# from . import db  # DB 제거됨
import json
import os

# 대부분 로컬 파일을 사용하므로 DB 부품에 대한 더미 대체
# 또는 ldr_loader.py의 기본 치수에 의존

def get_part_dims(part_id: str):
    # 현재로서는 ldr_loader가 기본값을 사용하도록 None 반환
    # 필요한 경우 간단한 조회 구현 가능
    # ldr_loader.py에서: dims가 없으면 기본값 1x1x1 사용
    return None

def get_part_geometry(part_id: str):
    # 메쉬 데이터를 가져오는 데 사용되었습니다.
    return None 

# 부품 치수를 위한 전역 캐시
DIMENSION_CACHE = {}

# LDraw 라이브러리 경로 (실제 위치로 조정됨)
LDRAW_LIB_PATH = r"C:\complete\ldraw"
PARTS_DIR = os.path.join(LDRAW_LIB_PATH, "parts")
P_DIR = os.path.join(LDRAW_LIB_PATH, "p")

def resolve_file_path(filename):
    """
    .dat 파일의 절대 경로를 찾습니다.
    'parts'를 먼저 확인한 다음 'p'를 확인합니다.
    백슬래시/슬래시를 처리합니다.
    """
    filename = filename.replace("\\", os.sep).replace("/", os.sep)
    
    # parts 폴더 확인
    path = os.path.join(PARTS_DIR, filename)
    if os.path.exists(path):
        return path
        
    # p (primitives) 폴더 확인
    path = os.path.join(P_DIR, filename)
    if os.path.exists(path):
        return path
        
    # 일부 하위 파일은 LDRAW 루트에 상대적일 수 있음 (드물지만 가능함)
    # 보통은 's\file.dat' (parts/s/file.dat 내부)
    # 또는 '4-4cyli.dat' (p/4-4cyli.dat 내부)
    
    # 대소문자 구분 없이 검색 시도
    clean_name = filename.lower()
    for root_dir in [PARTS_DIR, P_DIR]:
        try:
             path_lower = os.path.join(root_dir, clean_name)
             if os.path.exists(path_lower):
                 return path_lower
        except: pass
        
    # 유효한 디버그: 실패한 경우 시도한 내용 출력
    # print(f"[DEBUG] Failed to resolve: {filename}")
    return None

def parse_ldraw_part(part_id, depth=0, max_depth=3, accumulated_matrix=None):
    if part_id in DIMENSION_CACHE and depth == 0:
        pass

    file_path = resolve_file_path(part_id)
    if not file_path:
        # print(f"[DEBUG] Local parse failed - File not found: {part_id}")
        return None
    # ... (나머지 parse_ldraw_part 로직은 동일, 간결하게 유지)
    # 작동 보장을 위한 단순화된 파싱 로직 재구현
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
                
                if line_type == '1': # 서브파일
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
                        
                elif line_type in ('3', '4'): # 삼각형/사각형
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
    
    # 2. MongoDB 쿼리 - 독립 실행형을 위해 제거됨
    # try:
    #     coll = db.get_parts_collection()
    #     # ...
    # except Exception as e:
    #     print(f"[ERROR] DB Metadata Read Error: {e}")

    # 3. 대체 방법 (Fallback)
    # print(f"[Fallback] Parsing local file for {clean_id}")
    clean_filename = f"{clean_id}.dat"
    bbox = get_raw_bbox(clean_filename, 0, 10)
    
    if bbox:
        DIMENSION_CACHE[clean_id] = bbox
        return bbox
    else:
        print(f"[ERROR] 모든 방법이 {clean_id}에 대해 실패했습니다. DB 실패, 로컬 파일 실패.")

    return None

# 부품 설명 및 메타데이터를 위한 캐시
DESC_CACHE = {}
META_CACHE = {}

def get_part_metadata_from_db(part_id: str):
    """
    MongoDB에서 부품 메타데이터(설명, 카테고리, 키워드)를 검색합니다.
    딕셔너리 또는 None을 반환합니다.
    """
    # 1. 인스턴스 접미사 제거
    base_id = re.sub(r'_\d+$', '', part_id)
    # 2. 정규화
    clean_id = base_id.lower().replace(".dat", "").strip()
    
    # 캐시 확인 (단, "*.dat"가 아닌 실제 설명이 있는 경우에만)
    if clean_id in META_CACHE:
        cached = META_CACHE[clean_id]
        # 캐시된 설명이 단순히 파일명인 경우 캐시 무시하고 다시 가져오기
        if not cached["description"].lower().endswith(".dat"):
            return cached
        
    try:
        coll = db.get_parts_collection()
        # 가능한 모든 필드 시도
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
            # 문서에 실제로 어떤 필드가 있는지 확인
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
    DB를 먼저 사용하여 부품 설명을 가져오고, 로컬 파일로 대체합니다.
    개선사항: DB가 실제 이름 대신 파일명(예: "3004.dat")을 반환하면
    로컬 파일을 강제로 사용하여 실제 영문 설명을 가져옵니다.
    """
    # 0. 인스턴스 접미사 먼저 제거 (예: "11477.dat_0" -> "11477.dat")
    base_id = re.sub(r'_\d+$', '', part_id)
    
    # 1. DB 시도
    meta = get_part_metadata_from_db(base_id)
    if meta:
        desc = meta["description"]
        # DB가 실제 이름을 제공한 경우( .dat로 끝나지 않음), 사용.
        # 그렇지 않으면 로컬 파일로 이동하여 실제 이름을 가져옵니다.
        if not desc.lower().endswith(".dat"):
            return desc

    # 2. 로컬 파일로 대체
    clean_id = base_id.lower().replace(".dat", "").strip()
    filename = f"{clean_id}.dat"
    
    if clean_id in DESC_CACHE:
        return DESC_CACHE[clean_id]
        
    file_path = resolve_file_path(filename)
    if not file_path:
        # DB 결과가 있는 경우(단순 파일명이라도), 마지막 수단으로 반환
        if meta:
            return meta["description"]
        return "Unknown Part"
        
    try:
        with open(file_path, 'r', encoding='latin-1') as f:
            first_line = f.readline().strip()
            # 형식: "0 Part Name" 또는 "0 Name"
            if first_line.startswith("0"):
                desc = first_line[1:].strip()
                # 파일에도 "0 3004.dat"라고만 되어 있으면(드묾) 어쩔 수 없지만, 보통은 "0 Brick 1 x 2"라고 되어 있음
                DESC_CACHE[clean_id] = desc
                return desc
    except:
        pass
    
    # 최후의 수단: DB 결과가 있었다면 반환
    if meta:
        return meta["description"]
        
    return "Unknown Part"

# 형상 캐시 (삼각형 목록)
GEO_CACHE = {}

def get_part_geometry(part_id: str, depth=0, max_depth=3):
    """
    로컬 LDU 좌표계의 삼각형 목록을 반환합니다.
    각 삼각형은 3개 점의 튜플입니다: ((x1,y1,z1), (x2,y2,z2), (x3,y3,z3))
    """
    clean_id = part_id.lower().replace(".dat", "").strip()
    filename = f"{clean_id}.dat"
    
    # 캐시 확인 (보통 최상위 호출만 캐시를 사용하지만, 원시 형상도 캐시 가능)
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
                
                # 타입 1: 서브 파일
                if line_type == '1' and depth < max_depth:
                     # 1 <colour> x y z a b c d e f g h <file>
                    try:
                        tx, ty, tz = float(parts[2]), float(parts[3]), float(parts[4])
                        a, b, c = float(parts[5]), float(parts[6]), float(parts[7])
                        d, e, f = float(parts[8]), float(parts[9]), float(parts[10])
                        g, h, i = float(parts[11]), float(parts[12]), float(parts[13])
                        sub_file = " ".join(parts[14:])
                        
                        sub_tris = get_part_geometry(sub_file, depth + 1, max_depth)
                        
                        # 서브 삼각형 변환
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

                # 타입 3: 삼각형
                elif line_type == '3':
                    try:
                        coords = [float(x) for x in parts[2:11]]
                        p1 = (coords[0], coords[1], coords[2])
                        p2 = (coords[3], coords[4], coords[5])
                        p3 = (coords[6], coords[7], coords[8])
                        triangles.append((p1, p2, p3))
                    except:
                        pass
                        
                # 타입 4: 사각형 (2개의 삼각형으로 분할)
                elif line_type == '4':
                     try:
                        coords = [float(x) for x in parts[2:14]]
                        p1 = (coords[0], coords[1], coords[2])
                        p2 = (coords[3], coords[4], coords[5])
                        p3 = (coords[6], coords[7], coords[8])
                        p4 = (coords[9], coords[10], coords[11])
                        # 분할: (p1, p2, p3) 및 (p3, p4, p1)
                        triangles.append((p1, p2, p3))
                        triangles.append((p3, p4, p1))
                     except:
                        pass
                        
    except:
        pass
        
    if depth == 0:
        GEO_CACHE[clean_id] = triangles
        
    return triangles
