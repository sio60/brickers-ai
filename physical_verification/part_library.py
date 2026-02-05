# ============================================================================
# LDraw 파트 라이브러리 및 데이터 접근 모듈
# 이 파일은 LDraw 파트 파일의 경로를 확인하고, 파일을 파싱하여 
# 파트의 경계 상자(bbox) 및 기하학적 데이터(geometry)를 추출합니다.
# 또한, MongoDB 데이터베이스에서 파트의 치수, 설명 및 기타 메타데이터를
# 조회하는 기능을 제공합니다.
# ============================================================================
import os
import math
import re
import sys
from pathlib import Path

# Import DB module from the same directory
try:
    from . import yang_db as db
except ImportError:
    import yang_db as db

# Global cache for part dimensions
DIMENSION_CACHE = {}

# Path to LDraw Library (Adjusted to actual location)
LDRAW_LIB_PATH = r"C:\complete\ldraw"
PARTS_DIR = os.path.join(LDRAW_LIB_PATH, "parts")
P_DIR = os.path.join(LDRAW_LIB_PATH, "p")

def resolve_file_path(filename):
    """
    .dat 파일의 절대 경로를 찾습니다.
    'parts' 디렉토리를 먼저 확인한 후 'p' 디렉토리를 확인합니다.
    역슬래시/슬래시 경로 구분자를 처리합니다.
    """
    filename = filename.replace("\\", os.sep).replace("/", os.sep)
    
    # parts 디렉토리 확인
    path = os.path.join(PARTS_DIR, filename)
    if os.path.exists(path):
        return path
        
    # p (primitives) 디렉토리 확인
    path = os.path.join(P_DIR, filename)
    if os.path.exists(path):
        return path
        
    # 일부 서브파일은 LDraw 루트 기준일 수 있음 (드물지만 발생 가능)
    # 보통 parts/s/file.dat 내부의 's\file.dat'이거나
    # p/4-4cyli.dat 내부의 '4-4cyli.dat'임
    
    # 대소문자 구분 없이 검색 시도
    clean_name = filename.lower()
    for root_dir in [PARTS_DIR, P_DIR]:
        try:
             path_lower = os.path.join(root_dir, clean_name)
             if os.path.exists(path_lower):
                 return path_lower
        except: pass
        
    return None

def parse_ldraw_part(part_id, depth=0, max_depth=3, accumulated_matrix=None):
    if part_id in DIMENSION_CACHE and depth == 0:
        pass

    file_path = resolve_file_path(part_id)
    if not file_path:
        return None
    # 정확한 작동을 위해 간소화된 파싱 로직 재구현
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
    
    # 2. MongoDB 쿼리
    try:
        coll = db.get_parts_collection()
        if not coll:
            return None

        query = {
            "$or": [
                {"partFile": f"{clean_id}.dat"},
                {"filename": f"{clean_id}.dat"},
                {"partPath": {"$regex": rf"/{clean_id}\.dat$", "$options": "i"}}
            ]
        }
        doc = coll.find_one(query)
        if doc and "bbox" in doc:
            b_min, b_max = doc["bbox"]["min"], doc["bbox"]["max"]
            result = (b_min[0], b_min[1], b_min[2], b_max[0], b_max[1], b_max[2])
            DIMENSION_CACHE[clean_id] = result
            return result
    except Exception as e:
        print(f"[오류] DB 메타데이터 읽기 오류: {e}")

    # 3. 폴백 (로컬 파일 파싱)
    clean_filename = f"{clean_id}.dat"
    bbox = get_raw_bbox(clean_filename, 0, 10)
    
    if bbox:
        DIMENSION_CACHE[clean_id] = bbox
        return bbox
    else:
        print(f"[오류] 모든 방법 실패: {clean_id}. DB 및 로컬 파일 접근 불가.")

    return None

# 파트 설명 및 메타데이터 캐시
DESC_CACHE = {}
META_CACHE = {}

def get_part_metadata_from_db(part_id: str):
    """
    MongoDB에서 파트 메타데이터(설명, 카테고리, 키워드)를 가져옵니다.
    딕셔너리 또는 None을 반환합니다.
    """
    # 1. 인스턴스 접미사 제거
    base_id = re.sub(r'_\d+$', '', part_id)
    # 2. 정규화
    clean_id = base_id.lower().replace(".dat", "").strip()
    
    # 캐시 확인 (실제 설명이 있는 경우에만 사용)
    if clean_id in META_CACHE:
        cached = META_CACHE[clean_id]
        # 캐시된 설명이 파일 이름인 경우 캐시를 무시하고 다시 가져옴
        if not cached["description"].lower().endswith(".dat"):
            return cached
        
    try:
        coll = db.get_parts_collection()
        if not coll:
            return None

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
            desc = doc.get("name") or doc.get("description") or "알 수 없는 부품"
            meta = {
                "description": desc,
                "category": doc.get("category", ""),
                "keywords": doc.get("keywords", [])
            }
            META_CACHE[clean_id] = meta
            return meta
    except Exception as e:
        print(f"[오류] DB 메타데이터 읽기 오류: {e}")
        pass
        
    return None

def get_part_description(part_id: str) -> str:
    """
    DB를 우선으로 사용하고 로컬 파일로 폴백하여 파트 설명을 가져옵니다.
    개선: DB가 실제 이름 대신 파일 이름(예: "3004.dat")을 반환하는 경우,
    로컬 파일에서 실제 영문 설명을 가져오도록 폴백을 강제합니다.
    """
    # 0. 인스턴스 접미사 제거 (예: "11477.dat_0" -> "11477.dat")
    base_id = re.sub(r'_\d+$', '', part_id)
    
    # 1. DB 시도
    meta = get_part_metadata_from_db(base_id)
    if meta:
        desc = meta["description"]
        # DB가 실제 이름을 반환한 경우(.dat로 끝나지 않음) 사용함
        # 그렇지 않으면 로컬 파일에서 실제 이름을 가져오기 위해 다음 단계로 진행
        if not desc.lower().endswith(".dat"):
            return desc

    # 2. 로컬 파일로 폴백
    clean_id = base_id.lower().replace(".dat", "").strip()
    filename = f"{clean_id}.dat"
    
    if clean_id in DESC_CACHE:
        return DESC_CACHE[clean_id]
        
    file_path = resolve_file_path(filename)
    if not file_path:
        # DB 결과라도 있는 경우 최후의 수단으로 반환
        if meta:
            return meta["description"]
        return "알 수 없는 부품"
        
    try:
        with open(file_path, 'r', encoding='latin-1') as f:
            first_line = f.readline().strip()
            # 형식: "0 Part Name" 또는 "0 Name"
            if first_line.startswith("0"):
                desc = first_line[1:].strip()
                # 파일마저 "0 3004.dat"인 경우(드묾) 어쩔 수 없지만, 대개 "0 Brick 1 x 2" 형식임
                DESC_CACHE[clean_id] = desc
                return desc
    except:
        pass
    
    if meta:
        return meta["description"]
        
    return "알 수 없는 부품"

# 지오메트리 캐시 (삼각형 목록)
GEO_CACHE = {}

def get_part_geometry(part_id: str, depth=0, max_depth=3):
    """
    로컬 LDU 좌표계의 삼각형 목록을 반환합니다.
    각 삼각형은 3개의 점으로 구성된 튜플입니다: ((x1,y1,z1), (x2,y2,z2), (x3,y3,z3))
    """
    clean_id = part_id.lower().replace(".dat", "").strip()
    filename = f"{clean_id}.dat"
    
    # 캐시 확인
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
                
                # 타입 1: 서브파일
                if line_type == '1' and depth < max_depth:
                     # 1 <색상> x y z a b c d e f g h <파일>
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
