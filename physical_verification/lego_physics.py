"""
LEGO Physics Module - Stud/Tube Connection Logic

레고 브릭은 스터드(Stud, 상단 돌기)와 튜브(Tube, 하단 홈)의 
마찰력(Clutch Power)으로 결합됩니다.

LDU 단위:
- X/Z 그리드: 20 LDU (1 스터드 간격)
- Y (높이): 24 LDU (일반 브릭), 8 LDU (판 브릭)
- 스터드 높이: 약 4 LDU
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
import re

# LDU Constants
STUD_SPACING = 20.0  # X/Z grid
BRICK_HEIGHT = 24.0  # Normal brick
PLATE_HEIGHT = 8.0   # Plate
STUD_HEIGHT = 4.0    # Stud protrusion

# Brick size database (part_id -> (studs_x, studs_z, is_plate))
# studs_x * studs_z = number of studs
BRICK_SIZES = {
    # Basic Bricks (height = 24 LDU)
    # In LDraw, X is typically the LONG axis!
    "3001.dat": (4, 2, False),  # 2x4 Brick (4 studs in X, 2 in Z)
    "3002.dat": (3, 2, False),  # 2x3 Brick
    "3003.dat": (2, 2, False),  # 2x2 Brick
    "3004.dat": (2, 1, False),  # 1x2 Brick
    "3005.dat": (1, 1, False),  # 1x1 Brick
    "3006.dat": (10, 2, False), # 2x10 Brick
    "3007.dat": (8, 2, False),  # 2x8 Brick
    "3008.dat": (8, 1, False),  # 1x8 Brick
    "3009.dat": (6, 1, False),  # 1x6 Brick
    "3010.dat": (4, 1, False),  # 1x4 Brick
    "2456.dat": (6, 2, False),  # 2x6 Brick
    
    # Plates (height = 8 LDU)
    "3020.dat": (4, 2, True),   # 2x4 Plate
    "3021.dat": (3, 2, True),   # 2x3 Plate
    "3022.dat": (2, 2, True),   # 2x2 Plate
    "3023.dat": (2, 1, True),   # 1x2 Plate
    "3024.dat": (1, 1, True),   # 1x1 Plate
    "3795.dat": (6, 2, True),   # 2x6 Plate
    "3034.dat": (8, 2, True),   # 2x8 Plate
    "3832.dat": (10, 2, True),  # 2x10 Plate
}

# Realistic Mass Calculation
# Reference: 2x4 brick = 19200 LDU³ volume, weighs 2.3 grams
# Density constant: 2.3 / 19200 ≈ 0.00012 g/LDU³
BRICK_DENSITY = 2.3 / (40 * 20 * 24)  # grams per LDU³ (approx 0.00012)

def get_brick_mass_kg(part_file: str) -> float:
    """
    Returns realistic mass in KILOGRAMS for a brick.
    PyBullet uses SI units (kg, meters, seconds).
    
    Examples:
    - 2x4 brick: ~2.3g = 0.0023 kg
    - 1x1 brick: ~0.4g = 0.0004 kg
    """
    studs_x, studs_z, is_plate = get_brick_studs_count(part_file)
    height = PLATE_HEIGHT if is_plate else BRICK_HEIGHT
    
    # Volume in LDU³
    volume = (studs_x * STUD_SPACING) * (studs_z * STUD_SPACING) * height
    
    # Mass in grams, then convert to kg
    mass_grams = volume * BRICK_DENSITY
    return mass_grams / 1000.0  # Convert to kg

def get_brick_studs_count(part_file: str) -> Tuple[int, int, bool]:
    """
    Returns (studs_x, studs_z, is_plate) for a given part.
    Falls back to parsing part ID for common patterns.
    """
    part_file = part_file.lower().strip()
    
    if part_file in BRICK_SIZES:
        return BRICK_SIZES[part_file]
    
    # Fallback: Try to parse from part ID (e.g., "3001" -> 2x4 brick)
    # This is a heuristic for unknown parts
    match = re.match(r'^(\d+)\.dat$', part_file)
    if match:
        part_num = int(match.group(1))
        # Common fallback: assume 2x4 brick
        return (2, 4, False)
    
    # Default fallback
    return (2, 4, False)


def get_stud_positions_local(part_file: str) -> List[Tuple[float, float, float]]:
    """
    Returns list of stud center positions in LOCAL coordinates (part origin at 0,0,0).
    Studs are on the TOP of the brick (Y = 0 in LDraw, since Y points DOWN).
    
    LDraw coordinate system: Y is vertical (DOWN is positive)
    So studs are at Y = 0 (top surface), tubes are at Y = height (bottom)
    """
    studs_x, studs_z, is_plate = get_brick_studs_count(part_file)
    
    # Calculate stud positions centered around origin
    # For a 2x4 brick: studs at X = [-30, -10, 10, 30] (not quite, let me recalculate)
    # Actually: X range = studs_x * 20 LDU, centered at 0
    # For 2 studs in X: X = [-10, 10]
    # For 4 studs in Z: Z = [-30, -10, 10, 30]
    
    positions = []
    
    # Calculate starting positions (centered)
    start_x = -((studs_x - 1) * STUD_SPACING) / 2.0
    start_z = -((studs_z - 1) * STUD_SPACING) / 2.0
    
    for i in range(studs_x):
        for j in range(studs_z):
            x = start_x + i * STUD_SPACING
            z = start_z + j * STUD_SPACING
            y = 0.0  # Top surface in LDraw (Y-down)
            positions.append((x, y, z))
    
    return positions


def get_tube_positions_local(part_file: str) -> List[Tuple[float, float, float]]:
    """
    Returns list of tube positions in LOCAL coordinates.
    Tubes are on the BOTTOM of the brick (Y = height in LDraw).
    """
    studs_x, studs_z, is_plate = get_brick_studs_count(part_file)
    height = PLATE_HEIGHT if is_plate else BRICK_HEIGHT
    
    # Tubes mirror stud positions but at Y = height
    positions = []
    start_x = -((studs_x - 1) * STUD_SPACING) / 2.0
    start_z = -((studs_z - 1) * STUD_SPACING) / 2.0
    
    for i in range(studs_x):
        for j in range(studs_z):
            x = start_x + i * STUD_SPACING
            z = start_z + j * STUD_SPACING
            y = height  # Bottom in LDraw (Y-down)
            positions.append((x, y, z))
    
    return positions


def transform_positions(positions: List[Tuple[float, float, float]], 
                        matrix: np.ndarray, 
                        origin: np.ndarray) -> List[np.ndarray]:
    """
    Transforms local positions to global LDraw coordinates.
    
    Args:
        positions: List of (x, y, z) in local coords
        matrix: 3x3 rotation matrix
        origin: [x, y, z] translation
    
    Returns:
        List of [x, y, z] in global coords
    """
    result = []
    for pos in positions:
        local = np.array(pos)
        rotated = matrix @ local
        global_pos = rotated + origin
        result.append(global_pos)
    return result


def check_stud_tube_connection(brick_a, brick_b, tolerance: float = 5.0) -> bool:
    """
    Simplified Connectivity Check:
    Checks if Brick A is directly on top of Brick B (or vice-versa) based on bounding boxes.
    
    Rules:
    1. Y-difference matches brick height (Stacked)
    2. X/Z Bounding Boxes overlap (Touching horizontally)
    """
    if brick_a.origin is None or brick_b.origin is None:
        return False
    
    # 1. Get Global Bounding Boxes (Approximate from studs/dims)
    # Origin is Top-Center (LDraw standard)
    # Height: 24 (Brick) or 8 (Plate)
    # Width/Depth from BRICK_SIZES
    
    def get_bbox(brick):
        studs_x, studs_z, is_plate = get_brick_studs_count(brick.part_file or "3001.dat")
        height = PLATE_HEIGHT if is_plate else BRICK_HEIGHT
        
        # Origin is Top-Center. 
        # LDraw Y-down: Top=Y, Bottom=Y+height
        y_top = brick.origin[1]
        y_bottom = brick.origin[1] + height
        
        # Local Width/Depth
        # X: [-sx*10, sx*10], Z: [-sz*10, sz*10] (approx)
        w = studs_x * STUD_SPACING
        d = studs_z * STUD_SPACING
        
        # Global Bounds (assuming simplified rotation 0/90)
        # For simplicity, using simple center distance check which mimics overlap
        return {
            "y_top": y_top,
            "y_bottom": y_bottom,
            "xz_center": brick.origin[[0, 2]],
            "xz_dims": np.array([w, d])
        }

    bb_a = get_bbox(brick_a)
    bb_b = get_bbox(brick_b)
    
    # Check Vertical Stacking
    # Case 1: A on B (A.bottom ≈ B.top)
    a_on_b = abs(bb_a["y_bottom"] - bb_b["y_top"]) < tolerance
    # Case 2: B on A (B.bottom ≈ A.top)
    b_on_a = abs(bb_b["y_bottom"] - bb_a["y_top"]) < tolerance
    
    if not (a_on_b or b_on_a):
        return False
        
    # Check Horizontal Overlap
    # Simple Rectangle Overlap Check?
    # Or just center distance vs sum of half-sizes?
    # Use rotated overlap logic? 
    # Let's simple check: if distance < (size_a + size_b)/2 then overlap essentially.
    
    # More accurate: Check if XZ rectangles intersect.
    # Assuming Axis-Aligned for now (since models are usually 90 deg rotations)
    
    # A's Center & Half-Extents
    # Need to account for rotation (swap w/d if 90 deg)
    # Extract Yaw from matrix? Too complex?
    # Just assume overlap if distance is small enough.
    
    # Better: explicit AABB check in XZ plane
    # But need rotated bbox.
    # Let's use simple Radius check for robustness? 
    # No, simple "is touching" logic requested.
    
    # [Simple approach]: Check if centers are within combined half-dimensions
    # Ignoring rotation for a moment -> false positives possible but acceptable for "simple logic"
    # Actually, let's just check if ANY stud of A is inside B's footprint?
    
    dist_x = abs(bb_a["xz_center"][0] - bb_b["xz_center"][0])
    dist_z = abs(bb_a["xz_center"][1] - bb_b["xz_center"][1])
    
    # Crude overlap threshold: (width_a + width_b) / 2
    # But we don't know rotation easily without parsing matrix.
    # Let's take the MAX dimension to be safe (generous connection)
    max_dim_a = max(bb_a["xz_dims"])
    max_dim_b = max(bb_b["xz_dims"])
    
    threshold = (max_dim_a + max_dim_b) / 2.0 * 0.9 # 90% overlap allowed
    
    # If centers are closer than sum of half-widths
    # Ideally: intersection area > 0
    # Let's stick to the user's request: "simple touch"
    
    # Using existing utility? No.
    # New Logic:
    dx = dist_x
    dz = dist_z
    
    # Assume worst case alignment (min dimension) to be conservative? 
    # Or max dimension to be permissive? User wants PERMISSIVE ("just connect them")
    limit_x = (bb_a["xz_dims"][0] + bb_b["xz_dims"][0])/2.0
    limit_z = (bb_a["xz_dims"][1] + bb_b["xz_dims"][1])/2.0
    
    # Make it permissive: if close enough in X AND Z
    # Allow some slack
    is_overlapping = (dx < limit_x - 0.1) and (dz < limit_z - 0.1)
    
    # If rotated 90 deg, dims are swapped. 
    # Since we don't track rotation perfectly here, try both orientations?
    # Actually, LDR usually aligns to grid.
    # Let's just trust X/Z distance.
    
    return is_overlapping


def find_all_connections(bricks: list) -> List[Tuple[str, str]]:
    """
    Finds all stud-tube connections between bricks.
    
    Args:
        bricks: List of Brick objects
    
    Returns:
        List of (brick_id_a, brick_id_b) tuples for connected pairs
    """
    connections = []
    
    print(f"[DEBUG] Checking {len(bricks)} bricks for connections...")
    
    for i, brick_a in enumerate(bricks):
        for brick_b in bricks[i+1:]:
            # Print debug info for this pair
            # print(f"\n[DEBUG] Checking pair: {brick_a.id} vs {brick_b.id}")
            # print(f"  A origin: {brick_a.origin}")
            # print(f"  B origin: {brick_b.origin}")
            
            connected, reason = check_stud_tube_connection_debug(brick_a, brick_b)
            if connected:
                print(f"[DEBUG] CONNECTED: {brick_a.id} <-> {brick_b.id} | {reason}")
                connections.append((brick_a.id, brick_b.id))
            else:
                # Show why not connected - print ALL 4 surfaces
                studs_a = transform_positions(get_stud_positions_local(brick_a.part_file), brick_a.matrix, brick_a.origin)
                tubes_a = transform_positions(get_tube_positions_local(brick_a.part_file), brick_a.matrix, brick_a.origin)
                studs_b = transform_positions(get_stud_positions_local(brick_b.part_file), brick_b.matrix, brick_b.origin)
                tubes_b = transform_positions(get_tube_positions_local(brick_b.part_file), brick_b.matrix, brick_b.origin)
                print(f"  A studs (Y): {[f'{s[1]:.0f}' for s in studs_a[:2]]}, A tubes (Y): {[f'{t[1]:.0f}' for t in tubes_a[:2]]}")
                print(f"  B studs (Y): {[f'{s[1]:.0f}' for s in studs_b[:2]]}, B tubes (Y): {[f'{t[1]:.0f}' for t in tubes_b[:2]]}")
                print(f"  A tubes X: {[f'{t[0]:.0f}' for t in tubes_a[:2]]}, B studs X: {[f'{s[0]:.0f}' for s in studs_b[:2]]}")
    
    print(f"[DEBUG] Total connections found: {len(connections)}")
    return connections


def check_stud_tube_connection_debug(brick_a, brick_b, tolerance: float = 5.0) -> Tuple[bool, str]:
    """Debug version of check_stud_tube_connection that returns reason."""
    if brick_a.origin is None or brick_b.origin is None:
        return False, ""
    if brick_a.matrix is None or brick_b.matrix is None:
        return False, ""
    if brick_a.part_file is None or brick_b.part_file is None:
        return False, ""
    
    # Get stud and tube positions in global coords
    studs_a = transform_positions(
        get_stud_positions_local(brick_a.part_file),
        brick_a.matrix, brick_a.origin
    )
    tubes_a = transform_positions(
        get_tube_positions_local(brick_a.part_file),
        brick_a.matrix, brick_a.origin
    )
    
    studs_b = transform_positions(
        get_stud_positions_local(brick_b.part_file),
        brick_b.matrix, brick_b.origin
    )
    tubes_b = transform_positions(
        get_tube_positions_local(brick_b.part_file),
        brick_b.matrix, brick_b.origin
    )
    
    # Check if A's studs connect to B's tubes (y_diff should be ~0 when aligned)
    for stud in studs_a:
        for tube in tubes_b:
            y_diff = abs(tube[1] - stud[1])
            xz_dist = np.sqrt((stud[0] - tube[0])**2 + (stud[2] - tube[2])**2)
            
            if y_diff < tolerance and xz_dist < tolerance:
                return True, f"A_stud({stud[0]:.1f},{stud[1]:.1f},{stud[2]:.1f})->B_tube({tube[0]:.1f},{tube[1]:.1f},{tube[2]:.1f}) y_diff={y_diff:.1f} xz_dist={xz_dist:.1f}"
    
    # Check if B's studs connect to A's tubes
    for stud in studs_b:
        for tube in tubes_a:
            y_diff = abs(tube[1] - stud[1])
            xz_dist = np.sqrt((stud[0] - tube[0])**2 + (stud[2] - tube[2])**2)
            
            if y_diff < tolerance and xz_dist < tolerance:
                # print(f"[DEBUG] CONNECTED: {b1['id']} <-> {b2['id']} | A_stud({stud[0]:.1f},{stud[1]:.1f},{stud[2]:.1f})->B_tube({tube[0]:.1f},{tube[1]:.1f},{tube[2]:.1f}) y_diff={y_diff:.1f} xz_dist={xz_dist:.1f}")
                return True, f"B_stud({stud[0]:.1f},{stud[1]:.1f},{stud[2]:.1f})->A_tube({tube[0]:.1f},{tube[1]:.1f},{tube[2]:.1f}) y_diff={y_diff:.1f} xz_dist={xz_dist:.1f}"
    
    return False, ""


def find_floating_bricks(bricks: list, ground_y: float = 0.0, tolerance: float = 5.0) -> List[str]:
    """
    Finds bricks that are not connected to anything and not on the ground.
    
    Args:
        bricks: List of Brick objects
        ground_y: Y coordinate of ground (LDraw Y-down, so max Y = ground)
        tolerance: Y tolerance for ground contact
    
    Returns:
        List of floating brick IDs
    """
    if not bricks:
        return []
    
    # Find connections
    connections = find_all_connections(bricks)
    connected_bricks = set()
    for a, b in connections:
        connected_bricks.add(a)
        connected_bricks.add(b)
    
    # Find ground-level bricks (max Y value = bottom of brick)
    ground_bricks = set()
    max_y = max(b.origin[1] for b in bricks if b.origin is not None)
    
    for brick in bricks:
        if brick.origin is not None:
            # Get bottom Y (origin Y + height for LDraw)
            _, _, is_plate = get_brick_studs_count(brick.part_file or "3001.dat")
            height = PLATE_HEIGHT if is_plate else BRICK_HEIGHT
            bottom_y = brick.origin[1] + height
            
            # Check if on ground (within tolerance of max Y)
            if abs(bottom_y - max_y) < tolerance:
                ground_bricks.add(brick.id)
    
    # Propagate connectivity from ground
    visited = set(ground_bricks)
    to_visit = list(ground_bricks)
    
    while to_visit:
        current = to_visit.pop()
        for a, b in connections:
            if a == current and b not in visited:
                visited.add(b)
                to_visit.append(b)
            elif b == current and a not in visited:
                visited.add(a)
                to_visit.append(a)
    
    # Floating = not visited
    floating = [b.id for b in bricks if b.id not in visited]
    return floating
