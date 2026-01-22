# 이 파일은 LDraw 파일을 3D 메쉬로 변환하여 시각화하고 PyBullet을 통합하여 물리적 검증을 수행하는 스크립트입니다.
import trimesh
import numpy as np
import os
import sys

# Add physical_verification to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'physical_verification'))
try:
    import part_library
    import part_library
    from ldr_loader import LdrLoader
    from verifier import PhysicalVerifier
    from pybullet_verifier import PyBulletVerifier
except ImportError as e:
    print(f"Error importing physical_verification modules: {e}")
    sys.exit(1)

# LDraw Color ID -> RGB (0-255)
# Reference: glb_to_ldr_v3.py
LDRAW_COLORS = {
    0:  (33, 33, 33),
    1:  (0, 85, 191),
    2:  (0, 123, 40),
    3:  (0, 131, 138),
    4:  (180, 0, 0),
    5:  (171, 67, 183),
    6:  (91, 28, 12),
    7:  (156, 146, 145),
    8:  (99, 95, 82),
    9:  (107, 171, 220),
    10: (97, 189, 76),
    11: (0, 170, 164),
    12: (255, 99, 71),
    13: (255, 148, 194),
    14: (255, 220, 0),
    15: (255, 255, 255),
    17: (173, 221, 80),
    18: (251, 171, 24),
    19: (215, 197, 153),
    20: (215, 240, 215),
    21: (255, 240, 60), # Bright Yellow
    22: (88, 42, 18), # Dark Purple
    23: (0, 72, 150), # Medium Blue
    25: (245, 134, 36),
    26: (202, 31, 123),
    27: (159, 195, 65),
    28: (33, 55, 23),
    29: (160, 188, 172),
    31: (208, 208, 208), # Medium Lavender
    33: (252, 252, 252, 128), # Trans-Clear
    34: (35, 120, 65, 128),   # Trans-Green
    35: (50, 205, 50, 128),   # Trans-Bright Green
    36: (200, 50, 50, 128),   # Trans-Red
    37: (100, 100, 200, 128), # Trans-Dark Blue
    38: (255, 100, 0, 128),   # Trans-Neon Orange
    39: (252, 252, 252, 128), # Trans-Clear (Old/Custom mapping for car.ldr)
    40: (100, 100, 100, 128), # Trans-Black (Smoke)
    41: (150, 200, 255, 128), # Trans-Light Blue
    42: (200, 255, 50, 128),  # Trans-Neon Green
    43: (200, 220, 255, 128), # Trans-Very Light Blue
    46: (255, 220, 0, 128),   # Trans-Yellow
    47: (255, 255, 255, 128), # Trans-Clear (Old)
    52: (150, 50, 200, 128),  # Trans-Purple
    54: (255, 255, 50, 128),  # Trans-Neon Yellow
    57: (255, 150, 0, 128),   # Trans-Orange
    70: (89, 47, 14),
    71: (175, 181, 199),
    72: (108, 110, 104),
    73: (117, 142, 220),
    74: (183, 212, 37),
    77: (249, 164, 199), # Light Pink
    78: (254, 186, 189),
    84: (170, 125, 85),
    85: (89, 39, 115),
    89: (44, 21, 119), # Blue Violet
    92: (208, 208, 208), # Flesh
    100: (254, 196, 1), # Light Yellow
    110: (67, 84, 163), # Violet
    112: (104, 116, 159), # Medium Bluish Violet
    114: (100, 100, 100), # Medium Gray
    115: (0, 204, 0), # Medium Green
    118: (179, 215, 209), # Light Aqua
    120: (147, 185, 60), # Light Lime
    125: (249, 164, 199), # Light Pink
    129: (166, 202, 240), # Pale Blue
    134: (158, 163, 20), # Olive Green
    135: (91, 93, 58), # Pearl Grey
    137: (223, 102, 149), # Medium Nougat
    142: (179, 139, 23), # Gold
    148: (99, 95, 97), # Dark Pearl Gray
    151: (95, 117, 140), # Sand Blue
    178: (140, 0, 255), # Flat Dark Gold
    179: (204, 204, 204), # Flat Silver
    183: (255, 255, 255), # Pearl White
    191: (248, 187, 61),
    212: (175, 217, 255), # Light Royal Blue
    216: (174, 164, 111), # Rust
    226: (255, 255, 153), # Cool Yellow
    232: (135, 192, 234), # Sky Blue
    272: (46, 85, 197), # Dark Blue
    288: (30, 90, 168), # Dark Green
    308: (50, 48, 47), # Dark Brown
    313: (206, 233, 242), # Maersk Blue
    320: (120, 27, 33),
    321: (64, 90, 155), # Dark Azure
    322: (85, 140, 244), # Medium Azure
    323: (206, 234, 243), # Light Azure
    326: (231, 242, 167), # Spring Yellowish Green
    330: (128, 8, 27), # Olive Green
    335: (223, 223, 102), # Sand Red
    351: (247, 133, 177), # Medium Dark Pink
    353: (220, 96, 174), # Coral
    366: (178, 116, 145), # Earth Orange
    373: (178, 190, 197), # Sand Purple
    378: (163, 193, 173),
    379: (208, 219, 97), # Sand Blue
    450: (250, 213, 166), # Fabuland Brown
    462: (211, 211, 101), # Medium Orange
    484: (179, 62, 0),
}

def ldr_to_mesh(file_path):
    """
    Loads an LDR file and converts it to a single Trimesh Scene object.
    Uses 'part_library' to fetch geometry (DB cached or local file).
    """
    
    # 1. Check file existence
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    print(f"Loading {file_path}...")
    
    # ---------------------------------------------------------
    # [Physical Verification] Run checks first
    # ---------------------------------------------------------
    print("Running Physical Verification...")
    failed_brick_ids = set()
    try:
        loader = LdrLoader()
        plan = loader.load_from_file(file_path)
        
        # [Legacy Check] Suppressed for now to avoid confusion with PyBullet results
        # verifier = PhysicalVerifier(plan)
        # result = verifier.run_all_checks(mode="ADULT") 
        # if not result.is_valid: ... 
        
        # [PyBullet Check] Precise Mesh Collision
        print("Running PyBullet Precision Check (GUI Mode)...")
        pb_verifier = PyBulletVerifier(plan, gui=True)
        # Tolerance 0 = Maximum sensitivity (any touch is collision)
        pb_result = pb_verifier.run_collision_check(tolerance=0) 
        
        if not pb_result.is_valid:
            print(f"PyBullet Collision Detected!")
            for ev in pb_result.evidence:
                print(f"  [PyBullet] {ev.message}")
                for bid in ev.brick_ids:
                    failed_brick_ids.add(bid)
                    
        # [PyBullet Check] Stability (Gravity)
        print("Running PyBullet Stability Check (Gravity) - 10 seconds...")
        stab_result = pb_verifier.run_stability_check(duration=10.0)
        
        if not stab_result.is_valid:
             print(f"PyBullet Instability Detected!")
             for ev in stab_result.evidence:
                 print(f"  [Stability] {ev.message}")
                 for bid in ev.brick_ids:
                     failed_brick_ids.add(bid)

        if not failed_brick_ids:
            print("Physical Verification PASSED (All Clear)!")
        else:
             print(f"Total Failed Bricks to Highlight: {len(failed_brick_ids)}")
             print(f"Failed IDs: {list(failed_brick_ids)[:10]}...")  # Show first 10
            
    except Exception as e:
        print(f"Verification Skipped due to error: {e}")
        import traceback
        traceback.print_exc()

    # ---------------------------------------------------------
    # [Visualization] Build Scene
    # ---------------------------------------------------------
    scene = trimesh.Scene()
    
    brick_counter = 0 
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('0'): continue
            
            parts = line.split()
            if not parts: continue
            
            line_type = parts[0]
            
            # Line Type 1: Sub-file (Part)
            if line_type == '1':
                # Format: 1 <colour> x y z a b c d e f g h <file>
                try:
                    # Parse Position & Rotation
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    a, b, c = float(parts[5]), float(parts[6]), float(parts[7])
                    d, e, f = float(parts[8]), float(parts[9]), float(parts[10])
                    g, h, i = float(parts[11]), float(parts[12]), float(parts[13])
                    
                    part_id = " ".join(parts[14:]) 
                    
                    # Mapping ID for Verification
                    verifier_id = f"{parts[14]}_{brick_counter}"
                    brick_counter += 1
                    
                    # ---------------------------------------------------------
                    # [Core Logic] Fetch Geometry (Restored)
                    # ---------------------------------------------------------
                    triangles = part_library.get_part_geometry(part_id)
                    
                    if not triangles:
                        continue
                        
                    # Convert triangles to Trimesh object
                    vertices = np.array([p for tri in triangles for p in tri])
                    faces = np.arange(len(vertices)).reshape(-1, 3)
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                    
                    # ---------------------------------------------------------
                    # [Transform] Apply LDraw Position & Rotation
                    # ---------------------------------------------------------
                    # 1. LDraw Transform Matrix
                    ldraw_matrix = np.eye(4)
                    ldraw_matrix[0, :3] = [a, b, c]
                    ldraw_matrix[1, :3] = [d, e, f]
                    ldraw_matrix[2, :3] = [g, h, i]
                    ldraw_matrix[0, 3] = x
                    ldraw_matrix[1, 3] = y
                    ldraw_matrix[2, 3] = z
                    
                    # 2. Conversion Matrix (LDraw -> Model)
                    scale = 1/20.0
                    conv_matrix = np.array([
                        [scale, 0,     0,     0],
                        [0,     0,     scale, 0],
                        [0,     -scale, 0,     0], 
                        [0,     0,     0,     1]
                    ])
                    
                    # Apply transformation
                    final_matrix = conv_matrix @ ldraw_matrix
                    mesh.apply_transform(final_matrix)
                    
                    # Debug matching
                    if verifier_id in failed_brick_ids:
                        print(f"!!! MATCH Found: Highlighting Failed Brick: {verifier_id}")
                        final_color = [255, 0, 0, 255] # RED
                    else:
                        rgb_or_rgba = LDRAW_COLORS.get(int(parts[1]), (128, 128, 128)) # Normal Color
                        if len(rgb_or_rgba) == 4:
                             final_color = list(rgb_or_rgba)
                        else:
                             final_color = [*rgb_or_rgba, 255]

                    mesh.visual.face_colors = final_color
                    
                    scene.add_geometry(mesh)
                    
                except Exception as e:
                    print(f"Error parsing line: {line}\n{e}")
                    continue

    return scene

if __name__ == "__main__":
    # Default test target
    target_ldr = os.path.join("ldr", "car.ldr")
    
    # CLI Argument override
    if len(sys.argv) > 1:
        target_ldr = sys.argv[1]
        
    print(f"Target LDR: {target_ldr}")
    
    # Run Conversion
    scene = ldr_to_mesh(target_ldr)
    
    if scene:
        print("Success! Scene loaded.")
        print("Opening 3D Viewer...")
        scene.show()
    else:
        print("Failed to load scene.")
