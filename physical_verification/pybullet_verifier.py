import pybullet as p
import pybullet_data
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import List, Dict, Set, Tuple

try:
    from .models import Brick, BrickPlan, VerificationResult, Evidence
    from .lego_physics import check_stud_tube_connection, find_floating_bricks, find_all_connections, get_brick_mass_kg, get_brick_studs_count, STUD_SPACING, BRICK_HEIGHT, PLATE_HEIGHT
    from .part_library import get_part_geometry
except ImportError:
    from models import Brick, BrickPlan, VerificationResult, Evidence
    from lego_physics import check_stud_tube_connection, find_floating_bricks, find_all_connections, get_brick_mass_kg, get_brick_studs_count, STUD_SPACING, BRICK_HEIGHT, PLATE_HEIGHT
    from part_library import get_part_geometry

class PyBulletVerifier:
    SCALE = 0.01  # Scale down LDU to manageable range for PyBullet camera
    
    def __init__(self, plan: BrickPlan = None, gui: bool = False):
        self.plan = plan
        self.gui = gui
        self.physicsClient = None
        self.brick_bodies = {} # brick_id -> body_id
        self.cached_shapes = {} # part_file -> collision_shape_id

    def _init_simulation(self):
        if self.physicsClient is None:
            # GUI mode helps debugging, DIRECT is faster
            mode = p.GUI if self.gui else p.DIRECT
            self.physicsClient = p.connect(mode)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        p.resetSimulation()
        # Gravity: scaled world means 1 unit = 40mm now. Real gravity is fine.
        p.setGravity(0, 0, -9.8)
        
        # High quality physics settings
        p.setPhysicsEngineParameter(
            numSolverIterations=100,  # Default is 50, higher = more stable constraints
            numSubSteps=4,            # More sub-steps per frame
            erp=0.1,                  # Error Reduction Parameter (keep constraints tight)
            contactERP=0.1
        )

    def _close_simulation(self):
        if self.physicsClient is not None:
            p.disconnect()
            self.physicsClient = None

    def _get_collision_shape(self, part_file: str):
        """Creates a simplified BOX collision shape for stability."""
        # Clean filename
        part_file = part_file.lower().strip()
        
        if part_file in self.cached_shapes:
            return self.cached_shapes[part_file]

        # Get Dimensions from Library
        try:
             # Dynamically import to ensure we have the dict
             from lego_physics import get_brick_studs_count, STUD_SPACING, BRICK_HEIGHT, PLATE_HEIGHT
             studs_x, studs_z, is_plate = get_brick_studs_count(part_file)
             height = PLATE_HEIGHT if is_plate else BRICK_HEIGHT
             
             # Calculate Half Extents (PyBullet wants half-sizes)
             # X total = studs_x * 20
             # Y total = height (24 or 8)
             # Z total = studs_z * 20
             
             # Apply Scale
             # Shrink by 1% (0.99) to avoid friction rub on vertical neighbors? 
             # Or just 1.0 since we disable collision on connected?
             # Let's use 0.99 for safety against horizontal neighbors.
             
             safe_factor = 0.99
             x_half = (studs_x * STUD_SPACING * self.SCALE * safe_factor) / 2.0
             y_half = (height * self.SCALE * safe_factor) / 2.0  # LDraw Y is Height
             z_half = (studs_z * STUD_SPACING * self.SCALE * safe_factor) / 2.0
             
             # PyBullet GEOM_BOX takes halfExtents
             # Note: We swap Y/Z later in placement, but here simpler to just create box
             # Width(X), Height(Y), Depth(Z) locally
             # Wait, in LDraw local:
             # X is Width
             # Y is Height
             # Z is Depth
             # So box extents should be [x, y, z]
             
             colShapeId = p.createCollisionShape(
                 p.GEOM_BOX, 
                 halfExtents=[x_half, y_half, z_half]
             )
             self.cached_shapes[part_file] = colShapeId
             return colShapeId
             
        except Exception as e:
            print(f"[WARN] Failed to create box for {part_file}: {e}")
            # Fallback
            colShapeId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
            self.cached_shapes[part_file] = colShapeId
            return colShapeId

    def load_bricks(self, plan: BrickPlan = None):
        """Loads bricks into PyBullet as Static Bodies."""
        if plan:
            self.plan = plan
        
        self._init_simulation()
        
        bricks = self.plan.get_all_bricks()
        
        # PRE-PASS: Calculate Z positions and find the minimum to offset to ground
        # In LDraw, Y is down. In PyBullet, Z is up.
        # LDraw origin for a brick is typically at the TOP surface.
        # So we need to offset up by brick height so bottom sits on ground.
        
        BRICK_HEIGHT_LDU = 24.0  # Standard brick height in LDU
        
        # Find the lowest point in PyBullet coords (considering brick bottom)
        z_positions = []
        for b in bricks:
            if b.origin is not None:
                # pb_z = -ldr_y * SCALE, then subtract half height to get bottom
                pb_z = -b.origin[1] * self.SCALE
                # LDraw origin at top, so bottom is pb_z - height
                bottom_z = pb_z - (BRICK_HEIGHT_LDU * self.SCALE)
                z_positions.append(bottom_z)
        
        # Offset to lift all bricks so lowest bottom = 0
        if z_positions:
            min_z = min(z_positions)
            z_offset = -min_z  # How much to add to each Z position
        else:
            z_offset = 0
            
        print(f"[PyBullet] Z Offset applied: {z_offset:.4f} (to place model on ground)")
        
        for b in bricks:
            # Check if we have original data
            if b.part_file is None or b.origin is None or b.matrix is None:
                print(f"[WARN] Brick {b.id} missing original LDraw data. Skipping PyBullet load.")
                continue
                
            shape_id = self._get_collision_shape(b.part_file)
            
            # Rotation Matrix (3x3) -> Quaternion
            # b.matrix is 3x3 numpy array
            # Scipy rotation
            try:
                r = R.from_matrix(b.matrix)
                # PyBullet Order: x, y, z, w
                quat = r.as_quat() 
            except Exception as e:
                print(f"[ERR] Matrix conversion failed for {b.id}: {e}")
                quat = [0, 0, 0, 1]

            # Create Body with REALISTIC MASS
            # Calculate actual brick weight based on volume (2x4 brick ≈ 2.3g)
            brick_mass = get_brick_mass_kg(b.part_file)
            
            # Add random color for better visibility
            import random
            col = [random.random(), random.random(), random.random(), 1.0]

            # Coordinate Transformation: LDraw (X, Y-down, Z) -> PyBullet (X, Z, -Y)
            # LDraw: Y is vertical (down positive), Z is depth
            # PyBullet: Z is vertical (up positive), Y is depth
            ldr_x, ldr_y, ldr_z = b.origin[0], b.origin[1], b.origin[2]
            pb_x = ldr_x * self.SCALE
            pb_y = ldr_z * self.SCALE
            pb_z = -ldr_y * self.SCALE + z_offset  # Apply offset to sit on ground
            
            # Box Primitive Center Adjustment
            # LDraw Origin: Top Surface Center (Y=0)
            # Box Primitive Origin: Geometric Center (Y=Height/2)
            # So if we place Box at (0,0,0), its top is at -Height/2, bottom at +Height/2? No.
            # Box centered at 0 extends from -H/2 to +H/2.
            # We want Top to be at 0. So Center must be at +H/2.
            # Wait, LDraw Y-down. 
            #   Top = Origin Y
            #   Bottom = Origin Y + Height
            #   Center = Origin Y + Height/2
            
            # PyBullet Setup:
            #   pb_x = ldr_x * SCALE
            #   pb_y = ldr_z * SCALE 
            #   pb_z = -ldr_y * SCALE + z_offset (This puts Origin at correct Z height)
            
            # BUT, p.createMultiBody places the COM/Link Frame.
            # If our visual/collision shape is a centered box, and we put Body at `pb_z` (which is Top surface in global z),
            # the box will stick UP half-height and DOWN half-height.
            # We want the box to extend DOWN from `pb_z`.
            # So we must shift the Box CENTER down by half-height in PyBullet Z.
            # PyBullet Z is Up. So "Down" is -Z.
            
            studs_x, studs_z, is_plate = get_brick_studs_count(b.part_file)
            height_val = PLATE_HEIGHT if is_plate else BRICK_HEIGHT
            half_h_scaled = (height_val * self.SCALE) / 2.0
            
            # Shift Z down by half height so the "Top" of the box aligns with the Body Origin
            pb_z -= half_h_scaled
            
            # Coordinate Transformation: LDraw (X, Y-down, Z) -> PyBullet (X, Z, -Y)
            # We already computed base (pb_x, pb_y)
            # Rotation is tricky. 
            # LDraw Matrix rotates vectors. 
            # LDraw (1,0,0) -> X, (0,1,0) -> Down, (0,0,1) -> Z
            # Our Box is defined in Local axes X=width, Y=height, Z=depth.
            # We need to map Local Y (height) to Global Down (-Z).
            # And Local X/Z to Global X/Y.
            
            # Just relying on standard coordinate conversion logic:
            coord_convert = R.from_euler('x', -90, degrees=True)
            try:
                brick_rotation = R.from_matrix(b.matrix)
                final_rotation = coord_convert * brick_rotation
                quat = final_rotation.as_quat()  # x, y, z, w
            except:
                quat = coord_convert.as_quat()
            
            body_id = p.createMultiBody(
                baseMass=brick_mass,  # Realistic mass (2x4 ≈ 0.0023kg)
                baseCollisionShapeIndex=shape_id,
                baseVisualShapeIndex=-1, 
                basePosition=[pb_x, pb_y, pb_z],
                baseOrientation=quat
            )
            p.changeVisualShape(body_id, -1, rgbaColor=col)
            
            self.brick_bodies[b.id] = body_id
            
        # 3. Auto-Camera Setup - ZOOM WAY OUT
        if self.gui and bricks:
            all_pos = [b.origin for b in bricks if b.origin is not None]
            if all_pos:
                all_pos = np.array(all_pos) * self.SCALE
                min_b = np.min(all_pos, axis=0)
                max_b = np.max(all_pos, axis=0)
                center = (min_b + max_b) / 2.0
                extent = np.linalg.norm(max_b - min_b)
                
                # Distance: zoom out far enough to see everything
                cam_dist = max(extent * 3.0, 10.0)  # At least 10 units back
                p.resetDebugVisualizerCamera(
                    cameraDistance=cam_dist, 
                    cameraYaw=45, 
                    cameraPitch=-30, 
                    cameraTargetPosition=center
                )

    def run_collision_check(self, tolerance: float = -0.05) -> VerificationResult:
        """
        Runs collision detection using Global Contact Points (Fast).
        """
        self.load_bricks()
        result = VerificationResult()
        
        # 1. Global Collision Check (Broadphase + Narrowphase)
        # performCollisionDetection is implicit in getContactPoints, but good to call explicitly
        p.performCollisionDetection()
        points = p.getContactPoints()
        
        collisions = []
        checked_pairs = set()

        if points:
            for pt in points:
                # pt[1] = bodyUniqueIdA, pt[2] = bodyUniqueIdB
                b1, b2 = pt[1], pt[2]
                
                # Avoid duplicates (A-B and B-A)
                if b1 > b2: b1, b2 = b2, b1
                if (b1, b2) in checked_pairs: continue
                checked_pairs.add((b1, b2))
                
                # pt[8] = contactDistance
                dist = pt[8]
                
                # Filter for significant penetration
                if dist < tolerance:
                     bid1 = [k for k, v in self.brick_bodies.items() if v == b1][0]
                     bid2 = [k for k, v in self.brick_bodies.items() if v == b2][0]
                     
                     msg = f"Mesh Collision: {bid1} <-> {bid2} (Depth: {abs(dist):.2f} LDU)"
                     collisions.append(msg)
                     result.add_hard_fail(Evidence(
                         type="COLLISION",
                         severity="CRITICAL",
                         brick_ids=[bid1, bid2],
                         message=msg
                     ))

        # Do NOT close simulation here, keep it open for stability check if needed
        # self._close_simulation() 
        
        if not collisions:
            print("PyBullet Verification Passed (No Collisions)")
            result.score = 100
        else:
            result.is_valid = False
            result.score = 0
            
        return result

    def run_stability_check(self, duration: float = 2.0) -> VerificationResult:
        """
        Runs gravity simulation to check for stability.
        Uses Contact Points to auto-generate constraints (Glue).
        """
        print("Initializing Stability Simulation...")
        result = VerificationResult()  # Initialize result at the start
        
        # Ensure simulation is valid. If run_collision_check was called, it's open.
        # If not, init it.
        if self.physicsClient is None:
            self._init_simulation()
            self.load_bricks()
             
        # Reset gravity for stability (scaled world, so use real gravity)
        p.setGravity(0, 0, -9.8) 
        
        # 1. Ground Plane (if not exists)
        # Check if plane is loaded? Just load it, it's fine.
        try:
            planeId = p.loadURDF("plane.urdf")
        except:
             pass # Maybe already loaded or file missing. Plane is builtin usually.

        # 2. Dynamic Mass & Constraints
        # We need to switch bodies to dynamic? They were created with mass=0.
        # PyBullet: changeDynamics can change mass!
        
        brick_bodies = self.brick_bodies
        brick_ids = list(brick_bodies.keys())
        original_positions = {}
        
        # Determine ground threshold (lowest Z)
        # We already offset so lowest is 0, but let's be safe.
        all_z = []
        for body_id in brick_bodies.values():
             pos, _ = p.getBasePositionAndOrientation(body_id)
             all_z.append(pos[2])
        
        min_z = min(all_z) if all_z else 0.0
        ground_threshold = min_z + 0.05 # Within 5cm (scaled) or 0.2 studs
        
        for bid, body_id in brick_bodies.items():
            pos, orn = p.getBasePositionAndOrientation(body_id)
            original_positions[body_id] = (pos, orn)
            
            # DYNAMIC BODIES STABILIZATION
            # All bricks participate in physics (no infinite mass anchoring)
            # This allows checking for global instability (toppling/tipping)
            # High friction prevents sliding, but allows rolling/tipping.
            p.changeDynamics(
                body_id, 
                -1, 
                mass=0.1,  # All bricks have mass
                lateralFriction=0.9,     # High friction to grip ground
                rollingFriction=0.1,
                spinningFriction=0.1,
                restitution=0.0,         # No bouncing
                linearDamping=0.5,       # Air resistance equivalent
                angularDamping=0.5
            )

        # 3. Create Constraints using STUD-TUBE CONNECTION LOGIC
        # Only connect bricks that are properly connected via stud-tube alignment
        constraints_count = 0
        
        # Get all bricks and find proper connections
        bricks = self.plan.get_all_bricks()
        print(f"[Stability] Checking {len(bricks)} bricks for STUD-TUBE connections...")
        
        # Use lego_physics module to find proper connections
        connections = find_all_connections(bricks)
        print(f"[Stability] Found {len(connections)} stud-tube connections.")
        
        # Create id -> body_id mapping
        id_to_body = brick_bodies
        
        # Create constraints only for properly connected bricks
        for brick_id_a, brick_id_b in connections:
            if brick_id_a not in id_to_body or brick_id_b not in id_to_body:
                continue
                
            body_a = id_to_body[brick_id_a]
            body_b = id_to_body[brick_id_b]
            
            # Calculate relative transform to preserve initial offset
            # We want to lock A to B in their CURRENT relative position.
            # Pivot at Center of A (Local A = [0,0,0])
            # We need Center of A expressed in Local B.
            
            pos_a, orn_a = p.getBasePositionAndOrientation(body_a)
            pos_b, orn_b = p.getBasePositionAndOrientation(body_b)
            
            # Transform P_a into B's local frame
            # Local_Pos = Rotate_Inv(World_Pos - Body_Pos)
            
            # Inverse of B's rotation
            inv_orn_b = p.invertTransform([0,0,0], orn_b)[1] # Only care about rotation inverse
            
            # Vector from B to A
            diff_pos = np.array(pos_a) - np.array(pos_b)
            
            # Rotate into B's frame
            # p.multiplyTransforms is easy way
            # But diff_pos is a vector.
            # Using multiplyTransforms:
            # T_world_to_b = (pos_b, orn_b)^-1
            # P_a_in_b = T_world_to_b * P_a
            
            # Use PyBullet helper:
            # invertTransform returns (invPos, invOrn)
            invPosB, invOrnB = p.invertTransform(pos_b, orn_b)
            localPosA_in_B, localOrnA_in_B = p.multiplyTransforms(invPosB, invOrnB, pos_a, orn_a)
            
            p.createConstraint(
                parentBodyUniqueId=body_a,
                parentLinkIndex=-1,
                childBodyUniqueId=body_b,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0],     # Pivot on A center
                childFramePosition=localPosA_in_B, # Pivot relative to B
                parentFrameOrientation=[0,0,0,1],  # Identity (keep A aligned with itself)
                childFrameOrientation=localOrnA_in_B # A's relative orientation in B
            )
            # IMPORTANT: Disable collision between connected bricks!
            # LDraw geometry overlaps (studs inside tubes), causing physics explosions if not disabled.
            p.setCollisionFilterPair(body_a, body_b, -1, -1, enableCollision=0)
            constraints_count += 1
        
        # Check for floating bricks (not connected to anything and not on ground)
        floating = find_floating_bricks(bricks)
        if floating:
            print(f"[Stability] WARNING: {len(floating)} floating bricks detected: {floating[:5]}...")
            for fid in floating:
                result.evidence.append(Evidence(
                    type="FLOATING_BRICK",
                    severity="CRITICAL",
                    brick_ids=[fid],
                    message=f"Brick {fid} is not connected to any structure"
                ))
        
        print(f"[Stability] Created {constraints_count} constraints (Stud-Tube connections).")
        
        # 4. Run Simulation
        steps = int(240 * duration)
        print(f"[Stability] Simulating {duration}s ({steps} steps)...")
        
        first_failure_id = None
        first_failure_step = -1
        
        # Real-time monitoring loop
        frame_skip = 10 if not self.gui else 1 # Check less often in Headless for speed, every frame in GUI? No, 10 is fine.
        
        print(f"[Stability] Starting simulation loop for {steps} steps...")
        
        for step in range(steps):
            p.stepSimulation()
            
            # Check for failure every few steps
            if step % 10 == 0:
                current_max_drift = 0.0
                worst_brick = None
                
                for bid, body_id in brick_bodies.items():
                    current_pos, _ = p.getBasePositionAndOrientation(body_id)
                    start_pos, _ = original_positions[body_id]
                    dist = np.linalg.norm(np.array(current_pos) - np.array(start_pos))
                    
                    if dist > current_max_drift:
                        current_max_drift = dist
                        worst_brick = bid
                
                # Debug Output every 60 steps (0.25s) or if drift > 0.1
                if step % 60 == 0 or current_max_drift > 0.1:
                    # Only print if relevant
                    if current_max_drift > 0.05:
                        print(f"   [Step {step}] Max Drift: {current_max_drift:.2f} (Brick {worst_brick})")

                # Threshold: 0.5 (approx 50 LDU = 2.5 studs displacement)
                # If a brick moves more than 2.5 studs, it's definitely falling.
                fail_threshold_val = 0.5 
                
                if current_max_drift > fail_threshold_val and first_failure_id is None:
                        first_failure_id = worst_brick
                        first_failure_step = step
                        print(f"[Stability] FAILED at step {step} ({step/240:.2f}s): {worst_brick} moved {current_max_drift:.2f}")
                        # break # Don't break immediately, let it fall a bit more to see aftermath?
                        # Actually for pipeline speed, break is better.
                        break
            
            if self.gui:
                import time
                time.sleep(1./240.)
                
        # 5. Check Displacement & Report
        # (result initialization moved to start)
        result.is_valid = not bool(first_failure_id) # Valid only if no bricks fell
        failed_bricks = list() # Use list for compatibility
        max_drift = 0.0
        drift_threshold = 0.5 # Same threshold for final check
        
        # Add First Failure Evidence if detected
        if first_failure_id:
            result.evidence.append(Evidence(
                type="FIRST_FAILURE",
                severity="CRITICAL",
                brick_ids=[first_failure_id],
                message=f"Structural collapse started at {first_failure_id} (t={first_failure_step/240:.2f}s)"
            ))
        
        for bid, body_id in brick_bodies.items():
            current_pos, _ = p.getBasePositionAndOrientation(body_id)
            start_pos, _ = original_positions[body_id]
            dist = np.linalg.norm(np.array(current_pos) - np.array(start_pos))
            max_drift = max(max_drift, dist)
            
            if dist > drift_threshold:
                failed_bricks.append(bid)
                # Only add detail evidence if it's NOT the first failure (to avoid dupes)
                if bid != first_failure_id:
                    result.evidence.append(Evidence(
                        type="COLLAPSE_AFTERMATH",
                        severity="ERROR",
                        brick_ids=[bid],
                        message=f"Brick moved {dist:.1f} units after collapse started"
                    ))

        if failed_bricks:
            result.is_valid = False
            result.score = 0
            print(f"[Stability] FAILED. Max drift: {max_drift:.2f}")
        else:
            print(f"[Stability] PASSED. Max drift: {max_drift:.2f}")
            result.score = 100
        
        # --- REPORT CARD ---
        print("\n" + "="*40)
        print(" 🏭 물리 검증 리포트 (Physics Report)")
        print("="*40)
        print(f" - 🧱 총 브릭 수: {len(brick_bodies)}") # Changed self.brick_bodies to brick_bodies
        print(f" - 🔗 연결 상태: {constraints_count}개 본드 결합 완료") # Changed self.constraints to constraints_count
        
        # Re-evaluate floating bricks for report, using ground_threshold from earlier
        # connected_bricks needs to be derived from constraints
        connected_brick_ids = set()
        for brick_id_a, brick_id_b in connections:
            connected_brick_ids.add(brick_id_a)
            connected_brick_ids.add(brick_id_b)

        # Assuming 'bricks' is a list of brick IDs from self.plan.get_all_bricks()
        # And 'brick_bodies' maps brick IDs to PyBullet body IDs
        # We need brick_plans and brick_ids to get brick positions for floating check
        # This information is not directly available in this scope.
        # For now, let's use the 'floating' variable already computed.
        # If the original 'floating' check was sufficient, we can reuse it.
        # The original 'floating' check was: floating = find_floating_bricks(bricks)
        # This 'floating' variable is already available.
        
        if floating: # Reusing the 'floating' variable from earlier check
             print(f" - ⚠️ 위험 요소: Floating Brick {len(floating)}개 발견! (주의)")
        else:
             print(f" - ✨ 구조 상태: 모든 브릭이 잘 연결됨")
             
        print("-" * 40)
        print(f" [시뮬레이션 결과]")
        print(f" - 🕒 진행 시간: {duration:.1f}초")
        print(f" - 📏 최대 이동(Drift): {max_drift:.2f} (허용치: {drift_threshold})") # Changed threshold to drift_threshold
        print("-" * 40)
        
        if result.score == 100: # Changed score to result.score
            print(" ✅ 최종 판정: [합격] (SUCCESS)")
            print("    \"이 모델은 튼튼합니다!\"")
        else:
            print(" ❌ 최종 판정: [불합격] (FAIL)")
            # Find the culprit
            culprit = "Unknown"
            for ev in result.evidence:
                if ev.type == "FIRST_FAILURE" and ev.brick_ids:
                    culprit = ev.brick_ids[0]
                    break
            print(f"    💥 최초 붕괴: {culprit}")
            
            # List other victims
            victims = []
            for ev in result.evidence:
                if ev.type == "COLLAPSE_AFTERMATH" and ev.brick_ids:
                    victims.append(ev.brick_ids[0])
            
            if victims:
                print(f"    📉 추가 붕괴 ({len(victims)}개): {', '.join(victims[:5])}" + (f"...외 {len(victims)-5}개" if len(victims)>5 else ""))
                
            print("    \"구조가 불안정하여 무너졌습니다.\"")
        print("="*40 + "\n")
        
        # If GUI, keep window open to let user see
        if self.gui:
            print("[PyBullet] Simulation finished. Press Enter to close window...")
            input()

        self._close_simulation()
        return result

# Simple Test
if __name__ == "__main__":
    # Mock
    pass
