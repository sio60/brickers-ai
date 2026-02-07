use crate::types::{Brick, Issue, IssueType, Severity};
use glam::{DVec2, DVec3};

/// 브릭 무게 데이터 (g) - Luo et al. "Legolization" 논문 기반
fn get_brick_mass(brick: &Brick) -> f64 {
    // 스터드 개수 기반 근사치 (개당 약 0.275g)
    let studs = brick.studs() as f64;
    let height_ratio = brick.h as f64 / 3.0; // brick=1.0, plate=0.33
    studs * 0.275 * height_ratio
}

/// 전체 무게중심(COM) 계산
fn calculate_com(bricks: &[Brick], group_ids: &std::collections::HashSet<i32>) -> (DVec3, f64) {
    let mut total_mass = 0.0;
    let mut weighted_sum = DVec3::ZERO;

    for b in bricks {
        if !group_ids.contains(&b.id) {
            continue;
        }

        let mass = get_brick_mass(b);
        // LDraw 좌표계: Y가 아래쪽.
        // 브릭의 물리적 중심 = y - (height / 2)
        let center = DVec3::new(
            b.x,
            b.y - (b.height_ldu() / 2.0),
            b.z,
        );
        
        weighted_sum += center * mass;
        total_mass += mass;
    }

    if total_mass == 0.0 {
        (DVec3::ZERO, 0.0)
    } else {
        (weighted_sum / total_mass, total_mass)
    }
}

/// Cross product of 2D vectors (z-component)
fn cross_product(o: DVec2, a: DVec2, b: DVec2) -> f64 {
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
}

/// 2D Convex Hull (Monotone Chain Algorithm)
fn convex_hull(mut points: Vec<DVec2>) -> Vec<DVec2> {
    if points.len() < 3 {
        return points;
    }

    // Sort by x, then y
    points.sort_by(|a, b| {
        a.x.partial_cmp(&b.x).unwrap()
            .then(a.y.partial_cmp(&b.y).unwrap())
    });
    // Remove duplicates
    points.dedup();

    let mut hull = Vec::new();

    // Lower hull
    for p in &points {
        while hull.len() >= 2 && cross_product(hull[hull.len() - 2], hull[hull.len() - 1], *p) <= 0.0 {
            hull.pop();
        }
        hull.push(*p);
    }

    // Upper hull
    let lower_len = hull.len();
    for p in points.iter().rev() {
        while hull.len() > lower_len && cross_product(hull[hull.len() - 2], hull[hull.len() - 1], *p) <= 0.0 {
            hull.pop();
        }
        hull.push(*p);
    }

    // Remove duplicate end point (start point is repeated at end by algorithm, logic handles it)
    // Monotone chain output: start -> ... -> end -> ... -> start.
    // We want the polygon vertices. pop the last one which is duplicate of first.
    if hull.len() > 1 {
        hull.pop(); 
    }

    hull
}

/// Point in Convex Polygon Test
/// For a convex polygon (CCW), point must be to the left (or right) of all edges.
fn is_inside_convex(point: DVec2, hull: &[DVec2]) -> bool {
    if hull.len() < 3 {
        return false;
    }

    for i in 0..hull.len() {
        let p1 = hull[i];
        let p2 = hull[(i + 1) % hull.len()];
        // If cross product is negative (assuming CCW and standard coords), it's outside.
        // Or positive depending on system.
        // Monotone chain produces CCW order (if X is right, Y is up).
        // LDraw XZ plane: X right, Z forward (or back). 
        // Let's rely on consistency: if it's INSIDE, signs should be SAME for all edges.
        
        let cp = cross_product(p1, p2, point);
        if cp <= 0.0 { 
            // If any edge has point on right (or line), checks logic.
            // Actually, we should check if ALL are positive (or all negative).
            // But Monotone Chain guarantees order. 
            // Let's assume CCW. If cp < 0, it's strictly outside (right turn).
            // == 0 is on edge (stable).
            return false;
        }
    }
    true
}

/// 지지면(Base Polygon) 계산
fn get_support_polygon(bricks: &[Brick], ground_bricks: &std::collections::HashSet<i32>) -> Vec<DVec2> {
    let mut points = Vec::new();

    for b in bricks {
        if !ground_bricks.contains(&b.id) {
            continue;
        }

        let w = b.width_ldu(); // rotated considered inside brick
        let d = b.depth_ldu();

        let h_x = w / 2.0;
        let h_z = d / 2.0;

        // Brick center (x, z)
        let cx = b.x;
        let cz = b.z;

        points.push(DVec2::new(cx - h_x, cz - h_z));
        points.push(DVec2::new(cx + h_x, cz - h_z));
        points.push(DVec2::new(cx + h_x, cz + h_z));
        points.push(DVec2::new(cx - h_x, cz + h_z));
    }

    convex_hull(points)
}

/// 안정성 검사 실행
pub fn check_stability(
    bricks: &[Brick], 
    main_group: &std::collections::HashSet<i32>, 
    ground_bricks: &std::collections::HashSet<i32>
) -> Option<Issue> {
    
    if main_group.is_empty() || ground_bricks.is_empty() {
        return None;
    }

    // 1. 전체 COM 계산
    let (com_3d, _mass) = calculate_com(bricks, main_group);
    let com_2d = DVec2::new(com_3d.x, com_3d.z);

    // 2. 바닥 지지면 Polygon
    let hull = get_support_polygon(bricks, ground_bricks);

    // 3. 포함 여부 확인
    if !is_inside_convex(com_2d, &hull) {
        println!("Rust Debug: Unstable Base Detected! COM=({:.2}, {:.2})", com_2d.x, com_2d.y);
        
        // Tipping Vector 계산 (Polygon 중심 -> COM)
        // 간단히 Hull의 AABB 중심이나 평균점을 사용
        let center = hull.iter().fold(DVec2::ZERO, |acc, p| acc + *p) / hull.len() as f64;
        let tipping_vec = com_2d - center;
        let tipping_dir = tipping_vec.normalize_or_zero();

        println!("Rust Debug: Tipping Vector Calculated: x={:.2}, z={:.2}", tipping_dir.x, tipping_dir.y);


        let _msg = format!(
            "무게중심(x:{:.1}, z:{:.1})이 바닥 지지면을 벗어났습니다. 전복 위험!",
            com_3d.x, com_3d.z
        );
        
        // 데이터 추가
        let data = serde_json::json!({
            "tipping": {
                "x": tipping_dir.x,
                "z": tipping_dir.y // DVec2.y corresponds to Z in 3D
            }
        });

        return Some(Issue::new(
            None, // Global issue
            IssueType::UnstableBase,
            Severity::Critical,
        ).with_data(data));
    }

    None
}
