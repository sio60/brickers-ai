//! 물리 검증 - 스터드 위치 기반 연결 체크

use crate::types::{Brick, Issue, IssueType, Severity};
use std::collections::{HashMap, HashSet, VecDeque};
use glam::DVec3;

/// 높이 허용 오차 (LDU)
const HEIGHT_TOL: f64 = 5.0;

/// 브릭의 스터드 위치들 반환 (x, z 좌표 목록)
fn get_stud_positions(brick: &Brick) -> Vec<(i64, i64)> {
    let mut positions = Vec::new();

    // 스터드 단위 크기 (회전 고려)
    let (w, d) = if brick.rotated {
        (brick.d, brick.w)
    } else {
        (brick.w, brick.d)
    };

    // 중심 좌표 (정수로 반올림)
    let cx = brick.x.round() as i64;
    let cz = brick.z.round() as i64;

    // 스터드 간격 20 LDU
    // 첫 스터드 오프셋: -(w-1)*10, 그 다음 +20씩
    for i in 0..w {
        for j in 0..d {
            let sx = cx + (i as i64 * 20) - ((w as i64 - 1) * 10);
            let sz = cz + (j as i64 * 20) - ((d as i64 - 1) * 10);
            positions.push((sx, sz));
        }
    }

    positions
}

/// 두 브릭이 스터드로 연결되는지 (XZ 위치 겹침)
fn has_stud_connection(b1: &Brick, b2: &Brick) -> bool {
    let studs1 = get_stud_positions(b1);
    let studs2 = get_stud_positions(b2);

    // 하나라도 같은 위치면 연결
    for s1 in &studs1 {
        for s2 in &studs2 {
            if s1.0 == s2.0 && s1.1 == s2.1 {
                return true;
            }
        }
    }
    false
}

/// 메인 검증 함수
pub fn full_judge(bricks: &[Brick]) -> Vec<Issue> {
    let mut issues = Vec::new();

    if bricks.is_empty() {
        return issues;
    }

    // 연결 그래프 (위/아래 구분)
    // supports_me: 나를 아래에서 지지해주는 브릭들
    // i_support: 내가 아래에서 지지하는 브릭들 (내 위에 있는)
    let mut supports_me: HashMap<i32, HashSet<i32>> = HashMap::new();
    let mut i_support: HashMap<i32, HashSet<i32>> = HashMap::new();
    let mut all_connections: HashMap<i32, HashSet<i32>> = HashMap::new();

    for b in bricks {
        supports_me.insert(b.id, HashSet::new());
        i_support.insert(b.id, HashSet::new());
        all_connections.insert(b.id, HashSet::new());
    }

    for brick in bricks {
        let brick_h = brick.height_ldu();

        for other in bricks {
            if other.id == brick.id {
                continue;
            }

            let is_stud_connected = has_stud_connection(brick, other);
            let mut is_connected = false;

            let other_h = other.height_ldu();
            
            // 1. 스터드 기반 수직 연결 확인
            if is_stud_connected {
                let other_top = other.y - other_h;
                if (other_top - brick.y).abs() < HEIGHT_TOL {
                    // other가 brick을 지지
                    supports_me.get_mut(&brick.id).unwrap().insert(other.id);
                    i_support.get_mut(&other.id).unwrap().insert(brick.id);
                    is_connected = true;
                }
                
                let brick_top = brick.y - brick_h;
                if (brick_top - other.y).abs() < HEIGHT_TOL {
                    // brick이 other를 지지
                    i_support.get_mut(&brick.id).unwrap().insert(other.id);
                    supports_me.get_mut(&other.id).unwrap().insert(brick.id);
                    is_connected = true;
                }
            }



            if is_connected {
                all_connections.get_mut(&brick.id).unwrap().insert(other.id);
                all_connections.get_mut(&other.id).unwrap().insert(brick.id);
            }
        }
    }

    // 전체 브릭에서 연결 그룹 찾기 (Union-Find 방식)
    let mut all_groups: Vec<HashSet<i32>> = Vec::new();
    let mut visited: HashSet<i32> = HashSet::new();

    for brick in bricks {
        if visited.contains(&brick.id) {
            continue;
        }

        // BFS로 이 브릭과 연결된 모든 브릭 찾기
        let mut group: HashSet<i32> = HashSet::new();
        let mut q: VecDeque<i32> = VecDeque::new();
        q.push_back(brick.id);

        while let Some(current) = q.pop_front() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current);
            group.insert(current);

            if let Some(neighbors) = all_connections.get(&current) {
                for &neighbor in neighbors {
                    if !visited.contains(&neighbor) {
                        q.push_back(neighbor);
                    }
                }
            }
        }

        all_groups.push(group);
    }

    // 바닥 브릭 (가장 큰 Y값 = LDR에서 바닥)
    let max_y = bricks.iter().map(|b| b.y).fold(f64::NEG_INFINITY, f64::max);
    let ground_bricks: HashSet<i32> = bricks
        .iter()
        .filter(|b| b.y >= max_y - 10.0)
        .map(|b| b.id)
        .collect();

    // 메인 그룹 = 가장 큰 그룹 (스터드 총합 > 브릭 수 > 바닥 브릭 수 > 가장 작은 id)
    let main_group: HashSet<i32> = all_groups.iter()
        .max_by_key(|g| {
            let ground_count = g.intersection(&ground_bricks).count();
            let total_studs: i32 = g.iter()
                .filter_map(|id| bricks.iter().find(|b| b.id == *id))
                .map(|b| b.studs())
                .sum();
            // tie-breaker: 가장 작은 id를 가진 그룹 (음수로 변환하여 max가 min id가 되도록)
            let min_id = g.iter().min().copied().unwrap_or(i32::MAX);
            (total_studs, g.len(), ground_count, -min_id)
        })
        .cloned()
        .unwrap_or_default();

    // 메인 그룹에 속한 브릭만 connected_to_ground
    let connected_to_ground: HashSet<i32> = main_group.clone();

    // 이슈 감지
    for brick in bricks {
        let has_support_below = !supports_me.get(&brick.id).unwrap().is_empty();
        let has_bricks_above = !i_support.get(&brick.id).unwrap().is_empty();
        let is_ground = ground_bricks.contains(&brick.id);
        let conn_count = all_connections.get(&brick.id).map_or(0, |c| c.len());

        // 1. floating: 바닥과 연결 안 됨
        if !connected_to_ground.contains(&brick.id) {
            issues.push(Issue::new(
                Some(brick.id),
                IssueType::Floating,
                Severity::Critical,
            ));
            continue;
        }

        // 2. isolated: 연결 없음 (바닥 브릭 제외 - 바닥은 자체로 안정)
        if !is_ground && conn_count == 0 && bricks.len() > 1 {
            issues.push(Issue::new(
                Some(brick.id),
                IssueType::Isolated,
                Severity::High,
            ));
            continue;
        }

        // 3. top_only: 아래 지지 없이 위에만 연결됨 (바닥 제외)
        if !is_ground && !has_support_below && has_bricks_above {
            issues.push(Issue::new(
                Some(brick.id),
                IssueType::TopOnly,
                Severity::High,
            ));
        }
    }

    // 4. stability: 무게중심 검사 (연결 여부 무관하게 전체 브릭 대상)
    let all_brick_ids: HashSet<i32> = bricks.iter().map(|b| b.id).collect();
    if let Some(issue) = crate::stability::check_stability(bricks, &all_brick_ids, &ground_bricks) {
        issues.push(issue);
    }


    issues
}

/// 안정성 점수 계산
pub fn calc_stability_score(issues: &[Issue]) -> f64 {
    let mut score: f64 = 100.0;

    for issue in issues {
        match issue.severity.as_str() {
            "critical" => score -= 30.0,
            "high" => score -= 15.0,
            "medium" => score -= 5.0,
            _ => score -= 2.0,
        }
    }

    score.max(0.0).min(100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_brick(id: i32, x: f64, y: f64, z: f64, w: i32, d: i32) -> Brick {
        Brick { id, x, y, z, w, d, h: 3, rotated: false }
    }

    #[test]
    fn test_stacked_bricks() {
        let bricks = vec![
            make_brick(0, 0.0, 0.0, 0.0, 2, 2),
            make_brick(1, 0.0, -24.0, 0.0, 2, 2),
        ];

        let issues = full_judge(&bricks);
        assert!(issues.is_empty(), "정상 스택은 이슈 없어야 함");
    }

    #[test]
    fn test_top_only_brick() {
        // 구조:
        //       [D]  ← C와 B 위에 걸침
        //     [C]    ← A 위에
        // [A]   [B]  ← B는 아래 지지 없고 D만 위에 있음 = top_only
        //
        // 연결: A → C → D → B (모두 바닥과 연결됨)
        // B: 아래 지지 없음, D를 위에서 지지 → top_only!
        let bricks = vec![
            make_brick(0, 0.0, 0.0, 0.0, 2, 2),       // A: 바닥
            make_brick(1, 60.0, -24.0, 0.0, 2, 2),    // B: 아래 지지 없음
            make_brick(2, 20.0, -24.0, 0.0, 4, 2),    // C: A 위에
            make_brick(3, 40.0, -48.0, 0.0, 4, 2),    // D: C와 B 위에 걸침
        ];

        let issues = full_judge(&bricks);
        println!("Issues: {:?}", issues);
        // B(id=1)는 아래 지지 없이 D만 위에 있음 = top_only
        assert!(issues.iter().any(|i| i.brick_id == Some(1) && i.issue_type == "top_only"),
            "B should be top_only: {:?}", issues);
    }

    #[test]
    fn test_separated_ground_bricks() {
        // 두 개의 바닥 브릭이 떨어져 있음 (스터드 연결 없음)
        // 큰 그룹이 메인, 작은 그룹은 floating
        let bricks = vec![
            make_brick(0, 0.0, 0.0, 0.0, 2, 2),
            make_brick(1, 200.0, 0.0, 0.0, 2, 2),
        ];

        let issues = full_judge(&bricks);
        // 둘 다 바닥이고 크기 같으니까, 하나는 메인, 하나는 floating
        assert_eq!(issues.len(), 1);
        assert!(issues.iter().any(|i| i.issue_type == "floating"));
    }
}
