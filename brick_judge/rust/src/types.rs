//! 공통 타입 정의 - 간단 버전

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// 브릭 데이터 (스터드 단위)
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Brick {
    pub id: i32,
    pub x: f64,      // LDR X
    pub y: f64,      // LDR Y (0=바닥, -24=위층)
    pub z: f64,      // LDR Z
    pub w: i32,      // 스터드 너비
    pub d: i32,      // 스터드 깊이
    pub h: i32,      // 높이 단위 (브릭=3, 플레이트=1)
    #[serde(default)]
    pub rotated: bool,
}

impl Brick {
    /// 실제 너비 (LDU) - 회전 고려
    pub fn width_ldu(&self) -> f64 {
        if self.rotated { self.d as f64 * 20.0 } else { self.w as f64 * 20.0 }
    }

    /// 실제 깊이 (LDU) - 회전 고려
    pub fn depth_ldu(&self) -> f64 {
        if self.rotated { self.w as f64 * 20.0 } else { self.d as f64 * 20.0 }
    }

    /// 실제 높이 (LDU)
    pub fn height_ldu(&self) -> f64 {
        self.h as f64 * 8.0
    }

    /// 스터드 개수
    pub fn studs(&self) -> i32 {
        self.w * self.d
    }

    /// 무게 (g) - 스터드당 약 0.3g
    pub fn weight(&self) -> f64 {
        self.studs() as f64 * 0.3 * (self.h as f64 / 3.0)
    }
}

/// 문제 심각도
#[derive(Clone, Debug, PartialEq)]
pub enum Severity {
    Critical,
    High,
}

impl Severity {
    pub fn as_str(&self) -> &'static str {
        match self {
            Severity::Critical => "critical",
            Severity::High => "high",
        }
    }
}

/// 문제 유형
#[derive(Clone, Debug, PartialEq)]
pub enum IssueType {
    Floating,    // 공중에 떠있음
    Isolated,    // 연결 없음
    TopOnly,     // 위쪽으로만 연결 (조립 어려움)
    UnstableBase, // 무게중심이 지지면 밖 (전복 위험)
}

impl IssueType {
    pub fn as_str(&self) -> &'static str {
        match self {
            IssueType::Floating => "floating",
            IssueType::Isolated => "isolated",
            IssueType::TopOnly => "top_only",
            IssueType::UnstableBase => "unstable_base",
        }
    }
}

/// 발견된 문제 (Rust → Python 반환용)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Issue {
    pub brick_id: Option<i32>,
    pub issue_type: String,
    pub severity: String,
    pub data: Option<serde_json::Value>,
}

impl Issue {
    pub fn new(brick_id: Option<i32>, issue_type: IssueType, severity: Severity) -> Self {
        Self {
            brick_id,
            issue_type: issue_type.as_str().to_string(),
            severity: severity.as_str().to_string(),
            data: None,
        }
    }

    pub fn with_data(mut self, data: serde_json::Value) -> Self {
        self.data = Some(data);
        self
    }
}
