//! brick_judge_rs - 간단한 브릭 물리 검증

use pyo3::prelude::*;
use serde_json;

mod types;
mod judge;

use types::{Brick, Issue};

/// JSON 문자열에서 브릭 배열 파싱
fn parse_bricks(json: &str) -> PyResult<Vec<Brick>> {
    serde_json::from_str(json)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("JSON 파싱 실패: {}", e)))
}

/// Issue 배열을 JSON 문자열로 변환
fn issues_to_json(issues: &[Issue]) -> PyResult<String> {
    serde_json::to_string(issues)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("JSON 직렬화 실패: {}", e)))
}

/// 물리 검증 (JSON 인터페이스)
#[pyfunction]
fn full_judge_json(bricks_json: &str) -> PyResult<String> {
    let bricks = parse_bricks(bricks_json)?;
    let issues = judge::full_judge(&bricks);
    issues_to_json(&issues)
}

/// 안정성 점수 계산
#[pyfunction]
fn calc_score_json(issues_json: &str) -> PyResult<f64> {
    let issues: Vec<Issue> = serde_json::from_str(issues_json)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("JSON 파싱 실패: {}", e)))?;
    Ok(judge::calc_stability_score(&issues))
}

/// 버전
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Python 모듈
#[pymodule]
fn brick_judge_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(full_judge_json, m)?)?;
    m.add_function(wrap_pyfunction!(calc_score_json, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
