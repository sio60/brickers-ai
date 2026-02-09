"""
brick-judge/physics
물리 테스트 모듈 (Rust 전용)
"""

import json
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..parser import ParsedModel, Brick

import brick_judge_rs as _rust_module


class IssueType(Enum):
    floating = "floating"
    isolated = "isolated"
    top_only = "top_only"
    unstable_base = "unstable_base"


class Severity(Enum):
    critical = "critical"
    high = "high"


@dataclass
class Issue:
    brick_id: Optional[int]
    issue_type: IssueType
    severity: Severity
    message: str = ""
    data: dict = None


def _bricks_to_json(bricks: List["Brick"]) -> str:
    data = []
    for b in bricks:
        studs_w = int(b.width / 20)
        studs_d = int(b.depth / 20)
        h_units = int(b.height / 8)
        rot = getattr(b, 'rotation', [1,0,0,0,1,0,0,0,1])
        is_rotated = abs(rot[0]) < 0.5 and abs(rot[2]) > 0.5 if len(rot) >= 3 else False
        data.append({
            "id": b.id, "x": b.ldr_x, "y": b.ldr_y, "z": b.ldr_z,
            "w": studs_w, "d": studs_d, "h": h_units, "rotated": is_rotated,
        })
    return json.dumps(data)


def _parse_rust_issues(json_str: str, bricks: List["Brick"] = None) -> List[Issue]:
    brick_map = {b.id: b for b in bricks} if bricks else {}
    issues = []
    for item in json.loads(json_str):
        issue_type = IssueType(item["issue_type"])
        severity = Severity(item["severity"])
        brick_id = item.get("brick_id")
        
        # 이슈 타입별 한국어 설명
        ISSUE_DESC = {
            "unstable_base": "⚖️ 무게중심이 지지면을 벗어남 (전복 위험)",
            "off_balance": "⚖️ 무게중심이 불안정함",
            "floating": "공중에 떠있음",
            "isolated": "다른 브릭과 연결 없음",
            "top_only": "위에서만 결합됨",
            "weak": "결합력 약함",
        }
        
        # 메시지 생성: brick_map에서 이름 찾거나, 기본 메시지 생성
        if brick_id is not None and brick_id in brick_map:
            msg = f"브릭 #{brick_id} ({brick_map[brick_id].name}) - {ISSUE_DESC.get(issue_type.value, issue_type.value)}"
        elif brick_id is not None:
            msg = f"브릭 #{brick_id} - {ISSUE_DESC.get(issue_type.value, issue_type.value)}"
        else:
            # 전역 이슈 (brick_id가 None)
            msg = item.get("message") or ISSUE_DESC.get(issue_type.value, f"전역 문제: {issue_type.value}")
        
        issues.append(Issue(
            brick_id=brick_id,
            issue_type=issue_type,
            severity=severity,
            message=msg,
            data=item.get("data")
        ))

    return issues


def full_judge(model: "ParsedModel") -> List[Issue]:
    json_str = _bricks_to_json(model.bricks)
    result = _rust_module.full_judge_json(json_str)
    return _parse_rust_issues(result, model.bricks)


def calc_score_from_issues(issues: List[Issue], total_bricks: int = 0) -> int:
    """점수 계산: 문제 브릭 비율 기반
    
    - 무게중심 불안정(unstable_base) → 0점
    - 그 외: 100 - (문제 브릭 수 / 전체 브릭 수 * 100)
    """
    # 무게중심 불안정이면 0점
    if any(i.issue_type.value in ("unstable_base", "off_balance") for i in issues):
        return 0
    
    # 문제 브릭 수 (중복 제거, top_only 제외)
    problem_brick_ids = set(
        i.brick_id for i in issues 
        if i.brick_id is not None and i.issue_type.value != "top_only"
    )
    
    if total_bricks <= 0:
        # 폴백: 기존 로직 (브릭 수 모를 때)
        score = 100
        for issue in issues:
            if issue.severity.value == "critical":
                score -= 30
            elif issue.severity.value == "high":
                score -= 15
        return max(0, min(100, score))
    
    # 문제 브릭 비율로 점수 계산
    problem_ratio = len(problem_brick_ids) / total_bricks
    score = int(100 * (1 - problem_ratio))
    
    return max(0, min(100, score))


def get_backend_info() -> dict:
    return {"backend": "rust", "version": _rust_module.version(), "module": "brick_judge_rs"}


__all__ = ["full_judge", "Issue", "IssueType", "Severity", "calc_score_from_issues", "get_backend_info"]
