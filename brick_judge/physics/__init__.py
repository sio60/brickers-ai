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
        if brick_id is not None and str(brick_id) in brick_map:
            msg = f"브릭 #{brick_id} ({brick_map[str(brick_id)].name}) - {issue_type.value}"
        else:
            msg = item.get("message", "")
        
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


def calc_score_from_issues(issues: List[Issue]) -> float:
    score = 100.0
    for issue in issues:
        if issue.severity.value == "critical":
            score -= 30
        elif issue.severity.value == "high":
            score -= 15
    return max(0, min(100, score))


def get_backend_info() -> dict:
    return {"backend": "rust", "version": _rust_module.version(), "module": "brick_judge_rs"}


__all__ = ["full_judge", "Issue", "IssueType", "Severity", "calc_score_from_issues", "get_backend_info"]
