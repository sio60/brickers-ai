# brick-judge: LDR 물리 검증
from .parser import parse_ldr, parse_ldr_string, Brick, ParsedModel, Point3D
from .physics import full_judge, Issue, IssueType, Severity, calc_score_from_issues

__version__ = "0.1.0"

__all__ = [
    # Parser
    "parse_ldr",
    "parse_ldr_string",
    "Brick",
    "ParsedModel",
    "Point3D",
    # Physics
    "full_judge",
    "Issue",
    "IssueType",
    "Severity",
    "calc_score_from_issues",
]
