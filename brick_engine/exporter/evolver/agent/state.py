"""Agent State Definition"""
from typing import Dict, List, Any, TypedDict, Annotated
import operator
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class Proposal(TypedDict):
    id: str
    type: str  # "relocate", "rotate", "rebuild", "symmetry_fix", "add_support", "remove", "bridge", "rollback"
    description: str
    brick_id: str | None
    position: Dict | None  # {"x": int, "y": int, "z": int}
    part_id: str | None
    risk: str  # "low", "medium", "high"
    score: float | None
    # Multi-Agent Debate 점수
    scores: Dict | None  # {"shape": 80, "physics": 70, "total": 150}
    reasoning: Dict | None  # {"shape": "이유", "physics": "이유"}

class Memory(TypedDict):
    failed_approaches: List[str]
    successful_patterns: List[str]
    lessons: List[str]
    consecutive_failures: int

class AgentState(TypedDict):
    # Model
    model: Any
    model_backup: Any
    # parts_db 제거 - config.py에서 get_config().parts_db 사용
    original_brick_count: int

    # GLB Reference (from vision analysis)
    glb_reference: Dict | None  # {model_type, name, legs, key_features, structure_notes}

    # Current state (from PhysicalVerifier)
    floating_count: int
    collision_count: int
    floating_bricks: List[Dict]
    verification_result: Any | None  # VerificationResult
    verification_score: float
    verification_evidence: List[Any]

    # Vision analysis (형태 분석)
    vision_quality_score: int | None  # 0-100, None=미분석
    vision_problems: List[Dict]  # [{location, issue}, ...]

    # Symmetry analysis (대칭 분석)
    symmetry_issues: List[Dict]  # [{missing_side, suggested_pos, part_id, color}, ...]
    model_type: str  # animal, vehicle, building, etc

    # Tracking
    iteration: int
    session_id: str
    total_removed: int
    action_history: List[Dict]

    # CoScientist
    strategy: str  # 단일 전략
    target_brick: Dict | None  # 전략 타겟
    proposals: List[Proposal]
    selected_proposal: Proposal | None

    # Memory
    memory: Memory

    # Finish
    should_finish: bool
    finish_reason: str

    # LangSmith 트레이싱용 대화 기록
    messages: Annotated[List[BaseMessage], add_messages]
