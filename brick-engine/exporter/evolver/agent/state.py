"""Agent State Definition"""
from typing import Dict, List, Any, TypedDict, Annotated
import operator

class Proposal(TypedDict):
    id: str
    type: str  # "add_support", "remove", "bridge", "rollback"
    description: str
    brick_id: str | None
    position: Dict | None  # {"x": int, "y": int, "z": int}
    part_id: str | None
    risk: str  # "low", "medium", "high"
    score: float | None

class Memory(TypedDict):
    failed_approaches: List[str]
    successful_patterns: List[str]
    lessons: List[str]
    consecutive_failures: int

class AgentState(TypedDict):
    # Model
    model: Any
    model_backup: Any
    parts_db: Dict
    original_brick_count: int

    # GLB Reference (from vision analysis)
    glb_reference: Dict | None  # {model_type, name, legs, key_features, structure_notes}

    # Current state (from 승준's PhysicalVerifier)
    floating_count: int
    collision_count: int
    floating_bricks: List[Dict]
    verification_result: Any | None  # 승준's VerificationResult
    verification_score: float
    verification_evidence: List[Any]

    # Tracking
    iteration: int
    total_removed: int
    action_history: List[Dict]

    # CoScientist
    strategy: str
    target_brick: Dict | None
    proposals: List[Proposal]
    selected_proposal: Proposal | None

    # Memory
    memory: Memory

    # Finish
    should_finish: bool
    finish_reason: str
