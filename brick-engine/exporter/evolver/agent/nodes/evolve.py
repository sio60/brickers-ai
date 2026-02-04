"""EVOLVE Node - Execute selected proposal"""
import copy
from ..state import AgentState
from ..tools import (
    remove_brick, add_brick, rollback_model, get_model_state,
    try_add_brick_with_validation
)

MAX_REMOVAL_RATIO = 0.10

def node_evolve(state: AgentState) -> AgentState:
    """Execute the selected proposal with try-validate-rollback loop"""
    proposal = state["selected_proposal"]
    if not proposal:
        print("\n[EVOLVE] No proposal to execute")
        return state

    print(f"\n[EVOLVE] Executing: {proposal['id']}")

    action = {"type": proposal["type"], "proposal": proposal, "success": False}
    before_floating = state["floating_count"]

    if proposal["type"] == "auto_evolve":
        try:
            from evolver_cos import CoScientistEvolver
            from ldr_converter import validate_physics

            evolver = CoScientistEvolver(state["parts_db"], model_name=None)
            before = len(state["model"].bricks)

            result = evolver.evolve(
                state["model"],
                validate_physics(state["model"], state["parts_db"]),
                analyze_design=False
            )

            after = len(result.evolved_model.bricks)
            net_loss = before - after

            if net_loss > before * MAX_REMOVAL_RATIO:
                print(f"  BLOCKED: Would remove {net_loss} ({net_loss/before*100:.0f}%)")
                action["reason"] = "Too many removals"
            else:
                state["model"] = result.evolved_model
                state["total_removed"] += max(0, net_loss)
                action["success"] = True
                action["removed"] = net_loss
                print(f"  OK: Removed {net_loss} bricks")

        except Exception as e:
            print(f"  ERROR: {e}")
            action["error"] = str(e)

    elif proposal["type"] == "remove":
        result = remove_brick(state["model"], proposal["brick_id"])
        if result["success"]:
            state["total_removed"] += 1
            action["success"] = True
            action["backup"] = result["backup"]
            print(f"  Removed: {proposal['brick_id']}")
        else:
            print(f"  Failed: {result['error']}")

    elif proposal["type"] == "add_support" and proposal.get("candidates"):
        # 후보 위치들을 순회하면서 시도 → 검증 → 성공하면 종료
        candidates = proposal["candidates"]
        print(f"  Trying {len(candidates)} candidate positions...")

        best_result = None
        best_brick_id = None
        best_candidate = None

        for i, candidate in enumerate(candidates):
            # 모델 백업
            model_backup = copy.deepcopy(state["model"])

            # 브릭 추가 + 검증
            result = try_add_brick_with_validation(
                state["model"],
                candidate,
                state["parts_db"]
            )

            if result["success"]:
                new_floating = result["floating_count"]
                print(f"    Candidate {i+1}: floating {before_floating} -> {new_floating}")

                # 개선됐으면 채택
                if new_floating < before_floating:
                    print(f"    ✓ Improved! Keeping this candidate")
                    action["success"] = True
                    action["brick_id"] = result["brick_id"]
                    action["position"] = candidate
                    action["params"] = {"candidate_index": i, "position": candidate, "brick_id": result["brick_id"]}
                    break
                else:
                    # 안 좋아졌으면 롤백하고 다음 후보 시도
                    print(f"    ✗ No improvement, rolling back")
                    state["model"] = model_backup
            else:
                print(f"    Candidate {i+1}: Failed to add")

        if not action["success"]:
            print(f"  All candidates failed")
            action["params"] = {"candidates_count": len(candidates), "result": "all_failed"}

    elif proposal["type"] == "add_support" and proposal.get("position"):
        # 기존 방식 (단일 위치)
        pos = proposal["position"]
        color = 15
        if state["floating_bricks"]:
            color = state["floating_bricks"][0].get("color", 15)

        result = add_brick(
            state["model"],
            proposal.get("part_id", "3023"),
            pos["x"], pos["y"], pos["z"],
            color
        )
        if result["success"]:
            action["success"] = True
            action["brick_id"] = result["brick_id"]
            action["params"] = {"position": pos, "brick_id": result["brick_id"]}
            print(f"  Added: {result['brick_id']} at ({pos['x']}, {pos['y']}, {pos['z']})")

    elif proposal["type"] == "rollback":
        state["model"] = copy.deepcopy(state["model_backup"])
        state["total_removed"] = 0
        state["action_history"] = []
        action["success"] = True
        print(f"  Rolled back to {state['original_brick_count']} bricks")

    state["action_history"].append(action)

    return state
