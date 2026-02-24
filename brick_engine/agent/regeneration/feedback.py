# ============================================================================
# 검증 피드백 변환/포맷 함수
# ============================================================================

from ..core.llm_state import VerificationFeedback


def extract_verification_feedback(result, total_bricks: int) -> VerificationFeedback:
    """PyBullet VerificationResult를 LLM 피드백 형식으로 변환"""
    feedback = VerificationFeedback()
    feedback.total_bricks = total_bricks
    feedback.stable = result.is_valid

    fallen_bricks = set()
    floating_bricks = set()
    first_failure = None
    collision_count = 0

    for ev in result.evidence:
        if ev.type == "FIRST_FAILURE":
            if ev.brick_ids:
                first_failure = ev.brick_ids[0]
                fallen_bricks.update(ev.brick_ids)
        elif ev.type == "COLLAPSE_AFTERMATH":
            if ev.brick_ids:
                fallen_bricks.update(ev.brick_ids)
        elif ev.type in ("FLOATING_BRICK", "FLOATING"):
            if ev.brick_ids:
                floating_bricks.update(ev.brick_ids)
        elif ev.type == "COLLISION":
            collision_count += 1

    feedback.fallen_bricks = len(fallen_bricks)
    feedback.floating_bricks = len(floating_bricks)
    feedback.floating_brick_ids = list(floating_bricks)
    feedback.fallen_brick_ids = list(fallen_bricks)
    feedback.first_failure_brick = first_failure
    feedback.collision_count = collision_count

    if total_bricks > 0:
        feedback.failure_ratio = (len(fallen_bricks) + len(floating_bricks)) / total_bricks

    feedback.stability_grade = getattr(result, 'stability_grade', 'STABLE')
    feedback.stability_score = int(getattr(result, 'score', 100))

    return feedback


def format_feedback(feedback: VerificationFeedback) -> str:
    """VerificationFeedback을 사람/LLM이 읽을 수 있는 텍스트로 변환"""
    GRADE_EMOJI = {"STABLE": "🟢", "MEDIUM": "🟡", "UNSTABLE": "🔴"}
    GRADE_LABEL = {"STABLE": "안정", "MEDIUM": "중간", "UNSTABLE": "불안정"}

    grade = feedback.stability_grade
    emoji = GRADE_EMOJI.get(grade, "⚪")
    label = GRADE_LABEL.get(grade, grade)

    lines = [
        f"검증 결과:",
        f"- {emoji} 안정성 등급: {label} ({grade})",
        f"- 📊 점수: {feedback.stability_score}/100",
        f"- 총 브릭 수: {feedback.total_bricks}개",
        f"- 최대 변위(Drift): {feedback.max_drift:.3f}",
    ]

    if feedback.small_brick_count > 0:
        lines.append(f"- 1x1 브릭: {feedback.small_brick_count}개 ({feedback.small_brick_ratio * 100:.1f}%)")

    if grade != "STABLE" or feedback.floating_bricks > 0:
        lines.extend([
            f"- 떨어진 브릭: {feedback.fallen_bricks}개",
            f"- 공중부양 브릭: {feedback.floating_bricks}개",
            f"- 실패율: {feedback.failure_ratio * 100:.1f}%",
        ])
        if feedback.first_failure_brick:
            lines.append(f"- 최초 붕괴 브릭: {feedback.first_failure_brick}")

    if feedback.collision_count > 0:
        lines.append(f"- 충돌 감지: {feedback.collision_count}건")

    return "\n".join(lines)
