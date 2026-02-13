# ============================================================================
# 세션 리포트 생성
# ============================================================================

import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger("CoScientistMemory")


def generate_session_report(collection_exps, collection_sessions, session_id: str) -> Dict[str, Any]:
    """
    세션의 모든 실험을 분석하여 피드백 보고서 생성

    Returns:
        Dict containing:
        - model_id, total_iterations, success_rate
        - tools_analysis, key_lessons, timeline
        - final_recommendation
    """
    if collection_exps is None:
        return {"error": "Database not connected"}

    try:
        experiments = list(collection_exps.find(
            {"session_id": session_id}
        ).sort("iteration", 1))

        if not experiments:
            return {"error": "No experiments found for session"}

        # 기본 정보
        model_id = experiments[0].get("model_id", "unknown")
        agent_type = experiments[0].get("agent_type", "unknown")

        # 통계 계산
        total = len(experiments)
        successful = sum(1 for e in experiments if e.get("result_success", False))
        failed = total - successful

        # 도구 사용 통계
        tools_used = {}
        for exp in experiments:
            tool = exp.get("experiment", {}).get("tool", "unknown")
            if tool not in tools_used:
                tools_used[tool] = {"count": 0, "success": 0, "fail": 0}
            tools_used[tool]["count"] += 1
            if exp.get("result_success"):
                tools_used[tool]["success"] += 1
            else:
                tools_used[tool]["fail"] += 1

        for tool in tools_used:
            stats = tools_used[tool]
            stats["success_rate"] = round(stats["success"] / stats["count"] * 100, 1) if stats["count"] > 0 else 0

        # 핵심 교훈 추출
        lessons = []
        for exp in experiments:
            lesson = exp.get("improvement", {}).get("lesson_learned", "")
            if lesson and lesson not in lessons:
                lessons.append(lesson)

        # 타임라인
        timeline = []
        for exp in experiments:
            timeline.append({
                "iteration": exp.get("iteration", 0),
                "tool": exp.get("experiment", {}).get("tool", "unknown"),
                "success": exp.get("result_success", False),
                "analysis": exp.get("verification", {}).get("numerical_analysis", ""),
                "delta": exp.get("verification", {}).get("delta", {})
            })

        # 초기/최종 상태 비교
        first_exp = experiments[0]
        last_exp = experiments[-1]
        initial_metrics = first_exp.get("verification", {}).get("metrics_before", {})
        final_metrics = last_exp.get("verification", {}).get("metrics_after", {})

        improvement_by_metric = {}
        for key in initial_metrics:
            if key in final_metrics:
                try:
                    start_val = initial_metrics[key]
                    end_val = final_metrics[key]
                    if isinstance(start_val, (int, float)) and isinstance(end_val, (int, float)):
                        delta = end_val - start_val
                        pct = round((start_val - end_val) / start_val * 100, 1) if start_val != 0 else 0
                        improvement_by_metric[key] = {
                            "start": start_val,
                            "end": end_val,
                            "delta": delta,
                            "improvement_pct": pct
                        }
                except:
                    pass

        # 성공/실패 패턴 분석
        tool_sequence = [t["tool"] for t in timeline]
        successful_sequences = []
        failed_sequences = []

        for i in range(len(timeline) - 1):
            pair = [timeline[i]["tool"], timeline[i+1]["tool"]]
            if timeline[i+1]["success"]:
                if pair not in successful_sequences:
                    successful_sequences.append(pair)
            else:
                if pair not in failed_sequences:
                    failed_sequences.append(pair)

        # 메트릭별 최적 도구 분석
        best_tool_by_metric = {}
        for metric in improvement_by_metric:
            best_delta = 0
            best_tool_for_metric = "none"
            for exp in experiments:
                exp_delta = exp.get("verification", {}).get("delta", {}).get(metric, 0)
                if exp_delta < best_delta:
                    best_delta = exp_delta
                    best_tool_for_metric = exp.get("experiment", {}).get("tool", "unknown")
            if best_tool_for_metric != "none":
                best_tool_by_metric[metric] = best_tool_for_metric

        # 최종 권장사항
        best_tool = max(tools_used.items(), key=lambda x: x[1]["success_rate"])[0] if tools_used else "none"
        worst_tool = min(tools_used.items(), key=lambda x: x[1]["success_rate"])[0] if tools_used else "none"

        recommendation = f"가장 효과적인 도구: {best_tool} ({tools_used.get(best_tool, {}).get('success_rate', 0)}% 성공률)"
        if worst_tool != best_tool:
            recommendation += f" | 피해야 할 도구: {worst_tool} ({tools_used.get(worst_tool, {}).get('success_rate', 0)}% 성공률)"

        # RAG용 임베딩 요약
        success_pct = round(successful / total * 100) if total > 0 else 0
        tools_summary = " ".join([f"{t}:{s['success']}/{s['count']}" for t, s in tools_used.items()])
        embedding_summary = f"{model_id} {agent_type} iter={total} success={success_pct}% {tools_summary}"

        report = {
            "session_id": session_id,
            "model_id": model_id,
            "agent_type": agent_type,
            "generated_at": datetime.utcnow().isoformat(),
            "statistics": {
                "total_iterations": total,
                "successful_count": successful,
                "failed_count": failed,
                "success_rate": round(successful / total * 100, 1) if total > 0 else 0
            },
            "detailed_metrics": {
                "initial_state": initial_metrics,
                "final_state": final_metrics,
                "improvement_by_metric": improvement_by_metric
            },
            "tools_analysis": tools_used,
            "patterns": {
                "tool_sequence": tool_sequence,
                "successful_sequences": successful_sequences[-3:],
                "failed_sequences": failed_sequences[-3:],
                "best_tool_by_metric": best_tool_by_metric
            },
            "key_lessons": lessons[-5:],
            "timeline": timeline,
            "final_recommendation": recommendation,
            "embedding_summary": embedding_summary
        }

        # DB에 보고서 저장
        if collection_sessions is not None:
            collection_sessions.update_one(
                {"_id": session_id},
                {"$set": {
                    "report": report,
                    "status": "COMPLETED",
                    "end_time": datetime.utcnow().isoformat()
                }}
            )

        logger.info(f"Session report generated: {model_id} - {successful}/{total} success")
        return report

    except Exception as e:
        logger.error(f"Failed to generate session report: {e}")
        return {"error": str(e)}
