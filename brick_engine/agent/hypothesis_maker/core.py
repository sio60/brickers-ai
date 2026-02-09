
import os
import json
import asyncio
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

# External Libraries
import trimesh
import numpy as np

# Project Modules
try:
    from ..llm_clients import OpenAIClient
    from ..memory_utils import MemoryUtils, build_hypothesis
    import config
except ImportError:
    # For standalone testing
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    from llm_clients import OpenAIClient
    from memory_utils import MemoryUtils, build_hypothesis
    import config

logger = logging.getLogger("HypothesisMaker")


class HypothesisMaker:
    """
    Co-Scientist V2 핵심 로직
    - Shape Analysis: GLB 형태 분석 (부피, 비율)
    - Dual-Model RAG: 
        1. Gemini (Creator): 성공 사례 기반 계획 수립
        2. GPT (Critic): 실패 사례 기반 비판 및 개선
    """
    def __init__(self, memory_manager: MemoryUtils, gemini_client: Any):
        self.memory = memory_manager
        self.gemini_client = gemini_client
        
        # GPT Client (Critic)
        self.gpt_client = OpenAIClient(model_name="gpt-4o-mini")
        if not self.gpt_client.client:
            logger.warning("⚠️ GPT Client not available. Critic mode disabled.")

    def analyze_shape(self, glb_path: str) -> Dict[str, Any]:
        """
        GLB 파일의 기하학적 특징(Shape Fingerprint) 추출
        """
        try:
            mesh = trimesh.load(glb_path, force='mesh')
            
            # 1. Bounds & Dimensions
            bounds = mesh.bounds
            extents = mesh.extents
            
            # 2. Volume (Watertight가 아닐 수 있으므로 Convex Hull 사용)
            try:
                volume = mesh.volume
            except:
                volume = mesh.convex_hull.volume
                
            # 3. Aspect Ratio (Width / Depth)
            # Y is up in GLB usually, but check extents logic
            width, height, depth = extents
            aspect_ratio = width / depth if depth > 0 else 1.0
            
            # 4. Center of Mass
            center_mass = mesh.center_mass.tolist() if hasattr(mesh, 'center_mass') else [0,0,0]

            metrics = {
                "volume": float(volume),
                "bbox_size": [float(x) for x in extents],
                "aspect_ratio": float(aspect_ratio),
                "center_mass": center_mass,
                "is_watertight": mesh.is_watertight
            }
            return metrics

        except Exception as e:
            logger.error(f"Shape analysis failed: {e}")
            return {}

    async def make_hypothesis(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        [Advanced Dual-Model Flow]
        1. Search: Success & Failure cases (Parallel)
        2. Draft (Gemini): Create initial plan based on Success cases
        3. Critique (GPT): Analyze Draft against Failure cases ("Does this plan look like a known failure?")
        4. Refine (Gemini): Finalize plan considering Critique
        """
        observation = state.get("observation", "")
        verification = state.get("verification_result", {})
        
        # 1. Shape Analysis (if new GLB available)
        shape_metrics = {} 
        # TODO: Integrate shape analysis if GLB path is available in state
        
        # 2. Dual-Search (Async)
        logger.info(f"🔎 이중 검색(Dual-Search) 시작: {observation[:50]}...")
        rag_results = await asyncio.to_thread(
            self.memory.search_success_and_failure,
            observation=observation,
            limit=3,
            min_score=0.4,
            verification_metrics=verification,
            shape_metrics=shape_metrics
        )
        
        success_cases = rag_results.get("success", [])
        failure_cases = rag_results.get("failure", [])
        logger.info(f"✅ 검색 결과: 성공 사례 {len(success_cases)}건 / 실패 사례 {len(failure_cases)}건 발견")
        
        # 3. Gemini Draft (Based on Success)
        draft_hypothesis = await self._run_draft_creator(observation, success_cases, verification)
        logger.info(f"📝 Gemini 초안(Draft): {draft_hypothesis.get('hypothesis')}")

        # 4. GPT Critic (Based on Failure + Draft)
        # GPT에게 "Gemini가 이런 계획을 짰는데, 과거 실패 사례랑 비슷하니?" 라고 물어봄
        critique_result = await self._run_critic(failure_cases, draft_hypothesis, observation)
        logger.info(f"🧐 GPT 비평(Critique): {critique_result}")
        
        # 5. Gemini Refine (Final Synthesis)
        final_hypothesis = await self._run_final_creator(
            observation, 
            draft_hypothesis, 
            critique_result, 
            verification
        )
        logger.info(f"💡 최종 가설(Final Hypothesis): {final_hypothesis.get('hypothesis')}")
        
        return final_hypothesis

    async def _run_draft_creator(
        self, 
        observation: str, 
        success_cases: List[Dict], 
        verification: Dict
    ) -> Dict[str, Any]:
        """
        Gemini (Draft): 성공 사례를 바탕으로 초안 작성
        """
        success_text = "\n".join([
            f"- Case {i+1}: Algo={c.get('algorithm', 'unknown')}, Params={c.get('experiment', {}).get('parameters')}, Shape={c.get('shape_metrics', {})}"
            for i, c in enumerate(success_cases)
        ])
        
        if not success_text:
            success_text = "직접적인 성공 사례가 없습니다. 일반적인 물리학 원리에 의존하세요."

        prompt = f"""
        You are a Structural Engineer Expert.
        
        [Current Context]
        Observation: {observation}
        Current Metrics: {json.dumps(verification.get('metrics_after', {}), indent=2)}
        
        [Tweakable Parameters]
        - target (int): Overall size. Decrease to reduce weight.
        - shrink (float: 0.1-1.0): Scales the model. Lower = smaller/lighter.
        - plates_per_voxel (int: 1-3): Vertical density. Higher = stronger but heavier.
        - support_ratio (float: 0.0-2.0): Support density. Higher = more stability.
        - fill (bool): Fill internal cavities. True = stronger.
        - interlock (bool): Overlap bricks. True = much stronger.
        - erosion_iters (int: 0-3): Removes thin parts. Higher = cleaner but might lose detail.
        - auto_remove_1x1 (bool): Removes weak 1x1 bricks. True = safer.
        - smart_fix (bool): Enable algorithmic repair logic.
        
        [Success Patterns]
        {success_text}
        
        Based ONLY on success patterns and physics, propose an initial hypothesis to fix the issue.
        Write the 'hypothesis' and 'reasoning' in Korean.
        Respond in JSON: 
        {{ 
            "hypothesis": "가설 (한국어)", 
            "reasoning": "근거 (한국어)", 
            "proposed_params": {{
                "target": 60,
                "support_ratio": 1.2,
                ...
            }} 
        }}
        """
        try:
            # Fix: Wrap synchronous call in to_thread
            return await asyncio.to_thread(self.gemini_client.generate_json, prompt)
        except Exception as e:
            logger.error(f"초안 생성 실패: {e}")
            return {"hypothesis": "General adjustment", "reasoning": "Draft failed"}

    async def _run_critic(self, failure_cases: List[Dict], draft: Dict, current_observation: str) -> str:
        """
        GPT-4o-mini (Critic): 드래프트가 과거 실패와 유사한지 검증
        """
        if not self.gpt_client.client:
            return "비평 기능 비활성화됨."
            
        if not failure_cases:
            return "유사한 실패 사례가 없습니다. 주의해서 진행하세요."

        failures_text = "\n".join([
            f"- Failure {i+1}: Algo={c.get('algorithm', 'unknown')}, Params={c.get('experiment', {}).get('parameters')}, Cause={c.get('improvement', {}).get('lesson_learned')}"
            for i, c in enumerate(failure_cases)
        ])
        
        draft_summary = f"Plan: {draft.get('hypothesis')}, Params: {draft.get('proposed_params')}"
        
        prompt = f"""
        Review this proposed plan against past failures.
        
        [Proposed Plan]
        {draft_summary}
        
        [Past Failures for similar issue "{current_observation}"]
        {failures_text}
        
        Task:
        1. Does the proposed plan resemble any past failure?
        2. Identify specific risks.
        3. Suggest 1 concrete modification to avoid failure (Write this in Korean!).
        
        Be concise. Write everything in Korean.
        """
        
        try:
            return await asyncio.to_thread(self.gpt_client.generate, prompt)
        except Exception as e:
            logger.error(f"비평 생성 실패: {e}")
            return "비평 분석 중 오류 발생."

    async def _run_final_creator(
        self, 
        observation: str, 
        draft: Dict, 
        critique: str, 
        verification: Dict
    ) -> Dict[str, Any]:
        """
        Gemini (Refine): 초안 + 비판을 반영하여 최종 가설 수립
        """
        prompt = f"""
        Finalize the structural hypothesis.
        
        [Draft Plan]
        {draft.get('hypothesis')} (Reason: {draft.get('reasoning')})
        
        [Critic Feedback (Risk Analysis)]
        {critique}
        
        [Task]
        Refine the draft plan to address the critic's warnings. 
        If the critic found no issues, expand on the draft.
        Ensure you provide a complete set of proposed parameters in JSON format.
        
        CRITICAL: All text fields ("hypothesis", "reasoning", "prediction") MUST be in Korean.
        
        Respond in JSON:
        {{
            "hypothesis": "최종 가설 (한국어)",
            "reasoning": "왜 이 방법이 안전하고 효과적인지 (한국어, 비평가 내용 인용)",
            "prediction": "예상되는 결과 (한국어)",
            "proposed_params": {{
                "target": 60,
                "shrink": 0.8,
                "plates_per_voxel": 3,
                "support_ratio": 1.2,
                "fill": true,
                "interlock": true,
                "erosion_iters": 1,
                "auto_remove_1x1": true,
                "smart_fix": true,
                "step_order": "bottomup"
            }}
        }}
        """
        try:
            # Fix: Wrap synchronous call in to_thread
            return await asyncio.to_thread(self.gemini_client.generate_json, prompt)
        except Exception as e:
            logger.error(f"최종 생성 실패: {e}")
            return draft # Fallback to draft
