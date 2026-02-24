
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
    from ..core.llm_clients import OpenAIClient
    from ..core.memory_utils import MemoryUtils, build_hypothesis
    import config
except ImportError:
    # For standalone testing
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    from core.llm_clients import OpenAIClient
    from core.memory_utils import MemoryUtils, build_hypothesis
    import config

from . import prompts

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
        [Simplified Single-Shot Flow]
        1. Search: Success cases
        2. Draft (Gemini): Create plan based on Success cases → 바로 반환
        (Critic/Refine 제거 — 속도 우선)
        """
        observation = state.get("observation", "")
        verification = state.get("verification_result", {})
        
        # 1. Search
        shape_metrics = {} 
        logger.info(f"🔎 사례 검색 시작: {observation[:50]}...")
        rag_results = await asyncio.to_thread(
            self.memory.search_success_and_failure,
            observation=observation,
            limit=3,
            min_score=0.4,
            verification_metrics=verification,
            shape_metrics=shape_metrics
        )
        
        success_cases = rag_results.get("success", [])
        logger.info(f"✅ 검색 결과: 성공 사례 {len(success_cases)}건 발견")
        
        # 2. Gemini Draft (1회) → 바로 최종 가설로 사용
        draft_hypothesis = await self._run_draft_creator(observation, success_cases, verification)
        logger.info(f"📝 최종 가설: {draft_hypothesis.get('hypothesis')}")
        
        return draft_hypothesis

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

        prompt = prompts.get_draft_creator_prompt(observation, success_text, verification)
        
        try:
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
        
        prompt = prompts.get_critic_prompt(failures_text, draft_summary, current_observation)
        
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
        prompt = prompts.get_final_refine_prompt(draft, critique)
        
        try:
            return await asyncio.to_thread(self.gemini_client.generate_json, prompt)
        except Exception as e:
            logger.error(f"최종 생성 실패: {e}")
            return draft # Fallback to draft
