# screenshot-server/route/render.py
"""Evolver Vision용 멀티앵글 렌더링 API"""
from __future__ import annotations

import base64
import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from service.render_client import RENDER_ENABLED, render_custom_angles

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/render", tags=["render"])

# Evolver가 보내는 앵글명 → render_client의 (lat, lon) 매핑
# Evolver: FRONT(Z-), BACK(Z+), LEFT(X-), RIGHT(X+), BOTTOM
ANGLE_MAP: Dict[str, Dict[str, float]] = {
    "FRONT":  {"lat": 20,  "lon": 0},
    "BACK":   {"lat": 20,  "lon": 180},
    "LEFT":   {"lat": 20,  "lon": -90},
    "RIGHT":  {"lat": 20,  "lon": 90},
    "BOTTOM": {"lat": -90, "lon": 0},
}


class MultiAngleRequest(BaseModel):
    ldr_text: str
    angles: List[str]
    width: int = 512
    height: int = 512


class MultiAngleResponse(BaseModel):
    images: Dict[str, Optional[str]]


@router.post("/multi-angle", response_model=MultiAngleResponse)
async def render_multi_angle(req: MultiAngleRequest):
    if not RENDER_ENABLED:
        raise HTTPException(503, "LDView not available")

    # 요청된 앵글만 필터링
    angles_to_render = {}
    for name in req.angles:
        upper = name.upper()
        if upper in ANGLE_MAP:
            angles_to_render[upper] = ANGLE_MAP[upper]
        else:
            logger.warning("Unknown angle '%s', skipping", name)

    if not angles_to_render:
        raise HTTPException(400, "No valid angles provided")

    # LDView 렌더링 (async → thread)
    raw: Dict[str, bytes] = await render_custom_angles(
        req.ldr_text, angles_to_render, req.width, req.height
    )

    # bytes → base64 문자열
    images: Dict[str, Optional[str]] = {}
    for name, data in raw.items():
        if data:
            images[name] = base64.b64encode(data).decode("utf-8")
        else:
            images[name] = None

    return MultiAngleResponse(images=images)


@router.get("/health")
def render_health():
    return {"ldview": RENDER_ENABLED, "angles": list(ANGLE_MAP.keys())}
