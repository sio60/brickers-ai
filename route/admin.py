# route/admin.py
"""Admin API Hub - Central Dispatcher for Admin Modules"""
from __future__ import annotations
from datetime import datetime
from fastapi import APIRouter
import logging

from . import admin_analytics, admin_moderation, admin_logs

logger = logging.getLogger("api.admin")
router = APIRouter(tags=["admin"])

logger.info("Initializing unified admin routes...")

@router.get("/ai-admin/ping")
def ping_admin():
    """Simple health check for admin server"""
    return {"status": "ai_admin_ok", "timestamp": datetime.now().isoformat()}

# --- Include Modular Routers ---

# 1. Analytics Routes (/ai-admin/analytics/...)
router.include_router(admin_analytics.router, prefix="/ai-admin/analytics")

# 2. Moderation Routes (/ai-admin/...)
router.include_router(admin_moderation.router, prefix="/ai-admin")

# 3. Logging & System Routes (Mixed Prefixes)
# NOTE: admin_logs handles paths like /logs, /analyze, /ai-admin/archive
router.include_router(admin_logs.router) 
