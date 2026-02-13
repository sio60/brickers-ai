# screenshot-server/celery_app.py
"""Celery app configuration - SQS broker for screenshot/background tasks"""
from __future__ import annotations

import os

from celery import Celery

app = Celery("screenshot_worker")

app.conf.update(
    broker_url="sqs://",
    broker_transport_options={
        "region": os.environ.get("AWS_REGION", "ap-northeast-2"),
        "predefined_queues": {
            "celery-screenshots": {
                "url": os.environ.get("CELERY_SQS_SCREENSHOT_QUEUE_URL", ""),
            }
        },
        "queue_name_prefix": "",
        "visibility_timeout": 600,
        "polling_interval": 1,
        "wait_time_seconds": 10,
    },
    task_serializer="json",
    accept_content=["json"],
    result_backend=None,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    worker_concurrency=1,
    task_default_queue="celery-screenshots",
    timezone="Asia/Seoul",
)
