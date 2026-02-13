"""
Celery Screenshot Producer - AI Server에서 Screenshot Server로 Celery 태스크 디스패치
"""
from __future__ import annotations

import os
from datetime import datetime


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{ts}] [CeleryProducer] {msg}")


def _is_truthy(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


CELERY_ENABLED = _is_truthy(os.environ.get("CELERY_SCREENSHOT_ENABLED", "false"))

_celery_app = None


def _get_celery_app():
    """Lazy singleton Celery app (producer-only, no worker)"""
    global _celery_app
    if _celery_app is not None:
        return _celery_app

    from celery import Celery

    queue_url = os.environ.get("CELERY_SQS_SCREENSHOT_QUEUE_URL", "").strip()
    region = os.environ.get("AWS_REGION", "ap-northeast-2").strip()

    if not queue_url:
        raise RuntimeError("CELERY_SQS_SCREENSHOT_QUEUE_URL is not set")

    app = Celery("screenshot_producer")
    app.conf.update(
        broker_url="sqs://",
        broker_transport_options={
            "region": region,
            "predefined_queues": {
                "celery-screenshots": {
                    "url": queue_url,
                }
            },
            "queue_name_prefix": "",
        },
        task_serializer="json",
        accept_content=["json"],
        result_backend=None,
        task_default_queue="celery-screenshots",
    )

    _celery_app = app
    _log(f"Celery producer initialized | queue={queue_url[:60]}...")
    return _celery_app


def send_screenshot_task(
    job_id: str,
    ldr_url: str,
    model_name: str,
    source: str = "job",
    gallery_post_id: str = "",
) -> None:
    """Send screenshot task to Celery worker"""
    if not CELERY_ENABLED:
        return

    app = _get_celery_app()
    app.send_task(
        "tasks.process_screenshot",
        kwargs={
            "job_id": job_id,
            "ldr_url": ldr_url,
            "model_name": model_name,
            "source": source,
            "gallery_post_id": gallery_post_id,
        },
        queue="celery-screenshots",
    )
    _log(f"screenshot task sent | jobId={job_id} | source={source}")


def send_background_task(
    job_id: str,
    subject: str,
) -> None:
    """Send background generation task to Celery worker"""
    if not CELERY_ENABLED:
        return

    app = _get_celery_app()
    app.send_task(
        "tasks.process_background",
        kwargs={
            "job_id": job_id,
            "subject": subject,
        },
        queue="celery-screenshots",
    )
    _log(f"background task sent | jobId={job_id} | subject={subject}")
