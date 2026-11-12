"""Приложение Celery и расписание beat."""
from __future__ import annotations

from celery import Celery
from celery.schedules import crontab

from backend.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "product_descriptions",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["backend.worker.tasks"],
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Europe/Moscow",
    enable_utc=True,
    task_track_started=True,
    task_always_eager=settings.celery_task_always_eager,
    task_eager_propagates=False,
    result_expires=60 * 60 * 24,
    worker_prefetch_multiplier=1,
    beat_schedule={
        "cleanup-cache-nightly": {"task": "tasks.cleanup_cache", "schedule": crontab(hour=3, minute=0)},
        "reindex-knowledge-nightly": {"task": "tasks.reindex_knowledge", "schedule": crontab(hour=4, minute=0)},
    },
)
