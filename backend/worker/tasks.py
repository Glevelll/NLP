"""Фоновые задачи Celery.

Каждая задача создаёт собственный контейнер с NullPool-подключением к БД и запускает асинхронный код
через asyncio.run — так нет конфликтов event loop между задачами.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from backend.core.config import get_settings
from backend.core.models import ProductInput
from backend.data.examples_store import import_glossary, import_knowledge, read_records
from backend.evaluation.evaluator import Evaluator, load_dataset
from backend.services.container import AppContainer
from backend.worker.celery_app import celery_app

logger = logging.getLogger(__name__)


def _run(coro_factory: Callable[[AppContainer], Awaitable[Any]]) -> Any:
    async def runner():
        container = AppContainer(get_settings(), null_pool=True)
        try:
            await container.startup()
            return await coro_factory(container)
        finally:
            await container.shutdown()

    return asyncio.run(runner())


def _progress(task, current: int, total: int, **meta) -> None:
    if task.request.id and not task.request.called_directly:
        try:
            task.update_state(state="PROGRESS", meta={"current": current, "total": total, **meta})
        except Exception:  # в eager-режиме без backend
            pass


def _dataset(path: str | None) -> list[dict]:
    s = get_settings()
    return load_dataset(path or Path(s.data_dir) / "test.json")


@celery_app.task(name="tasks.reindex_knowledge")
def reindex_knowledge_task() -> dict:
    return _run(lambda c: c.reload_knowledge())


@celery_app.task(name="tasks.cleanup_cache")
def cleanup_cache_task(older_than_hours: int | None = None) -> dict:
    async def job(c: AppContainer):
        return {"deleted": await c.cache.cleanup(older_than_hours)}

    return _run(job)


@celery_app.task(name="tasks.bulk_import")
def bulk_import_task(kind: str, records: list[dict] | None = None, path: str | None = None) -> dict:
    async def job(c: AppContainer):
        recs = records if records is not None else read_records(path)
        async with c.db.session_factory() as s:
            added = await (import_knowledge(s, recs) if kind == "knowledge" else import_glossary(s, recs))
        await c.reload_knowledge()
        return {"kind": kind, "added": added, "total": len(recs)}

    return _run(job)


@celery_app.task(bind=True, name="tasks.batch_generate")
def batch_generate_task(self, products: list[dict], mode: str = "+agents", provider: str | None = None,
                        model: str | None = None, user_id: int | None = None) -> dict:
    async def job(c: AppContainer):
        results = []
        for i, p in enumerate(products, 1):
            res = await c.generation.generate(ProductInput(**p), mode, provider, model, user_id=user_id)
            results.append(res.model_dump(mode="json"))
            _progress(self, i, len(products))
        return {"results": results, "count": len(results)}

    return _run(job)


@celery_app.task(bind=True, name="tasks.evaluate_model")
def evaluate_model_task(self, model: str, mode: str, sample_size: int, seed: int = 42,
                        data_path: str | None = None) -> dict:
    async def job(c: AppContainer):
        _progress(self, 0, 1, model=model, mode=mode)
        ev = Evaluator(c.generation)
        ev.fit_style_classifier(_kb_records(c))
        result = await ev.evaluate(_dataset(data_path), model, mode, sample_size, seed)
        async with c.db.session_factory() as s:
            row = await ev.save_result(s, result)
        _progress(self, 1, 1)
        return {"id": row.id, **{k: v for k, v in result.items() if k != "per_sample"}}

    return _run(job)


@celery_app.task(bind=True, name="tasks.compare_models")
def compare_models_task(self, models: list[str], mode: str, sample_size: int, seed: int = 42,
                        data_path: str | None = None) -> dict:
    async def job(c: AppContainer):
        ev = Evaluator(c.generation)
        ev.fit_style_classifier(_kb_records(c))
        _progress(self, 0, len(models))
        comparison = await ev.compare(_dataset(data_path), models, mode, sample_size, seed)
        async with c.db.session_factory() as s:
            for r in comparison["results"].values():
                if "metrics" in r:
                    await ev.save_result(s, r)
            row = await ev.save_comparison(s, comparison)
        return {"id": row.id, "winner": comparison["winner"], "significance": comparison["significance"],
                "results": {k: {kk: vv for kk, vv in v.items() if kk != "per_sample"}
                            for k, v in comparison["results"].items()}}

    return _run(job)


@celery_app.task(bind=True, name="tasks.batch_evaluate")
def batch_evaluate_task(self, models: list[str], modes: list[str], sample_sizes: list[int], seed: int = 42,
                        data_path: str | None = None) -> dict:
    async def job(c: AppContainer):
        ev = Evaluator(c.generation)
        ev.fit_style_classifier(_kb_records(c))
        data = _dataset(data_path)
        total = len(models) * len(modes) * len(sample_sizes)
        done, out = 0, []
        for size in sample_sizes:
            for model in models:
                for mode in modes:
                    try:
                        result = await ev.evaluate(data, model, mode, size, seed)
                        async with c.db.session_factory() as s:
                            row = await ev.save_result(s, result)
                        out.append({"id": row.id, **{k: v for k, v in result.items() if k != "per_sample"}})
                    except Exception as exc:
                        out.append({"model_name": model, "mode": mode, "sample_size": size, "error": str(exc)})
                    done += 1
                    _progress(self, done, total)
        return {"results": out}

    return _run(job)


def _kb_records(c: AppContainer) -> list[dict]:
    path = Path(c.settings.data_dir) / "knowledge_base.json"
    return read_records(path) if path.exists() else []
