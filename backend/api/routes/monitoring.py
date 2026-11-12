"""Мониторинг: состояние сервисов, статистика кэша и LLM-вызовов, статус фоновых задач."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_container, get_current_user, get_db, require_admin
from backend.db.models import GenerationRequest, LLMCall, User
from backend.services.container import AppContainer

router = APIRouter(tags=["monitoring"])


async def services_status(c: AppContainer) -> dict:
    status = {"database": await c.db.ping(), "vector_index": c.knowledge.retriever.vector.health(),
              "knowledge_loaded": c.knowledge.loaded}
    status["llm"] = await c.llm.available_providers()
    if c.settings.use_redis_cache or not c.settings.celery_task_always_eager:
        try:
            import redis.asyncio as aioredis

            r = aioredis.from_url(c.settings.redis_url, socket_connect_timeout=2)
            status["redis"] = bool(await r.ping())
            await r.aclose()
        except Exception:
            status["redis"] = False
    return status


@router.get("/health")
async def health(c: AppContainer = Depends(get_container)) -> dict:
    services = await services_status(c)
    ok = services["database"] and services["vector_index"] and any(services["llm"].values())
    return {"status": "ok" if ok else "degraded", "services": services}


@router.get("/monitoring/stats")
async def stats(_: User = Depends(get_current_user), db: AsyncSession = Depends(get_db),
                c: AppContainer = Depends(get_container)) -> dict:
    llm_rows = (await db.execute(
        select(LLMCall.provider, LLMCall.model_name, LLMCall.status, func.count(), func.avg(LLMCall.latency_ms),
               func.sum(LLMCall.tokens_used))
        .group_by(LLMCall.provider, LLMCall.model_name, LLMCall.status)
    )).all()
    gen_rows = (await db.execute(
        select(GenerationRequest.mode, func.count(), func.avg(GenerationRequest.critic_score),
               func.avg(GenerationRequest.elapsed_ms)).group_by(GenerationRequest.mode)
    )).all()
    return {
        "cache": await c.cache.stats(),
        "llm_calls": [{"provider": p, "model": m, "status": s, "count": n, "avg_latency_ms": round(lat or 0, 1),
                       "tokens": int(t or 0)} for p, m, s, n, lat, t in llm_rows],
        "generations": [{"mode": m, "count": n, "avg_critic_score": round(sc, 2) if sc is not None else None,
                         "avg_elapsed_ms": round(el or 0, 1)} for m, n, sc, el in gen_rows],
        "knowledge": c.knowledge.stats(),
    }


@router.get("/llm/providers")
async def providers(c: AppContainer = Depends(get_container)) -> dict:
    s = c.settings
    return {"available": await c.llm.available_providers(), "default": s.llm_default_provider,
            "ollama_models": s.ollama_models, "ensemble_models": s.ensemble_models}


@router.post("/monitoring/cache/clear")
async def clear_cache(_: User = Depends(require_admin), c: AppContainer = Depends(get_container)) -> dict:
    return {"deleted": await c.cache.clear()}


@router.get("/tasks/{task_id}")
async def task_status(task_id: str, _: User = Depends(get_current_user)) -> dict:
    from celery.result import AsyncResult

    from backend.worker.celery_app import celery_app

    res = AsyncResult(task_id, app=celery_app)
    try:
        state = res.state
        info = res.info if isinstance(res.info, dict) else (str(res.info) if res.info else None)
    except Exception as exc:  # backend недоступен (локальный eager-режим)
        return {"task_id": task_id, "state": "UNKNOWN", "info": str(exc)}
    return {"task_id": task_id, "state": state, "info": info if state != "SUCCESS" else None,
            "result": res.result if state == "SUCCESS" else None}
