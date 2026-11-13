"""Кэш результатов генерации: таблица cache в БД (основной) + Redis (опционально, если включён)."""
from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from datetime import timedelta
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import CacheEntry, utcnow

logger = logging.getLogger(__name__)


def make_cache_key(*parts: Any) -> str:
    raw = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class CacheService:
    def __init__(self, session_factory: Callable[[], AsyncSession], ttl_hours: int = 72,
                 redis_url: str | None = None, enabled: bool = True):
        self.session_factory = session_factory
        self.ttl = timedelta(hours=ttl_hours)
        self.enabled = enabled
        self._redis = None
        if redis_url:
            try:
                import redis.asyncio as aioredis

                self._redis = aioredis.from_url(redis_url, decode_responses=True)
            except Exception as exc:  # pragma: no cover
                logger.warning("Redis недоступен: %s", exc)

    async def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        if self._redis is not None:
            try:
                raw = await self._redis.get(f"cache:{key}")
                if raw:
                    await self._redis.incr(f"cache_hits:{key}")
                    return json.loads(raw)
            except Exception as exc:
                logger.debug("Redis get: %s", exc)
        async with self.session_factory() as s:
            entry = (await s.execute(select(CacheEntry).where(CacheEntry.input_key == key))).scalar_one_or_none()
            if entry is None:
                return None
            if utcnow() - entry.created_at > self.ttl:
                await s.delete(entry)
                await s.commit()
                return None
            entry.hits += 1
            entry.last_hit_at = utcnow()
            await s.commit()
            return json.loads(entry.output)

    async def set(self, key: str, value: dict[str, Any], namespace: str = "generation") -> None:
        if not self.enabled:
            return
        payload = json.dumps(value, ensure_ascii=False, default=str)
        async with self.session_factory() as s:
            query = select(CacheEntry).where(CacheEntry.input_key == key)
            entry = (await s.execute(query)).scalar_one_or_none()
            if entry is None:
                s.add(CacheEntry(input_key=key, namespace=namespace, output=payload))
                try:
                    await s.commit()
                except IntegrityError:
                    await s.rollback()
                    if (await s.execute(query)).scalar_one_or_none() is None:
                        raise
                    return
            else:
                entry.output = payload
                entry.created_at = utcnow()
                await s.commit()
        if self._redis is not None:
            try:
                await self._redis.set(f"cache:{key}", payload, ex=int(self.ttl.total_seconds()))
            except Exception as exc:
                logger.debug("Redis set: %s", exc)

    async def cleanup(self, older_than_hours: int | None = None) -> int:
        border = utcnow() - (timedelta(hours=older_than_hours) if older_than_hours is not None else self.ttl)
        async with self.session_factory() as s:
            res = await s.execute(delete(CacheEntry).where(CacheEntry.created_at < border))
            await s.commit()
            return res.rowcount or 0

    async def clear(self) -> int:
        async with self.session_factory() as s:
            res = await s.execute(delete(CacheEntry))
            await s.commit()
        if self._redis is not None:
            try:
                keys = [k async for k in self._redis.scan_iter("cache*")]
                if keys:
                    await self._redis.delete(*keys)
            except Exception:  # pragma: no cover
                pass
        return res.rowcount or 0

    async def stats(self) -> dict[str, Any]:
        async with self.session_factory() as s:
            total = (await s.execute(select(func.count()).select_from(CacheEntry))).scalar_one()
            hits = (await s.execute(select(func.coalesce(func.sum(CacheEntry.hits), 0)))).scalar_one()
            top = (await s.execute(select(CacheEntry).order_by(CacheEntry.hits.desc()).limit(5))).scalars().all()
        return {
            "entries": total,
            "total_hits": int(hits),
            "backend": "db+redis" if self._redis is not None else "db",
            "top": [{"key": e.input_key[:12], "hits": e.hits, "namespace": e.namespace} for e in top],
        }
