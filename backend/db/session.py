"""Асинхронное подключение к БД (PostgreSQL в Docker, SQLite локально и в тестах)."""
from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from backend.db.models import Base


class Database:
    """Обёртка над движком и фабрикой сессий. Для Celery-задач создаётся отдельный экземпляр с NullPool."""

    def __init__(self, url: str, echo: bool = False, null_pool: bool = False):
        self.url = url
        if url.startswith("sqlite"):
            db_path = url.split("///", 1)[-1]
            if db_path and db_path != ":memory:":
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        kwargs: dict = {"echo": echo}
        if null_pool or url.startswith("sqlite"):
            kwargs["poolclass"] = NullPool
        else:
            kwargs.update(pool_size=10, max_overflow=20, pool_pre_ping=True)
        self.engine: AsyncEngine = create_async_engine(url, **kwargs)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)

    async def create_all(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def drop_all(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    async def ping(self) -> bool:
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.session_factory() as s:
            yield s

    async def dispose(self) -> None:
        await self.engine.dispose()
