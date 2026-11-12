"""Асинхронные утилиты: retry-декоратор с экспоненциальной задержкой."""
from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Awaitable, Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")
logger = logging.getLogger(__name__)


def async_retry(retries: int = 2, delay: float = 0.5, backoff: float = 2.0,
                exceptions: tuple[type[BaseException], ...] = (Exception,),
                no_retry: tuple[type[BaseException], ...] = ()):
    """Повторяет корутину при ошибке. Исключения из no_retry пробрасываются сразу."""

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            wait = delay
            for attempt in range(retries + 1):
                try:
                    return await func(*args, **kwargs)
                except no_retry:
                    raise
                except exceptions as exc:
                    if attempt >= retries:
                        raise
                    logger.warning("%s: попытка %d не удалась (%s), повтор через %.1fс",
                                   func.__qualname__, attempt + 1, exc, wait)
                    await asyncio.sleep(wait)
                    wait *= backoff
            raise RuntimeError("unreachable")  # pragma: no cover

        return wrapper

    return decorator


async def gather_limited(coros: list[Awaitable[T]], limit: int = 4) -> list[T]:
    """asyncio.gather с ограничением числа одновременных задач."""
    sem = asyncio.Semaphore(limit)

    async def run(c: Awaitable[T]) -> T:
        async with sem:
            return await c

    return await asyncio.gather(*(run(c) for c in coros))
