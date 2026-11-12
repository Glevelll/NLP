"""Логирование всех вызовов LLM: файл llm_calls.log + таблица llm_calls + трасса текущего агента."""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import LLMCall
from backend.utils.llm_processing import text_hash
from backend.utils.tracing import agent_name_var, record_llm_call, request_id_var

logger = logging.getLogger("backend.llm")


class LLMCallLogger:
    def __init__(self, session_factory: Callable[[], AsyncSession] | None = None, preview_chars: int = 300):
        self.session_factory = session_factory
        self.preview_chars = preview_chars

    async def log(self, *, provider: str, model: str, prompt: str, response: str | None, latency_ms: float,
                  status: str, tokens: int | None = None, parameters: dict[str, Any] | None = None,
                  error: str | None = None, cached: bool = False) -> dict[str, Any]:
        info = {
            "request_id": request_id_var.get(),
            "agent_name": agent_name_var.get(),
            "provider": provider,
            "model_name": model,
            "prompt_hash": text_hash(prompt),
            "response_hash": text_hash(response) if response is not None else None,
            "tokens_used": tokens,
            "latency_ms": round(latency_ms, 2),
            "status": status,
            "error": error,
            "parameters": {**(parameters or {}), "cached": cached},
        }
        record_llm_call(info)
        level = logging.INFO if status == "success" else logging.WARNING
        logger.log(level, "LLM %s/%s %s за %.0f мс", provider, model, status, latency_ms, extra={"extra_data": info})

        if self.session_factory is not None:
            try:
                async with self.session_factory() as session:
                    session.add(LLMCall(
                        **{k: v for k, v in info.items()},
                        prompt_preview=prompt[: self.preview_chars],
                        response_preview=(response or "")[: self.preview_chars],
                    ))
                    await session.commit()
            except Exception as exc:  # логирование не должно ломать генерацию
                logging.getLogger(__name__).warning("Не удалось записать llm_call в БД: %s", exc)
        return info
