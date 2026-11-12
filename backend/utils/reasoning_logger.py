"""Сохранение и чтение лога рассуждений агентов (таблица reasoning_logs + JSON-лог в файл)."""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.models import ReasoningStep
from backend.db.models import LLMCall, ReasoningLog

logger = logging.getLogger("backend.reasoning")


class ReasoningLogStore:
    def __init__(self, session_factory: Callable[[], AsyncSession]):
        self.session_factory = session_factory

    async def save(self, request_id: str, steps: list[ReasoningStep]) -> int:
        for s in steps:
            logger.info("[%s] шаг %d %s: %s", request_id, s.step, s.agent_name, s.status,
                        extra={"extra_data": s.model_dump(mode="json", exclude={"llm_calls"})})
        try:
            async with self.session_factory() as session:
                for s in steps:
                    session.add(ReasoningLog(
                        request_id=request_id, step=s.step, agent_name=s.agent_name,
                        input_summary=s.input_summary[:4000], output_summary=s.output_summary[:4000],
                        model_name=s.model_name, parameters={**s.parameters, "llm_calls": len(s.llm_calls)},
                        started_at=s.started_at, finished_at=s.finished_at, status=s.status, error=s.error,
                    ))
                await session.commit()
        except Exception as exc:
            logging.getLogger(__name__).warning("Не удалось сохранить reasoning_log: %s", exc)
            return 0
        return len(steps)

    async def get(self, request_id: str) -> list[dict[str, Any]]:
        async with self.session_factory() as session:
            rows = (await session.execute(
                select(ReasoningLog).where(ReasoningLog.request_id == request_id).order_by(ReasoningLog.step)
            )).scalars().all()
        return [
            {
                "step": r.step, "agent_name": r.agent_name, "input_summary": r.input_summary,
                "output_summary": r.output_summary, "model_name": r.model_name, "parameters": r.parameters,
                "started_at": r.started_at, "finished_at": r.finished_at, "status": r.status, "error": r.error,
                "duration_ms": round((r.finished_at - r.started_at).total_seconds() * 1000, 1)
                if r.finished_at else None,
            }
            for r in rows
        ]

    async def llm_calls(self, request_id: str) -> list[dict[str, Any]]:
        async with self.session_factory() as session:
            rows = (await session.execute(
                select(LLMCall).where(LLMCall.request_id == request_id).order_by(LLMCall.id)
            )).scalars().all()
        return [
            {
                "id": r.id, "agent_name": r.agent_name, "provider": r.provider, "model_name": r.model_name,
                "prompt_hash": r.prompt_hash, "response_hash": r.response_hash, "prompt_preview": r.prompt_preview,
                "response_preview": r.response_preview, "tokens_used": r.tokens_used, "latency_ms": r.latency_ms,
                "status": r.status, "error": r.error, "parameters": r.parameters, "created_at": r.created_at,
            }
            for r in rows
        ]
