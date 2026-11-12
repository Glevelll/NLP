"""Active Learning: очередь неоднозначных результатов (низкая оценка Critic / не прошёл валидацию)."""
from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.constants import ReviewStatus
from backend.core.exceptions import NotFoundError, ValidationAppError
from backend.db.models import UncertainExample, utcnow


class ActiveLearningQueue:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def add(self, *, request_id: str | None, input_text: str, input_json: dict[str, Any] | None,
                  model_output: str, critic_score: float | None, reason: str) -> UncertainExample:
        item = UncertainExample(request_id=request_id, input_text=input_text, input_json=input_json,
                                model_output=model_output, critic_score=critic_score, reason=reason)
        self.session.add(item)
        await self.session.commit()
        await self.session.refresh(item)
        return item

    async def list(self, status: str | None = ReviewStatus.PENDING, limit: int = 50, offset: int = 0
                   ) -> list[UncertainExample]:
        q = select(UncertainExample).order_by(UncertainExample.critic_score.asc().nulls_first(),
                                              UncertainExample.created_at.desc())
        if status:
            q = q.where(UncertainExample.status == status)
        return list((await self.session.execute(q.limit(limit).offset(offset))).scalars())

    async def get(self, item_id: int) -> UncertainExample:
        item = await self.session.get(UncertainExample, item_id)
        if item is None:
            raise NotFoundError(f"Пример {item_id} не найден")
        return item

    async def _decide(self, item_id: int, status: str, reviewer_id: int | None,
                      corrected: str | None = None) -> UncertainExample:
        item = await self.get(item_id)
        if item.status != ReviewStatus.PENDING:
            raise ValidationAppError(f"Пример {item_id} уже обработан ({item.status})")
        item.status = status
        item.reviewed_by = reviewer_id
        item.reviewed_at = utcnow()
        if corrected:
            item.corrected_output = corrected
        await self.session.commit()
        return item

    async def approve(self, item_id: int, reviewer_id: int | None, corrected: str | None = None) -> UncertainExample:
        return await self._decide(item_id, ReviewStatus.APPROVED, reviewer_id, corrected)

    async def reject(self, item_id: int, reviewer_id: int | None) -> UncertainExample:
        return await self._decide(item_id, ReviewStatus.REJECTED, reviewer_id)

    async def stats(self) -> dict[str, int]:
        rows = (await self.session.execute(
            select(UncertainExample.status, func.count()).group_by(UncertainExample.status)
        )).all()
        return {status: count for status, count in rows}
