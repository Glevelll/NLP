"""Обратная связь пользователей: оценки и исправления сгенерированных описаний."""
from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import Feedback


class FeedbackService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def add(self, request_id: str, rating: int, user_id: int | None = None, comment: str | None = None,
                  corrected_text: str | None = None) -> Feedback:
        fb = Feedback(request_id=request_id, rating=rating, user_id=user_id, comment=comment,
                      corrected_text=corrected_text)
        self.session.add(fb)
        await self.session.commit()
        await self.session.refresh(fb)
        return fb

    async def list(self, request_id: str | None = None, limit: int = 100) -> list[Feedback]:
        q = select(Feedback).order_by(Feedback.created_at.desc()).limit(limit)
        if request_id:
            q = q.where(Feedback.request_id == request_id)
        return list((await self.session.execute(q)).scalars())

    async def average_rating(self) -> float | None:
        value = (await self.session.execute(select(func.avg(Feedback.rating)))).scalar_one()
        return round(float(value), 2) if value is not None else None
