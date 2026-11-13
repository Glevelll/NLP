"""Human-in-the-Loop: очередь неоднозначных примеров и обратная связь."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_container, get_current_user, get_db, require_editor
from backend.api.routes.requests import get_accessible_request
from backend.core.exceptions import AppError
from backend.core.models import FeedbackIn, ProductInput, ReviewDecision
from backend.db.models import UncertainExample, User
from backend.hitl.active_learning import ActiveLearningQueue
from backend.hitl.feedback import FeedbackService
from backend.hitl.updater import KnowledgeUpdater
from backend.services.container import AppContainer

router = APIRouter(prefix="/hitl", tags=["hitl"])


def _out(item: UncertainExample) -> dict:
    return {"id": item.id, "request_id": item.request_id, "input_text": item.input_text, "input_json": item.input_json,
            "model_output": item.model_output, "critic_score": item.critic_score, "reason": item.reason,
            "status": item.status, "corrected_output": item.corrected_output, "reviewed_by": item.reviewed_by,
            "reviewed_at": item.reviewed_at, "created_at": item.created_at}


@router.get("/uncertain")
async def list_uncertain(status: str | None = "pending", limit: int = 50, offset: int = 0,
                         _: User = Depends(require_editor), db: AsyncSession = Depends(get_db)) -> list[dict]:
    return [_out(i) for i in await ActiveLearningQueue(db).list(status or None, limit, offset)]


@router.post("/uncertain/{item_id}/approve")
async def approve(item_id: int, decision: ReviewDecision, user: User = Depends(require_editor),
                  db: AsyncSession = Depends(get_db), c: AppContainer = Depends(get_container)) -> dict:
    try:
        item = await ActiveLearningQueue(db).approve(item_id, user.id, decision.corrected_output)
    except AppError as exc:
        raise HTTPException(exc.status_code, exc.message) from exc
    kb_row = await KnowledgeUpdater(db, c.knowledge).add_approved(item)
    return {**_out(item), "knowledge_id": kb_row.id}


@router.post("/uncertain/{item_id}/reject")
async def reject(item_id: int, user: User = Depends(require_editor), db: AsyncSession = Depends(get_db)) -> dict:
    try:
        return _out(await ActiveLearningQueue(db).reject(item_id, user.id))
    except AppError as exc:
        raise HTTPException(exc.status_code, exc.message) from exc


@router.get("/stats")
async def stats(_: User = Depends(require_editor), db: AsyncSession = Depends(get_db)) -> dict:
    return {"queue": await ActiveLearningQueue(db).stats(),
            "average_rating": await FeedbackService(db).average_rating()}


@router.post("/feedback", status_code=201)
async def feedback(data: FeedbackIn, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
                   ) -> dict:
    req = await get_accessible_request(data.request_id, user, db)
    fb = await FeedbackService(db).add(data.request_id, data.rating, user.id, data.comment, data.corrected_text)
    if data.rating <= 2 or data.corrected_text:
        # плохая оценка или правка пользователя — повод показать пример редактору
        await ActiveLearningQueue(db).add(
            request_id=data.request_id, input_text=ProductInput(**req.input_json).to_text(),
            input_json=req.input_json, model_output=req.output, critic_score=req.critic_score,
            reason=f"Отзыв пользователя: {data.rating}/5. {data.comment or ''}".strip(),
        )
    return {"id": fb.id, "request_id": fb.request_id, "rating": fb.rating}
