"""История запросов, лог рассуждений и трассировка LLM-вызовов."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import ROLE_LEVEL, get_container, get_current_user, get_db
from backend.core.constants import Role
from backend.db.models import GenerationRequest, User
from backend.services.container import AppContainer

router = APIRouter(prefix="/requests", tags=["requests"])


def _can_see_all(user: User) -> bool:
    return ROLE_LEVEL.get(Role(user.role), 0) >= ROLE_LEVEL[Role.EDITOR]


@router.get("")
async def list_requests(limit: int = 50, offset: int = 0, user: User = Depends(get_current_user),
                        db: AsyncSession = Depends(get_db)) -> list[dict]:
    q = select(GenerationRequest).order_by(GenerationRequest.created_at.desc()).limit(limit).offset(offset)
    if not _can_see_all(user):
        q = q.where(GenerationRequest.user_id == user.id)
    rows = (await db.execute(q)).scalars()
    return [{"request_id": r.request_id, "product_name": r.product_name, "mode": r.mode, "provider": r.provider,
             "model_name": r.model_name, "critic_score": r.critic_score, "validation_passed": r.validation_passed,
             "iterations": r.iterations, "elapsed_ms": r.elapsed_ms, "created_at": r.created_at} for r in rows]


async def _get_request(request_id: str, user: User, db: AsyncSession) -> GenerationRequest:
    row = (await db.execute(select(GenerationRequest).where(GenerationRequest.request_id == request_id))
           ).scalar_one_or_none()
    if row is None:
        raise HTTPException(404, "Запрос не найден")
    if not _can_see_all(user) and row.user_id != user.id:
        raise HTTPException(403, "Нет доступа к запросу")
    return row


@router.get("/{request_id}")
async def get_request(request_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
                      ) -> dict:
    r = await _get_request(request_id, user, db)
    return {"request_id": r.request_id, "product": r.input_json, "output": r.output, "mode": r.mode,
            "provider": r.provider, "model_name": r.model_name, "critic_score": r.critic_score,
            "validation_passed": r.validation_passed, "iterations": r.iterations, "elapsed_ms": r.elapsed_ms,
            "created_at": r.created_at}


@router.get("/{request_id}/reasoning_log")
async def reasoning_log(request_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db),
                        c: AppContainer = Depends(get_container)) -> dict:
    await _get_request(request_id, user, db)
    steps = await c.log_store.get(request_id)
    return {"request_id": request_id, "steps": steps}


@router.get("/{request_id}/llm_calls")
async def llm_calls(request_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db),
                    c: AppContainer = Depends(get_container)) -> dict:
    await _get_request(request_id, user, db)
    return {"request_id": request_id, "calls": await c.log_store.llm_calls(request_id)}
