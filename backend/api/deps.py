"""Зависимости FastAPI: контейнер, сессия БД, текущий пользователь и проверка ролей."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.constants import Role
from backend.core.exceptions import AuthError
from backend.db.models import User
from backend.services.container import AppContainer
from backend.utils.security import decode_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)
ROLE_LEVEL = {Role.USER: 0, Role.EDITOR: 1, Role.ADMIN: 2}


def get_container(request: Request) -> AppContainer:
    return request.app.state.container


async def get_db(container: AppContainer = Depends(get_container)) -> AsyncIterator[AsyncSession]:
    async with container.db.session_factory() as session:
        yield session


async def get_current_user(token: str | None = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)) -> User:
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Требуется авторизация",
                            headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = decode_token(token)
    except AuthError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, exc.message,
                            headers={"WWW-Authenticate": "Bearer"}) from exc
    user = await db.get(User, int(payload["sub"]))
    if user is None or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Пользователь не найден или заблокирован")
    return user


def require_role(role: Role):
    async def checker(user: User = Depends(get_current_user)) -> User:
        if ROLE_LEVEL.get(Role(user.role), 0) < ROLE_LEVEL[role]:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"Требуется роль {role.value}")
        return user

    return checker


require_editor = require_role(Role.EDITOR)
require_admin = require_role(Role.ADMIN)


async def submit_task(task, container: AppContainer, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Отправляет задачу в Celery. В eager-режиме (без брокера) выполняет её в отдельном потоке,
    чтобы asyncio.run внутри задачи не конфликтовал с event loop FastAPI."""
    if container.settings.celery_task_always_eager:
        result = await asyncio.to_thread(task.apply, args=args, kwargs=kwargs)
        return {"task_id": result.id, "status": result.status,
                "result": result.result if result.successful() else None,
                "error": str(result.result) if result.failed() else None}
    async_result = task.apply_async(args=args, kwargs=kwargs)
    return {"task_id": async_result.id, "status": "PENDING", "result": None, "error": None}
