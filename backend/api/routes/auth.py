"""Аутентификация: регистрация, вход, профиль, управление ролями."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_current_user, get_db, require_admin
from backend.core.constants import Role
from backend.core.models import RoleUpdate, Token, UserCreate, UserOut
from backend.db.models import User
from backend.utils.security import create_access_token, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(data: UserCreate, db: AsyncSession = Depends(get_db)) -> User:
    exists = (await db.execute(select(User).where(or_(User.email == data.email, User.username == data.username))
                               )).scalar_one_or_none()
    if exists:
        raise HTTPException(status.HTTP_409_CONFLICT, "Пользователь с таким email или именем уже существует")
    # Первый зарегистрированный пользователь становится администратором
    users_count = (await db.execute(select(func.count()).select_from(User))).scalar_one()
    user = User(email=data.email, username=data.username, hashed_password=hash_password(data.password),
                role=Role.ADMIN.value if users_count == 0 else Role.USER.value)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@router.post("/login", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)) -> Token:
    user = (await db.execute(select(User).where(or_(User.username == form.username, User.email == form.username))
                             )).scalar_one_or_none()
    if user is None or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Неверное имя пользователя или пароль")
    if not user.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Пользователь заблокирован")
    return Token(access_token=create_access_token(str(user.id), user.role), role=user.role)


@router.get("/me", response_model=UserOut)
async def me(user: User = Depends(get_current_user)) -> User:
    return user


@router.get("/users", response_model=list[UserOut])
async def list_users(_: User = Depends(require_admin), db: AsyncSession = Depends(get_db)) -> list[User]:
    return list((await db.execute(select(User).order_by(User.id))).scalars())


@router.patch("/users/{user_id}/role", response_model=UserOut)
async def set_role(user_id: int, data: RoleUpdate, _: User = Depends(require_admin),
                   db: AsyncSession = Depends(get_db)) -> User:
    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Пользователь не найден")
    user.role = data.role.value
    await db.commit()
    await db.refresh(user)
    return user
