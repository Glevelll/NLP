"""Хэширование паролей (bcrypt) и JWT-токены."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import bcrypt
import jwt

from backend.core.config import get_settings
from backend.core.exceptions import AuthError


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8")[:72], hashed.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(subject: str, role: str, expires_minutes: int | None = None) -> str:
    s = get_settings()
    now = datetime.now(UTC)
    payload = {
        "sub": subject,
        "role": role,
        "iat": now,
        "exp": now + timedelta(minutes=expires_minutes or s.jwt_expire_minutes),
    }
    return jwt.encode(payload, s.jwt_secret, algorithm=s.jwt_algorithm)


def decode_token(token: str) -> dict:
    s = get_settings()
    try:
        return jwt.decode(token, s.jwt_secret, algorithms=[s.jwt_algorithm])
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("Срок действия токена истёк") from exc
    except jwt.PyJWTError as exc:
        raise AuthError("Неверный токен") from exc
