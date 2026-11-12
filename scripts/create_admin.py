"""Создание (или повышение до admin) пользователя.

Пример: python scripts/create_admin.py --email admin@example.com --username admin --password secret123
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy import select  # noqa: E402

from backend.core.config import get_settings  # noqa: E402
from backend.db.models import User  # noqa: E402
from backend.db.session import Database  # noqa: E402
from backend.utils.security import hash_password  # noqa: E402


async def create_admin(email: str, username: str, password: str, database_url: str | None = None) -> str:
    db = Database(database_url or get_settings().database_url)
    await db.create_all()
    async with db.session_factory() as s:
        user = (await s.execute(select(User).where(User.username == username))).scalar_one_or_none()
        if user:
            user.role = "admin"
            msg = f"Пользователь {username} повышен до admin"
        else:
            s.add(User(email=email, username=username, hashed_password=hash_password(password), role="admin"))
            msg = f"Создан администратор {username}"
        await s.commit()
    await db.dispose()
    return msg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--email", required=True)
    ap.add_argument("--username", required=True)
    ap.add_argument("--password", required=True)
    args = ap.parse_args()
    print(asyncio.run(create_admin(args.email, args.username, args.password)))


if __name__ == "__main__":
    main()
