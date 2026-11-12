"""Работа с таблицами knowledge_base и glossary: CRUD и массовый импорт без дубликатов."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import Glossary, KnowledgeBase

KB_FIELDS = ("source_text", "target_text", "category", "platform", "style", "product_name")
GLOSSARY_FIELDS = ("term", "definition", "kind", "category")


def read_records(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else data.get("records", [])
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_csv_bytes(content: bytes) -> list[dict[str, Any]]:
    return list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))


async def import_knowledge(session: AsyncSession, records: list[dict[str, Any]], source: str = "import") -> int:
    existing = {
        (r.source_text, r.platform, r.style)
        for r in (await session.execute(select(KnowledgeBase))).scalars()
    }
    added = 0
    for rec in records:
        if not rec.get("source_text") or not rec.get("target_text"):
            continue
        key = (rec["source_text"], rec.get("platform") or None, rec.get("style") or None)
        if key in existing:
            continue
        existing.add(key)
        session.add(KnowledgeBase(**{k: (rec.get(k) or None) for k in KB_FIELDS}, source=source))
        added += 1
    await session.commit()
    return added


async def import_glossary(session: AsyncSession, records: list[dict[str, Any]]) -> int:
    existing = {(g.term.lower(), g.kind, g.category) for g in (await session.execute(select(Glossary))).scalars()}
    added = 0
    for rec in records:
        term = (rec.get("term") or "").strip()
        kind = (rec.get("kind") or "selling_phrase").strip()
        category = rec.get("category") or None
        if not term or (term.lower(), kind, category) in existing:
            continue
        existing.add((term.lower(), kind, category))
        session.add(Glossary(term=term, definition=rec.get("definition") or "", kind=kind, category=category))
        added += 1
    await session.commit()
    return added


async def count_rows(session: AsyncSession) -> dict[str, int]:
    kb = (await session.execute(select(func.count()).select_from(KnowledgeBase))).scalar_one()
    gl = (await session.execute(select(func.count()).select_from(Glossary))).scalar_one()
    return {"knowledge_base": kb, "glossary": gl}


def kb_to_record(row: KnowledgeBase) -> dict[str, Any]:
    return {"id": f"kb:{row.id}", **{k: getattr(row, k) for k in KB_FIELDS}}


def glossary_to_record(row: Glossary) -> dict[str, Any]:
    return {"id": row.id, **{k: getattr(row, k) for k in GLOSSARY_FIELDS}}
