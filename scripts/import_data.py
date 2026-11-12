"""Импорт базы знаний и справочника из CSV/JSON в БД (без дубликатов).

Пример: python scripts/import_data.py --knowledge data/import/knowledge_base.json --glossary data/import/glossary.json
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.config import get_settings  # noqa: E402
from backend.data.examples_store import count_rows, import_glossary, import_knowledge, read_records  # noqa: E402
from backend.db.session import Database  # noqa: E402


async def run(knowledge: Path | None, glossary: Path | None, database_url: str | None = None) -> dict:
    db = Database(database_url or get_settings().database_url)
    await db.create_all()
    result = {}
    async with db.session_factory() as s:
        if knowledge:
            result["knowledge_added"] = await import_knowledge(s, read_records(knowledge))
        if glossary:
            result["glossary_added"] = await import_glossary(s, read_records(glossary))
        result["totals"] = await count_rows(s)
    await db.dispose()
    return result


def main() -> None:
    data = ROOT / "data" / "import"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--knowledge", type=Path, default=data / "knowledge_base.json")
    ap.add_argument("--glossary", type=Path, default=data / "glossary.json")
    ap.add_argument("--database-url", default=None)
    args = ap.parse_args()
    print(asyncio.run(run(args.knowledge, args.glossary, args.database_url)))


if __name__ == "__main__":
    main()
