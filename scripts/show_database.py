"""Просмотр локальной базы SQLite в терминале."""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

DATABASE = Path(__file__).resolve().parents[1] / "data" / "app.db"
QUERIES = {
    "users": "SELECT id, username, role FROM users ORDER BY id",
    "knowledge_base": "SELECT id, category, platform, substr(target_text, 1, 100) AS description FROM knowledge_base "
                      "ORDER BY id DESC",
    "glossary": "SELECT id, kind, term FROM glossary ORDER BY id DESC",
    "generation_requests": "SELECT request_id, mode, provider, model_name FROM generation_requests ORDER BY id DESC",
    "reasoning_logs": "SELECT request_id, step, agent_name, status FROM reasoning_logs ORDER BY id DESC",
    "llm_calls": "SELECT id, request_id, provider, model_name, status FROM llm_calls ORDER BY id DESC",
    "uncertain_examples": "SELECT id, request_id, status, critic_score FROM uncertain_examples ORDER BY id DESC",
    "evaluation_results": "SELECT id, model_name, mode, sample_size FROM evaluation_results ORDER BY id DESC",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Показать таблицы проекта или первые 10 записей.")
    parser.add_argument("table", nargs="?", choices=QUERIES)
    args = parser.parse_args()
    try:
        with sqlite3.connect(DATABASE.as_uri() + "?mode=ro", uri=True) as db:
            if args.table:
                rows = db.execute(QUERIES[args.table] + " LIMIT 10")
                print(" | ".join(column[0] for column in rows.description))
                for row in rows:
                    print(" | ".join(str(value).replace("\n", " ") for value in row))
            else:
                print(f"SQLite: {DATABASE}")
                tables = db.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
                for (table,) in tables:
                    print(f"  {table}")
    except sqlite3.Error as exc:
        parser.exit(1, f"Не удалось прочитать {DATABASE}: {exc}\n")


if __name__ == "__main__":
    main()
