"""Настройка логирования: консоль + файлы с ротацией, JSON-формат по желанию, уровни по модулям."""
from __future__ import annotations

import json
import logging
import logging.handlers
from datetime import UTC, datetime
from pathlib import Path

from backend.utils.tracing import agent_name_var, request_id_var

_CONFIGURED = False
TEXT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | req=%(request_id)s agent=%(agent)s | %(message)s"


class ContextFilter(logging.Filter):
    """Добавляет в каждую запись request_id и имя агента из contextvars."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get() or "-"
        record.agent = agent_name_var.get() or "-"
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
            "agent": getattr(record, "agent", "-"),
        }
        extra = getattr(record, "extra_data", None)
        if extra:
            payload["data"] = extra
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(log_dir: Path | str = "logs", level: str = "INFO", json_logs: bool = False,
                  module_levels: dict[str, str] | None = None, force: bool = False) -> None:
    global _CONFIGURED
    if _CONFIGURED and not force:
        return
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level.upper())
    for h in list(root.handlers):
        root.removeHandler(h)

    formatter: logging.Formatter = JsonFormatter() if json_logs else logging.Formatter(TEXT_FORMAT)
    ctx = ContextFilter()

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.addFilter(ctx)
    root.addHandler(console)

    app_file = logging.handlers.RotatingFileHandler(log_dir / "app.log", maxBytes=5_000_000, backupCount=5,
                                                    encoding="utf-8")
    app_file.setFormatter(formatter)
    app_file.addFilter(ctx)
    root.addHandler(app_file)

    # Отдельный файл трассировки LLM-вызовов (всегда JSON — удобно анализировать)
    llm_file = logging.handlers.RotatingFileHandler(log_dir / "llm_calls.log", maxBytes=5_000_000, backupCount=5,
                                                    encoding="utf-8")
    llm_file.setFormatter(JsonFormatter())
    llm_file.addFilter(ctx)
    llm_logger = logging.getLogger("backend.llm")
    for h in list(llm_logger.handlers):
        llm_logger.removeHandler(h)
    llm_logger.addHandler(llm_file)

    for name, lvl in (module_levels or {}).items():
        logging.getLogger(name).setLevel(lvl.upper())
    _CONFIGURED = True
