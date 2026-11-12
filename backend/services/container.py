"""Контейнер зависимостей приложения: БД, LLM, база знаний, кэш, сервисы."""
from __future__ import annotations

import logging
from pathlib import Path

from backend.core.config import Settings
from backend.data.cache import CacheService
from backend.data.examples_store import count_rows, import_glossary, import_knowledge, read_records
from backend.data.knowledge_base import KnowledgeService, get_knowledge_service
from backend.data.llm_manager import LLMManager
from backend.db.session import Database
from backend.services.generation import GenerationService
from backend.utils.llm_logger import LLMCallLogger
from backend.utils.reasoning_logger import ReasoningLogStore

logger = logging.getLogger(__name__)


class AppContainer:
    def __init__(self, settings: Settings, db: Database | None = None, knowledge: KnowledgeService | None = None,
                 llm: LLMManager | None = None, null_pool: bool = False):
        self.settings = settings
        self.db = db or Database(settings.database_url, settings.db_echo, null_pool=null_pool)
        self.call_logger = LLMCallLogger(self.db.session_factory)
        self.llm = llm or LLMManager(settings, self.call_logger)
        self.llm.call_logger = self.call_logger
        self.knowledge = knowledge or get_knowledge_service(settings)
        self.cache = CacheService(self.db.session_factory, settings.cache_ttl_hours,
                                  settings.redis_url if settings.use_redis_cache else None, settings.cache_enabled)
        self.log_store = ReasoningLogStore(self.db.session_factory)
        self.generation = GenerationService(settings, self.llm, self.knowledge, self.cache,
                                            self.db.session_factory, self.log_store)

    async def import_initial_data(self, data_dir: Path | None = None) -> dict[str, int]:
        data_dir = Path(data_dir or self.settings.data_dir)
        added = {"knowledge_base": 0, "glossary": 0}
        async with self.db.session_factory() as s:
            counts = await count_rows(s)
            kb_file, gl_file = data_dir / "knowledge_base.json", data_dir / "glossary.json"
            if counts["knowledge_base"] == 0 and kb_file.exists():
                added["knowledge_base"] = await import_knowledge(s, read_records(kb_file))
            if counts["glossary"] == 0 and gl_file.exists():
                added["glossary"] = await import_glossary(s, read_records(gl_file))
        if any(added.values()):
            logger.info("Импортированы начальные данные: %s", added)
        return added

    async def startup(self, load_knowledge: bool = True) -> None:
        await self.db.create_all()
        if self.settings.auto_import_data:
            await self.import_initial_data()
        if load_knowledge and not self.knowledge.loaded:
            await self.reload_knowledge()

    async def reload_knowledge(self) -> dict:
        async with self.db.session_factory() as s:
            await self.knowledge.load_from_db(s)
        return self.knowledge.stats()

    async def shutdown(self) -> None:
        await self.db.dispose()
