"""Обновление базы знаний подтверждёнными человеком примерами."""
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.models import ProductInput
from backend.data.examples_store import kb_to_record
from backend.data.knowledge_base import KnowledgeService
from backend.db.models import KnowledgeBase, UncertainExample


class KnowledgeUpdater:
    def __init__(self, session: AsyncSession, knowledge: KnowledgeService):
        self.session = session
        self.knowledge = knowledge

    async def add_approved(self, item: UncertainExample) -> KnowledgeBase:
        """Добавляет пример (с исправлением редактора, если оно есть) в БД и в поисковые индексы."""
        text = item.corrected_output or item.model_output
        product = ProductInput(**item.input_json) if item.input_json else None
        row = KnowledgeBase(
            source_text=item.input_text,
            target_text=text,
            category=product.category if product else None,
            platform=product.platform.value if product else None,
            style=product.style.value if product else None,
            product_name=product.name if product else None,
            source="hitl",
        )
        self.session.add(row)
        await self.session.commit()
        await self.session.refresh(row)
        self.knowledge.add_example(kb_to_record(row))
        return row
