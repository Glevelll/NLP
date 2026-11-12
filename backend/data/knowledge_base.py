"""Сервис базы знаний: гибридный поиск похожих товаров и доступ к справочнику."""
from __future__ import annotations

import logging
import random
import threading
from collections import defaultdict
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import Settings
from backend.core.constants import DEFAULT_FORBIDDEN, GlossaryKind
from backend.core.models import ProductInput, RetrievedExample
from backend.data.embeddings import EmbeddingModel
from backend.data.examples_store import glossary_to_record, kb_to_record
from backend.db.models import Glossary, KnowledgeBase
from backend.retrieval.bm25_index import BM25Index
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.reranker import Reranker
from backend.retrieval.types import Document
from backend.retrieval.vector_index import VectorIndex
from backend.utils.text_processing import feature_coverage

logger = logging.getLogger(__name__)


def record_to_document(rec: dict[str, Any]) -> Document:
    text = f"{rec.get('category') or ''}\n{rec['source_text']}"
    return Document(
        id=str(rec["id"]),
        text=text,
        metadata={
            "category": rec.get("category"),
            "platform": rec.get("platform"),
            "style": rec.get("style"),
            "description": rec.get("target_text", ""),
            "source_text": rec["source_text"],
        },
    )


class KnowledgeService:
    def __init__(self, settings: Settings, embedder: EmbeddingModel | None = None, vector_client: Any = None):
        self.settings = settings
        self.embedder = embedder or EmbeddingModel.get_instance(settings.embedding_model, settings.embedding_backend,
                                                                settings.embedding_dim)
        self.retriever = HybridRetriever(
            BM25Index(),
            VectorIndex(self.embedder, settings.qdrant_collection, settings.qdrant_url, client=vector_client),
            rrf_k=settings.rrf_k,
            candidates=settings.retrieval_candidates,
            reranker=Reranker(),
        )
        self.glossary: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.categories: set[str] = set()
        self.loaded = False
        self._lock = threading.Lock()

    # ------------------- загрузка -------------------
    def load_records(self, kb_records: list[dict[str, Any]], glossary_records: list[dict[str, Any]]) -> None:
        with self._lock:
            docs = [record_to_document({**r, "id": r.get("id", f"rec:{i}")}) for i, r in enumerate(kb_records)]
            self.retriever.build(docs)
            self.categories = {d.metadata["category"] for d in docs if d.metadata.get("category")}
            self.glossary = defaultdict(list)
            for g in glossary_records:
                self.glossary[g.get("kind") or GlossaryKind.SELLING_PHRASE].append(g)
            self.loaded = True
        logger.info("База знаний загружена: %d примеров, %d записей справочника (эмбеддинги: %s, индекс: %s)",
                    len(docs), len(glossary_records), self.embedder.name, self.retriever.vector.backend)

    async def load_from_db(self, session: AsyncSession) -> None:
        kb = [kb_to_record(r) for r in (await session.execute(select(KnowledgeBase))).scalars()]
        gl = [glossary_to_record(r) for r in (await session.execute(select(Glossary))).scalars()]
        self.load_records(kb, gl)

    def add_example(self, rec: dict[str, Any]) -> None:
        doc = record_to_document(rec)
        self.retriever.add([doc])
        if doc.metadata.get("category"):
            self.categories.add(doc.metadata["category"])

    def remove_example(self, doc_id: str) -> None:
        self.retriever.remove(doc_id)

    # ------------------- поиск -------------------
    def search_examples(self, product: ProductInput, top_k: int | None = None,
                        same_category: bool = True) -> list[RetrievedExample]:
        top_k = top_k or self.settings.rag_top_k
        if not len(self.retriever):
            return []
        filters = {"category": product.category} if same_category and product.category in self.categories else None
        context = {"platform": product.platform.value, "style": product.style.value}
        hits = self.retriever.search(product.to_text(), top_k=top_k, filters=filters, context=context)
        return [
            RetrievedExample(
                id=h.doc.id, text=h.doc.metadata.get("source_text", ""), description=h.doc.metadata.get("description", ""),
                category=h.doc.metadata.get("category"), platform=h.doc.metadata.get("platform"),
                style=h.doc.metadata.get("style"), score=round(h.score, 4), bm25_rank=h.bm25_rank,
                vector_rank=h.vector_rank,
            )
            for h in hits
        ]

    # ------------------- справочник -------------------
    def forbidden_phrases(self) -> list[str]:
        terms = [g["term"] for g in self.glossary.get(GlossaryKind.FORBIDDEN, [])]
        return sorted(set(terms) | set(DEFAULT_FORBIDDEN))

    def selling_phrases(self, category: str | None, k: int = 6, seed: str = "") -> list[str]:
        items = self.glossary.get(GlossaryKind.SELLING_PHRASE, [])
        own = [g["term"] for g in items if g.get("category") == category]
        common = [g["term"] for g in items if not g.get("category")]
        pool = own or common
        rng = random.Random(seed or category or "")
        return rng.sample(pool, min(k, len(pool))) if pool else []

    def benefits(self, category: str | None) -> dict[str, str]:
        return {g["term"]: g.get("definition", "") for g in self.glossary.get(GlossaryKind.BENEFIT, [])
                if g.get("category") == category}

    def benefits_for(self, category: str | None, features: list[str]) -> dict[str, str]:
        """Подбирает из справочника выгоду для каждой особенности товара (по совпадению основ слов)."""
        known = self.benefits(category) or {g["term"]: g.get("definition", "")
                                            for g in self.glossary.get(GlossaryKind.BENEFIT, [])}
        result: dict[str, str] = {}
        for f in features:
            best, best_ratio = None, 0.0
            for term, benefit in known.items():
                ok, ratio = feature_coverage(term, f)
                if ok and ratio > best_ratio and benefit:
                    best, best_ratio = benefit, ratio
            if best:
                result[f] = best
        return result

    def templates(self, platform: str | None, style: str | None, k: int = 2) -> list[str]:
        items = self.glossary.get(GlossaryKind.TEMPLATE, [])

        def ok(g: dict) -> bool:
            cat = g.get("category") or ""
            return (not platform or platform in cat) and (not style or style in cat)

        matched = [g["definition"] or g["term"] for g in items if ok(g)]
        return matched[:k]

    def stats(self) -> dict[str, Any]:
        return {
            "examples": len(self.retriever),
            "vector_count": self.retriever.vector.count(),
            "vector_backend": self.retriever.vector.backend,
            "embedding_model": self.embedder.name,
            "glossary": {k: len(v) for k, v in self.glossary.items()},
            "categories": sorted(self.categories),
        }


_service: KnowledgeService | None = None
_service_lock = threading.Lock()


def get_knowledge_service(settings: Settings) -> KnowledgeService:
    """Один экземпляр на процесс (индексы и модель эмбеддингов не зависят от event loop)."""
    global _service
    with _service_lock:
        if _service is None:
            _service = KnowledgeService(settings)
        return _service


def reset_knowledge_service() -> None:
    global _service
    _service = None
