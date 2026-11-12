"""Простое переранжирование: бонус за совпадение площадки и стиля с запросом."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.retrieval.hybrid_retriever import HybridHit


class Reranker:
    def __init__(self, platform_bonus: float = 0.3, style_bonus: float = 0.2):
        self.platform_bonus = platform_bonus
        self.style_bonus = style_bonus

    def rerank(self, hits: list[HybridHit], context: dict) -> list[HybridHit]:
        if not hits:
            return hits
        top = max(h.score for h in hits) or 1.0
        for h in hits:
            bonus = 0.0
            if context.get("platform") and h.doc.metadata.get("platform") == context["platform"]:
                bonus += self.platform_bonus
            if context.get("style") and h.doc.metadata.get("style") == context["style"]:
                bonus += self.style_bonus
            h.score = h.score / top + bonus
        return sorted(hits, key=lambda h: h.score, reverse=True)
