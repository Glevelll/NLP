"""Гибридный поиск: BM25 + векторный индекс, объединение через Reciprocal Rank Fusion."""
from __future__ import annotations

from dataclasses import dataclass

from backend.retrieval.bm25_index import BM25Index
from backend.retrieval.reranker import Reranker
from backend.retrieval.types import Document, SearchHit
from backend.retrieval.vector_index import VectorIndex


def reciprocal_rank_fusion(rankings: list[list[SearchHit]], k: int = 60,
                           weights: list[float] | None = None) -> dict[str, float]:
    """RRF: score(d) = Σ w_i / (k + rank_i(d)). Документ, найденный обоими индексами, поднимается выше."""
    weights = weights or [1.0] * len(rankings)
    fused: dict[str, float] = {}
    for w, hits in zip(weights, rankings, strict=True):
        for h in hits:
            fused[h.id] = fused.get(h.id, 0.0) + w / (k + h.rank)
    return fused


@dataclass
class HybridHit:
    doc: Document
    score: float
    bm25_rank: int | None
    vector_rank: int | None


class HybridRetriever:
    def __init__(self, bm25: BM25Index, vector: VectorIndex, rrf_k: int = 60, candidates: int = 30,
                 reranker: Reranker | None = None):
        self.bm25 = bm25
        self.vector = vector
        self.rrf_k = rrf_k
        self.candidates = candidates
        self.reranker = reranker
        self._docs: dict[str, Document] = {}

    def __len__(self) -> int:
        return len(self._docs)

    def build(self, docs: list[Document]) -> None:
        self._docs = {d.id: d for d in docs}
        self.bm25.build(docs)
        self.vector.build(docs)

    def add(self, docs: list[Document]) -> None:
        for d in docs:
            self._docs[d.id] = d
        self.bm25.add(docs)
        self.vector.add(docs)

    def remove(self, doc_id: str) -> None:
        self._docs.pop(doc_id, None)
        self.bm25.remove(doc_id)
        self.vector.remove(doc_id)

    def get(self, doc_id: str) -> Document | None:
        return self._docs.get(doc_id)

    def search(self, query: str, top_k: int = 5, filters: dict | None = None,
               weights: tuple[float, float] = (1.0, 1.0), context: dict | None = None) -> list[HybridHit]:
        bm25_hits = self.bm25.search(query, self.candidates, filters)
        vec_hits = self.vector.search(query, self.candidates, filters)
        fused = reciprocal_rank_fusion([bm25_hits, vec_hits], k=self.rrf_k, weights=list(weights))
        bm25_rank = {h.id: h.rank for h in bm25_hits}
        vec_rank = {h.id: h.rank for h in vec_hits}
        hits = [
            HybridHit(self._docs[i], s, bm25_rank.get(i), vec_rank.get(i))
            for i, s in fused.items() if i in self._docs
        ]
        hits.sort(key=lambda h: h.score, reverse=True)
        if self.reranker is not None and context:
            hits = self.reranker.rerank(hits, context)
        return hits[:top_k]
