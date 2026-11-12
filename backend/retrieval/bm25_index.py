"""Лексический индекс BM25 (rank_bm25) по основам слов."""
from __future__ import annotations

import threading

import numpy as np
from rank_bm25 import BM25Okapi

from backend.retrieval.types import Document, SearchHit, match_filters
from backend.utils.text_processing import stems


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: list[Document] = []
        self._terms: list[set[str]] = []
        self._bm25: BM25Okapi | None = None
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self.docs)

    def build(self, docs: list[Document]) -> None:
        with self._lock:
            self.docs = list(docs)
            self._rebuild()

    def add(self, docs: list[Document]) -> None:
        with self._lock:
            existing = {d.id for d in self.docs}
            self.docs.extend(d for d in docs if d.id not in existing)
            self._rebuild()

    def remove(self, doc_id: str) -> None:
        with self._lock:
            self.docs = [d for d in self.docs if d.id != doc_id]
            self._rebuild()

    def _rebuild(self) -> None:
        corpus = [stems(d.text) or ["_"] for d in self.docs]
        self._terms = [set(c) for c in corpus]
        self._bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b) if corpus else None

    def search(self, query: str, top_k: int = 10, filters: dict | None = None) -> list[SearchHit]:
        if not self._bm25 or not self.docs:
            return []
        q = stems(query)
        if not q:
            return []
        q_set = set(q)
        scores = self._bm25.get_scores(q)
        order = np.argsort(-scores, kind="stable")
        hits: list[SearchHit] = []
        for idx in order:
            # IDF частого термина может быть нулевым, поэтому кандидатом считаем любой документ с общим термином
            if not (self._terms[idx] & q_set):
                continue
            doc = self.docs[idx]
            if not match_filters(doc.metadata, filters):
                continue
            hits.append(SearchHit(id=doc.id, score=float(scores[idx]), rank=len(hits) + 1))
            if len(hits) >= top_k:
                break
        return hits
