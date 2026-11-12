"""Векторный индекс в Qdrant (cosine). Без сервера используется встроенный in-memory режим qdrant-client,
а если библиотеки нет — простой numpy-индекс с тем же интерфейсом."""
from __future__ import annotations

import logging
import uuid
from typing import Any

import numpy as np

from backend.data.embeddings import EmbeddingModel
from backend.retrieval.types import Document, SearchHit, match_filters

logger = logging.getLogger(__name__)


def _point_id(doc_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id))


class VectorIndex:
    def __init__(self, embedder: EmbeddingModel, collection: str = "product_descriptions",
                 url: str | None = None, client: Any | None = None):
        self.embedder = embedder
        self.collection = collection
        self.url = url
        self._client = client
        self._dim: int | None = None
        self.backend = "qdrant"
        # numpy-запасной вариант
        self._np_docs: list[Document] = []
        self._np_vecs: np.ndarray | None = None
        if self._client is None:
            try:
                from qdrant_client import QdrantClient

                self._client = QdrantClient(url=url, timeout=10) if url else QdrantClient(location=":memory:")
            except Exception as exc:
                logger.warning("Qdrant недоступен (%s) — используется numpy-индекс", exc)
                self.backend = "numpy"

    # ---------------- служебное ----------------
    def _ensure_collection(self, dim: int, recreate: bool = False) -> None:
        from qdrant_client import models

        exists = self._client.collection_exists(self.collection)
        if exists and recreate:
            self._client.delete_collection(self.collection)
            exists = False
        if not exists:
            self._client.create_collection(
                self.collection, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
            )
            for field in ("category", "platform", "style"):
                try:
                    self._client.create_payload_index(self.collection, field, models.PayloadSchemaType.KEYWORD)
                except Exception:  # in-memory режим не поддерживает индексы payload
                    pass
        self._dim = dim

    def _qdrant_filter(self, filters: dict | None):
        if not filters:
            return None
        from qdrant_client import models

        must = []
        for key, value in filters.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple, set)):
                must.append(models.FieldCondition(key=key, match=models.MatchAny(any=list(value))))
            else:
                must.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
        return models.Filter(must=must) if must else None

    # ---------------- API ----------------
    def build(self, docs: list[Document], batch_size: int = 256) -> None:
        if self.backend == "numpy":
            self._np_docs = []
            self._np_vecs = None
            self.add(docs)
            return
        vecs = self.embedder.encode([d.text for d in docs[:1]] or ["x"])
        self._ensure_collection(vecs.shape[1], recreate=True)
        self.add(docs, batch_size=batch_size)

    def add(self, docs: list[Document], batch_size: int = 256) -> None:
        if not docs:
            return
        if self.backend == "numpy":
            vecs = self.embedder.encode([d.text for d in docs])
            known = {d.id for d in self._np_docs}
            keep = [i for i, d in enumerate(docs) if d.id not in known]
            self._np_docs.extend(docs[i] for i in keep)
            new = vecs[keep]
            self._np_vecs = new if self._np_vecs is None else np.vstack([self._np_vecs, new])
            return
        from qdrant_client import models

        for i in range(0, len(docs), batch_size):
            chunk = docs[i : i + batch_size]
            vecs = self.embedder.encode([d.text for d in chunk])
            if self._dim is None:
                self._ensure_collection(vecs.shape[1])
            points = [
                models.PointStruct(id=_point_id(d.id), vector=v.tolist(), payload={"doc_id": d.id, **d.metadata})
                for d, v in zip(chunk, vecs, strict=True)
            ]
            self._client.upsert(self.collection, points=points)

    def remove(self, doc_id: str) -> None:
        if self.backend == "numpy":
            idx = [i for i, d in enumerate(self._np_docs) if d.id != doc_id]
            self._np_docs = [self._np_docs[i] for i in idx]
            self._np_vecs = self._np_vecs[idx] if self._np_vecs is not None and idx else None
            return
        from qdrant_client import models

        if self._dim is not None:
            self._client.delete(self.collection, points_selector=models.PointIdsList(points=[_point_id(doc_id)]))

    def count(self) -> int:
        if self.backend == "numpy":
            return len(self._np_docs)
        if self._dim is None and not self._client.collection_exists(self.collection):
            return 0
        return self._client.count(self.collection).count

    def search(self, query: str, top_k: int = 10, filters: dict | None = None) -> list[SearchHit]:
        qv = self.embedder.encode_one(query)
        if self.backend == "numpy":
            if self._np_vecs is None or not len(self._np_docs):
                return []
            sims = self._np_vecs @ qv
            hits: list[SearchHit] = []
            for idx in np.argsort(-sims):
                doc = self._np_docs[idx]
                if match_filters(doc.metadata, filters):
                    hits.append(SearchHit(id=doc.id, score=float(sims[idx]), rank=len(hits) + 1))
                    if len(hits) >= top_k:
                        break
            return hits
        if self._dim is None and not self._client.collection_exists(self.collection):
            return []
        res = self._client.query_points(self.collection, query=qv.tolist(), limit=top_k,
                                        query_filter=self._qdrant_filter(filters), with_payload=True)
        return [SearchHit(id=str(p.payload.get("doc_id")), score=float(p.score), rank=i + 1)
                for i, p in enumerate(res.points)]

    def health(self) -> bool:
        if self.backend == "numpy":
            return True
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False
