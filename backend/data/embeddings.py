"""Эмбеддинги: синглтон SentenceTransformer с ленивой загрузкой и запасной hashing-моделью."""
from __future__ import annotations

import hashlib
import logging
import threading

import numpy as np

from backend.utils.text_processing import char_ngrams, stems

logger = logging.getLogger(__name__)


class HashingEmbedder:
    """Лёгкая детерминированная модель (feature hashing основ слов + символьных триграмм).

    Нужна для тестов и окружений без torch; семантику ловит хуже, чем трансформер.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in stems(text):
            h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
            v[h % self.dim] += 2.0 if (h >> 1) & 1 else -2.0
        for ng in char_ngrams(text):
            h = int(hashlib.md5(ng.encode()).hexdigest()[:8], 16)
            v[h % self.dim] += 0.5 if (h >> 1) & 1 else -0.5
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), dtype=np.float32)


class EmbeddingModel:
    """Синглтон: модель загружается один раз на процесс при первом обращении."""

    _instance: EmbeddingModel | None = None
    _lock = threading.Lock()

    def __init__(self, model_name: str, backend: str = "sentence-transformers", dim: int = 384):
        self.model_name = model_name
        self.backend = backend
        self.dim = dim
        self._model = None

    @classmethod
    def get_instance(cls, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                     backend: str = "sentence-transformers", dim: int = 384) -> EmbeddingModel:
        with cls._lock:
            if cls._instance is None or cls._instance.model_name != model_name or cls._instance.backend != backend:
                cls._instance = cls(model_name, backend, dim)
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def _load(self):
        if self._model is not None:
            return self._model
        if self.backend == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer

                logger.info("Загрузка модели эмбеддингов %s", self.model_name)
                self._model = SentenceTransformer(self.model_name)
                self.dim = self._model.get_sentence_embedding_dimension()
                return self._model
            except Exception as exc:
                logger.warning("sentence-transformers недоступен (%s), используется HashingEmbedder", exc)
                self.backend = "hashing"
        self._model = HashingEmbedder(self.dim)
        return self._model

    @property
    def name(self) -> str:
        return self.model_name if self.backend == "sentence-transformers" else f"hashing-{self.dim}"

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        model = self._load()
        if isinstance(model, HashingEmbedder):
            return model.encode(texts)
        vecs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
