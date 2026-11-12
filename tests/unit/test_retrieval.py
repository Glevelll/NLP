import numpy as np
import pytest

from backend.data.embeddings import EmbeddingModel, HashingEmbedder
from backend.retrieval.bm25_index import BM25Index
from backend.retrieval.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion
from backend.retrieval.reranker import Reranker
from backend.retrieval.types import Document, SearchHit, match_filters
from backend.retrieval.vector_index import VectorIndex

DOCS = [
    Document("1", "беспроводные наушники с шумоподавлением", {"category": "Электроника", "platform": "ozon"}),
    Document("2", "термокружка держит тепло 8 часов", {"category": "Дом и кухня", "platform": "site"}),
    Document("3", "наушники для спорта защита от воды", {"category": "Электроника", "platform": "site"}),
    Document("4", "кроссовки с амортизирующей подошвой", {"category": "Обувь", "platform": "wildberries"}),
]


@pytest.fixture
def embedder():
    return EmbeddingModel("hashing", backend="hashing", dim=128)


def test_hashing_embedder_normalized():
    vecs = HashingEmbedder(64).encode(["наушники", "наушники", "кружка"])
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)
    assert vecs[0] @ vecs[1] == pytest.approx(1.0)
    assert HashingEmbedder(8).encode([]).shape == (0, 8)


def test_embedding_singleton_and_fallback():
    EmbeddingModel.reset()
    a = EmbeddingModel.get_instance("hashing", "hashing", 32)
    assert EmbeddingModel.get_instance("hashing", "hashing", 32) is a
    assert a.encode_one("тест").shape == (32,) and a.name == "hashing-32"
    broken = EmbeddingModel("nonexistent/model-xyz", backend="sentence-transformers", dim=16)
    assert broken.encode(["x"]).shape == (1, 16)  # нет torch / модели -> hashing
    EmbeddingModel.reset()


def test_match_filters():
    assert match_filters({"a": 1}, None)
    assert match_filters({"a": 1}, {"a": 1, "b": None})
    assert match_filters({"a": 1}, {"a": [1, 2]})
    assert not match_filters({"a": 1}, {"a": 2})
    assert not match_filters({"a": 1}, {"a": [3]})


def test_bm25_search_and_filters():
    idx = BM25Index()
    assert idx.search("наушники") == []
    idx.build(DOCS)
    hits = idx.search("наушники шумоподавление", top_k=2)
    assert hits[0].id == "1" and hits[0].rank == 1
    assert {h.id for h in idx.search("наушники", filters={"platform": "site"})} == {"3"}
    assert idx.search("и или") == []
    idx.add([Document("5", "наушники детские", {})])
    assert len(idx) == 5
    idx.remove("5")
    assert len(idx) == 4


@pytest.mark.parametrize("use_numpy", [True, False])
def test_vector_index(embedder, use_numpy):
    vi = VectorIndex(embedder, "test_coll")
    if use_numpy:
        vi.backend = "numpy"
    assert vi.search("наушники") == []
    vi.build(DOCS)
    assert vi.count() == 4 and vi.health()
    hits = vi.search("термокружка тепло", top_k=2)
    assert hits[0].id == "2"
    filtered = vi.search("наушники", top_k=5, filters={"category": "Электроника"})
    assert {h.id for h in filtered} <= {"1", "3"}
    assert vi.search("наушники", filters={"category": ["Обувь"]})[0].id == "4"
    vi.remove("2")
    assert vi.count() == 3


def test_rrf_prefers_documents_in_both_lists():
    a = [SearchHit("x", 5, 1), SearchHit("y", 4, 2)]
    b = [SearchHit("y", 0.9, 1), SearchHit("z", 0.8, 2)]
    fused = reciprocal_rank_fusion([a, b], k=60)
    assert max(fused, key=fused.get) == "y"
    assert fused["x"] == pytest.approx(1 / 61)
    weighted = reciprocal_rank_fusion([a, b], k=60, weights=[2.0, 0.0])
    assert max(weighted, key=weighted.get) == "x"


def test_hybrid_retriever_and_reranker(embedder):
    hr = HybridRetriever(BM25Index(), VectorIndex(embedder, "hybrid"), rrf_k=60, reranker=Reranker())
    hr.build(DOCS)
    hits = hr.search("наушники", top_k=2)
    assert {h.doc.id for h in hits} <= {"1", "3"}
    assert hits[0].bm25_rank is not None
    reranked = hr.search("наушники", top_k=2, context={"platform": "site", "style": None})
    assert reranked[0].doc.id == "3"
    hr.add([Document("9", "наушники премиум", {"platform": "ozon"})])
    assert hr.get("9") is not None and len(hr) == 5
    hr.remove("9")
    assert hr.get("9") is None
    assert Reranker().rerank([], {}) == []
