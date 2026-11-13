import json
from datetime import timedelta

import pytest
from sqlalchemy import select, update

from backend.core.exceptions import LLMUnavailableError, NotFoundError, ValidationAppError
from backend.core.models import ProductInput
from backend.data.cache import make_cache_key
from backend.data.examples_store import import_glossary, import_knowledge, parse_csv_bytes, read_records
from backend.db.models import CacheEntry, GenerationRequest, KnowledgeBase, UncertainExample, utcnow
from backend.evaluation.evaluator import Evaluator, load_dataset, sample_dataset
from backend.evaluation.report import markdown_table, save_charts, save_results, summary_rows
from backend.hitl.active_learning import ActiveLearningQueue
from backend.hitl.feedback import FeedbackService
from backend.hitl.updater import KnowledgeUpdater
from backend.services.generation import ensemble_score


async def test_container_imports_initial_data(container):
    stats = container.knowledge.stats()
    assert stats["examples"] == 120 and stats["glossary"]["forbidden"] == 50
    assert stats["vector_backend"] in ("qdrant", "numpy")
    assert await container.import_initial_data() == {"knowledge_base": 0, "glossary": 0}  # повторно не импортирует


async def test_knowledge_service_search(container, product):
    examples = container.knowledge.search_examples(product, top_k=3)
    assert len(examples) == 3 and all(e.category == "Электроника" for e in examples)
    assert container.knowledge.selling_phrases("Электроника", k=3)
    assert container.knowledge.selling_phrases("Неизвестная", k=3) == []  # общих фраз в справочнике нет
    hints = container.knowledge.benefits_for("Электроника", ["активное шумоподавление", "нечто непонятное"])
    assert list(hints) == ["активное шумоподавление"]
    assert container.knowledge.templates("ozon", "нейтральный")
    assert "лучший в мире" in container.knowledge.forbidden_phrases()


@pytest.mark.parametrize("mode", ["baseline", "+dict", "+rag", "+agents"])
async def test_generation_modes(container, product, mode):
    res = await container.generation.generate(product, mode, use_cache=False)
    assert res.description and res.provider == "mock" and res.mode == mode
    assert bool(res.examples_used) == (mode in ("+rag", "+agents"))
    assert (res.critic_score is not None) == (mode == "+agents")
    steps = await container.log_store.get(res.request_id)
    assert steps and steps[0]["agent_name"] == "DescriptionGenerator"
    async with container.db.session_factory() as s:
        row = (await s.execute(select(GenerationRequest).where(GenerationRequest.request_id == res.request_id))
               ).scalar_one()
    assert row.mode == mode


async def test_generation_cache_and_ensemble(container, product):
    first = await container.generation.generate(product, "+rag")
    second = await container.generation.generate(product, "+rag")
    assert not first.cached and second.cached and second.description == first.description
    stats = await container.cache.stats()
    assert stats["entries"] >= 1 and stats["total_hits"] >= 1

    ens = await container.generation.generate(product, "+ensemble")
    assert ens.mode == "+ensemble" and {c["model"] for c in ens.candidates} == {"mock:a", "mock:b"}
    steps = await container.log_store.get(ens.request_id)
    assert steps[-1]["agent_name"] == "EnsembleSelector" and "ollama" in steps[-1]["input_summary"]
    with pytest.raises(LLMUnavailableError):
        await container.generation.ensemble(product, ["ollama:qwen2.5:7b"])

    multi = await container.generation.generate_multi(product, ["ozon", "site"], ["нейтральный"], "+dict", None,
                                                      None)
    assert {r.description.count("Ozon") > 0 for r in multi} == {True, False}
    batch = await container.generation.generate_batch([product, product.model_copy(update={"name": "Другое"})],
                                                      "baseline")
    assert len(batch) == 2


def test_ensemble_score(state):
    state.critic_score, state.validation_passed, state.errors = 8.0, True, []
    good = ensemble_score(state)
    state.validation_passed, state.errors = False, ["e1", "e2"]
    assert ensemble_score(state) < good


async def test_cache_service(container):
    key = make_cache_key("x", {"a": 1})
    assert key == make_cache_key("x", {"a": 1}) and await container.cache.get(key) is None
    await container.cache.set(key, {"v": 1})
    await container.cache.set(key, {"v": 2})
    assert await container.cache.get(key) == {"v": 2}
    async with container.db.session_factory() as s:
        await s.execute(update(CacheEntry).values(created_at=utcnow() - timedelta(days=30)))
        await s.commit()
    assert await container.cache.get(key) is None  # истёк TTL
    await container.cache.set(key, {"v": 3})
    assert await container.cache.cleanup(older_than_hours=0) == 1
    await container.cache.set(key, {"v": 4})
    assert await container.cache.clear() == 1
    container.cache.enabled = False
    await container.cache.set(key, {"v": 5})
    assert await container.cache.get(key) is None


async def test_hitl_queue_feedback_and_updater(container, product):
    async with container.db.session_factory() as s:
        q = ActiveLearningQueue(s)
        item = await q.add(request_id="r1", input_text=product.to_text(), input_json=product.model_dump(mode="json"),
                           model_output="Слабое описание", critic_score=4.0, reason="низкая оценка")
        other = await q.add(request_id="r2", input_text="x", input_json=None, model_output="y", critic_score=None,
                            reason="r")
        assert [i.id for i in await q.list()] == [other.id, item.id]
        approved = await q.approve(item.id, None, "Исправленное описание товара")
        with pytest.raises(ValidationAppError):
            await q.approve(item.id, None)
        with pytest.raises(NotFoundError):
            await q.get(999)
        await q.reject(other.id, None)
        assert await q.stats() == {"approved": 1, "rejected": 1}

        before = len(container.knowledge.retriever)
        row = await KnowledgeUpdater(s, container.knowledge).add_approved(approved)
        assert row.source == "hitl" and row.target_text == "Исправленное описание товара"
        assert len(container.knowledge.retriever) == before + 1

        fb = FeedbackService(s)
        await fb.add("r1", 5)
        await fb.add("r1", 2, comment="плохо")
        assert await fb.average_rating() == 3.5 and len(await fb.list("r1")) == 2


async def test_low_quality_result_goes_to_review(container, product, monkeypatch):
    from backend.agents import critic as critic_module

    monkeypatch.setattr(critic_module, "parse_critic_response", lambda text: None)  # Critic -> fallback 5.0
    res = await container.generation.generate(product, "+agents", use_cache=False)
    assert res.needs_review
    async with container.db.session_factory() as s:
        items = (await s.execute(select(UncertainExample))).scalars().all()
    assert items and items[0].request_id == res.request_id


async def test_examples_store_import(container, tmp_path):
    async with container.db.session_factory() as s:
        added = await import_knowledge(s, [{"source_text": "a", "target_text": "b"}, {"source_text": "a",
                                                                                      "target_text": "b"}, {}])
        assert added == 1
        assert await import_glossary(s, [{"term": "Новая", "kind": "selling_phrase"}, {"term": "новая"}]) == 1
        assert (await s.execute(select(KnowledgeBase).where(KnowledgeBase.source_text == "a"))).scalar_one()
    csv_path = tmp_path / "x.csv"
    csv_path.write_text("term,kind\nфраза,forbidden\n", encoding="utf-8")
    assert read_records(csv_path) == [{"term": "фраза", "kind": "forbidden"}]
    assert parse_csv_bytes(b"a,b\n1,2\n") == [{"a": "1", "b": "2"}]


# ------------------------- оценка -------------------------
def test_sample_dataset_stratified(dataset_dir):
    data = load_dataset(dataset_dir / "test.json")
    sample = sample_dataset(data, 12, seed=1)
    assert len(sample) == 12 and len({r["category"] for r in sample}) == 12
    assert sample_dataset(data, 12, seed=1) == sample
    assert len(sample_dataset(data, 5, stratify=False)) == 5
    assert len(sample_dataset(data, 1000)) == len(data)


async def test_evaluator_evaluate_compare_and_save(container, dataset_dir, tmp_path):
    data = load_dataset(dataset_dir / "test.json")
    ev = Evaluator(container.generation, ensemble_models=["mock:a", "mock:b"], concurrency=2)
    ev.fit_style_classifier(read_records(dataset_dir / "knowledge_base.json"))
    base = await ev.evaluate(data, "mock", "baseline", 6)
    rag = await ev.evaluate(data, "mock", "rag", 6)
    ens = await ev.evaluate(data, "mock", "+ensemble", 3)
    assert base["sample_size"] == 6 and 0 <= base["metrics"]["bleu"] <= 1
    assert rag["metrics"]["composite"] >= base["metrics"]["composite"]
    assert "style_accuracy" in base["metrics"] and base["confidence_intervals"]["bleu"][0] <= base["metrics"]["bleu"]
    assert ens["mode"] == "+ensemble"
    with pytest.raises(LLMUnavailableError):
        await ev.evaluate(data, "ollama:qwen2.5:7b", "baseline", 2)

    comp = await ev.compare(data, ["mock:a", "mock:b", "openai:gpt-4o-mini"], "baseline", 4)
    assert comp["winner"] in ("mock:a", "mock:b") and "error" in comp["results"]["openai:gpt-4o-mini"]
    async with container.db.session_factory() as s:
        row = await ev.save_result(s, base)
        cmp_row = await ev.save_comparison(s, comp)
        hist = await ev.history(s, model="mock")
    assert row.id and cmp_row.winner == comp["winner"] and hist[0].model_name == "mock"

    paths = save_results([base, rag, {"model_name": "x", "error": "e"}], tmp_path, extra={"k": 1})
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["k"] == 1
    assert "baseline" in paths["md"].read_text(encoding="utf-8")
    rows = summary_rows([base, rag])
    assert len(rows) == 2 and "| model |" in markdown_table(rows) and markdown_table([]) == "_нет результатов_"
    charts = save_charts([base, rag], tmp_path)
    assert any(p.suffix == ".html" for p in charts)
    assert save_charts([], tmp_path) == []


async def test_reasoning_store_handles_db_errors(container, state):
    from backend.utils.reasoning_logger import ReasoningLogStore

    def broken_factory():
        raise RuntimeError("БД недоступна")

    assert await ReasoningLogStore(broken_factory).save("r", []) == 0
    assert await container.log_store.llm_calls("нет") == []
    assert ProductInput(**state.product.model_dump()).name == state.product.name
