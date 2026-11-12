import asyncio
import json

from backend.worker import tasks
from scripts import create_admin, import_data, run_eval
from scripts.generate_data import build_glossary, generate


def test_generate_data_volumes(tmp_path):
    stats = generate(tmp_path, kb_size=24, test_size=12, seed=3)
    assert stats == {"knowledge_base": 24, "test": 12, "glossary": len(build_glossary())}
    assert len(build_glossary()) >= 500
    kb = json.loads((tmp_path / "knowledge_base.json").read_text(encoding="utf-8"))
    test = json.loads((tmp_path / "test.json").read_text(encoding="utf-8"))
    assert {r["product_name"] for r in kb}.isdisjoint({r["product"]["name"] for r in test})
    assert all(r["target_text"] and r["source_text"].startswith("Название:") for r in kb)
    assert (tmp_path / "batch_example.csv").exists()


async def test_import_and_create_admin(tmp_path, dataset_dir):
    url = f"sqlite+aiosqlite:///{(tmp_path / 's.db').as_posix()}"
    res = await import_data.run(dataset_dir / "knowledge_base.json", dataset_dir / "glossary.json", url)
    assert res["knowledge_added"] == 120 and res["totals"]["glossary"] >= 500
    again = await import_data.run(dataset_dir / "knowledge_base.json", None, url)
    assert again["knowledge_added"] == 0
    assert "Создан" in await create_admin.create_admin("a@b.ru", "boss", "secret123", url)
    assert "повышен" in await create_admin.create_admin("a@b.ru", "boss", "secret123", url)


async def test_run_eval_script(settings, dataset_dir, tmp_path):
    args = run_eval.parse_args([
        "--data", str(dataset_dir / "test.json"), "--kb", str(dataset_dir / "knowledge_base.json"),
        "--glossary", str(dataset_dir / "glossary.json"), "--sample-sizes", "4", "--modes", "baseline", "+rag",
        "+ensemble", "--models", "mock:a", "ollama:none", "--ensemble-models", "mock:a", "mock:b",
        "--output", str(tmp_path / "res"), "--save-db",
    ])
    results = await run_eval.run(args, settings)
    ok = [r for r in results if "metrics" in r]
    assert {r["mode"] for r in ok} == {"baseline", "+rag", "+ensemble"}
    assert any("error" in r for r in results)  # ollama недоступен
    out = tmp_path / "res"
    assert (out / "eval_results.json").exists() and (out / "eval_summary.csv").exists()
    sig = json.loads((out / "significance.json").read_text(encoding="utf-8"))
    assert "mock:a|n=4" in sig


def test_celery_tasks_eager(task_env, product):
    assert tasks.reindex_knowledge_task.apply().get()["examples"] == 120
    assert tasks.cleanup_cache_task.apply(args=(0,)).get() == {"deleted": 0}
    res = tasks.batch_generate_task.apply(args=([product.model_dump(mode="json")], "baseline")).get()
    assert res["count"] == 1 and res["results"][0]["description"]
    added = tasks.bulk_import_task.apply(args=("glossary", [{"term": "новая фраза", "kind": "selling_phrase"}])).get()
    assert added["added"] == 1
    ev = tasks.evaluate_model_task.apply(args=("mock", "baseline", 3)).get()
    assert ev["id"] and ev["sample_size"] == 3
    cmp = tasks.compare_models_task.apply(args=(["mock:a", "mock:b"], "+dict", 3)).get()
    assert cmp["winner"] in ("mock:a", "mock:b")
    batch = tasks.batch_evaluate_task.apply(args=(["mock", "groq:x"], ["baseline"], [2])).get()
    assert len(batch["results"]) == 2 and "error" in batch["results"][1]


def test_progress_helper_is_safe():
    class Req:
        id = "x"
        called_directly = False

    class FakeTask:
        request = Req()

        def update_state(self, **kw):
            raise RuntimeError("нет backend")

    tasks._progress(FakeTask(), 1, 2)  # не падает
    assert asyncio.iscoroutinefunction(import_data.run)
