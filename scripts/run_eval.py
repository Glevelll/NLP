"""Оценка качества и сравнение моделей/режимов на выборках разного размера.

Пример:
  python scripts/run_eval.py --data data/import/test.json --sample-sizes 20 50 100 \
      --modes baseline +dict +rag +agents +ensemble --models ollama:qwen2.5:7b ollama:qwen2.5:3b --output results/

Модели задаются как provider:model (ollama:qwen2.5:7b, openai:gpt-4o-mini, anthropic:claude-haiku-5-5,
groq:llama-3.1-8b-instant, hf:Qwen/Qwen2.5-0.5B-Instruct, mock). Без префикса считается, что это модель Ollama.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.config import Settings, get_settings  # noqa: E402
from backend.core.models import normalize_mode  # noqa: E402
from backend.data.examples_store import read_records  # noqa: E402
from backend.evaluation.evaluator import Evaluator, load_dataset  # noqa: E402
from backend.evaluation.report import markdown_table, save_charts, save_results, summary_rows  # noqa: E402
from backend.evaluation.significance import compare_systems  # noqa: E402
from backend.services.container import AppContainer  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "import" / "test.json")
    ap.add_argument("--kb", type=Path, default=ROOT / "data" / "import" / "knowledge_base.json",
                    help="база знаний для RAG и обучения классификатора стиля")
    ap.add_argument("--glossary", type=Path, default=ROOT / "data" / "import" / "glossary.json")
    ap.add_argument("--sample-sizes", type=int, nargs="+", default=[20, 50, 100])
    ap.add_argument("--modes", nargs="+", default=["baseline", "+dict", "+rag", "+agents", "+ensemble"])
    ap.add_argument("--models", nargs="+", default=["mock"])
    ap.add_argument("--ensemble-models", nargs="+", default=None,
                    help="модели для режима +ensemble (по умолчанию — --models)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--output", type=Path, default=ROOT / "results")
    ap.add_argument("--database-url", default=None, help="куда писать llm_calls (по умолчанию — временная SQLite)")
    ap.add_argument("--save-db", action="store_true", help="сохранить результаты в evaluation_results")
    ap.add_argument("--no-charts", action="store_true")
    return ap.parse_args(argv)


async def run(args: argparse.Namespace, settings: Settings | None = None) -> list[dict]:
    base = settings or get_settings()
    db_url = args.database_url or f"sqlite+aiosqlite:///{(args.output / 'eval.db').as_posix()}"
    args.output.mkdir(parents=True, exist_ok=True)
    settings = base.model_copy(update={"database_url": db_url, "auto_import_data": False, "cache_enabled": False})
    container = AppContainer(settings)
    await container.db.create_all()
    kb = read_records(args.kb)
    container.knowledge.load_records(kb, read_records(args.glossary) if args.glossary.exists() else [])

    evaluator = Evaluator(container.generation, ensemble_models=args.ensemble_models or args.models,
                          concurrency=args.concurrency)
    evaluator.fit_style_classifier(kb)
    dataset = load_dataset(args.data)
    modes = [normalize_mode(m) for m in args.modes]

    results: list[dict] = []
    for size in args.sample_sizes:
        for model in args.models:
            for mode in modes:
                if mode == "+ensemble" and model != args.models[0]:
                    continue  # ансамбль один на все модели
                label = "ensemble(" + ",".join(evaluator.ensemble_models) + ")" if mode == "+ensemble" else model
                t0 = time.perf_counter()
                try:
                    res = await evaluator.evaluate(dataset, model, mode, size, args.seed)
                except Exception as exc:
                    print(f"[!] {label} {mode} n={size}: {exc}")
                    results.append({"model_name": label, "mode": mode, "sample_size": size, "error": str(exc)})
                    continue
                res["model_name"] = label
                results.append(res)
                mm = res["metrics"]
                print(f"[{time.perf_counter() - t0:6.1f}s] {label:28s} {mode:10s} n={size:<4d} "
                      f"BLEU={mm['bleu']:.3f} chrF++={mm['chrf_pp']:.3f} ROUGE-L={mm['rouge_l']:.3f} "
                      f"METEOR={mm['meteor']:.3f} BERTScore={mm['bertscore']:.3f} "
                      f"KW={mm.get('keyword_coverage', 0):.3f} composite={mm['composite']:.3f}")
                if args.save_db:
                    async with container.db.session_factory() as s:
                        await evaluator.save_result(s, res)

    # Значимость: для каждого размера сравниваем режимы с baseline той же модели
    significance = {}
    for size in args.sample_sizes:
        for model in args.models:
            per = {r["mode"]: r["per_sample"] for r in results
                   if r.get("sample_size") == size and "per_sample" in r
                   and (r["model_name"] == model or r["mode"] == "+ensemble")}
            if "baseline" in per and len(per) > 1:
                significance[f"{model}|n={size}"] = {
                    metric: compare_systems(per, metric, "baseline") for metric in ("chrf_pp", "keyword_coverage")
                }
    paths = save_results(results, args.output, extra={"significance": significance, "args": {
        k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}})
    if not args.no_charts:
        save_charts(results, args.output)
    (args.output / "significance.json").write_text(json.dumps(significance, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
    print("\n" + markdown_table(summary_rows(results)))
    print(f"\nРезультаты сохранены: {paths['json']}")
    await container.shutdown()
    return results


def main(argv: list[str] | None = None) -> None:
    asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    main()
