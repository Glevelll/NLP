"""Evaluator: выборки 20/50/100, прогон модели в нужном режиме, метрики, сохранение результатов."""
from __future__ import annotations

import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.constants import Mode
from backend.core.exceptions import LLMUnavailableError
from backend.core.models import ProductInput, normalize_mode
from backend.data.llm_manager import parse_model_spec
from backend.db.models import EvaluationResult, ModelComparison
from backend.evaluation.metrics_calculator import COMPOSITE_KEYS, MetricsCalculator
from backend.evaluation.significance import bootstrap_ci, compare_systems
from backend.services.generation import GenerationService
from backend.utils.async_utils import gather_limited

logger = logging.getLogger(__name__)


def load_dataset(path: Path | str) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data["records"]


def sample_dataset(dataset: list[dict[str, Any]], size: int, seed: int = 42, stratify: bool = True
                   ) -> list[dict[str, Any]]:
    """Стратифицированная по категориям выборка фиксированного размера (воспроизводима по seed)."""
    if size >= len(dataset):
        return list(dataset)
    rng = random.Random(seed)
    if not stratify:
        return rng.sample(dataset, size)
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in dataset:
        groups[rec.get("category") or rec["product"].get("category", "")].append(rec)
    for g in groups.values():
        rng.shuffle(g)
    result: list[dict] = []
    keys = sorted(groups)
    i = 0
    while len(result) < size:
        g = groups[keys[i % len(keys)]]
        if g:
            result.append(g.pop())
        i += 1
    return result


class Evaluator:
    def __init__(self, generation: GenerationService, metrics: MetricsCalculator | None = None,
                 ensemble_models: list[str] | None = None, concurrency: int = 4):
        self.generation = generation
        self.metrics = metrics or MetricsCalculator(generation.settings.bertscore_backend,
                                                    generation.settings.bertscore_model,
                                                    generation.knowledge.embedder)
        self.ensemble_models = ensemble_models
        self.concurrency = concurrency

    def fit_style_classifier(self, kb_records: list[dict[str, Any]]) -> None:
        texts = [r["target_text"] for r in kb_records if r.get("style")]
        styles = [r["style"] for r in kb_records if r.get("style")]
        if texts:
            self.metrics.fit_style_classifier(texts, styles)

    async def _predict(self, rec: dict[str, Any], provider: str | None, model: str | None, mode: str) -> dict:
        product = ProductInput(**rec["product"])
        if mode == Mode.ENSEMBLE:
            res = await self.generation.ensemble(product, self.ensemble_models, persist=False)
        else:
            res = await self.generation.generate(product, mode, provider, model, use_cache=False, persist=False)
        actual_model = f"{res.provider or ''}:{res.model or ''}"
        if mode != Mode.ENSEMBLE and (not res.description
                                     or not self.generation.llm.matches_model(actual_model, provider, model)):
            raise LLMUnavailableError(f"Модель {provider}{f':{model}' if model else ''} не вернула описание")
        return {"prediction": res.description, "critic_score": res.critic_score, "iterations": res.iterations,
                "model": actual_model}

    async def evaluate(self, dataset: list[dict[str, Any]], model_spec: str, mode: str, sample_size: int,
                       seed: int = 42) -> dict[str, Any]:
        mode = normalize_mode(mode)
        provider, model = parse_model_spec(model_spec)
        if mode != Mode.ENSEMBLE and not await self.generation.llm.model_available(provider, model):
            raise LLMUnavailableError(f"Модель {model_spec} недоступна")
        samples = sample_dataset(dataset, sample_size, seed)
        start = time.perf_counter()
        preds = await gather_limited([self._predict(r, provider, model, mode) for r in samples], self.concurrency)
        elapsed = time.perf_counter() - start

        computed = self.metrics.compute(
            [p["prediction"] for p in preds], [r["reference"] for r in samples],
            [r["product"].get("features", []) for r in samples], [r["product"].get("style") for r in samples],
        )
        ci = {k: [round(x, 4) for x in bootstrap_ci(v, seed=seed)] for k, v in computed["per_sample"].items()
              if k in COMPOSITE_KEYS}
        actual = sorted({p["model"] for p in preds})
        return {
            "model_name": model_spec,
            "mode": mode,
            "sample_size": len(samples),
            "metrics": computed["metrics"],
            "confidence_intervals": ci,
            "per_sample": computed["per_sample"],
            "evaluation_time": round(elapsed, 2),
            "bertscore_backend": computed["bertscore_backend"],
            "models_actually_used": actual,
            "examples": [{"id": r.get("id"), "prediction": p["prediction"][:600], "reference": r["reference"][:600]}
                         for r, p in list(zip(samples, preds, strict=True))[:3]],
        }

    async def compare(self, dataset: list[dict[str, Any]], models: list[str], mode: str, sample_size: int,
                      seed: int = 42, metric: str = "composite") -> dict[str, Any]:
        results = {}
        for spec in models:
            try:
                results[spec] = await self.evaluate(dataset, spec, mode, sample_size, seed)
            except LLMUnavailableError as exc:
                results[spec] = {"error": str(exc)}
        ok = {k: v for k, v in results.items() if "metrics" in v}
        winner = max(ok, key=lambda k: ok[k]["metrics"].get(metric, 0)) if ok else None
        significance = {}
        if winner and len(ok) > 1:
            per = {k: {**v["per_sample"], "composite": _composite_per_sample(v["per_sample"])} for k, v in ok.items()}
            significance = compare_systems(per, metric if metric in per[winner] else "bleu", winner)
        return {"models": models, "mode": normalize_mode(mode), "sample_size": sample_size, "results": results,
                "winner": winner, "significance": significance}

    # ------------------------- БД -------------------------
    @staticmethod
    async def save_result(session: AsyncSession, result: dict[str, Any]) -> EvaluationResult:
        row = EvaluationResult(
            model_name=result["model_name"], mode=result["mode"], sample_size=result["sample_size"],
            metrics_json=result["metrics"], evaluation_time=result["evaluation_time"],
            details_json={k: result.get(k) for k in ("confidence_intervals", "bertscore_backend",
                                                       "models_actually_used", "examples")},
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return row

    @staticmethod
    async def save_comparison(session: AsyncSession, comparison: dict[str, Any]) -> ModelComparison:
        slim = {k: ({kk: vv for kk, vv in v.items() if kk != "per_sample"} if isinstance(v, dict) else v)
                for k, v in comparison["results"].items()}
        row = ModelComparison(models_compared=comparison["models"], sample_size=comparison["sample_size"],
                              mode=comparison["mode"], results_json={"results": slim,
                                                                     "significance": comparison["significance"]},
                              winner=comparison["winner"])
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return row

    @staticmethod
    async def history(session: AsyncSession, limit: int = 100, model: str | None = None) -> list[EvaluationResult]:
        q = select(EvaluationResult).order_by(EvaluationResult.created_at.desc()).limit(limit)
        if model:
            q = q.where(EvaluationResult.model_name == model)
        return list((await session.execute(q)).scalars())


def _composite_per_sample(per: dict[str, list[float]]) -> list[float]:
    keys = [k for k in COMPOSITE_KEYS if k in per]
    n = len(per[keys[0]]) if keys else 0
    return [sum(per[k][i] for k in keys) / len(keys) for i in range(n)]
