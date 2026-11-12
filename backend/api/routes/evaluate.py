"""Оценка качества: запуск, сравнение моделей, пакетная оценка, история."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_container, get_current_user, get_db, submit_task
from backend.core.exceptions import AppError
from backend.core.models import BatchEvaluateRequest, EvaluateRequest, normalize_mode
from backend.data.examples_store import read_records
from backend.db.models import ModelComparison, User
from backend.evaluation.evaluator import Evaluator, load_dataset
from backend.services.container import AppContainer
from backend.worker.tasks import batch_evaluate_task, compare_models_task, evaluate_model_task

router = APIRouter(prefix="/evaluate", tags=["evaluate"])


def _evaluator(c: AppContainer) -> tuple[Evaluator, list[dict]]:
    data_dir = Path(c.settings.data_dir)
    test_path = data_dir / "test.json"
    if not test_path.exists():
        raise HTTPException(404, "Тестовый набор не найден — выполните scripts/generate_data.py")
    ev = Evaluator(c.generation, ensemble_models=c.settings.ensemble_models)
    kb_path = data_dir / "knowledge_base.json"
    if kb_path.exists():
        ev.fit_style_classifier(read_records(kb_path))
    return ev, load_dataset(test_path)


@router.post("")
async def evaluate(req: EvaluateRequest, _: User = Depends(get_current_user), db: AsyncSession = Depends(get_db),
                   c: AppContainer = Depends(get_container)) -> dict:
    if req.background:
        return await submit_task(evaluate_model_task, c, req.model, req.mode, req.sample_size, req.seed)
    ev, data = _evaluator(c)
    try:
        result = await ev.evaluate(data, req.model, req.mode, req.sample_size, req.seed)
    except AppError as exc:
        raise HTTPException(exc.status_code, exc.message) from exc
    row = await ev.save_result(db, result)
    return {"id": row.id, "created_at": row.created_at, **{k: v for k, v in result.items() if k != "per_sample"}}


@router.get("/compare")
async def compare(models: str = Query(..., description="через запятую: mock,ollama:qwen2.5:7b"),
                  sample_size: int = Query(20, ge=1, le=500), mode: str = "baseline", seed: int = 42,
                  background: bool = False, _: User = Depends(get_current_user), db: AsyncSession = Depends(get_db),
                  c: AppContainer = Depends(get_container)) -> dict:
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    mode = normalize_mode(mode)
    if background:
        return await submit_task(compare_models_task, c, model_list, mode, sample_size, seed)
    ev, data = _evaluator(c)
    comparison = await ev.compare(data, model_list, mode, sample_size, seed)
    for r in comparison["results"].values():
        if "metrics" in r:
            await ev.save_result(db, r)
    row = await ev.save_comparison(db, comparison)
    return {"id": row.id, "winner": comparison["winner"], "mode": mode, "sample_size": sample_size,
            "significance": comparison["significance"],
            "results": {k: {kk: vv for kk, vv in v.items() if kk != "per_sample"}
                        for k, v in comparison["results"].items()}}


@router.post("/batch")
async def batch(req: BatchEvaluateRequest, _: User = Depends(get_current_user),
                c: AppContainer = Depends(get_container)) -> dict:
    if req.background:
        return await submit_task(batch_evaluate_task, c, req.models, req.modes, req.sample_sizes, req.seed)
    ev, data = _evaluator(c)
    out = []
    async with c.db.session_factory() as db:
        for size in req.sample_sizes:
            for model in req.models:
                for mode in req.modes:
                    try:
                        res = await ev.evaluate(data, model, mode, size, req.seed)
                        row = await ev.save_result(db, res)
                        out.append({"id": row.id, **{k: v for k, v in res.items() if k != "per_sample"}})
                    except AppError as exc:
                        out.append({"model_name": model, "mode": mode, "sample_size": size, "error": exc.message})
    return {"results": out}


@router.get("/history")
async def history(limit: int = Query(100, le=1000), model: str | None = None, _: User = Depends(get_current_user),
                  db: AsyncSession = Depends(get_db)) -> list[dict]:
    rows = await Evaluator.history(db, limit, model)
    return [{"id": r.id, "model_name": r.model_name, "mode": r.mode, "sample_size": r.sample_size,
             "metrics": r.metrics_json, "evaluation_time": r.evaluation_time, "created_at": r.created_at,
             "details": r.details_json} for r in rows]


@router.get("/comparisons")
async def comparisons(limit: int = 20, _: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
                      ) -> list[dict]:
    rows = (await db.execute(select(ModelComparison).order_by(ModelComparison.created_at.desc()).limit(limit))
            ).scalars()
    return [{"id": r.id, "models": r.models_compared, "sample_size": r.sample_size, "mode": r.mode,
             "winner": r.winner, "results": r.results_json, "created_at": r.created_at} for r in rows]
