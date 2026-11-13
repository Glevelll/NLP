"""Генерация описаний: одиночная, мульти-вариант, пакетная, ансамбль."""
from __future__ import annotations

import csv

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from backend.api.deps import get_container, get_current_user, submit_task
from backend.core.exceptions import AppError
from backend.core.models import (
    BatchGenerateRequest,
    EnsembleRequest,
    GenerateRequest,
    GenerationResult,
    MultiGenerateRequest,
    ProductInput,
    normalize_mode,
)
from backend.data.examples_store import parse_csv_bytes
from backend.db.models import User
from backend.services.container import AppContainer
from backend.worker.tasks import batch_generate_task

router = APIRouter(prefix="/generate", tags=["generate"])


def _http(exc: Exception) -> HTTPException:
    code = exc.status_code if isinstance(exc, AppError) else 500
    return HTTPException(code, str(exc))


@router.post("", response_model=GenerationResult)
async def generate(req: GenerateRequest, user: User = Depends(get_current_user),
                   c: AppContainer = Depends(get_container)) -> GenerationResult:
    try:
        return await c.generation.generate(req.product, req.mode, req.provider, req.model, req.use_cache, user.id)
    except Exception as exc:
        raise _http(exc) from exc


@router.post("/multi", response_model=list[GenerationResult])
async def generate_multi(req: MultiGenerateRequest, user: User = Depends(get_current_user),
                         c: AppContainer = Depends(get_container)) -> list[GenerationResult]:
    if len(req.platforms or [1]) * len(req.styles or [1]) > 15:
        raise HTTPException(422, "Не более 15 вариантов за запрос")
    try:
        return await c.generation.generate_multi(req.product, req.platforms, req.styles, req.mode, req.provider,
                                                 req.model, user.id, use_cache=req.use_cache)
    except Exception as exc:
        raise _http(exc) from exc


@router.post("/batch")
async def generate_batch(req: BatchGenerateRequest, user: User = Depends(get_current_user),
                         c: AppContainer = Depends(get_container)) -> dict:
    if req.background:
        return await submit_task(batch_generate_task, c, [p.model_dump(mode="json") for p in req.products],
                                 req.mode, req.provider, req.model, user.id)
    results = await c.generation.generate_batch(req.products, req.mode, req.provider, req.model, user.id)
    return {"count": len(results), "results": [r.model_dump(mode="json") for r in results]}


@router.post("/batch/upload")
async def generate_batch_upload(file: UploadFile = File(...), mode: str = Form("+agents"),
                                provider: str | None = Form(None), background: bool = Form(True),
                                user: User = Depends(get_current_user), c: AppContainer = Depends(get_container)
                                ) -> dict:
    """CSV с колонками name, category, features (через ';'), price, audience, brand, style, platform."""
    try:
        rows = parse_csv_bytes(await file.read())
    except (UnicodeError, csv.Error) as exc:
        raise HTTPException(422, "Не удалось прочитать CSV. Проверьте формат и кодировку UTF-8.") from exc
    products = []
    for i, row in enumerate(rows, 1):
        try:
            products.append(ProductInput(
                name=row["name"], category=row["category"],
                features=[f.strip() for f in (row.get("features") or "").split(";") if f.strip()],
                price=float(row["price"]) if row.get("price") else None, audience=row.get("audience") or None,
                brand=row.get("brand") or None, style=row.get("style") or "нейтральный",
                platform=row.get("platform") or "ozon",
            ))
        except Exception as exc:
            raise HTTPException(422, f"Строка {i}: {exc}") from exc
    req = BatchGenerateRequest(products=products, mode=normalize_mode(mode), provider=provider or None,
                               background=background)
    return await generate_batch(req, user, c)


@router.post("/ensemble", response_model=GenerationResult)
async def generate_ensemble(req: EnsembleRequest, user: User = Depends(get_current_user),
                            c: AppContainer = Depends(get_container)) -> GenerationResult:
    try:
        return await c.generation.ensemble(req.product, req.models, req.use_agents, user.id)
    except Exception as exc:
        raise _http(exc) from exc
