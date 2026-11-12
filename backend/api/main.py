"""Точка входа FastAPI."""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routes import admin, auth, evaluate, generate, hitl, monitoring, requests
from backend.core.config import Settings, get_settings
from backend.core.exceptions import AppError
from backend.services.container import AppContainer
from backend.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, container: AppContainer | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_dir, settings.log_level, settings.log_json, settings.log_levels)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.container = container or AppContainer(settings)
        await app.state.container.startup()
        logger.info("Приложение запущено (%s)", settings.environment)
        yield
        await app.state.container.shutdown()

    app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan,
                  description="Мультиагентный генератор продающих описаний товаров (RAG + LLM + оценка качества)")
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        logger.info("%s %s -> %d (%.0f мс)", request.method, request.url.path, response.status_code,
                    (time.perf_counter() - start) * 1000)
        return response

    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    @app.get("/healthz", tags=["monitoring"])
    async def healthz() -> dict:
        return {"status": "ok"}

    for module in (auth, generate, admin, evaluate, hitl, requests, monitoring):
        app.include_router(module.router, prefix=settings.api_prefix)
    return app


app = create_app()
