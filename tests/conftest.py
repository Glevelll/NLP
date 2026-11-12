"""Общие фикстуры: изолированные настройки, маленький датасет, контейнер, тестовый клиент API."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.config import Settings  # noqa: E402
from backend.core.models import AgentState, ProductInput  # noqa: E402
from backend.data.knowledge_base import KnowledgeService  # noqa: E402
from backend.data.llm_manager import LLMManager  # noqa: E402
from backend.services.container import AppContainer  # noqa: E402
from scripts.generate_data import generate  # noqa: E402


@pytest.fixture(scope="session")
def dataset_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("data")
    generate(out, kb_size=120, test_size=30, seed=7)
    return out


@pytest.fixture
def settings(tmp_path, dataset_dir) -> Settings:
    return Settings(
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'test.db').as_posix()}",
        embedding_backend="hashing",
        llm_fallback_order=["mock"],
        llm_default_provider="auto",
        ollama_url="http://127.0.0.1:9",
        log_dir=tmp_path / "logs",
        data_dir=dataset_dir,
        results_dir=tmp_path / "results",
        celery_task_always_eager=True,
        bertscore_backend="embedding",
        ensemble_models=["mock:a", "mock:b", "ollama:qwen2.5:7b"],
        jwt_secret="test-secret-key-for-pytest-only",
    )


@pytest.fixture
def product() -> ProductInput:
    return ProductInput(
        name="Беспроводные наушники SoundMax Air 3", category="Электроника",
        features=["активное шумоподавление", "до 30 часов автономной работы", "защита от воды IPX5"],
        price=4990, audience="любители музыки", brand="SoundMax", style="нейтральный", platform="ozon",
    )


@pytest.fixture
def state(product) -> AgentState:
    return AgentState(product=product)


@pytest.fixture
def llm(settings) -> LLMManager:
    return LLMManager(settings)


@pytest_asyncio.fixture
async def container(settings):
    c = AppContainer(settings, knowledge=KnowledgeService(settings))
    await c.startup()
    yield c
    await c.shutdown()


@pytest.fixture
def task_env(settings, monkeypatch):
    """Celery-задачи берут настройки через get_settings() и общий KnowledgeService — подменяем на тестовые."""
    from backend.worker import tasks

    monkeypatch.setattr(tasks, "get_settings", lambda: settings)
    monkeypatch.setattr("backend.services.container.get_knowledge_service", lambda s: KnowledgeService(settings))
    return settings


@pytest.fixture
def api(settings, task_env):
    from backend.api.main import create_app

    container = AppContainer(settings, knowledge=KnowledgeService(settings))
    app = create_app(settings, container)
    with TestClient(app) as client:
        client.container = container
        yield client


def auth_headers(client: TestClient, username: str = "admin", password: str = "secret123") -> dict[str, str]:
    client.post("/api/v1/auth/register", json={"email": f"{username}@example.com", "username": username,
                                               "password": password})
    token = client.post("/api/v1/auth/login", data={"username": username, "password": password}).json()
    return {"Authorization": f"Bearer {token['access_token']}"}


@pytest.fixture
def admin_headers(api) -> dict[str, str]:
    return auth_headers(api)
