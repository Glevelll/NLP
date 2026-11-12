import json

import httpx
import pytest
from sqlalchemy import select

from backend.core.exceptions import LLMError, LLMTimeoutError, LLMUnavailableError, ProviderNotConfiguredError
from backend.data.api_llm import APILLM
from backend.data.hf_model import HFModel
from backend.data.llm_manager import LLMManager, parse_model_spec
from backend.data.mock_llm import MockLLM
from backend.db.models import LLMCall
from backend.db.session import Database
from backend.utils.llm_client import OllamaClient
from backend.utils.llm_logger import LLMCallLogger
from backend.utils.tracing import trace_agent, trace_request


def ollama_transport(status=200, body=None, exc=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if exc:
            raise exc
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen2.5:7b"}]})
        sent = json.loads(request.content)
        assert sent["stream"] is False and "temperature" in sent["options"]
        return httpx.Response(status, json=body or {"response": "Описание: Отличный товар", "prompt_eval_count": 10,
                                                    "eval_count": 5})

    return httpx.MockTransport(handler)


async def test_ollama_success_and_models():
    client = OllamaClient("http://ollama", transport=ollama_transport(), retries=0)
    resp = await client.generate("промпт", system="sys", json_mode=True)
    assert resp.text == "Отличный товар" and resp.total_tokens == 15 and resp.provider == "ollama"
    assert await client.is_available() and await client.list_models() == ["qwen2.5:7b"]


@pytest.mark.parametrize("status,error", [(404, LLMError), (500, LLMUnavailableError), (400, LLMError)])
async def test_ollama_http_errors(status, error):
    client = OllamaClient("http://ollama", transport=ollama_transport(status=status, body={"error": "x"}), retries=0)
    with pytest.raises(error):
        await client.generate("p")


async def test_ollama_timeout_and_unavailable():
    client = OllamaClient("http://ollama", transport=ollama_transport(exc=httpx.ReadTimeout("t")), retries=0)
    with pytest.raises(LLMTimeoutError):
        await client.generate("p")
    down = OllamaClient("http://ollama", transport=ollama_transport(exc=httpx.ConnectError("no")), retries=0)
    with pytest.raises(LLMUnavailableError):
        await down.generate("p")
    assert not await down.is_available()


def api_transport(provider: str, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        if status != 200:
            return httpx.Response(status, json={"error": "x"})
        body = json.loads(request.content)
        if provider == "anthropic":
            assert request.headers["x-api-key"] == "key" and body["system"] == "sys"
            return httpx.Response(200, json={"content": [{"type": "text", "text": "Ответ Claude"}],
                                             "usage": {"input_tokens": 3, "output_tokens": 4}})
        assert request.headers["Authorization"] == "Bearer key"
        return httpx.Response(200, json={"choices": [{"message": {"content": "Ответ GPT"}}],
                                         "usage": {"prompt_tokens": 1, "completion_tokens": 2}})

    return httpx.MockTransport(handler)


@pytest.mark.parametrize("provider,expected", [("openai", "Ответ GPT"), ("anthropic", "Ответ Claude"),
                                               ("groq", "Ответ GPT"), ("openrouter", "Ответ GPT")])
async def test_api_llm_providers(settings, provider, expected):
    s = settings.model_copy(update={f"{provider}_api_key": "key", "llm_max_retries": 0})
    api = APILLM(s, transport=api_transport(provider))
    assert provider in api.configured_providers()
    resp = await api.generate("привет", provider, system="sys", json_mode=True)
    assert resp.text == expected and resp.total_tokens in (3, 7)


async def test_api_llm_errors(settings):
    api = APILLM(settings)
    with pytest.raises(ProviderNotConfiguredError):
        await api.generate("p", "openai")
    with pytest.raises(ProviderNotConfiguredError):
        await api.generate("p", "unknown")
    s = settings.model_copy(update={"openai_api_key": "key", "llm_max_retries": 0})
    with pytest.raises(LLMUnavailableError):
        await APILLM(s, transport=api_transport("openai", 503)).generate("p", "openai")
    with pytest.raises(LLMError):
        await APILLM(s, transport=api_transport("openai", 401)).generate("p", "openai")
    with pytest.raises(LLMError):
        APILLM._parse("openai", {"bad": 1})


async def test_hf_model_with_fake_pipeline(monkeypatch):
    hf = HFModel("fake-model")
    calls = {"n": 0}

    class FakePipe:
        task = "text-generation"

        def __call__(self, inputs, **kwargs):
            calls["n"] += 1
            assert inputs[-1]["role"] == "user"
            return [{"generated_text": [{"role": "assistant", "content": "Сгенерированный текст"}]}]

    monkeypatch.setattr(hf, "_load", lambda name: FakePipe())
    r1 = await hf.generate("промпт", system="sys", temperature=0.5)
    r2 = await hf.generate("промпт", system="sys", temperature=0.5)
    assert r1.text == "Сгенерированный текст" and r2.cached and calls["n"] == 1


async def test_hf_t5_and_load_error(monkeypatch):
    hf = HFModel("google/flan-t5-small")

    class T5Pipe:
        task = "text2text-generation"

        def __call__(self, inputs, **kwargs):
            assert isinstance(inputs, str)
            return [{"generated_text": "t5 ответ"}]

    monkeypatch.setattr(hf, "_load", lambda name: T5Pipe())
    assert (await hf.generate("p", temperature=0)).text == "t5 ответ"

    broken = HFModel("x")
    monkeypatch.setattr(broken, "_load", lambda name: (_ for _ in ()).throw(OSError("нет модели")))
    with pytest.raises(LLMUnavailableError):
        await broken.generate("p")


async def test_mock_llm_tasks(product):
    mock = MockLLM()
    payload = {"product": product.model_dump(mode="json")}
    text = (await mock.generate("p", task="generate", payload=payload)).text
    assert product.name in text and "Преимущества" in text
    critic = json.loads((await mock.generate("p", task="critic", payload={**payload, "description": text,
                                                                          "forbidden_phrases": []})).text)
    assert critic["overall"] > 7
    bad = json.loads((await mock.generate("p", task="critic", payload={**payload, "description": "Лучший в мире",
                                                                       "forbidden_phrases": ["лучший в мире"]})).text)
    assert bad["suggestions"] and bad["overall"] < critic["overall"]
    edited = (await mock.generate("p", task="edit", payload={**payload, "description": "Лучший в мире товар.",
                                                             "forbidden_phrases": ["лучший в мире"]})).text
    assert "лучший в мире" not in edited.lower() and "шумоподавление" in edited.lower()
    assert json.loads((await mock.generate("p", task="validate")).text) == {"issues": []}
    assert (await mock.generate("строка1\nстрока2", task="other")).text == "строка2"


def test_parse_model_spec():
    assert parse_model_spec("ollama:qwen2.5:7b") == ("ollama", "qwen2.5:7b")
    assert parse_model_spec("openai:gpt-4o-mini") == ("openai", "gpt-4o-mini")
    assert parse_model_spec("mock") == ("mock", None)
    assert parse_model_spec("qwen2.5") == ("ollama", "qwen2.5")
    assert parse_model_spec(None) == (None, None)


async def test_llm_manager_fallback_and_logging(settings, tmp_path):
    db = Database(f"sqlite+aiosqlite:///{(tmp_path / 'llm.db').as_posix()}")
    await db.create_all()
    s = settings.model_copy(update={"llm_fallback_order": ["ollama", "mock"], "llm_max_retries": 0})
    down = OllamaClient("http://ollama", transport=ollama_transport(exc=httpx.ConnectError("no")), retries=0)
    manager = LLMManager(s, LLMCallLogger(db.session_factory), ollama=down)
    with trace_request("rid-1"), trace_agent("Critic") as calls:
        resp = await manager.generate("строка", task="other")
    assert resp.provider == "mock" and calls and calls[0]["agent_name"] == "Critic"
    async with db.session_factory() as session:
        rows = (await session.execute(select(LLMCall))).scalars().all()
    assert rows[-1].request_id == "rid-1" and rows[-1].status == "success" and rows[-1].prompt_hash

    # Ollama «доступен», но падает при генерации -> ошибка логируется, срабатывает fallback
    flaky = OllamaClient("http://ollama", transport=ollama_transport(status=500), retries=0)
    flaky._available, flaky._checked_at = True, 1e18
    manager2 = LLMManager(s, LLMCallLogger(db.session_factory), ollama=flaky)
    resp2 = await manager2.generate("p", provider="ollama", task="other")
    assert resp2.provider == "mock"
    async with db.session_factory() as session:
        statuses = [r.status for r in (await session.execute(select(LLMCall))).scalars()]
    assert "error" in statuses
    assert (await manager2.available_providers())["mock"]
    assert manager2.default_model("hf") == s.hf_model and manager2.default_model("mock") == "rule-based-v1"
    with pytest.raises(ProviderNotConfiguredError):
        await manager2.generate("p", provider="nope")
    strict = LLMManager(s.model_copy(update={"llm_fallback_order": ["ollama"], "llm_allow_fallback": False}),
                        ollama=down)
    with pytest.raises(LLMError):
        await strict.generate("p")
    await db.dispose()


async def test_llm_manager_routes_providers(settings):
    s = settings.model_copy(update={"openai_api_key": "key", "llm_max_retries": 0, "llm_default_provider": "openai",
                                    "llm_fallback_order": ["openai", "hf", "mock"]})
    manager = LLMManager(s, api=APILLM(s, transport=api_transport("openai")))
    assert (await manager.generate("p")).provider == "openai"
    assert await manager.resolve("hf") in (["hf", "openai", "mock"], ["openai", "mock"])
