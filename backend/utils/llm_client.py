"""Асинхронный клиент Ollama (HTTP API /api/generate) с таймаутами и повторами."""
from __future__ import annotations

import time
from typing import Any

import httpx

from backend.core.exceptions import LLMError, LLMTimeoutError, LLMUnavailableError
from backend.data.llm_base import LLMResponse
from backend.utils.async_utils import async_retry
from backend.utils.llm_processing import clean_response


class OllamaClient:
    provider = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434", default_model: str = "qwen2.5:7b",
                 timeout: float = 120.0, retries: int = 2, transport: httpx.AsyncBaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout
        self.retries = retries
        self._transport = transport
        self._available: bool | None = None
        self._checked_at = 0.0

    def _client(self, timeout: float | None = None) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.base_url, timeout=timeout or self.timeout, transport=self._transport)

    async def is_available(self, ttl: float = 30.0) -> bool:
        """Проверка доступности с кэшированием результата на ttl секунд."""
        if self._available is not None and time.monotonic() - self._checked_at < ttl:
            return self._available
        try:
            async with self._client(timeout=3.0) as c:
                r = await c.get("/api/tags")
                self._available = r.status_code == 200
        except httpx.HTTPError:
            self._available = False
        self._checked_at = time.monotonic()
        return self._available

    async def list_models(self) -> list[str]:
        async with self._client(timeout=5.0) as c:
            r = await c.get("/api/tags")
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]

    async def generate(self, prompt: str, *, model: str | None = None, system: str | None = None,
                       temperature: float = 0.7, max_tokens: int = 800, **kwargs: Any) -> LLMResponse:
        model = model or self.default_model

        @async_retry(retries=self.retries, delay=1.0, exceptions=(LLMUnavailableError, LLMTimeoutError))
        async def _call() -> LLMResponse:
            body: dict[str, Any] = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            }
            if system:
                body["system"] = system
            if kwargs.get("json_mode"):
                body["format"] = "json"
            start = time.perf_counter()
            try:
                async with self._client() as c:
                    r = await c.post("/api/generate", json=body)
            except httpx.TimeoutException as exc:
                raise LLMTimeoutError(f"Ollama не ответил за {self.timeout}с") from exc
            except httpx.HTTPError as exc:
                raise LLMUnavailableError(f"Ollama недоступен: {exc}") from exc
            if r.status_code == 404:
                raise LLMError(f"Модель {model} не найдена в Ollama (выполните `ollama pull {model}`)")
            if r.status_code >= 500:
                raise LLMUnavailableError(f"Ollama вернул {r.status_code}")
            if r.status_code >= 400:
                raise LLMError(f"Ollama вернул {r.status_code}: {r.text[:200]}")
            data = r.json()
            return LLMResponse(
                text=clean_response(data.get("response", "")),
                provider=self.provider,
                model=model,
                prompt_tokens=data.get("prompt_eval_count"),
                completion_tokens=data.get("eval_count"),
                latency_ms=(time.perf_counter() - start) * 1000,
                raw={"done_reason": data.get("done_reason")},
            )

        return await _call()
