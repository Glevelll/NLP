"""Клиенты внешних LLM API: OpenAI, Anthropic, OpenRouter, Groq.

Единый интерфейс: ``await api.generate(prompt, provider="openai", model=..., temperature=...)``.
OpenRouter и Groq совместимы с OpenAI Chat Completions, поэтому используют общий код.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from backend.core.config import Settings
from backend.core.exceptions import (
    LLMError,
    LLMTimeoutError,
    LLMUnavailableError,
    ProviderNotConfiguredError,
)
from backend.data.llm_base import LLMResponse
from backend.utils.async_utils import async_retry
from backend.utils.llm_processing import clean_response

OPENAI_COMPATIBLE = {"openai", "openrouter", "groq"}
SUPPORTED_PROVIDERS = OPENAI_COMPATIBLE | {"anthropic"}


@dataclass
class ProviderConfig:
    api_key: str | None
    base_url: str
    default_model: str


class APILLM:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.timeout = settings.llm_timeout
        self.retries = settings.llm_max_retries
        self._transport = transport
        self.providers: dict[str, ProviderConfig] = {
            "openai": ProviderConfig(settings.openai_api_key, settings.openai_base_url, settings.openai_model),
            "anthropic": ProviderConfig(settings.anthropic_api_key, settings.anthropic_base_url,
                                        settings.anthropic_model),
            "openrouter": ProviderConfig(settings.openrouter_api_key, settings.openrouter_base_url,
                                         settings.openrouter_model),
            "groq": ProviderConfig(settings.groq_api_key, settings.groq_base_url, settings.groq_model),
        }

    def is_configured(self, provider: str) -> bool:
        cfg = self.providers.get(provider)
        return bool(cfg and cfg.api_key)

    def configured_providers(self) -> list[str]:
        return [p for p in self.providers if self.is_configured(p)]

    def default_model(self, provider: str) -> str:
        return self.providers[provider].default_model

    async def generate(self, prompt: str, provider: str, *, model: str | None = None, system: str | None = None,
                       temperature: float = 0.7, max_tokens: int = 800, **kwargs: Any) -> LLMResponse:
        if provider not in SUPPORTED_PROVIDERS:
            raise ProviderNotConfiguredError(f"Неизвестный провайдер: {provider}")
        if not self.is_configured(provider):
            raise ProviderNotConfiguredError(f"Не задан API-ключ для провайдера {provider}")
        cfg = self.providers[provider]
        model = model or cfg.default_model

        @async_retry(retries=self.retries, delay=1.0, exceptions=(LLMUnavailableError, LLMTimeoutError))
        async def _call() -> LLMResponse:
            start = time.perf_counter()
            if provider == "anthropic":
                url, headers, body = self._anthropic_request(cfg, model, prompt, system, temperature, max_tokens)
            else:
                url, headers, body = self._openai_request(provider, cfg, model, prompt, system, temperature,
                                                          max_tokens, kwargs.get("json_mode", False))
            try:
                async with httpx.AsyncClient(timeout=self.timeout, transport=self._transport) as c:
                    r = await c.post(url, headers=headers, json=body)
            except httpx.TimeoutException as exc:
                raise LLMTimeoutError(f"{provider}: таймаут") from exc
            except httpx.HTTPError as exc:
                raise LLMUnavailableError(f"{provider}: ошибка соединения {exc}") from exc
            if r.status_code in (429, 500, 502, 503, 504, 529):
                raise LLMUnavailableError(f"{provider}: HTTP {r.status_code}")
            if r.status_code >= 400:
                raise LLMError(f"{provider}: HTTP {r.status_code} {r.text[:300]}")
            data = r.json()
            text, pt, ct = self._parse(provider, data)
            return LLMResponse(text=clean_response(text), provider=provider, model=model, prompt_tokens=pt,
                               completion_tokens=ct, latency_ms=(time.perf_counter() - start) * 1000)

        return await _call()

    @staticmethod
    def _openai_request(provider, cfg, model, prompt, system, temperature, max_tokens, json_mode):
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature,
                                "max_tokens": max_tokens}
        if json_mode and provider == "openai":
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
        if provider == "openrouter":
            headers["X-Title"] = "Product Description Generator"
        return f"{cfg.base_url.rstrip('/')}/chat/completions", headers, body

    @staticmethod
    def _anthropic_request(cfg, model, prompt, system, temperature, max_tokens):
        body: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "temperature": temperature,
                                "messages": [{"role": "user", "content": prompt}]}
        if system:
            body["system"] = system
        headers = {"x-api-key": cfg.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        return f"{cfg.base_url.rstrip('/')}/messages", headers, body

    @staticmethod
    def _parse(provider: str, data: dict) -> tuple[str, int | None, int | None]:
        try:
            if provider == "anthropic":
                text = "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
                usage = data.get("usage", {})
                return text, usage.get("input_tokens"), usage.get("output_tokens")
            text = data["choices"][0]["message"]["content"] or ""
            usage = data.get("usage") or {}
            return text, usage.get("prompt_tokens"), usage.get("completion_tokens")
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"{provider}: неожиданный формат ответа") from exc
