"""LLMManager: единая точка вызова LLM с выбором бэкенда (Ollama -> внешние API -> HF -> mock).

Строка модели в формате ``provider:model`` (например ``ollama:qwen2.5:7b``, ``openai:gpt-4o-mini``, ``mock``).
"""
from __future__ import annotations

import logging
import time
from typing import Any

from backend.core.config import Settings
from backend.core.exceptions import LLMError, ProviderNotConfiguredError
from backend.data.api_llm import APILLM, SUPPORTED_PROVIDERS
from backend.data.hf_model import HFModel
from backend.data.llm_base import LLMResponse
from backend.data.mock_llm import MockLLM
from backend.utils.llm_client import OllamaClient
from backend.utils.llm_logger import LLMCallLogger

logger = logging.getLogger(__name__)
KNOWN_PROVIDERS = {"ollama", "hf", "mock"} | SUPPORTED_PROVIDERS


def parse_model_spec(spec: str | None) -> tuple[str | None, str | None]:
    """'ollama:qwen2.5:7b' -> ('ollama', 'qwen2.5:7b'); 'qwen2.5' -> ('ollama', 'qwen2.5'); 'mock' -> ('mock', None)."""
    if not spec:
        return None, None
    spec = spec.strip()
    head, _, rest = spec.partition(":")
    if head in KNOWN_PROVIDERS:
        return head, rest or None
    return "ollama", spec


class LLMManager:
    def __init__(self, settings: Settings, call_logger: LLMCallLogger | None = None,
                 ollama: OllamaClient | None = None, hf: HFModel | None = None, api: APILLM | None = None,
                 mock: MockLLM | None = None):
        self.settings = settings
        self.call_logger = call_logger or LLMCallLogger()
        self.ollama = ollama or OllamaClient(settings.ollama_url, settings.ollama_model, settings.llm_timeout,
                                             settings.llm_max_retries)
        self.hf = hf or HFModel(settings.hf_model, settings.hf_device)
        self.api = api or APILLM(settings)
        self.mock = mock or MockLLM()

    async def provider_available(self, provider: str) -> bool:
        if provider == "ollama":
            return await self.ollama.is_available()
        if provider == "hf":
            return await self.hf.is_available()
        if provider == "mock":
            return True
        return self.api.is_configured(provider)

    async def available_providers(self) -> dict[str, bool]:
        return {p: await self.provider_available(p) for p in self.settings.llm_fallback_order}

    async def resolve(self, provider: str | None) -> list[str]:
        """Возвращает упорядоченный список провайдеров, которые стоит попробовать."""
        default = self.settings.llm_default_provider
        if provider and provider != "auto":
            chain = [provider]
            if self.settings.llm_allow_fallback:
                chain += [p for p in self.settings.llm_fallback_order if p != provider]
        elif default and default != "auto":
            chain = [default] + [p for p in self.settings.llm_fallback_order if p != default]
        else:
            chain = list(self.settings.llm_fallback_order)
        return [p for p in chain if await self.provider_available(p)]

    def default_model(self, provider: str) -> str:
        if provider == "ollama":
            return self.settings.ollama_model
        if provider == "hf":
            return self.settings.hf_model
        if provider == "mock":
            return self.mock.model_name
        return self.api.default_model(provider)

    async def generate(self, prompt: str, *, system: str | None = None, provider: str | None = None,
                       model: str | None = None, temperature: float = 0.7, max_tokens: int | None = None,
                       task: str = "generate", payload: dict[str, Any] | None = None,
                       json_mode: bool = False) -> LLMResponse:
        max_tokens = max_tokens or self.settings.llm_max_tokens
        if provider and provider not in KNOWN_PROVIDERS and provider != "auto":
            raise ProviderNotConfiguredError(f"Неизвестный провайдер: {provider}")
        chain = await self.resolve(provider)
        if not chain:
            raise LLMError("Нет доступных LLM-провайдеров")

        last_exc: Exception | None = None
        for prov in chain:
            # явная модель относится только к явно выбранному провайдеру
            use_model = model if (model and prov == (provider or prov)) else self.default_model(prov)
            params = {"temperature": temperature, "max_tokens": max_tokens, "task": task}
            start = time.perf_counter()
            try:
                resp = await self._call(prov, prompt, use_model, system, temperature, max_tokens, task, payload,
                                        json_mode)
            except Exception as exc:
                last_exc = exc
                await self.call_logger.log(provider=prov, model=use_model, prompt=prompt, response=None,
                                           latency_ms=(time.perf_counter() - start) * 1000, status="error",
                                           parameters=params, error=str(exc))
                logger.warning("Провайдер %s не справился: %s", prov, exc)
                continue
            await self.call_logger.log(provider=prov, model=resp.model, prompt=prompt, response=resp.text,
                                       latency_ms=resp.latency_ms or (time.perf_counter() - start) * 1000,
                                       status="success", tokens=resp.total_tokens, parameters=params,
                                       cached=resp.cached)
            return resp
        raise LLMError(f"Все провайдеры недоступны: {last_exc}")

    async def _call(self, prov, prompt, model, system, temperature, max_tokens, task, payload, json_mode):
        if prov == "ollama":
            return await self.ollama.generate(prompt, model=model, system=system, temperature=temperature,
                                              max_tokens=max_tokens, json_mode=json_mode)
        if prov == "hf":
            return await self.hf.generate(prompt, model=model, system=system, temperature=temperature,
                                          max_tokens=max_tokens)
        if prov == "mock":
            return await self.mock.generate(prompt, model=model, system=system, temperature=temperature,
                                            max_tokens=max_tokens, task=task, payload=payload)
        return await self.api.generate(prompt, prov, model=model, system=system, temperature=temperature,
                                       max_tokens=max_tokens, json_mode=json_mode)
