"""Асинхронная обёртка над transformers.pipeline с ленивой загрузкой и кэшированием результатов."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from typing import Any

from backend.core.exceptions import LLMUnavailableError
from backend.data.llm_base import LLMResponse
from backend.utils.llm_processing import clean_response

logger = logging.getLogger(__name__)


class HFModel:
    provider = "hf"

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", device: int = -1, cache_size: int = 256):
        self.model_name = model_name
        self.device = device
        self._pipelines: dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._cache: OrderedDict[str, LLMResponse] = OrderedDict()
        self._cache_size = cache_size

    async def is_available(self) -> bool:
        try:
            import transformers  # noqa: F401
        except ImportError:
            return False
        return True

    def _load(self, model_name: str):
        from transformers import pipeline

        task = "text2text-generation" if "t5" in model_name.lower() else "text-generation"
        logger.info("Загрузка HF-модели %s (%s)", model_name, task)
        return pipeline(task, model=model_name, device=self.device)

    async def _get_pipeline(self, model_name: str):
        async with self._lock:
            if model_name not in self._pipelines:
                try:
                    self._pipelines[model_name] = await asyncio.to_thread(self._load, model_name)
                except Exception as exc:
                    raise LLMUnavailableError(f"Не удалось загрузить HF-модель {model_name}: {exc}") from exc
            return self._pipelines[model_name]

    def _cache_key(self, model: str, prompt: str, system: str | None, temperature: float, max_tokens: int) -> str:
        raw = f"{model}|{system}|{temperature}|{max_tokens}|{prompt}"
        return hashlib.sha256(raw.encode()).hexdigest()

    async def generate(self, prompt: str, *, model: str | None = None, system: str | None = None,
                       temperature: float = 0.7, max_tokens: int = 800, **kwargs: Any) -> LLMResponse:
        model = model or self.model_name
        key = self._cache_key(model, prompt, system, temperature, max_tokens)
        if key in self._cache:
            self._cache.move_to_end(key)
            cached = self._cache[key]
            return LLMResponse(**{**cached.__dict__, "cached": True})

        pipe = await self._get_pipeline(model)
        start = time.perf_counter()
        is_t5 = getattr(pipe, "task", "") == "text2text-generation"
        gen_kwargs: dict[str, Any] = {"max_new_tokens": max_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        if is_t5:
            inputs: Any = (system + "\n\n" if system else "") + prompt
        else:
            messages = ([{"role": "system", "content": system}] if system else []) + [
                {"role": "user", "content": prompt}
            ]
            inputs = messages
            gen_kwargs["return_full_text"] = False

        out = await asyncio.to_thread(pipe, inputs, **gen_kwargs)
        generated = out[0].get("generated_text", "")
        if isinstance(generated, list):  # формат чата: список сообщений
            generated = generated[-1].get("content", "") if generated else ""
        resp = LLMResponse(text=clean_response(str(generated)), provider=self.provider, model=model,
                           latency_ms=(time.perf_counter() - start) * 1000)
        self._cache[key] = resp
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return resp
