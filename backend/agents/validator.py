"""Validator — формальная проверка описания (rule-based + опциональная LLM-проверка фактов)."""
from __future__ import annotations

import re

from backend.agents import prompts
from backend.agents.base import Agent
from backend.core.constants import PLATFORM_RULES, STYLE_STOPWORDS, Style
from backend.core.models import AgentState
from backend.utils.llm_processing import extract_json
from backend.utils.text_processing import (
    EMOJI_RE,
    URL_RE,
    contains_phrase,
    feature_coverage,
    overused_words,
    repeated_sentences,
    stems,
)

PRICE_RE = re.compile(r"(\d[\d\s]*(?:[.,]\d+)?)\s*(?:₽|руб)", re.IGNORECASE)
CAPS_RE = re.compile(r"\b[А-ЯЁA-Z]{5,}\b")


class Validator(Agent):
    name = "Validator"

    def summarize_output(self, state: AgentState) -> str:
        return f"passed={state.validation_passed}; ошибки: {state.errors}; предупреждения: {state.warnings}"

    def check(self, state: AgentState) -> tuple[list[str], list[str], list[str]]:
        text = state.description or ""
        p = state.product
        rules = PLATFORM_RULES[p.platform]
        errors: list[str] = []
        warnings: list[str] = []

        if not text.strip():
            return ["Описание пустое"], [], list(p.features)

        n = len(text)
        if n < rules["min_chars"]:
            errors.append(f"Слишком короткое описание: {n} симв. (минимум {rules['min_chars']})")
        elif n > rules["max_chars"]:
            errors.append(f"Слишком длинное описание: {n} симв. (максимум {rules['max_chars']})")

        missing = [f for f in p.features if not feature_coverage(f, text)[0]]
        for f in missing:
            errors.append(f"Не упомянута ключевая особенность: «{f}»")

        for phrase in state.forbidden_phrases:
            if contains_phrase(text, phrase):
                errors.append(f"Запрещённая формулировка: «{phrase}»")

        if not rules["allow_urls"] and URL_RE.search(text):
            errors.append(f"Ссылки запрещены на площадке {rules['title']}")
        if not rules["allow_emoji"] and EMOJI_RE.search(text):
            warnings.append(f"Эмодзи нежелательны на площадке {rules['title']}")

        if p.price is not None:
            for m in PRICE_RE.finditer(text):
                try:
                    value = float(m.group(1).replace(" ", "").replace(",", "."))
                except ValueError:
                    continue
                if abs(value - p.price) > 0.5:
                    errors.append(f"Цена в тексте ({m.group(0).strip()}) не совпадает с ценой товара ({p.price:g} ₽)")

        for w in STYLE_STOPWORDS.get(p.style, []):
            if re.search(rf"\b{w}", text, re.IGNORECASE):
                warnings.append(f"Слово «{w}…» не соответствует тону «{p.style.value}»")

        name_stems = [s for s in stems(p.name) if len(s) > 3 and s.isalpha()]
        text_stems = set(stems(text))
        if name_stems and not any(s in text_stems for s in name_stems):
            warnings.append("В тексте не упомянуто название товара")
        if p.brand and p.brand.lower() not in text.lower():
            warnings.append(f"Не упомянут бренд {p.brand}")

        for s in repeated_sentences(text):
            warnings.append(f"Повтор предложения: «{s[:60]}…»")
        overused = overused_words(text)
        if overused:
            warnings.append("Слишком частые слова: " + ", ".join(overused))
        if text.count("!") > 3 and p.style != Style.EMOTIONAL:
            warnings.append("Слишком много восклицательных знаков")
        if len(CAPS_RE.findall(text)) > 2:
            warnings.append("Много слов КАПСОМ")
        return errors, warnings, missing

    async def process(self, state: AgentState) -> AgentState:
        errors, warnings, missing = self.check(state)
        if self.settings.validator_use_llm and self.llm is not None and state.description:
            try:
                resp = await self.call_llm(state, prompts.validator_llm_prompt(state.product, state.description),
                                           system=None, temperature=self.settings.temperature_validator,
                                           task="validate", payload={"description": state.description},
                                           json_mode=True, max_tokens=300)
                data = extract_json(resp.text) or {}
                warnings += [f"LLM-проверка: {i}" for i in data.get("issues", []) if isinstance(i, str)]
            except Exception as exc:
                warnings.append(f"LLM-проверка недоступна: {exc}")
        state.errors = errors
        state.warnings = warnings
        state.missing_features = missing
        state.validation_passed = not errors
        self.last_result = {"passed": state.validation_passed, "errors": errors, "warnings": warnings}
        return state
