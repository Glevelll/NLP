"""Детерминированный rule-based «LLM» для офлайн-режима, тестов и демонстрации без GPU.

Используется как последний вариант в цепочке fallback. Получает структурированный payload
(те же данные, что попадают в промпт) и имитирует поведение модели: генерирует описание,
оценивает его как Critic, исправляет как Editor.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import time
from typing import Any

from backend.core.constants import STYLE_STOPWORDS
from backend.data.llm_base import LLMResponse
from backend.utils.text_processing import contains_phrase, feature_coverage, split_sentences

OPENERS: dict[str, list[str]] = {
    "нейтральный": ["{name} — практичное решение для повседневных задач.",
                    "Представляем {name}: продуманный товар без лишних деталей."],
    "эмоциональный": ["Встречайте {name} — вещь, которая подарит настоящее удовольствие!",
                      "{name} создан, чтобы каждый день приносил радость."],
    "премиальный": ["{name} — воплощение безупречного качества и внимания к деталям.",
                    "{name} создан для тех, кто ценит изысканность и надёжность."],
    "молодёжный": ["{name} — твой новый must-have.", "Знакомься: {name}. Просто, стильно и по делу."],
    "экспертный": ["{name} — инженерно выверенное решение с понятными характеристиками.",
                   "Разберём, чем {name} выделяется среди аналогов."],
}
BENEFITS = [
    "это заметно упрощает ежедневное использование", "вы почувствуете разницу с первого дня",
    "это экономит время и силы", "это обеспечивает комфорт и уверенность", "товар прослужит долго",
    "это удобно дома и в поездках", "результат радует стабильно", "это продуманная деталь для вашего удобства",
]
CLOSERS: dict[str, str] = {
    "ozon": "Оформите заказ на Ozon — доставка в удобный пункт выдачи.",
    "wildberries": "Добавьте в корзину и забирайте в ближайшем пункте выдачи Wildberries.",
    "site": "Закажите сейчас на нашем сайте — мы бережно упакуем и быстро доставим ваш заказ.",
}


def _rng(*parts: str) -> random.Random:
    seed = int(hashlib.md5("|".join(parts).encode()).hexdigest()[:8], 16)
    return random.Random(seed)


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


NAME_RE = re.compile(r"Название:\s*(.+)")


def _borrow_from_example(example: dict[str, Any], name: str) -> tuple[str | None, str | None]:
    """Перенимает у найденного примера приёмы: вводную фразу (с подстановкой названия) и финальный призыв.
    Фрагменты с фактами чужого товара не переносятся."""
    desc = example.get("description", "")
    m = NAME_RE.search(example.get("source", ""))
    ex_name = m.group(1).strip() if m else None
    paragraphs = [p.strip() for p in desc.split("\n\n") if p.strip()]
    if not paragraphs:
        return None, None
    opener = None
    first = split_sentences(paragraphs[0])[0] if split_sentences(paragraphs[0]) else ""
    if ex_name and ex_name in first:
        opener = first.replace(ex_name, name)
    closer = paragraphs[-1] if not re.search(r"\d", paragraphs[-1]) and (not ex_name or ex_name not in paragraphs[-1]) \
        else None
    return opener, closer


class MockLLM:
    provider = "mock"
    model_name = "rule-based-v1"

    async def is_available(self) -> bool:
        return True

    async def generate(self, prompt: str, *, model: str | None = None, system: str | None = None,
                       temperature: float = 0.7, max_tokens: int = 800, task: str = "generate",
                       payload: dict[str, Any] | None = None, **kwargs: Any) -> LLMResponse:
        start = time.perf_counter()
        payload = payload or {}
        handler = {
            "generate": self._generate,
            "critic": self._critic,
            "edit": self._edit,
            "validate": self._validate,
        }.get(task, self._echo)
        text = handler(payload, prompt)
        return LLMResponse(text=text, provider=self.provider, model=model or self.model_name,
                           prompt_tokens=len(prompt.split()), completion_tokens=len(text.split()),
                           latency_ms=(time.perf_counter() - start) * 1000)

    # ------------------------------------------------------------------
    def _generate(self, payload: dict[str, Any], prompt: str) -> str:
        p = payload.get("product") or {}
        name = p.get("name", "Товар")
        style = p.get("style", "нейтральный")
        platform = p.get("platform", "ozon")
        features: list[str] = list(p.get("features") or [])
        rng = _rng(name, style, platform, str(payload.get("variant", "")))
        phrases: list[str] = list(payload.get("selling_phrases") or [])
        benefits: dict[str, str] = dict(payload.get("benefits") or {})
        examples: list[dict[str, Any]] = list(payload.get("examples") or [])

        # Без справочника и примеров «модель» иногда упускает особенности — как и маленькие LLM
        if not phrases and not examples and len(features) > 3 and rng.random() < 0.5:
            features = features[:-1]

        borrowed_opener, borrowed_closer = _borrow_from_example(examples[0], name) if examples else (None, None)
        opener = borrowed_opener or rng.choice(OPENERS.get(style, OPENERS["нейтральный"])).format(name=name)
        feature_lines = []
        for f in features:
            benefit = benefits.get(f) or rng.choice(BENEFITS)
            feature_lines.append(f"{_cap(f)} — {benefit.rstrip('.')}.")

        parts: list[str] = [opener]
        if p.get("brand"):
            parts.append(f"Бренд {p['brand']} отвечает за качество каждой детали.")
        if phrases:
            parts.append(f"Коротко о главном: {phrases[0].rstrip('.')}.")

        if platform == "ozon":
            body = "Преимущества:\n" + "\n".join(f"• {line}" for line in feature_lines)
        elif platform == "wildberries":
            body = "\n".join(feature_lines)
        else:
            body = " ".join(feature_lines)
        parts.append(body)

        if p.get("audience"):
            parts.append(f"Подойдёт для: {p['audience']}.")
        if p.get("price"):
            parts.append(f"Стоимость — {float(p['price']):g} ₽.")
        parts.append(borrowed_closer or CLOSERS.get(platform, CLOSERS["site"]))
        return "\n\n".join(parts)

    def _critic(self, payload: dict[str, Any], prompt: str) -> str:
        desc = payload.get("description", "")
        p = payload.get("product") or {}
        features = p.get("features") or []
        covered = [feature_coverage(f, desc)[0] for f in features] or [True]
        coverage = sum(covered) / len(covered)
        length_ok = 1.0 if 350 <= len(desc) <= 6000 else 0.6
        forbidden = [f for f in payload.get("forbidden_phrases", []) if contains_phrase(desc, f)]
        stop = [w for w in STYLE_STOPWORDS.get(p.get("style", ""), []) if w in desc.lower()]
        audience = p.get("audience")
        audience_ok = 1.0 if not audience or audience.lower()[:12] in desc.lower() else 0.5

        scores = {
            "attractiveness": round(6.0 + 3.0 * length_ok * coverage, 1),
            "persuasiveness": round(5.5 + 3.5 * coverage - 1.5 * bool(forbidden), 1),
            "audience_relevance": round(5.0 + 4.0 * audience_ok, 1),
            "factual_accuracy": round(4.0 + 5.5 * coverage - 2.0 * bool(forbidden), 1),
            "style_match": round(8.5 - 2.0 * bool(stop), 1),
        }
        suggestions = []
        if coverage < 1:
            suggestions.append("Упомяните все ключевые особенности товара")
        if audience_ok < 1:
            suggestions.append("Явно обратитесь к целевой аудитории")
        if forbidden:
            suggestions.append("Уберите запрещённые формулировки: " + ", ".join(forbidden))
        if stop:
            suggestions.append("Слова не соответствуют тону бренда: " + ", ".join(stop))
        overall = round(sum(scores.values()) / len(scores), 2)
        return json.dumps({"scores": scores, "overall": overall,
                           "reasoning": f"Покрытие особенностей {coverage:.0%}, длина {len(desc)} символов.",
                           "suggestions": suggestions}, ensure_ascii=False)

    def _edit(self, payload: dict[str, Any], prompt: str) -> str:
        text = payload.get("description", "")
        p = payload.get("product") or {}
        for phrase in payload.get("forbidden_phrases", []):
            text = re.sub(re.escape(phrase), "", text, flags=re.IGNORECASE)
        for w in STYLE_STOPWORDS.get(p.get("style", ""), []):
            text = re.sub(rf"\b{w}\w*", "", text, flags=re.IGNORECASE)
        missing = payload.get("missing_features") or [
            f for f in p.get("features") or [] if not feature_coverage(f, text)[0]
        ]
        additions = [f"{_cap(f)} — ещё одно важное преимущество модели." for f in missing]
        audience = p.get("audience")
        if audience and audience.lower()[:12] not in text.lower():
            additions.append(f"Подойдёт для: {audience}.")
        if additions:
            paragraphs = text.split("\n\n")
            insert_at = max(len(paragraphs) - 1, 1)
            paragraphs.insert(insert_at, " ".join(additions))
            text = "\n\n".join(paragraphs)
        if len(text) < 350:
            text += "\n\n" + CLOSERS.get(p.get("platform", "site"), CLOSERS["site"])
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r" +([.,!?])", r"\1", text)
        return text.strip()

    def _validate(self, payload: dict[str, Any], prompt: str) -> str:
        return json.dumps({"issues": []}, ensure_ascii=False)

    def _echo(self, payload: dict[str, Any], prompt: str) -> str:
        return prompt.strip().splitlines()[-1] if prompt.strip() else ""
