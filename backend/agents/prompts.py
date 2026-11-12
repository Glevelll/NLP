"""Шаблоны промптов для агентов."""
from __future__ import annotations

import json

from backend.core.constants import PLATFORM_RULES, STYLE_GUIDES
from backend.core.models import ProductInput, RetrievedExample

GENERATOR_SYSTEM = (
    "Ты — опытный копирайтер e-commerce. Пишешь продающие описания товаров на русском языке для карточек "
    "маркетплейсов и интернет-магазинов. Опираешься только на переданные характеристики, ничего не выдумываешь, "
    "не используешь запрещённые формулировки и не даёшь медицинских или абсолютных обещаний."
)

CRITIC_SYSTEM = (
    "Ты — строгий редактор маркетплейса и эксперт по конверсии. Оцениваешь описания товаров объективно "
    "и отвечаешь только валидным JSON."
)

EDITOR_SYSTEM = (
    "Ты — литературный редактор продающих текстов. Исправляешь описание по замечаниям, сохраняя факты, "
    "и возвращаешь только итоговый текст описания без комментариев."
)

CRITERIA = {
    "attractiveness": "привлекательность",
    "persuasiveness": "убедительность",
    "audience_relevance": "релевантность для ЦА",
    "factual_accuracy": "соответствие фактам",
    "style_match": "соответствие стилю бренда и площадке",
}


def product_block(p: ProductInput) -> str:
    lines = [f"- Название: {p.name}", f"- Категория: {p.category}"]
    if p.brand:
        lines.append(f"- Бренд: {p.brand}")
    if p.features:
        lines.append("- Ключевые особенности (упомяни КАЖДУЮ):")
        lines += [f"  • {f}" for f in p.features]
    if p.price is not None:
        lines.append(f"- Цена: {p.price:g} ₽")
    if p.audience:
        lines.append(f"- Целевая аудитория: {p.audience}")
    return "\n".join(lines)


def requirements_block(p: ProductInput) -> str:
    rules = PLATFORM_RULES[p.platform]
    return (
        f"- Площадка: {rules['title']} — {rules['hint']}.\n"
        f"- Длина: от {rules['min_chars']} до {min(rules['max_chars'], 2500)} символов.\n"
        f"- Стиль бренда: «{p.style.value}» — {STYLE_GUIDES[p.style]}.\n"
        + ("- Без ссылок, телефонов и эмодзи.\n" if not rules["allow_urls"] else "")
    )


def generator_prompt(p: ProductInput, examples: list[RetrievedExample], phrases: list[str],
                     forbidden: list[str], templates: list[str], benefits: dict[str, str] | None = None) -> str:
    parts = ["Напиши продающее описание товара для карточки.", "", "### ТОВАР", product_block(p), "",
             "### ТРЕБОВАНИЯ", requirements_block(p)]
    if benefits:
        parts += ["### ВЫГОДЫ ДЛЯ ПОКУПАТЕЛЯ (раскрой особенности через них)",
                  *[f"- {f} → {b}" for f, b in benefits.items()], ""]
    if phrases:
        parts += ["### УДАЧНЫЕ ПРОДАЮЩИЕ ФРАЗЫ (используй уместные)", *[f"- {x}" for x in phrases], ""]
    if templates:
        parts += ["### СТРУКТУРА ОПИСАНИЯ", *[f"- {t}" for t in templates], ""]
    if forbidden:
        parts += ["### ЗАПРЕЩЕНО ИСПОЛЬЗОВАТЬ", ", ".join(f"«{f}»" for f in forbidden[:40]), ""]
    if examples:
        parts.append("### ПРИМЕРЫ УДАЧНЫХ ОПИСАНИЙ ПОХОЖИХ ТОВАРОВ (перенимай приёмы, не копируй факты)")
        for i, ex in enumerate(examples[:3], 1):
            parts += [f"Пример {i}. Характеристики: {ex.text[:300]}", f"Описание: {ex.description[:700]}", ""]
    parts.append("Выведи только текст описания, без заголовков «Описание:» и пояснений.")
    return "\n".join(parts)


CRITIC_FEW_SHOT = """### ПРИМЕР 1 (хорошее описание)
Товар: Термокружка Thermo Go 450 мл; особенности: двойные стенки, держит тепло 8 часов, герметичная крышка.
Описание: «Thermo Go 450 мл сохранит кофе горячим до 8 часов благодаря двойным стенкам. Герметичная крышка не даст пролиться ни капле — кружку можно смело бросить в рюкзак. Подойдёт для поездок, офиса и прогулок.»
Ответ: {"scores": {"attractiveness": 8.5, "persuasiveness": 8.5, "audience_relevance": 8, "factual_accuracy": 9.5, "style_match": 8.5}, "overall": 8.6, "reasoning": "Все особенности раскрыты через выгоды, текст живой и точный.", "suggestions": ["Можно добавить сценарий использования зимой"]}

### ПРИМЕР 2 (плохое описание)
Товар: Термокружка Thermo Go 450 мл; особенности: двойные стенки, держит тепло 8 часов, герметичная крышка.
Описание: «Лучшая кружка в мире!!! Держит тепло сутки. Купите.»
Ответ: {"scores": {"attractiveness": 3, "persuasiveness": 2.5, "audience_relevance": 2, "factual_accuracy": 1.5, "style_match": 3}, "overall": 2.4, "reasoning": "Искажён факт (сутки вместо 8 часов), запрещённое «лучшая в мире», нет особенностей.", "suggestions": ["Убрать «лучшая в мире»", "Указать 8 часов", "Раскрыть двойные стенки и крышку", "Обратиться к аудитории"]}
"""


def critic_prompt(p: ProductInput, description: str) -> str:
    criteria = "\n".join(f"- {k} — {v}" for k, v in CRITERIA.items())
    return (
        "Оцени описание товара по шкале 0–10 по каждому критерию:\n"
        f"{criteria}\n\n{CRITIC_FEW_SHOT}\n"
        "### ОЦЕНИВАЕМОЕ ОПИСАНИЕ\n"
        f"Товар:\n{product_block(p)}\nСтиль: {p.style.value}; площадка: {PLATFORM_RULES[p.platform]['title']}\n"
        f"Описание: «{description}»\n\n"
        'Ответь строго JSON: {"scores": {...}, "overall": число, "reasoning": "...", "suggestions": ["..."]}'
    )


def editor_prompt(p: ProductInput, description: str, errors: list[str], warnings: list[str],
                  suggestions: list[str], missing: list[str], forbidden: list[str]) -> str:
    issues = [f"- [ошибка] {e}" for e in errors] + [f"- [замечание] {w}" for w in warnings] + [
        f"- [рекомендация] {s}" for s in suggestions
    ]
    parts = ["Улучши описание товара. Исправь все перечисленные проблемы, сохрани факты и требования.", "",
             "### ТОВАР", product_block(p), "", "### ТРЕБОВАНИЯ", requirements_block(p),
             "### ПРОБЛЕМЫ", *issues]
    if missing:
        parts += ["", "Обязательно добавь особенности: " + "; ".join(missing)]
    if forbidden:
        parts += ["", "Не используй: " + ", ".join(f"«{f}»" for f in forbidden[:40])]
    parts += ["", "### ТЕКУЩЕЕ ОПИСАНИЕ", description, "", "Верни только исправленный текст описания."]
    return "\n".join(parts)


def validator_llm_prompt(p: ProductInput, description: str) -> str:
    return (
        "Проверь, нет ли в описании фактов, которых нет в характеристиках (выдуманные цифры, материалы, гарантии).\n"
        f"Характеристики:\n{product_block(p)}\n\nОписание:\n{description}\n\n"
        'Ответь JSON: {"issues": ["краткое описание проблемы", ...]} (пустой список, если всё верно).'
    )


def product_payload(p: ProductInput) -> dict:
    return json.loads(p.model_dump_json())
