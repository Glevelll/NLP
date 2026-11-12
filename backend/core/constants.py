"""Константы предметной области: площадки, стили, категории, словари тональности."""
from __future__ import annotations

from enum import StrEnum


class Platform(StrEnum):
    OZON = "ozon"
    WILDBERRIES = "wildberries"
    SITE = "site"


class Style(StrEnum):
    NEUTRAL = "нейтральный"
    EMOTIONAL = "эмоциональный"
    PREMIUM = "премиальный"
    YOUTH = "молодёжный"
    EXPERT = "экспертный"


class Mode(StrEnum):
    BASELINE = "baseline"
    DICT = "+dict"
    RAG = "+rag"
    AGENTS = "+agents"
    ENSEMBLE = "+ensemble"


class Role(StrEnum):
    USER = "user"
    EDITOR = "editor"
    ADMIN = "admin"


class GlossaryKind(StrEnum):
    SELLING_PHRASE = "selling_phrase"
    FORBIDDEN = "forbidden"
    TEMPLATE = "template"
    BENEFIT = "benefit"


class ReviewStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


CATEGORIES: list[str] = [
    "Электроника",
    "Бытовая техника",
    "Одежда",
    "Обувь",
    "Дом и кухня",
    "Красота и уход",
    "Спорт и отдых",
    "Детские товары",
    "Автотовары",
    "Зоотовары",
    "Аксессуары",
    "Инструменты",
]

PLATFORM_RULES: dict[str, dict] = {
    Platform.OZON: {
        "title": "Ozon",
        "min_chars": 350,
        "max_chars": 6000,
        "allow_urls": False,
        "allow_emoji": False,
        "hint": "структурированный текст: вводный абзац, блок «Преимущества» списком, характеристики и сценарии использования",
    },
    Platform.WILDBERRIES: {
        "title": "Wildberries",
        "min_chars": 300,
        "max_chars": 5000,
        "allow_urls": False,
        "allow_emoji": False,
        "hint": "короткие абзацы, ключевые слова в начале, без ссылок, контактов и эмодзи",
    },
    Platform.SITE: {
        "title": "собственный сайт",
        "min_chars": 300,
        "max_chars": 8000,
        "allow_urls": True,
        "allow_emoji": True,
        "hint": "живой продающий текст с историей использования и призывом к действию",
    },
}

STYLE_GUIDES: dict[str, str] = {
    Style.NEUTRAL: "спокойный информативный тон, факты и практическая польза, без восклицаний",
    Style.EMOTIONAL: "живой эмоциональный тон, яркие образы, обращение к чувствам покупателя",
    Style.PREMIUM: "сдержанный премиальный тон, акцент на качестве, материалах и статусе, без слов «дёшево» и «скидка»",
    Style.YOUTH: "лёгкий молодёжный тон, короткие фразы, на «ты», уместный сленг без грубости",
    Style.EXPERT: "экспертный тон, точные характеристики, объяснение технологий и сравнение с аналогами",
}

# Слова, противоречащие тону бренда
STYLE_STOPWORDS: dict[str, list[str]] = {
    Style.PREMIUM: ["дешев", "копеечн", "распродаж", "скидк", "халяв", "круто", "огонь"],
    Style.EXPERT: ["вау", "огонь", "бомба", "круто"],
    Style.NEUTRAL: ["вау", "огонь", "бомба"],
}

# Короткий лексикон для анализа тональности
POSITIVE_WORDS: list[str] = [
    "удобн", "надёжн", "надежн", "качествен", "комфорт", "лёгк", "легк", "прочн", "стильн", "практичн",
    "идеальн", "отличн", "превосходн", "приятн", "безопасн", "эффективн", "долговечн", "современн",
    "мягк", "универсальн", "радост", "любим", "уверен", "вдохнов", "элегантн", "компактн", "быстр",
    "яркий", "ярк", "заботлив", "экономи", "тих", "выгод", "наслажд", "свобод", "уют",
]
NEGATIVE_WORDS: list[str] = [
    "плох", "слаб", "ломк", "неудобн", "дорог", "проблем", "недостат", "минус", "разочар", "хрупк",
    "громк", "тяжёл", "тяжел", "опасн", "брак", "ужасн", "скучн",
]

DEFAULT_FORBIDDEN: list[str] = [
    "лучший в мире", "100% гарантия", "гарантированно вылечит", "лечит", "излечивает", "без побочных эффектов",
    "самый дешёвый", "номер один в россии", "одобрено минздравом", "абсолютно безопасно", "вечный",
    "никогда не ломается", "панацея", "мгновенный результат",
]
