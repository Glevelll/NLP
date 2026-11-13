"""Справочники для интерфейса."""

CATEGORIES = [
    "Электроника", "Бытовая техника", "Одежда", "Обувь", "Дом и кухня", "Красота и уход", "Спорт и отдых",
    "Детские товары", "Автотовары", "Зоотовары", "Аксессуары", "Инструменты",
]
PLATFORMS = {"ozon": "Ozon", "wildberries": "Wildberries", "site": "Собственный сайт"}
STYLES = ["нейтральный", "эмоциональный", "премиальный", "молодёжный", "экспертный"]
MODES = {
    "+agents": "Поиск, справочник и проверка текста",
    "+rag": "Гибридный поиск и справочник",
    "+dict": "Модель со справочником",
    "baseline": "Только модель",
    "+ensemble": "Ансамбль моделей",
}
PROVIDERS = ["auto", "ollama", "openai", "anthropic", "openrouter", "groq", "hf", "mock"]
CRITERIA_LABELS = {
    "attractiveness": "Привлекательность",
    "persuasiveness": "Убедительность",
    "audience_relevance": "Релевантность ЦА",
    "factual_accuracy": "Соответствие фактам",
    "style_match": "Соответствие стилю",
}
METRICS = ["bleu", "chrf_pp", "rouge_l", "meteor", "bertscore", "keyword_coverage", "sentiment", "style_accuracy",
           "composite"]
