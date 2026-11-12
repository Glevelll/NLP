"""Генерация эталонного датасета.

Создаёт в data/import/:
  * knowledge_base.json / .csv — 1200 пар «характеристики → описание» (база знаний для RAG);
  * test.json — 200 тестовых пар для оценки качества, товары не пересекаются с базой знаний;
  * glossary.json / .csv — справочник (продающие фразы, выгоды, запрещённые формулировки, шаблоны), 500+ записей.

Описания собираются из шаблонов, различающихся по стилю бренда и площадке, с вариативностью
формулировок. Источник — каталог предметной области в scripts/data_catalog.py.

Пример: python scripts/generate_data.py --kb-size 1200 --test-size 200 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.data_catalog import (  # noqa: E402
    CATALOG,
    FORBIDDEN_PHRASES,
    PLATFORM_TEMPLATES,
    SELLING_ADJECTIVES,
    STYLE_TEMPLATE_NOTES,
    agree,
)

PLATFORMS = ["ozon", "wildberries", "site"]
STYLES = list(STYLE_TEMPLATE_NOTES)

OPENERS = {
    "нейтральный": ["{name} — удобный и практичный выбор для повседневного использования.",
                    "{name} от {brand} сочетает понятные характеристики и надёжность.",
                    "{name} — товар, который решает задачу без лишних сложностей."],
    "эмоциональный": ["Влюбитесь в {name} с первого использования!",
                      "{name} — это маленькая радость, которая делает каждый день ярче.",
                      "Представьте, как легко станет жить с {name}!"],
    "премиальный": ["{name} — безупречное качество от {brand} для тех, кто ценит лучшее в деталях.",
                    "{name} воплощает изысканный подход {brand} к материалам и исполнению.",
                    "Создано для взыскательных: {name} от {brand}."],
    "молодёжный": ["{name} — штука, которая реально упрощает жизнь.",
                   "Лови {name}: всё, что нужно, и ничего лишнего.",
                   "{name} — твой новый любимчик на каждый день."],
    "экспертный": ["{name} — инженерно продуманная модель с чёткими характеристиками.",
                   "Разберём {name} по пунктам: что умеет модель и почему это важно.",
                   "{name} от {brand} рассчитан на тех, кто выбирает по характеристикам."],
}
FEATURE_TEMPLATES = {
    "нейтральный": ["{F}: {benefit}.", "{F} — {benefit}.", "Благодаря тому, что есть {f}, {benefit}."],
    "эмоциональный": ["{F} — и {benefit}!", "А ещё {f}: {benefit}.", "{F}, а значит {benefit}."],
    "премиальный": ["{F} — {benefit}.", "Особого внимания заслуживает деталь: {f}. {Benefit}.",
                    "{F}: {benefit}, как и подобает вещи высокого класса."],
    "молодёжный": ["{F} — {benefit}, всё по делу.", "Есть {f}, так что {benefit}.", "{F}? Да! {Benefit}."],
    "экспертный": ["{F}. На практике это означает, что {benefit}.", "Параметр «{f}»: {benefit}.",
                   "{F} — {benefit}, что выгодно отличает модель от аналогов."],
}
AUDIENCE_TEMPLATES = {
    "нейтральный": ["Подойдёт для: {audience}.", "Товар рассчитан на такую аудиторию: {audience}."],
    "эмоциональный": ["Идеальный выбор, если вы из тех, кого можно назвать так: {audience}!",
                      "Отличный подарок для тех, кто дорог: {audience}."],
    "премиальный": ["Выбор для ценителей: {audience}.", "Создано специально для аудитории: {audience}."],
    "молодёжный": ["Для тех, кто в теме: {audience}.", "Зайдёт всем, кого можно назвать так: {audience}."],
    "экспертный": ["Целевая аудитория модели: {audience}.", "Рекомендуем для категории покупателей: {audience}."],
}
CLOSERS = {
    "ozon": ["Оформляйте заказ на Ozon — доставим в удобный пункт выдачи.",
             "Добавьте товар в корзину Ozon и получите его уже в ближайшие дни."],
    "wildberries": ["Добавляйте в корзину и забирайте в ближайшем пункте выдачи.",
                    "Закажите сейчас — товар приедет в ваш пункт выдачи Wildberries."],
    "site": ["Оформите заказ на нашем сайте — мы бережно упакуем и быстро доставим покупку.",
             "Закажите прямо сейчас, а мы позаботимся о быстрой доставке."],
}
SITE_STORY = {
    "нейтральный": "Мы подобрали характеристики так, чтобы товаром было удобно пользоваться каждый день.",
    "эмоциональный": "Это та вещь, о которой вы будете вспоминать с улыбкой каждый раз, когда берёте её в руки.",
    "премиальный": "Каждая деталь продумана, а качество исполнения чувствуется с первого прикосновения.",
    "молодёжный": "Никаких сложностей: взял, попробовал, понял, что без этого уже никак.",
    "экспертный": "Мы протестировали модель в реальных сценариях и собрали главное, что важно знать перед покупкой.",
}


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def write_description(p: dict, benefits: dict[str, str], rng: random.Random) -> str:
    style, platform = p["style"], p["platform"]
    ctx = {"name": p["name"], "brand": p["brand"]}
    parts = [rng.choice(OPENERS[style]).format(**ctx)]
    feature_lines = []
    for f in p["features"]:
        b = benefits[f]
        feature_lines.append(rng.choice(FEATURE_TEMPLATES[style]).format(F=_cap(f), f=f, benefit=b, Benefit=_cap(b)))
    if platform == "ozon":
        parts.append("Преимущества:\n" + "\n".join(f"• {line}" for line in feature_lines))
    elif platform == "wildberries":
        parts.extend(feature_lines)
    else:
        parts.append(SITE_STORY[style])
        parts.append(" ".join(feature_lines))
    parts.append(rng.choice(AUDIENCE_TEMPLATES[style]).format(audience=p["audience"]))
    if platform == "site":
        parts.append(f"Цена — {p['price']} ₽.")
    parts.append(rng.choice(CLOSERS[platform]))
    return "\n\n".join(parts)


def source_text(p: dict) -> str:
    return (f"Название: {p['name']}\nКатегория: {p['category']}\nБренд: {p['brand']}\n"
            f"Особенности: {'; '.join(p['features'])}\nЦена: {p['price']} ₽\nЦелевая аудитория: {p['audience']}\n"
            f"Стиль: {p['style']}\nПлощадка: {p['platform']}")


def make_products(n: int, rng: random.Random, used_names: set[str]) -> list[dict]:
    products = []
    categories = list(CATALOG)
    attempts = 0
    while len(products) < n and attempts < n * 50:
        attempts += 1
        cat = categories[len(products) % len(categories)]
        c = CATALOG[cat]
        brand = rng.choice(c["brands"])
        model = f"{rng.choice('ABCDEFGHKMPRSTVX')}{rng.randint(1, 99)}"
        name = f"{rng.choice(c['types'])} {brand} {model}"
        if name in used_names:
            continue
        used_names.add(name)
        features = [f for f, _ in rng.sample(c["features"], rng.randint(3, 5))]
        lo, hi = c["price"]
        price = int(round(rng.uniform(lo, hi), -1)) - 10
        products.append({"name": name, "category": cat, "brand": brand, "features": features,
                         "price": max(price, 90), "audience": rng.choice(c["audiences"]),
                         "style": rng.choice(STYLES), "platform": rng.choice(PLATFORMS)})
    return products


def build_glossary() -> list[dict]:
    items: list[dict] = []
    for cat, c in CATALOG.items():
        for adj in SELLING_ADJECTIVES:
            for noun, gender in c["nouns"]:
                items.append({"term": f"{agree(adj, gender)} {noun}", "definition": "продающая формулировка-позиционирование",
                              "kind": "selling_phrase", "category": cat})
        for feature, benefit in c["features"]:
            items.append({"term": feature, "definition": benefit, "kind": "benefit", "category": cat})
            items.append({"term": benefit, "definition": f"выгода для особенности «{feature}»",
                          "kind": "selling_phrase", "category": cat})
    for phrase in FORBIDDEN_PHRASES:
        items.append({"term": phrase, "definition": "недопустимое или непроверяемое утверждение",
                      "kind": "forbidden", "category": None})
    for platform, structure in PLATFORM_TEMPLATES.items():
        for style, note in STYLE_TEMPLATE_NOTES.items():
            items.append({"term": f"шаблон {platform}/{style}", "definition": f"{structure}; тон: {note}",
                          "kind": "template", "category": f"{platform}|{style}"})
    # уникальность по (term, kind, category)
    seen, unique = set(), []
    for it in items:
        key = (it["term"].lower(), it["kind"], it["category"])
        if key not in seen:
            seen.add(key)
            unique.append(it)
    return unique


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def generate(out_dir: Path, kb_size: int = 1200, test_size: int = 200, seed: int = 42) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    used: set[str] = set()
    benefits = {f: b for c in CATALOG.values() for f, b in c["features"]}

    kb_products = make_products(kb_size, rng, used)
    test_products = make_products(test_size, random.Random(seed + 1000), used)

    kb = []
    for i, p in enumerate(kb_products, 1):
        kb.append({"id": f"kb-{i:04d}", "source_text": source_text(p), "target_text": write_description(p, benefits, rng),
                   "category": p["category"], "platform": p["platform"], "style": p["style"],
                   "product_name": p["name"]})
    test = []
    trng = random.Random(seed + 2000)
    for i, p in enumerate(test_products, 1):
        test.append({"id": f"test-{i:04d}", "product": p, "reference": write_description(p, benefits, trng),
                     "category": p["category"], "platform": p["platform"], "style": p["style"]})
    glossary = build_glossary()

    (out_dir / "knowledge_base.json").write_text(json.dumps(kb, ensure_ascii=False, indent=1), encoding="utf-8")
    save_csv(out_dir / "knowledge_base.csv", kb)
    (out_dir / "test.json").write_text(json.dumps(test, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "glossary.json").write_text(json.dumps(glossary, ensure_ascii=False, indent=1), encoding="utf-8")
    save_csv(out_dir / "glossary.csv", glossary)

    # Пример CSV для пакетной генерации
    batch = [{"name": p["name"], "category": p["category"], "features": ";".join(p["features"]), "price": p["price"],
              "audience": p["audience"], "brand": p["brand"], "style": p["style"], "platform": p["platform"]}
             for p in test_products[:10]]
    save_csv(out_dir / "batch_example.csv", batch)

    stats = {"knowledge_base": len(kb), "test": len(test), "glossary": len(glossary)}
    print("Сгенерировано:", stats)
    print("Справочник по типам:", dict(Counter(g["kind"] for g in glossary)))
    print("База знаний по категориям:", dict(Counter(r["category"] for r in kb)))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "import")
    ap.add_argument("--kb-size", type=int, default=1200)
    ap.add_argument("--test-size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    generate(args.out, args.kb_size, args.test_size, args.seed)


if __name__ == "__main__":
    main()
