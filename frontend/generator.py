"""Создание и проверка описаний товаров."""
from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from frontend.utils.api_client import APIError
from frontend.utils.constants import CATEGORIES, CRITERIA_LABELS, MODES, PLATFORMS, PROVIDERS, STYLES
from frontend.utils.layout import ASSETS, page_heading
from frontend.utils.session import client


def product_form() -> None:
    with st.form("product"):
        st.subheader("Параметры товара")
        name = st.text_input("Название товара", "Беспроводные наушники SoundMax Air 3")
        category = st.selectbox("Категория", CATEGORIES)
        features = st.text_area(
            "Характеристики", "активное шумоподавление\nдо 30 часов работы с кейсом\nзащита от брызг IPX4",
            help="Укажите каждую характеристику с новой строки.", height=105,
        )
        brand_col, price_col = st.columns(2)
        brand = brand_col.text_input("Бренд", "SoundMax")
        price = price_col.number_input("Цена, ₽", min_value=0.0, value=4990.0, step=100.0)
        audience = st.text_input("Для кого товар", "любители музыки и активного образа жизни")
        platform_col, style_col = st.columns(2)
        platform = platform_col.selectbox("Площадка", list(PLATFORMS), format_func=PLATFORMS.get)
        style = style_col.selectbox("Стиль", STYLES)
        with st.expander("Настройки генерации"):
            mode = st.selectbox("Режим", list(MODES), format_func=MODES.get)
            provider = st.selectbox("Провайдер модели", PROVIDERS, format_func=lambda p: "Авто" if p == "auto" else p)
            model = st.text_input("Модель", placeholder="Например, qwen2.5:1.5b")
            use_cache = st.checkbox("Использовать кэш", True)
            multi = st.checkbox("Сгенерировать для всех площадок")
        submitted = st.form_submit_button("Сгенерировать", type="primary", use_container_width=True)

    if not submitted:
        return

    product = {
        "name": name, "category": category, "features": [item for item in features.splitlines() if item.strip()],
        "price": price or None, "brand": brand or None, "audience": audience or None, "platform": platform,
        "style": style,
    }
    api = client()
    try:
        with st.spinner("Готовим описание…"):
            if multi:
                results = api.generate_multi(product, list(PLATFORMS), [], mode)
            elif mode == "+ensemble":
                results = [api.ensemble(product, None, False)]
            else:
                results = [api.generate(product, mode, None if provider == "auto" else provider,
                                        model.strip() or None, use_cache)]
            st.session_state["results"] = results
    except APIError as exc:
        st.error(str(exc))


def result_panel(result: dict, index: int) -> None:
    request_id = result["request_id"]
    widget_key = f"{request_id}_{index}"
    with st.container(border=True):
        st.subheader("Описание товара")
        metadata = [f"{result['provider']}:{result['model']}", MODES.get(result["mode"], result["mode"])]
        if result.get("cached"):
            metadata.append("Из кэша")
        st.caption(" · ".join(metadata))
        st.html(f'<div class="description-copy">{escape(result["description"])}</div>')
        st.download_button("Скачать текст", result["description"], "description.txt", "text/plain",
                           key=f"download_{widget_key}", icon=":material/download:")
        score, iterations, duration = st.columns(3)
        rating = result.get("critic_score")
        score.metric("Оценка из 10", f"{rating:.1f}" if rating is not None else "—")
        iterations.metric("Итерации", result.get("iterations", 0))
        duration.metric("Время", f"{result.get('elapsed_ms', 0) / 1000:.1f} с")
        if result.get("needs_review"):
            st.warning("Описание отправлено редактору.")
        elif result["validation_passed"]:
            st.success("Проверка пройдена.")
        else:
            st.warning("Проверка выявила ошибки.")
        with st.expander("Проверка и рекомендации", expanded=bool(result.get("errors"))):
            for error in result.get("errors", []):
                st.error(error)
            for warning in result.get("warnings", []):
                st.warning(warning)
            scores = result.get("critic_scores")
            if scores:
                st.bar_chart(pd.Series({CRITERIA_LABELS.get(key, key): value for key, value in scores.items()}))
            for suggestion in result.get("critic_suggestions", []):
                st.info(suggestion)
            if not any(result.get(key) for key in ("errors", "warnings", "critic_scores", "critic_suggestions")):
                st.caption("Замечаний нет.")
        examples = result.get("examples_used")
        if examples:
            with st.expander(f"Примеры из базы знаний · {len(examples)}"):
                for example in examples:
                    st.caption(f"{example.get('platform') or '—'} · {example.get('style') or '—'}")
                    st.text(example["text"][:300])
                    st.caption(f"Оценка поиска {example['score']:.3f} · BM25 {example.get('bm25_rank') or '—'} · "
                               f"векторный поиск {example.get('vector_rank') or '—'}")
        if result.get("candidates"):
            with st.expander("Варианты ансамбля"):
                st.dataframe(pd.DataFrame(result["candidates"])[["model", "score", "validation_passed"]])
        log, feedback = st.tabs(["Журнал", "Отзыв"])
        with log:
            st.caption(f"Запрос {request_id}")
            if st.button("Открыть журнал запроса", key=f"log_{widget_key}", icon=":material/arrow_forward:"):
                st.session_state["journal_request"] = request_id
                st.switch_page("pages/5_Логи_рассуждений.py")
        with feedback:
            value = st.slider("Оценка", 1, 5, 4, key=f"rating_{widget_key}")
            comment = st.text_input("Комментарий", key=f"comment_{widget_key}")
            if st.button("Отправить отзыв", key=f"feedback_{widget_key}"):
                try:
                    client().feedback(request_id, value, comment or None)
                    st.success("Отзыв отправлен.")
                except APIError as exc:
                    st.error(str(exc))


with st.container(key="generator-hero"):
    intro, illustration = st.columns([1.6, 1], vertical_alignment="center")
    with intro:
        page_heading("Генератор описаний", "Укажите характеристики товара и выберите площадку.", "edit_note")
    with illustration:
        st.image(str(ASSETS / "workspace.webp"))
form_column, result_column = st.columns([.94, 1.06], gap="large")
with form_column, st.container(key="product-form"):
    product_form()
with result_column, st.container(key="description-panel"):
    results = st.session_state.get("results", [])
    if not results:
        st.html('<div class="description-empty"><strong>Описание товара</strong>'
                'Заполните форму и нажмите «Сгенерировать».</div>')
    for index, result in enumerate(results):
        result_panel(result, index)
