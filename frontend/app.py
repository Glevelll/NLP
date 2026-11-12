"""Главная страница: вход и генерация описания товара."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from frontend.utils.api_client import APIError  # noqa: E402
from frontend.utils.constants import CATEGORIES, CRITERIA_LABELS, MODES, PLATFORMS, PROVIDERS, STYLES  # noqa: E402
from frontend.utils.session import client, login_form, sidebar_user  # noqa: E402

st.set_page_config(page_title="Генератор описаний товаров", page_icon="🛒", layout="wide")
st.title("🛒 Генератор продающих описаний товаров")
sidebar_user()

if not st.session_state.get("user"):
    st.caption("Мультиагентная система: DescriptionGenerator → Validator → Critic → Editor, RAG по базе эталонных "
               "описаний и справочник продающих фраз.")
    login_form()
    st.stop()

with st.form("product"):
    c1, c2 = st.columns(2)
    name = c1.text_input("Название товара", "Беспроводные наушники SoundMax Air 3")
    category = c2.selectbox("Категория", CATEGORIES)
    features_raw = st.text_area("Ключевые особенности (по одной в строке)",
                                "активное шумоподавление\nдо 30 часов работы с кейсом\nзащита от брызг IPX4")
    c3, c4, c5 = st.columns(3)
    price = c3.number_input("Цена, ₽", min_value=0.0, value=4990.0, step=100.0)
    brand = c4.text_input("Бренд", "SoundMax")
    audience = c5.text_input("Целевая аудитория", "любители музыки и активного образа жизни")
    c6, c7, c8 = st.columns(3)
    platform = c6.selectbox("Площадка", list(PLATFORMS), format_func=PLATFORMS.get)
    style = c7.selectbox("Стиль бренда", STYLES)
    mode = c8.selectbox("Режим", list(MODES), format_func=MODES.get)
    c9, c10, c11 = st.columns(3)
    provider = c9.selectbox("Провайдер LLM", PROVIDERS)
    model = c10.text_input("Модель (необязательно)", "")
    use_cache = c11.checkbox("Использовать кэш", True)
    multi = st.checkbox("Сгенерировать варианты для всех площадок")
    submitted = st.form_submit_button("Сгенерировать", type="primary")

product = {"name": name, "category": category, "features": [f for f in features_raw.splitlines() if f.strip()],
           "price": price or None, "brand": brand or None, "audience": audience or None, "platform": platform,
           "style": style}

if submitted:
    try:
        with st.spinner("Агенты работают…"):
            if multi:
                st.session_state["results"] = client().generate_multi(product, list(PLATFORMS), [], mode)
            elif mode == "+ensemble":
                st.session_state["results"] = [client().ensemble(product, None, False)]
            else:
                st.session_state["results"] = [client().generate(product, mode,
                                                                  None if provider == "auto" else provider,
                                                                  model or None, use_cache)]
    except APIError as exc:
        st.error(str(exc))

for res in st.session_state.get("results", []):
    st.divider()
    head = st.columns([3, 1, 1, 1])
    head[0].subheader(f"Результат · {res['mode']} · {res.get('provider')}:{res.get('model')}")
    head[1].metric("Оценка Critic", f"{res['critic_score']:.1f}" if res.get("critic_score") is not None else "—")
    head[2].metric("Итераций", res.get("iterations", 0))
    head[3].metric("Время, мс", int(res.get("elapsed_ms", 0)))
    st.text_area("Описание", res["description"], height=280, key=f"desc_{res['request_id']}")
    st.caption(f"request_id: `{res['request_id']}` · длина {len(res['description'])} симв."
               + (" · из кэша" if res.get("cached") else "") + (" · отправлено на проверку редактору"
                                                                 if res.get("needs_review") else ""))
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Валидация:** " + ("✅ пройдена" if res["validation_passed"] else "❌ есть ошибки"))
        for e in res.get("errors", []):
            st.error(e)
        for w in res.get("warnings", []):
            st.warning(w)
    with cols[1]:
        if res.get("critic_scores"):
            st.markdown("**Оценки Critic**")
            st.bar_chart(pd.Series({CRITERIA_LABELS.get(k, k): v for k, v in res["critic_scores"].items()}))
        for s in res.get("critic_suggestions", []):
            st.info(s)
    if res.get("examples_used"):
        with st.expander(f"Найденные примеры из базы знаний ({len(res['examples_used'])})"):
            for ex in res["examples_used"]:
                st.markdown(f"**score {ex['score']:.3f}** · BM25 #{ex.get('bm25_rank')} · вектор "
                            f"#{ex.get('vector_rank')} · {ex.get('platform')}/{ex.get('style')}")
                st.caption(ex["text"][:300])
    if res.get("candidates"):
        with st.expander("Кандидаты ансамбля"):
            st.dataframe(pd.DataFrame(res["candidates"])[["model", "score", "critic_score", "validation_passed"]])
    with st.expander("Оценить результат"):
        rating = st.slider("Оценка", 1, 5, 4, key=f"rate_{res['request_id']}")
        comment = st.text_input("Комментарий", key=f"comm_{res['request_id']}")
        if st.button("Отправить отзыв", key=f"fb_{res['request_id']}"):
            try:
                client().feedback(res["request_id"], rating, comment or None)
                st.success("Спасибо за отзыв!")
            except APIError as exc:
                st.error(str(exc))
