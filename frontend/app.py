"""Навигация, вход и регистрация."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from frontend.utils.api_client import APIError  # noqa: E402
from frontend.utils.layout import ASSETS, apply_styles, sidebar_brand  # noqa: E402
from frontend.utils.session import client, login_form, sidebar_user  # noqa: E402

st.set_page_config(page_title="Карточки", page_icon=":material/inventory_2:", layout="wide")


def sign_in() -> None:
    form, cover = st.columns([.95, 1.05], gap="large", vertical_alignment="center")
    with form, st.container(key="sign-in"):
        st.html('<div class="studio-brand"><span class="studio-mark" aria-hidden="true">К</span>Карточки</div>')
        st.title("Генератор описаний товаров")
        st.html('<p class="auth-intro">Подготовьте описание для Ozon, Wildberries или своего магазина.</p>')
        login_form()
    with cover, st.container(key="auth-cover"):
        picture = ASSETS / "catalog.webp"
        if picture.is_file():
            st.image(str(picture))


def workspace() -> None:
    pages = [
        st.Page("generator.py", title="Генератор", icon=":material/edit_note:", default=True),
        st.Page("pages/1_Пакетная_обработка.py", title="Пакетная обработка", icon=":material/table_rows:"),
        st.Page("pages/2_Справочник_и_база.py", title="Справочник и база знаний", icon=":material/library_books:"),
        st.Page("pages/3_Проверка_редактором.py", title="Проверка редактором", icon=":material/fact_check:"),
        st.Page("pages/4_Оценка_качества.py", title="Оценка качества", icon=":material/bar_chart:"),
        st.Page("pages/5_Логи_рассуждений.py", title="Журнал генерации", icon=":material/receipt_long:"),
        st.Page("pages/6_Мониторинг.py", title="Мониторинг", icon=":material/monitor_heart:"),
    ]
    current = st.navigation(pages, position="hidden")
    sidebar_brand()
    with st.sidebar.container(key="workspace-nav"):
        st.caption("Рабочая область")
        for page in pages[:4]:
            st.page_link(page)
        st.caption("Аналитика")
        for page in pages[4:]:
            st.page_link(page)
    sidebar_user()
    current.run()


apply_styles()
if st.session_state.get("user"):
    try:
        st.session_state["user"] = client().me()
    except APIError as exc:
        if exc.status_code == 401:
            st.session_state.clear()
            st.rerun()
    workspace()
else:
    st.navigation([st.Page(sign_in, title="Вход", default=True)], position="hidden").run()
