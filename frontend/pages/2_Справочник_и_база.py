"""Управление справочником и базой знаний."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from frontend.utils.api_client import APIError  # noqa: E402
from frontend.utils.constants import CATEGORIES, PLATFORMS, STYLES  # noqa: E402
from frontend.utils.session import client, require_login, sidebar_user  # noqa: E402

st.set_page_config(page_title="Справочник и база знаний", page_icon="📚", layout="wide")
sidebar_user()
user = require_login("editor")
st.title("📚 Справочник и база знаний")

try:
    stats = client().admin_stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("Примеров в базе знаний", stats["rows"]["knowledge_base"])
    c2.metric("Записей справочника", stats["rows"]["glossary"])
    c3.metric("Векторов в индексе", stats["index"]["vector_count"])
except APIError as exc:
    st.error(str(exc))

tab_g, tab_k, tab_i = st.tabs(["Справочник", "База знаний", "Импорт"])

with tab_g:
    kinds = {"": "все", "selling_phrase": "продающие фразы", "forbidden": "запрещённые", "template": "шаблоны",
             "benefit": "выгоды"}
    c1, c2 = st.columns(2)
    kind = c1.selectbox("Тип", list(kinds), format_func=kinds.get)
    q = c2.text_input("Поиск")
    rows = client().glossary(kind=kind or None, q=q or None, limit=300)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=350)
    with st.form("add_g"):
        st.markdown("**Добавить запись**")
        a, b, cc = st.columns(3)
        term = a.text_input("Термин / фраза")
        kind_new = b.selectbox("Тип записи", [k for k in kinds if k])
        cat = cc.selectbox("Категория", [""] + CATEGORIES)
        definition = st.text_input("Описание / пояснение")
        if st.form_submit_button("Добавить") and term:
            client().add_glossary({"term": term, "kind": kind_new, "category": cat or None, "definition": definition})
            st.rerun()
    if user["role"] == "admin":
        del_id = st.number_input("ID для удаления", min_value=0, step=1, key="del_g")
        if st.button("Удалить запись справочника") and del_id:
            client().delete_glossary(int(del_id))
            st.rerun()

with tab_k:
    c1, c2, c3 = st.columns(3)
    cat = c1.selectbox("Категория", [""] + CATEGORIES, key="kcat")
    plat = c2.selectbox("Площадка", [""] + list(PLATFORMS), key="kplat")
    q = c3.text_input("Поиск по характеристикам")
    rows = client().knowledge(category=cat or None, platform=plat or None, q=q or None, limit=200)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=350)
    with st.form("add_k"):
        st.markdown("**Добавить эталонное описание**")
        src = st.text_area("Характеристики")
        tgt = st.text_area("Описание")
        a, b, cc = st.columns(3)
        kc = a.selectbox("Категория", CATEGORIES)
        kp = b.selectbox("Площадка", list(PLATFORMS))
        ks = cc.selectbox("Стиль", STYLES)
        if st.form_submit_button("Добавить") and src and tgt:
            client().add_knowledge({"source_text": src, "target_text": tgt, "category": kc, "platform": kp,
                                    "style": ks})
            st.rerun()
    if user["role"] == "admin":
        del_id = st.number_input("ID для удаления", min_value=0, step=1, key="del_k")
        if st.button("Удалить пример") and del_id:
            client().delete_knowledge(int(del_id))
            st.rerun()

with tab_i:
    kind = st.radio("Куда импортировать", ["knowledge", "glossary"], horizontal=True)
    st.caption("knowledge: source_text, target_text, category, platform, style, product_name · "
               "glossary: term, definition, kind, category")
    f = st.file_uploader("CSV или JSON", type=["csv", "json"])
    if f and st.button("Импортировать", type="primary"):
        try:
            st.success(client().import_file(kind, f.name, f.getvalue()))
        except APIError as exc:
            st.error(str(exc))
    if st.button("Переиндексировать базу знаний"):
        st.json(client().reindex())
