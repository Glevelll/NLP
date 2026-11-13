"""Мониторинг сервисов, кэша и LLM."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from frontend.utils.api_client import APIError  # noqa: E402
from frontend.utils.layout import page_heading  # noqa: E402
from frontend.utils.session import client, require_login  # noqa: E402

user = require_login()
page_heading("Мониторинг", "Проверьте доступность сервисов, состояние кэша и статистику генераций.")

try:
    health = client().health()
except APIError as exc:
    st.error(str(exc))
    st.stop()

st.subheader(f"Статус: {'🟢 ok' if health['status'] == 'ok' else '🟡 degraded'}")
services = health["services"]
cols = st.columns(4)
cols[0].metric("База данных", "✅" if services["database"] else "❌")
cols[1].metric("Векторный индекс", "✅" if services["vector_index"] else "❌")
cols[2].metric("База знаний", "✅" if services["knowledge_loaded"] else "❌")
if "redis" in services:
    cols[3].metric("Redis", "✅" if services["redis"] else "❌")
st.markdown("**LLM-провайдеры:** " + " · ".join(f"{p} {'✅' if ok else '—'}" for p, ok in services["llm"].items()))

stats = client().stats()
c1, c2, c3 = st.columns(3)
c1.metric("Записей в кэше", stats["cache"]["entries"])
c2.metric("Попаданий в кэш", stats["cache"]["total_hits"])
c3.metric("Примеров в индексе", stats["knowledge"]["examples"])
st.caption(f"Эмбеддинги: {stats['knowledge']['embedding_model']} · индекс: {stats['knowledge']['vector_backend']} · "
           f"кэш: {stats['cache']['backend']}")

st.subheader("Вызовы LLM")
if stats["llm_calls"]:
    st.dataframe(pd.DataFrame(stats["llm_calls"]), use_container_width=True)
st.subheader("Генерации по режимам")
if stats["generations"]:
    st.dataframe(pd.DataFrame(stats["generations"]), use_container_width=True)

if user["role"] == "admin" and st.button("Очистить кэш"):
    st.success(client().clear_cache())
