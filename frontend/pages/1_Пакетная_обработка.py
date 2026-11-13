"""Пакетная генерация из CSV."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from frontend.utils.api_client import APIError  # noqa: E402
from frontend.utils.constants import MODES  # noqa: E402
from frontend.utils.layout import page_heading  # noqa: E402
from frontend.utils.session import client, require_login  # noqa: E402

require_login()
page_heading("Пакетная обработка", "Загрузите CSV, чтобы подготовить описания для нескольких товаров сразу.")
st.markdown("CSV с колонками `name, category, features, price, audience, brand, style, platform`; "
            "особенности разделяются символом `;`.")
sample = pd.DataFrame([{"name": "Термокружка Thermo Go 450 мл", "category": "Дом и кухня",
                        "features": "двойные стенки;держит тепло 8 часов;герметичная крышка", "price": 1290,
                        "audience": "путешественники и офисные сотрудники", "brand": "Thermo Go",
                        "style": "нейтральный", "platform": "wildberries"}])
st.download_button("Скачать шаблон CSV", sample.to_csv(index=False).encode("utf-8"), "template.csv", "text/csv")

file = st.file_uploader("Загрузите CSV", type=["csv"])
mode = st.selectbox("Режим", list(MODES), format_func=MODES.get)
background = st.checkbox("Выполнить в фоне (Celery)", True)

if file and st.button("Запустить", type="primary"):
    try:
        resp = client().batch_upload(file.name, file.getvalue(), mode, background)
    except APIError as exc:
        st.error(str(exc))
        st.stop()
    if "task_id" in resp and resp.get("result") is None and resp.get("status") != "SUCCESS":
        st.info(f"Задача поставлена: `{resp['task_id']}`")
        bar = st.progress(0.0)
        while True:
            status = client().task(resp["task_id"])
            info = status.get("info") or {}
            if isinstance(info, dict) and info.get("total"):
                bar.progress(info["current"] / info["total"])
            if status["state"] in ("SUCCESS", "FAILURE", "UNKNOWN"):
                break
            time.sleep(2)
        results = (status.get("result") or {}).get("results", [])
    else:
        results = (resp.get("result") or resp).get("results", [])
    if results:
        df = pd.DataFrame([{"request_id": r["request_id"], "описание": r["description"],
                            "оценка": r.get("critic_score"), "валидация": r["validation_passed"]} for r in results])
        st.dataframe(df, use_container_width=True)
        st.download_button("Скачать результат", df.to_csv(index=False).encode("utf-8"), "descriptions.csv")
