"""Оценка качества: запуск, сравнение моделей и режимов, графики Plotly."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from frontend.utils.api_client import APIError  # noqa: E402
from frontend.utils.constants import METRICS, MODES  # noqa: E402
from frontend.utils.layout import page_heading  # noqa: E402
from frontend.utils.session import client, require_login  # noqa: E402

require_login()
page_heading("Оценка качества", "Сравните модели и режимы генерации на выборках из 20, 50 или 100 товаров.", "bar_chart")

try:
    prov = client().providers()
    available = [p for p, ok in prov["available"].items() if ok]
    st.caption("Доступные провайдеры: " + ", ".join(available))
    default_models = ["mock"] + [f"ollama:{m}" for m in prov["ollama_models"]] if "ollama" in available else ["mock"]
except APIError:
    default_models = ["mock"]

with st.form("eval"):
    models_raw = st.text_input("Модели (через запятую, формат provider:model)", ", ".join(default_models[:2]))
    modes = st.multiselect("Режимы", list(MODES), default=["baseline", "+rag", "+agents"], format_func=str)
    sizes = st.multiselect("Размеры выборки", [20, 50, 100], default=[20])
    background = st.checkbox("В фоне (Celery)", False)
    run = st.form_submit_button("Запустить оценку", type="primary")

if run:
    models = [m.strip() for m in models_raw.split(",") if m.strip()]
    try:
        with st.spinner("Идёт оценка… (для 100 образцов и локальных LLM это может занять долго)"):
            resp = client().evaluate_batch(models, modes, sizes, background)
            if background and resp.get("task_id"):
                while True:
                    s = client().task(resp["task_id"])
                    if s["state"] in ("SUCCESS", "FAILURE", "UNKNOWN"):
                        break
                    time.sleep(3)
                resp = s.get("result") or resp
            results = resp.get("results") or (resp.get("result") or {}).get("results", [])
        errors = [r for r in results if "error" in r]
        for e in errors:
            st.warning(f"{e['model_name']} {e['mode']} n={e['sample_size']}: {e['error']}")
        st.success(f"Готово: {len(results) - len(errors)} прогонов")
    except APIError as exc:
        st.error(str(exc))

st.subheader("История оценок")
history = client().eval_history(300)
if not history:
    st.info("Пока нет результатов.")
    st.stop()

df = pd.DataFrame([{"id": h["id"], "model": h["model_name"], "mode": h["mode"], "n": h["sample_size"],
                    "time_s": h["evaluation_time"], "created_at": h["created_at"], **h["metrics"]} for h in history])
c1, c2 = st.columns(2)
sel_n = c1.selectbox("Размер выборки", sorted(df["n"].unique()), index=len(df["n"].unique()) - 1)
metric = c2.selectbox("Метрика", [m for m in METRICS if m in df.columns])
latest = df[df["n"] == sel_n].sort_values("created_at").groupby(["model", "mode"], as_index=False).last()
cols = ["model", "mode", "n", "time_s"] + [m for m in METRICS if m in latest.columns]
st.dataframe(latest[cols].style.highlight_max(
    subset=[m for m in METRICS if m in latest.columns], props="background-color:#316349;color:#fff;font-weight:bold"
),
             use_container_width=True)

latest["system"] = latest["model"] + " " + latest["mode"]
st.plotly_chart(px.bar(latest, x="system", y=metric, color="mode", title=f"{metric}, n={sel_n}"),
                use_container_width=True)

radar_metrics = [m for m in ["bleu", "chrf_pp", "rouge_l", "meteor", "bertscore", "keyword_coverage"]
                 if m in latest.columns]
fig = go.Figure()
for _, row in latest.iterrows():
    vals = [row[m] for m in radar_metrics]
    fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=radar_metrics + radar_metrics[:1], fill="toself",
                                  name=row["system"]))
fig.update_layout(title="Профиль метрик", polar={"radialaxis": {"range": [0, 1]}})
st.plotly_chart(fig, use_container_width=True)

by_size = df.sort_values("created_at").groupby(["model", "mode", "n"], as_index=False).last()
by_size["system"] = by_size["model"] + " " + by_size["mode"]
st.plotly_chart(px.line(by_size, x="n", y=metric, color="system", markers=True,
                        title=f"{metric} в зависимости от размера выборки"), use_container_width=True)
