"""Просмотр лога рассуждений агентов и трассировки LLM-вызовов."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from frontend.utils.layout import page_heading  # noqa: E402
from frontend.utils.session import client, require_login  # noqa: E402

require_login()
page_heading("Журнал генерации", "Выберите запрос, чтобы увидеть шаги проверки и вызовы модели.")

reqs = client().requests(100)
if not reqs:
    st.info("Запросов пока нет — сгенерируйте описание на главной странице.")
    st.stop()

labels = {r["request_id"]: f"{r['created_at'][:19]} · {r['product_name']} · {r['mode']} · "
                           f"score {r['critic_score']}" for r in reqs}
request_ids = list(labels)
selected_id = st.session_state.pop("journal_request", None)
if selected_id in labels:
    st.session_state["selected_journal_request"] = selected_id
elif selected_id is not None:
    st.info("Этот запрос отсутствует в списке последних генераций.")
if st.session_state.get("selected_journal_request") not in labels:
    st.session_state["selected_journal_request"] = request_ids[0]
rid = st.selectbox("Запрос", request_ids, format_func=labels.get, key="selected_journal_request")
req = client().request(rid)
st.markdown(f"**Режим:** {req['mode']} · **модель:** {req['provider']}:{req['model_name']} · "
            f"**итераций:** {req['iterations']} · **время:** {req['elapsed_ms']:.0f} мс")
with st.expander("Итоговое описание"):
    st.write(req["output"])

steps = client().reasoning_log(rid)["steps"]
calls = client().llm_calls(rid)["calls"]

st.subheader("Шаги агентов")
if steps:
    timeline = pd.DataFrame([{"Шаг": f"{s['step']}. {s['agent_name']}", "start": s["started_at"],
                              "finish": s["finished_at"] or s["started_at"], "Статус": s["status"]} for s in steps])
    st.plotly_chart(px.timeline(timeline, x_start="start", x_end="finish", y="Шаг", color="Статус"),
                    use_container_width=True)
icons = {"ok": "✅", "warning": "⚠️", "error": "❌"}
for s in steps:
    with st.expander(f"{icons.get(s['status'], '•')} Шаг {s['step']}: {s['agent_name']} "
                     f"({s.get('duration_ms') or 0:.0f} мс) {s.get('model_name') or ''}"):
        st.markdown(f"**Вход:** {s['input_summary']}")
        st.markdown(f"**Выход:** {s['output_summary']}")
        st.json(s.get("parameters") or {})
        if s.get("error"):
            st.error(s["error"])

st.subheader("Вызовы LLM")
if calls:
    df = pd.DataFrame(calls)
    st.dataframe(df[["id", "agent_name", "provider", "model_name", "status", "latency_ms", "tokens_used",
                     "prompt_hash", "response_hash"]], use_container_width=True)
    for c in calls:
        with st.expander(f"#{c['id']} {c['agent_name']} → {c['provider']}:{c['model_name']} ({c['status']})"):
            st.markdown("**Промпт (начало):**")
            st.code(c["prompt_preview"] or "", language="text")
            st.markdown("**Ответ (начало):**")
            st.code(c["response_preview"] or c.get("error") or "", language="text")
else:
    st.caption("Для этого запроса нет LLM-вызовов (например, результат из кэша).")
