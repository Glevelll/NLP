"""Human-in-the-Loop: проверка неоднозначных описаний редактором."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from frontend.utils.api_client import APIError  # noqa: E402
from frontend.utils.layout import page_heading  # noqa: E402
from frontend.utils.session import client, require_login  # noqa: E402

require_login("editor")
page_heading("Проверка редактором", "Исправьте, подтвердите или отклоните описание из очереди.", "fact_check")
st.caption("Сюда попадают описания с низкой оценкой Critic, непрошедшие валидацию и с плохими отзывами. "
           "Подтверждённые (и исправленные) описания добавляются в базу знаний и сразу участвуют в RAG.")

statuses = {"pending": "Ожидают проверки", "approved": "Подтверждены", "rejected": "Отклонены"}
status = st.radio("Статус", list(statuses), format_func=statuses.get, horizontal=True)
try:
    items = client().uncertain(status)
except APIError as exc:
    st.error(str(exc))
    st.stop()
if not items:
    st.info("В этой очереди пока нет описаний.")
for item in items:
    title = f"#{item['id']} · оценка {item['critic_score']}" + (f" · {item['input_json']['name']}"
                                                                if item.get("input_json") else "")
    with st.expander(title, expanded=status == "pending" and item is items[0]):
        st.markdown(f"**Причина:** {item['reason']}")
        st.text(item["input_text"])
        corrected = st.text_area("Описание (можно исправить)", item.get("corrected_output") or item["model_output"],
                                 height=250, key=f"txt_{item['id']}")
        if status == "pending":
            c1, c2 = st.columns(2)
            if c1.button("✅ Подтвердить и добавить в базу", key=f"ok_{item['id']}"):
                try:
                    changed = corrected if corrected != item["model_output"] else None
                    client().approve(item["id"], changed)
                    st.rerun()
                except APIError as exc:
                    st.error(str(exc))
            if c2.button("❌ Отклонить", key=f"no_{item['id']}"):
                client().reject(item["id"])
                st.rerun()
