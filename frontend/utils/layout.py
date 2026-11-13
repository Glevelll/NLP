"""Общее оформление страниц."""
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).resolve().parents[1] / "assets"


def apply_styles() -> None:
    st.html(ASSETS / "style.css")


def page_heading(title: str, description: str) -> None:
    with st.container(key="page-heading"):
        st.title(title)
        st.markdown(description)


def sidebar_brand() -> None:
    st.sidebar.html(
        '<div class="studio-brand"><span class="studio-mark" aria-hidden="true">К</span>'
        '<span>Карточки<small>Генератор описаний</small></span></div>'
    )
