"""Общие функции страниц: авторизация и клиент API из session_state."""
from __future__ import annotations

import streamlit as st

from frontend.utils.api_client import APIClient, APIError


def client() -> APIClient:
    return APIClient(st.session_state.get("token"))


def require_login(min_role: str | None = None) -> dict:
    user = st.session_state.get("user")
    if not user:
        st.warning("Войдите в систему на главной странице.")
        st.stop()
    levels = {"user": 0, "editor": 1, "admin": 2}
    if min_role and levels.get(user["role"], 0) < levels[min_role]:
        st.error(f"Страница доступна для роли «{min_role}» и выше.")
        st.stop()
    return user


def _show_auth_error(error: APIError, placeholders: dict) -> None:
    if not error.field_errors:
        st.error(str(error))
        return
    for field, message in error.field_errors.items():
        placeholders[field].error(message)


def login_form() -> None:
    tab_login, tab_reg = st.tabs(["Вход", "Регистрация"])
    with tab_login, st.form("login"):
        username = st.text_input("Имя пользователя или email")
        username_error = st.empty()
        password = st.text_input("Пароль", type="password")
        password_error = st.empty()
        if st.form_submit_button("Войти", type="primary"):
            try:
                token = APIClient().login(username, password)
                st.session_state["token"] = token["access_token"]
                st.session_state["user"] = APIClient(token["access_token"]).me()
                st.rerun()
            except APIError as exc:
                _show_auth_error(exc, {"username": username_error, "password": password_error})
    with tab_reg, st.form("register"):
        email = st.text_input("Email")
        email_error = st.empty()
        username = st.text_input("Имя пользователя", key="reg_username")
        username_error = st.empty()
        password = st.text_input("Пароль (от 6 символов)", type="password", key="reg_password")
        password_error = st.empty()
        if st.form_submit_button("Зарегистрироваться"):
            try:
                APIClient().register(email, username, password)
                st.success("Аккаунт создан. Войдите на вкладке «Вход».")
            except APIError as exc:
                _show_auth_error(exc, {"email": email_error, "username": username_error, "password": password_error})


def sidebar_user() -> None:
    user = st.session_state.get("user")
    if user:
        st.sidebar.markdown(f"**{user['username']}** · роль `{user['role']}`")
        if st.sidebar.button("Выйти"):
            st.session_state.clear()
            st.rerun()
