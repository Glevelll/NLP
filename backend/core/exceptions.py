"""Собственные исключения приложения."""
from __future__ import annotations


class AppError(Exception):
    """Базовое исключение приложения."""

    status_code: int = 500

    def __init__(self, message: str = "", **details):
        super().__init__(message)
        self.message = message
        self.details = details


class LLMError(AppError):
    status_code = 502


class LLMUnavailableError(LLMError):
    status_code = 503


class LLMTimeoutError(LLMError):
    status_code = 504


class ProviderNotConfiguredError(LLMError):
    status_code = 400


class RetrievalError(AppError):
    status_code = 500


class NotFoundError(AppError):
    status_code = 404


class AuthError(AppError):
    status_code = 401


class PermissionDeniedError(AppError):
    status_code = 403


class ValidationAppError(AppError):
    status_code = 422
