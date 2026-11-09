"""Supported language codes and metadata normalization."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Book

LANGUAGE_NAMES = {
    "en": "English",
    "ru": "Russian",
    "es": "Spanish",
}

_ALIASES = {
    "en": "en",
    "eng": "en",
    "english": "en",
    "ru": "ru",
    "rus": "ru",
    "russian": "ru",
    "es": "es",
    "spa": "es",
    "spanish": "es",
    "español": "es",
    "castellano": "es",
}


class UnsupportedLanguageError(ValueError):
    """Raised when a language is outside the MVP's supported set."""


def normalize_language(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    tag = value.strip().lower().replace("_", "-").split("-", 1)[0]
    language = _ALIASES.get(tag)
    if language is None:
        supported = ", ".join(LANGUAGE_NAMES)
        raise UnsupportedLanguageError(
            f"unsupported language {value!r}; supported languages: {supported}"
        )
    return language


def language_name(code: str) -> str:
    normalized = normalize_language(code)
    if normalized is None:
        raise UnsupportedLanguageError("language code cannot be empty")
    return LANGUAGE_NAMES[normalized]


def resolve_language_pair(
    source_book: Book,
    translation_book: Book,
    source_override: str | None = None,
    translation_override: str | None = None,
) -> tuple[str, str]:
    """Use overrides, then metadata, with the legacy EN→RU fallback."""
    source_value = (
        source_override
        if source_override is not None
        else source_book.metadata.get("language")
    )
    translation_value = (
        translation_override
        if translation_override is not None
        else translation_book.metadata.get("language")
    )
    source_language = normalize_language(source_value) or "en"
    translation_language = normalize_language(translation_value) or "ru"
    return source_language, translation_language
