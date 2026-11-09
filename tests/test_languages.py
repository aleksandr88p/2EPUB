"""Tests for supported language codes and metadata resolution."""

from __future__ import annotations

import pytest

from bookparse.languages import (
    UnsupportedLanguageError,
    normalize_language,
    resolve_language_pair,
)
from bookparse.models import Book


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("en", "en"), ("eng", "en"), ("English", "en"),
        ("en-US", "en"), ("ru", "ru"), ("rus", "ru"),
        ("ru_RU", "ru"), ("es", "es"), ("spa", "es"),
        ("Spanish", "es"), ("es-ES", "es"), ("español", "es"),
    ],
)
def test_normalize_supported_language_codes(value: str, expected: str) -> None:
    assert normalize_language(value) == expected


@pytest.mark.parametrize("value", ["fr", "de", "xx"])
def test_reject_unsupported_language_codes(value: str) -> None:
    with pytest.raises(UnsupportedLanguageError, match="supported languages"):
        normalize_language(value)


def test_resolve_pair_uses_metadata_then_overrides() -> None:
    source = Book("source", "epub", {"language": "spa"})
    translation = Book("translation", "fb2", {"language": "rus"})

    assert resolve_language_pair(source, translation) == ("es", "ru")
    assert resolve_language_pair(
        source, translation, "en-US", "español",
    ) == ("en", "es")


def test_resolve_pair_preserves_legacy_fallback_without_metadata() -> None:
    source = Book("source", "fb2", {})
    translation = Book("translation", "epub", {})

    assert resolve_language_pair(source, translation) == ("en", "ru")
