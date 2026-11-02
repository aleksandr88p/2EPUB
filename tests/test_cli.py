"""Tests for the user-facing `python -m bookparse` CLI."""

from __future__ import annotations

from pathlib import Path

from bookparse import cli
from experiments import build_preview_epub


def test_build_defaults_to_full_book_and_universal_output(
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        build_preview_epub, "main",
        lambda argv: calls.append(argv) or 0,
    )

    result = cli.main(["build", "books/en.epub", "books/ru.fb2"])

    assert result == 0
    assert calls == [[
        "--en", "books/en.epub",
        "--ru", "books/ru.fb2",
        "--all",
        "--output", str(Path("output") / "2epub.epub"),
    ]]


def test_build_preview_delegates_preview_and_custom_output(
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        build_preview_epub, "main",
        lambda argv: calls.append(argv) or 0,
    )

    result = cli.main([
        "build", "en.fb2", "ru.epub", "--preview",
        "--output", "output/sample.epub",
    ])

    assert result == 0
    assert calls == [[
        "--en", "en.fb2",
        "--ru", "ru.epub",
        "--output", "output/sample.epub",
    ]]


def test_existing_check_command_keeps_positional_en_ru(
    monkeypatch, capsys,
) -> None:
    books = {"EN.epub": object(), "RU.fb2": object()}
    report_calls = []
    monkeypatch.setattr(cli, "_parse_any", lambda path: books[str(path)])
    monkeypatch.setattr(
        cli, "build_report",
        lambda en, ru: report_calls.append((en, ru)) or ("check-report", []),
    )

    result = cli.main(["check", "EN.epub", "RU.fb2"])

    assert result == 0
    assert report_calls == [(books["EN.epub"], books["RU.fb2"])]
    assert "check-report" in capsys.readouterr().out
