"""Tests for the user-facing `python -m bookparse` CLI."""

from __future__ import annotations

from pathlib import Path

from bookparse import cli
from bookparse.models import Book, Chapter, Paragraph, Sentence
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


def test_build_forwards_language_overrides(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        build_preview_epub, "main",
        lambda argv: calls.append(argv) or 0,
    )

    result = cli.main([
        "build", "books/source.epub", "books/translation.fb2",
        "--source-lang", "spa", "--translation-lang", "eng",
    ])

    assert result == 0
    assert calls == [[
        "--en", "books/source.epub",
        "--ru", "books/translation.fb2",
        "--all",
        "--source-lang", "spa",
        "--translation-lang", "eng",
        "--output", str(Path("output") / "2epub.epub"),
    ]]


def test_existing_check_command_keeps_positional_en_ru(
    monkeypatch, capsys,
) -> None:
    books = {
        "EN.epub": Book("EN.epub", "epub", {"language": "en"}),
        "RU.fb2": Book("RU.fb2", "fb2", {"language": "ru"}),
    }
    report_calls = []
    monkeypatch.setattr(
        cli, "_parse_any", lambda path, lang=None: books[str(path)],
    )
    monkeypatch.setattr(
        cli, "build_report",
        lambda source, translation, *labels: (
            report_calls.append((source, translation, *labels))
            or ("check-report", [])
        ),
    )

    result = cli.main(["check", "EN.epub", "RU.fb2"])

    assert result == 0
    assert report_calls == [
        (books["EN.epub"], books["RU.fb2"], "English", "Russian"),
    ]
    assert "check-report" in capsys.readouterr().out


def test_check_and_alignment_reports_use_metadata_languages(
    tmp_path, monkeypatch, capsys,
) -> None:
    source = Book(
        "source.epub", "epub", {"language": "spa"},
        chapters=[Chapter(
            "s1", 0, "Llegada",
            paragraphs=[Paragraph(
                "sp1", 0, "paragraph", "Un texto suficientemente largo.",
                sentences=[Sentence("ss1", 0, "Un texto suficientemente largo.")],
            )],
        )],
    )
    translation = Book(
        "translation.fb2", "fb2", {"language": "eng"},
        chapters=[Chapter(
            "t1", 0, "Arrival",
            paragraphs=[Paragraph(
                "tp1", 0, "paragraph", "A sufficiently long passage.",
                sentences=[Sentence("ts1", 0, "A sufficiently long passage.")],
            )],
        )],
    )
    books = {"source.epub": source, "translation.fb2": translation}
    monkeypatch.setattr(
        cli, "_parse_any",
        lambda path, lang=None: books[str(path)],
    )

    check_result = cli.main(["check", "source.epub", "translation.fb2"])
    check_report = capsys.readouterr().out
    assert check_result == 0
    assert "Spanish:" in check_report and "English:" in check_report

    output = tmp_path / "alignment.json"
    align_result = cli.main([
        "align-chapters", "source.epub", "translation.fb2",
        "--output", str(output),
    ])
    align_report = capsys.readouterr().out
    assert align_result == 0
    assert "Spanish: source.epub" in align_report
    assert "English: translation.fb2" in align_report
