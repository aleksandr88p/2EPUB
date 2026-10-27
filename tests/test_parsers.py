"""Tests for EPUB/FB2 structure extraction."""

from __future__ import annotations

from pathlib import Path

import pytest
from ebooklib import epub

from bookparse import parse_epub, parse_fb2
from bookparse.segment import count_words, split_sentences

BOOKS_DIR = Path(__file__).resolve().parent.parent / "books"


def make_epub(path: Path) -> None:
    """Build a small EPUB: one XHTML file containing two chapters
    delimited by anchors in the TOC, plus an image."""
    book = epub.EpubBook()
    book.set_identifier("test-id")
    book.set_title("Test Book")
    book.set_language("en")

    img = epub.EpubImage(
        uid="img1", file_name="cover.jpg", media_type="image/jpeg",
        content=b"\xff\xd8\xff\xd9",
    )
    doc = epub.EpubHtml(uid="doc1", file_name="text.xhtml", lang="en")
    doc.content = (
        '<html><body>'
        '<p><a id="c1"/><b>Chapter One</b></p>'
        '<p><a id="p1"/>First sentence. Second sentence!</p>'
        '<p><img src="cover.jpg" alt="cover"/></p>'
        '<p><a id="c2"/><b>Chapter Two</b></p>'
        '<p>Last paragraph.</p>'
        '</body></html>'
    )
    book.add_item(img)
    book.add_item(doc)
    book.toc = [epub.Link("text.xhtml#c1", "Chapter One", "c1"),
                epub.Link("text.xhtml#c2", "Chapter Two", "c2")]
    book.spine = ["doc1"]
    book.add_item(epub.EpubNcx())
    epub.write_epub(str(path), book)


FB2_SAMPLE = """<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0"
             xmlns:l="http://www.w3.org/1999/xlink">
<description><title-info>
  <author><first-name>Joe</first-name><last-name>Abercrombie</last-name></author>
  <book-title>Test FB2</book-title>
  <lang>ru</lang>
</title-info></description>
<body>
  <section>
    <title><p>Часть первая</p></title>
    <epigraph><p>Эпиграф первой части.</p></epigraph>
    <p>Первый абзац. Второе предложение?</p>
    <empty-line/>
    <image l:href="#img"/>
  </section>
  <section>
    <title><p>Внешняя</p></title>
    <section>
      <title><p>Вложенная</p></title>
      <poem><stanza><v>Строка стиха раз</v><v>Строка стиха два</v></stanza></poem>
      <p>Конец.</p>
    </section>
  </section>
</body>
<body name="notes">
  <section id="n1">
    <title><p>Примечания</p></title>
    <p>Текст примечания.</p>
  </section>
</body>
</FictionBook>
"""


def test_epub_chapters_from_toc_not_files(tmp_path: Path) -> None:
    src = tmp_path / "test.epub"
    make_epub(src)
    book = parse_epub(src)

    assert book.format == "epub"
    assert book.metadata["title"] == "Test Book"
    # Two chapters come from TOC anchors, not from file boundaries.
    assert [c.title for c in book.chapters] == ["Chapter One", "Chapter Two"]
    assert book.chapters[0].source_id == "text.xhtml#c1"

    ch1 = book.chapters[0]
    assert ch1.paragraphs[0].type == "heading"
    assert ch1.paragraphs[0].source_id == "c1"
    assert ch1.paragraphs[1].source_id == "p1"
    assert [s.text for s in ch1.paragraphs[1].sentences] == [
        "First sentence.", "Second sentence!"
    ]
    assert ch1.paragraphs[2].type == "image"
    assert ch1.paragraphs[2].images[0]["src"] == "cover.jpg"

    stats = book.stats()
    assert stats["chapters"] == 2
    assert stats["sentences"] >= 4


def test_fb2_structure(tmp_path: Path) -> None:
    src = tmp_path / "test.fb2"
    src.write_text(FB2_SAMPLE, encoding="utf-8")
    book = parse_fb2(src)

    assert book.format == "fb2"
    assert book.metadata["title"] == "Test FB2"
    assert book.metadata["author"] == "Joe Abercrombie"

    titles = [(c.title, c.level, c.body) for c in book.chapters]
    assert titles == [
        ("Часть первая", 1, "main"),
        ("Внешняя", 1, "main"),
        ("Вложенная", 2, "main"),
        ("Примечания", 1, "notes"),
    ]

    ch1 = book.chapters[0]
    types = [p.type for p in ch1.paragraphs]
    assert types == ["heading", "epigraph", "paragraph", "empty-line", "image"]
    assert ch1.paragraphs[4].images[0]["src"] == "#img"

    nested = book.chapters[2]
    verse_types = [p.type for p in nested.paragraphs]
    assert "verse" in verse_types and "heading" in verse_types

    notes = book.chapters[3]
    assert notes.source_id == "n1"
    assert notes.paragraphs[1].text == "Текст примечания."


def test_sentence_splitter() -> None:
    assert split_sentences("One. Two! Three?", "en") == ["One.", "Two!", "Three?"]
    assert split_sentences("Mr. Smith left. He came back.", "en") == [
        "Mr. Smith left.", "He came back."
    ]
    assert split_sentences("J. R. R. Tolkien wrote books.", "en") == [
        "J. R. R. Tolkien wrote books."
    ]
    ru = split_sentences("— Дерьмо! — выругался Логен. Он ушёл.", "ru")
    assert ru == ["— Дерьмо!", "— выругался Логен.", "Он ушёл."]
    assert count_words("Hello, world — test.") == 3


@pytest.mark.skipif(
    not BOOKS_DIR.is_dir() or not list(BOOKS_DIR.iterdir()),
    reason="real books not present",
)
def test_real_books() -> None:
    epubs = list(BOOKS_DIR.glob("*.epub"))
    fb2s = list(BOOKS_DIR.glob("*.fb2"))
    if epubs:
        book = parse_epub(epubs[0])
        stats = book.stats()
        assert stats["chapters"] > 10
        assert stats["words"] > 10000
    if fb2s:
        book = parse_fb2(fb2s[0])
        stats = book.stats()
        assert stats["chapters"] > 10
        assert stats["words"] > 10000
