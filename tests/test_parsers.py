"""Tests for EPUB/FB2 structure extraction."""

from __future__ import annotations

from pathlib import Path

import pytest
from ebooklib import epub

from bookparse import parse_epub, parse_fb2
from bookparse.models import Book, Chapter, Paragraph
from bookparse.paragraph_reconstruction import rebuild_chapter
from bookparse.segment import count_words, split_sentences
from bookparse.sentence_alignment import AlignOp

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


def write_single_section_fb2(
    path: Path,
    chapter_count: int = 4,
    with_part: bool = True,
) -> None:
    blocks = [
        '<p>Publisher title page.</p>',
        *['<empty-line/>'] * (4 if with_part else 1),
    ]
    if with_part:
        blocks.extend([
            '<p>Division Alpha</p>', '<empty-line/>',
            '<p>Orientation Level</p>', '<empty-line/>',
        ])
    for index in range(chapter_count):
        letter = chr(ord("A") + index)
        blocks.extend([
            '<empty-line/>',
            f'<p>Anchor {letter}</p>', '<empty-line/>',
            f'<p>Arbitrary Heading {letter}</p>',
            '<empty-line/>', '<empty-line/>',
            (
                "<p>The narrative begins with enough ordinary words to "
                "distinguish this passage from a compact heading."
                " <emphasis>Styled phrase.</emphasis></p>"
            ),
            '<p>Brief.</p>', '<p>Wait.</p>',
            (
                "<p>Another ordinary paragraph provides context and keeps "
                "the short dialogue from resembling a structural marker.</p>"
            ),
        ])
        if index == 1:
            blocks.append('<image l:href="#inline"/>')
        blocks.append('<empty-line/>')

    path.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0"
             xmlns:l="http://www.w3.org/1999/xlink">
  <description><title-info>
    <book-title>Structure Sample</book-title><lang>en</lang>
  </title-info></description>
  <body><section>"""
        + "\n".join(blocks)
        + """</section></body>
  <binary id="inline" content-type="image/png">aW1hZ2U=</binary>
</FictionBook>""",
        encoding="utf-8",
    )


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
    assert book.chapter_recovery is None


def test_fb2_single_section_recovers_repeated_structure_and_parts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "single-section.fb2"
    write_single_section_fb2(source)

    original = parse_fb2(source, recover_chapters=False)
    recovered = parse_fb2(source)
    main = [chapter for chapter in recovered.chapters if chapter.body == "main"]

    assert recovered.chapter_recovery["status"] == "recovered"
    assert recovered.chapter_recovery["recovered_chapters"] == 4
    assert recovered.chapter_recovery["recovered_parts"] == 1
    assert [chapter.level for chapter in main] == [1, 2, 2, 2, 2]
    assert main[0].title == "Division Alpha — Orientation Level"
    assert main[1].title == "Anchor A — Arbitrary Heading A"
    assert main[0].paragraphs[0].type == "heading"
    assert main[0].paragraphs[2].type == "heading"
    assert all(chapter.body == "main" for chapter in main)


def test_fb2_recovery_preserves_all_paragraphs_order_and_images(
    tmp_path: Path,
) -> None:
    source = tmp_path / "single-section.fb2"
    write_single_section_fb2(source)
    original = parse_fb2(source, recover_chapters=False)
    recovered = parse_fb2(source)

    before = original.chapters[0].paragraphs
    after = [
        paragraph
        for chapter in recovered.chapters
        for paragraph in chapter.paragraphs
    ]
    assert [(p.text, p.order, p.source_id, p.images) for p in after] == [
        (p.text, p.order, p.source_id, p.images) for p in before
    ]
    assert [p.text for p in after].count("Division Alpha") == 1
    assert [p.text for p in after].count("Orientation Level") == 1
    assert [image for p in after for image in p.images] == [
        image for p in before for image in p.images
    ]
    assert recovered.stats()["paragraphs"] == original.stats()["paragraphs"]
    assert recovered.stats()["sentences"] == original.stats()["sentences"]
    assert recovered.stats()["words"] == original.stats()["words"]
    assert any("Styled phrase." in p.text for p in after)


def test_fb2_recovery_does_not_promote_short_dialogue_paragraphs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "single-section.fb2"
    write_single_section_fb2(source)
    book = parse_fb2(source)
    recovered_titles = [
        chapter.title for chapter in book.chapters if chapter.body == "main"
    ]
    assert len(recovered_titles) == 5  # one part and four real headings
    assert not any(title in {"Brief.", "Wait."} for title in recovered_titles)


def test_fb2_recovery_requires_repeated_evidence(tmp_path: Path) -> None:
    from bookparse.checks import build_report

    source = tmp_path / "few-markers.fb2"
    write_single_section_fb2(source, chapter_count=2, with_part=False)
    book = parse_fb2(source)

    assert len(book.chapters) == 1
    assert book.chapters[0].title is None
    assert book.chapter_recovery["status"] == "insufficient_evidence"
    assert book.chapter_recovery["chapter_count"] == 2
    reference = Book(
        source_file="reference.fb2", format="fb2", metadata={},
        chapters=[Chapter(
            id="ref-1", order=0, title="Reference",
            paragraphs=[Paragraph(
                id="ref-p", order=0, type="paragraph",
                text="A complete reference chapter.",
            )],
        )],
    )
    report, issues = build_report(book, reference)
    assert "no-chapters" in {issue.code for issue in issues}
    assert "VERDICT: STOP" in report


def test_check_and_builder_share_recovered_structure(tmp_path: Path) -> None:
    from bookparse.checks import build_report
    from experiments.build_preview_epub import _parse_book

    source = tmp_path / "single-section.fb2"
    write_single_section_fb2(source)
    checked_book = parse_fb2(source)
    built_book = _parse_book(source)
    assert [
        (chapter.title, chapter.level, chapter.body, chapter.order)
        for chapter in checked_book.chapters
    ] == [
        (chapter.title, chapter.level, chapter.body, chapter.order)
        for chapter in built_book.chapters
    ]

    reference = Book(
        source_file="reference.fb2",
        format="fb2",
        metadata={},
        chapters=[
            Chapter(
                id=f"ref-{index}", order=index, title=f"Reference {index}",
                paragraphs=[Paragraph(
                    id=f"p-{index}", order=index, type="paragraph",
                    text=("A structured reference chapter with enough "
                          "complete text for a volume comparison."),
                )],
            )
            for index in range(2)
        ],
    )
    report, issues = build_report(checked_book, reference)
    codes = {issue.code for issue in issues}
    assert "no-chapters" not in codes
    assert "chapter-count" in codes
    assert "chapter-recovery-reference" in codes
    assert "does not force chapter counts" in report


def test_short_paragraph_mode_accepts_recovered_heading_blocks(
    tmp_path: Path,
) -> None:
    source = tmp_path / "single-section.fb2"
    write_single_section_fb2(source)
    recovered = parse_fb2(source)
    chapter = next(c for c in recovered.chapters if c.level == 2)
    headings = [p for p in chapter.paragraphs if p.type == "heading"]

    assert len(headings) == 2
    assert all(len(p.sentences) == 1 for p in headings)
    source_paragraphs = [
        [sentence.text for sentence in paragraph.sentences]
        for paragraph in chapter.paragraphs
    ]
    source_sentences = [
        sentence for paragraph in source_paragraphs for sentence in paragraph
    ]
    translation_sentences = [
        f"Translation {index}." for index in range(len(source_sentences))
    ]
    operations = [
        AlignOp(
            kind="1:1", en_idx=(index,), ru_idx=(index,), score=1.0,
            en_text=source, ru_text=translation_sentences[index],
            quality="confident",
        )
        for index, source in enumerate(source_sentences)
    ]

    blocks = rebuild_chapter(
        source_paragraphs,
        translation_sentences,
        operations,
        para_types=[paragraph.type for paragraph in chapter.paragraphs],
        short_paragraphs=True,
    )
    assert [
        sentence for block in blocks for sentence in block.en_sentences
    ] == source_sentences
    assert [
        sentence for block in blocks for sentence in block.ru_sentences
    ] == translation_sentences
    assert sum(block.type == "heading" for block in blocks) == 2


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
    es = split_sentences(
        "El Sr. García llegó. ¿Está aquí? ¡Sí!", "es",
    )
    assert es == ["El Sr. García llegó.", "¿Está aquí?", "¡Sí!"]
    assert split_sentences(
        "La Dra. López respondió. Después se fue.", "spa",
    ) == ["La Dra. López respondió.", "Después se fue."]
    assert split_sentences(
        "Á. Pérez llegó. Luego llamó.", "es",
    ) == ["Á. Pérez llegó.", "Luego llamó."]
    assert count_words("Hello, world — test.") == 3


@pytest.mark.skipif(
    not BOOKS_DIR.is_dir() or not list(BOOKS_DIR.iterdir()),
    reason="real books not present",
)
def test_real_books() -> None:
    epubs = list(BOOKS_DIR.glob("*.epub"))
    fb2s = list(BOOKS_DIR.glob("*.fb2"))
    for paths, parser in ((epubs, parse_epub), (fb2s, parse_fb2)):
        if paths:
            assert any(
                (stats := parser(path).stats())["chapters"] > 10
                and stats["words"] > 10000
                for path in paths
            )
