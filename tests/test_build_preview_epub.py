"""Tests for the EPUB builder script (experiments/build_preview_epub.py).

Covers the --all mode: CLI flag, output path, the chapter plan (EN
book order, unpaired chapters kept EN-only), paragraph rendering and
the post-write verification. No ML, no real book files.
"""

from __future__ import annotations

from ebooklib import epub

from bookparse.models import Book, Chapter, Paragraph, Sentence
from bookparse.paragraph_reconstruction import BilingualParagraph
from experiments.build_preview_epub import (
    FULL_PATH,
    PREVIEW_PATH,
    _build_parser,
    _chapter_plan,
    _en_only_paragraphs,
    _output_path,
    _render_paragraph,
    _unpaired_reason,
    _verify,
)


def _sentence(text: str, order: int = 0) -> Sentence:
    return Sentence(id=f"s-{order}", order=order, text=text)


def _paragraph(
    text: str,
    ptype: str = "paragraph",
    order: int = 0,
    images: list | None = None,
) -> Paragraph:
    p = Paragraph(
        id=f"p-{order}", order=order, type=ptype, text=text,
        images=images or [],
    )
    if text:
        p.sentences = [_sentence(text, order)]
    return p


def _chapter(
    order: int,
    title: str | None = None,
    body: str = "main",
    paras: list | None = None,
) -> Chapter:
    return Chapter(
        id=f"ch-{order:03d}", order=order, title=title, body=body,
        paragraphs=list(paras or []),
    )


def _book(chapters: list[Chapter]) -> Book:
    return Book(
        source_file="x.fb2", format="fb2", metadata={},
        chapters=list(chapters),
    )


# --- CLI / output path --------------------------------------------------

def test_all_flag_defaults_off() -> None:
    assert _build_parser().parse_args([]).all_chapters is False


def test_all_flag_on() -> None:
    assert _build_parser().parse_args(["--all"]).all_chapters is True


def test_output_paths() -> None:
    assert _output_path(False) == PREVIEW_PATH
    assert _output_path(True) == FULL_PATH
    assert PREVIEW_PATH.name == "2epub_preview.epub"
    assert FULL_PATH.name == "2epub_full.epub"


# --- chapter plan -------------------------------------------------------

def test_chapter_plan_covers_all_en_chapters_in_order() -> None:
    book = _book([
        _chapter(0),                    # untitled front matter
        _chapter(1, "One"),
        _chapter(2),                    # unpaired mid-book chapter
        _chapter(3, "Two"),
    ])
    pairs = [
        {"en_order": 1, "ru_order": 0},
        {"en_order": 3, "ru_order": 2},
    ]
    plan = _chapter_plan(book, pairs)
    assert [ch.order for ch, _ in plan] == [0, 1, 2, 3]
    assert [pair is not None for _, pair in plan] == [
        False, True, False, True,
    ]
    assert plan[1][1] is pairs[0]
    assert plan[3][1] is pairs[1]


def test_unpaired_reason() -> None:
    assert _unpaired_reason(_chapter(0, title="A")) == "unmatched"
    assert _unpaired_reason(_chapter(0)) == "untitled-front-matter"
    assert (
        _unpaired_reason(_chapter(0, title="N", body="notes"))
        == "auxiliary-body:notes"
    )


def test_en_only_paragraphs_keep_text_types_and_no_ru() -> None:
    ch = _chapter(0, "Ack", paras=[
        _paragraph("Head", "heading", 0),
        _paragraph("Body text.", "paragraph", 1),
        _paragraph("", "image", 2, images=[{"src": "#img1"}]),
    ])
    bps = _en_only_paragraphs(ch)
    assert [bp.index for bp in bps] == [0, 1, 2]
    assert [bp.type for bp in bps] == ["heading", "paragraph", "image"]
    assert bps[1].en_sentences == ["Body text."]
    assert all(bp.ru_sentences == [] for bp in bps)


# --- rendering -----------------------------------------------------------

def test_render_en_bold_then_ru_normal() -> None:
    src = _paragraph("", order=0)
    bp = BilingualParagraph(
        index=0, type="paragraph",
        en_sentences=["Hello."], ru_sentences=["Привет."],
    )
    html = _render_paragraph(bp, src, {})
    assert html.index('class="en') < html.index('class="ru')
    assert "<strong>Hello.</strong>" in html
    assert ">Привет.<" in html


def test_render_empty_ru_not_fabricated() -> None:
    src = _paragraph("", order=0)
    bp = BilingualParagraph(
        index=0, type="paragraph", en_sentences=["Solo."],
    )
    html = _render_paragraph(bp, src, {})
    assert "<strong>Solo.</strong>" in html
    assert 'class="ru"' not in html
    assert "no translation" not in html.lower()


def test_render_escapes_markup() -> None:
    src = _paragraph("", order=0)
    bp = BilingualParagraph(
        index=0, type="paragraph",
        en_sentences=["a < b & c"], ru_sentences=["х < у"],
    )
    html = _render_paragraph(bp, src, {})
    assert "a &lt; b &amp; c" in html
    assert "х &lt; у" in html


def test_render_heading_and_quote_types() -> None:
    src = _paragraph("", order=0)
    heading = _render_paragraph(
        BilingualParagraph(
            index=0, type="heading",
            en_sentences=["Title"], ru_sentences=["Титул"],
        ),
        src, {},
    )
    assert "<h2" in heading and 'class="ru"' in heading

    quote = _render_paragraph(
        BilingualParagraph(
            index=0, type="epigraph",
            en_sentences=["Ep."], ru_sentences=["Эп."],
        ),
        src, {},
    )
    assert "<blockquote>" in quote


# --- post-write verification --------------------------------------------

def _mini_epub(path, body_html: str) -> None:
    book = epub.EpubBook()
    book.set_language("en")
    book.set_identifier("test")
    book.set_title("test")
    c = epub.EpubHtml(uid="chap_1", file_name="chap_1.xhtml", title="T")
    c.content = f"<body>{body_html}</body>"
    book.add_item(c)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.toc = (c,)
    book.spine = ["nav", c]
    epub.write_epub(str(path), book)


def test_verify_passes_on_complete_text(tmp_path) -> None:
    bps = [
        BilingualParagraph(
            index=0, type="paragraph",
            en_sentences=["E0", "E1"], ru_sentences=["R0"],
        )
    ]
    path = tmp_path / "good.epub"
    _mini_epub(
        path,
        '<p class="en"><strong>E0 E1</strong></p><p class="ru">R0</p>',
    )
    assert all(ok for _, ok in _verify(path, [("T", bps)]))


def test_verify_detects_missing_ru(tmp_path) -> None:
    bps = [
        BilingualParagraph(
            index=0, type="paragraph",
            en_sentences=["E0", "E1"], ru_sentences=["R0"],
        )
    ]
    path = tmp_path / "bad.epub"
    _mini_epub(path, '<p class="en"><strong>E0 E1</strong></p>')
    assert not all(ok for _, ok in _verify(path, [("T", bps)]))
