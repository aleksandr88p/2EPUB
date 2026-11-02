"""Tests for the EPUB builder script (experiments/build_preview_epub.py).

Covers the --all mode: CLI flag, output path, the chapter plan (EN
book order, unpaired chapters kept EN-only), paragraph rendering and
the post-write verification. No ML, no real book files.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from ebooklib import epub

from bookparse.chapter_alignment import align_chapters
from bookparse.checks import ERROR, WARNING, Issue
from bookparse.models import Book, Chapter, Paragraph, Sentence
from bookparse.paragraph_reconstruction import BilingualParagraph
from experiments import build_preview_epub as builder
from experiments.build_preview_epub import (
    FULL_PATH,
    PREVIEW_PATH,
    _image_file_names,
    _build_parser,
    _chapter_plan,
    _en_only_paragraphs,
    _image_key,
    _output_path,
    _parse_book,
    _render_paragraph,
    _source_assets,
    _unpaired_reason,
    _validate_book_path,
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


def _write_minimal_fb2(path: Path) -> None:
    cover = base64.b64encode(b"fb2-cover").decode("ascii")
    inline = base64.b64encode(b"fb2-inline").decode("ascii")
    path.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0"
             xmlns:l="http://www.w3.org/1999/xlink">
  <description><title-info>
    <book-title>Sample</book-title><lang>en</lang>
    <coverpage><image l:href="#cover"/></coverpage>
  </title-info></description>
  <body><section>
    <title><p>Sample</p></title>
    <p>A sentence for alignment.</p>
    <image l:href="#inline"/>
  </section></body>
  <binary id="cover" content-type="image/jpeg">{cover}</binary>
  <binary id="inline" content-type="image/png">{inline}</binary>
</FictionBook>""",
        encoding="utf-8",
    )


def _write_minimal_epub(path: Path) -> None:
    book = epub.EpubBook()
    book.set_identifier("sample")
    book.set_title("Sample")
    book.set_language("en")
    book.set_cover("cover.jpg", b"epub-cover")
    image = epub.EpubImage(
        uid="inline", file_name="images/inline.png",
        media_type="image/png", content=b"epub-inline",
    )
    chapter = epub.EpubHtml(
        uid="chapter", file_name="text/chapter.xhtml",
        title="Sample", lang="en",
    )
    chapter.content = (
        '<html><body><h1 id="chapter">Sample</h1>'
        "<p>A sentence for alignment.</p>"
        '<p><img src="../images/inline.png" alt="inline"/></p>'
        "</body></html>"
    )
    book.add_item(image)
    book.add_item(chapter)
    book.toc = (epub.Link(
        "text/chapter.xhtml#chapter", "Sample", "chapter",
    ),)
    book.spine = [chapter]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(path), book)


# --- CLI / output path --------------------------------------------------

def test_all_flag_defaults_off() -> None:
    assert _build_parser().parse_args([
        "--en", "en.fb2", "--ru", "ru.fb2",
    ]).all_chapters is False


def test_all_flag_on() -> None:
    assert _build_parser().parse_args([
        "--all", "--en", "en.fb2", "--ru", "ru.fb2",
    ]).all_chapters is True


def test_output_paths() -> None:
    assert _output_path(False) == PREVIEW_PATH
    assert _output_path(True) == FULL_PATH
    assert PREVIEW_PATH.name == "2epub_preview.epub"
    assert FULL_PATH.name == "2epub_full.epub"


def test_book_paths_are_required() -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args([])


def test_custom_paths_parsed() -> None:
    args = _build_parser().parse_args([
        "--en", "books/english.fb2",
        "--ru", "books/russian.fb2",
        "--output", "output/my_book.epub",
        "--all",
    ])
    assert args.en == Path("books/english.fb2")
    assert args.ru == Path("books/russian.fb2")
    assert args.output == Path("output/my_book.epub")
    assert args.all_chapters is True


def test_validate_book_path_accepts_existing_fb2(tmp_path) -> None:
    f = tmp_path / "book.fb2"
    f.write_text("<FictionBook/>")
    assert _validate_book_path(f, "English") == f


def test_validate_book_path_accepts_epub(tmp_path) -> None:
    f = tmp_path / "book.epub"
    f.write_bytes(b"not parsed by path validation")
    assert _validate_book_path(f, "English") == f


def test_validate_book_path_rejects_missing(tmp_path) -> None:
    with pytest.raises(SystemExit, match="not found"):
        _validate_book_path(tmp_path / "missing.fb2", "English")


def test_validate_book_path_rejects_non_fb2(tmp_path) -> None:
    f = tmp_path / "book.txt"
    f.write_text("x")
    with pytest.raises(SystemExit, match=r"\.fb2 or \.epub"):
        _validate_book_path(f, "Russian")


@pytest.mark.parametrize(
    ("en_ext", "ru_ext"),
    [(".fb2", ".fb2"), (".fb2", ".epub"),
     (".epub", ".fb2"), (".epub", ".epub")],
)
def test_all_input_format_combinations(
    tmp_path, en_ext: str, ru_ext: str,
) -> None:
    en_path = tmp_path / f"en{en_ext}"
    ru_path = tmp_path / f"ru{ru_ext}"
    writers = {".fb2": _write_minimal_fb2, ".epub": _write_minimal_epub}
    writers[en_ext](en_path)
    writers[ru_ext](ru_path)

    en_book = _parse_book(_validate_book_path(en_path, "English"))
    ru_book = _parse_book(_validate_book_path(ru_path, "Russian"))

    assert en_book.format == en_ext[1:]
    assert ru_book.format == ru_ext[1:]
    pairs = align_chapters(en_book, ru_book)["pairs"]
    assert len(pairs) == 1
    assert pairs[0]["en_title"] == pairs[0]["ru_title"] == "Sample"


def test_source_assets_preserve_cover_and_inline_images(tmp_path) -> None:
    for ext, writer, inline_key, cover_key in (
        (".fb2", _write_minimal_fb2, "inline", "cover"),
        (
            ".epub", _write_minimal_epub,
            "images/inline.png", "cover.jpg",
        ),
    ):
        path = tmp_path / f"source{ext}"
        writer(path)
        book = _parse_book(path)
        images, actual_cover_key = _source_assets(path)
        paragraph_image_keys = {
            _image_key(image["src"])
            for chapter in book.chapters
            for paragraph in chapter.paragraphs
            for image in paragraph.images
        }

        assert actual_cover_key == cover_key
        assert inline_key in images
        assert inline_key in paragraph_image_keys
        assert images[actual_cover_key][1].endswith(b"cover")
        image_paths = _image_file_names(paragraph_image_keys, images)
        assert image_paths[inline_key].startswith("images/")


def test_missing_english_or_russian_file_is_reported(tmp_path) -> None:
    missing = tmp_path / "missing.fb2"
    for label in ("English", "Russian"):
        with pytest.raises(SystemExit, match=f"{label} book not found"):
            _validate_book_path(missing, label)


def test_preflight_ok_allows_conversion(monkeypatch, capsys) -> None:
    book = _book([])
    monkeypatch.setattr(
        builder, "build_report", lambda en, ru: ("VERDICT: OK", []),
    )

    assert builder._run_preflight(book, book) is True
    captured = capsys.readouterr().out
    assert "VERDICT: OK" in captured
    assert "Preflight: OK; continuing." in captured


def test_preflight_warning_allows_full_cli_generation(
    tmp_path, monkeypatch, capsys,
) -> None:
    en_path = tmp_path / "en.fb2"
    ru_path = tmp_path / "ru.epub"
    output_path = tmp_path / "nested" / "preview.epub"
    en_path.write_text("placeholder")
    ru_path.write_text("placeholder")

    en_book = _book([_chapter(
        0, "Sample", paras=[_paragraph("English sentence.")],
    )])
    ru_book = _book([])
    parsed = {en_path: en_book, ru_path: ru_book}
    report_calls = []

    monkeypatch.setattr(builder, "_parse_book", lambda path: parsed[path])
    monkeypatch.setattr(
        builder, "build_report",
        lambda en, ru: (
            report_calls.append((en, ru))
            or ("PREFLIGHT WARNING\nVERDICT: PROCEED WITH CAUTION", [
                Issue(WARNING, "english", "sample-warning", "Review this."),
            ])
        ),
    )
    monkeypatch.setattr(builder, "_source_assets", lambda path: ({}, None))

    result = builder.main([
        "--en", str(en_path), "--ru", str(ru_path),
        "--output", str(output_path), "--all",
    ])

    captured = capsys.readouterr().out
    assert result == 0
    assert output_path.is_file()
    assert report_calls == [(en_book, ru_book)]
    assert "PREFLIGHT WARNING" in captured
    assert "Preflight: 1 warning(s); continuing." in captured


def test_preflight_error_stops_before_alignment_or_generation(
    tmp_path, monkeypatch, capsys,
) -> None:
    en_path = tmp_path / "en.fb2"
    ru_path = tmp_path / "ru.fb2"
    output_path = tmp_path / "must-not-exist.epub"
    en_path.write_text("placeholder")
    ru_path.write_text("placeholder")
    book = _book([])

    monkeypatch.setattr(builder, "_parse_book", lambda _path: book)
    monkeypatch.setattr(
        builder, "build_report",
        lambda _en, _ru: ("PREFLIGHT ERROR\nVERDICT: STOP", [
            Issue(ERROR, "english", "invalid-source", "Cannot align."),
        ]),
    )
    monkeypatch.setattr(
        builder, "align_chapters",
        lambda *_args: pytest.fail("alignment must not start after ERROR"),
    )

    result = builder.main([
        "--en", str(en_path), "--ru", str(ru_path),
        "--output", str(output_path), "--all",
    ])

    assert result != 0
    assert not output_path.exists()
    assert "Preflight: ERROR — conversion stopped." in capsys.readouterr().out


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

def _mini_epub(path, bodies: list[str], spine_order=None) -> None:
    book = epub.EpubBook()
    book.set_language("en")
    book.set_identifier("test")
    book.set_title("test")
    items = []
    for k, body in enumerate(bodies):
        c = epub.EpubHtml(
            uid=f"chap_{k + 1}", file_name=f"chap_{k + 1}.xhtml",
            title=f"T{k + 1}",
        )
        c.content = f"<body>{body}</body>"
        book.add_item(c)
        items.append(c)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.toc = tuple(items)
    order = spine_order if spine_order is not None else range(len(items))
    book.spine = ["nav", *[items[i] for i in order]]
    epub.write_epub(str(path), book)


def _expected(n: int) -> list[tuple[str, list[BilingualParagraph]]]:
    return [
        (f"T{k + 1}", [
            BilingualParagraph(
                index=0, type="paragraph",
                en_sentences=[f"E{k}"], ru_sentences=[f"R{k}"],
            )
        ])
        for k in range(n)
    ]


def _bodies(n: int) -> list[str]:
    return [
        f'<p class="en"><strong>E{k}</strong></p><p class="ru">R{k}</p>'
        for k in range(n)
    ]


def test_verify_passes_on_complete_text(tmp_path) -> None:
    path = tmp_path / "good.epub"
    _mini_epub(path, _bodies(1))
    assert all(ok for _, ok in _verify(path, _expected(1)))


def test_verify_detects_missing_ru(tmp_path) -> None:
    path = tmp_path / "bad.epub"
    _mini_epub(path, ['<p class="en"><strong>E0</strong></p>'])
    assert not all(ok for _, ok in _verify(path, _expected(1)))


def test_verify_many_chapters_not_broken_by_name_sort(tmp_path) -> None:
    # Regression: lexicographic name sort misorders chap_10+ vs
    # chap_2 — presence must compare sets, order must use the spine.
    path = tmp_path / "many.epub"
    _mini_epub(path, _bodies(12))
    assert all(ok for _, ok in _verify(path, _expected(12)))


def test_verify_detects_wrong_spine_order(tmp_path) -> None:
    path = tmp_path / "reordered.epub"
    _mini_epub(path, _bodies(3), spine_order=[2, 1, 0])
    results = dict(_verify(path, _expected(3)))
    assert results["exactly the selected chapters present"] is True
    assert results["chapters in original order (spine)"] is False
