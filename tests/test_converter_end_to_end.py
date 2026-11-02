"""Full converter CLI-path tests using tiny real EPUB/FB2 books.

The parsers, preflight checks, alignment algorithms, EPUB writer and
post-write verifier run for real. Only model construction and embedding
inference are stubbed.
"""

from __future__ import annotations

import base64
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from bs4 import BeautifulSoup
from ebooklib import epub, ITEM_DOCUMENT

from experiments import build_preview_epub as builder

PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/"
    "x8AAwMCAO+/p5sAAAAASUVORK5CYII="
)
CHAPTERS = [
    ("Arrival", "Прибытие", "The traveler arrived. Rain covered the road.",
     "Путник прибыл. Дождь залил дорогу."),
    ("Night", "Ночь", "The fire burned. Stars filled the sky.",
     "Костёр горел. Звёзды заполнили небо."),
]


def _write_fb2(path: Path, language: str) -> None:
    title_index = 0 if language == "en" else 1
    text_index = 2 if language == "en" else 3
    sections = []
    for index, chapter in enumerate(CHAPTERS, 1):
        title, body = chapter[title_index], chapter[text_index]
        image = '<image l:href="#inline"/>' if index == 1 and language == "en" else ""
        sections.append(
            f"<section><title><p>{title}</p></title>"
            f"<p>{body}</p>{image}</section>"
        )
    cover = base64.b64encode(PIXEL_PNG).decode("ascii")
    inline = base64.b64encode(PIXEL_PNG).decode("ascii")
    path.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0"
             xmlns:l="http://www.w3.org/1999/xlink">
  <description><title-info>
    <book-title>End-to-end test</book-title><lang>{language}</lang>
    <coverpage><image l:href="#cover"/></coverpage>
  </title-info></description>
  <body>{"".join(sections)}</body>
  <binary id="cover" content-type="image/png">{cover}</binary>
  <binary id="inline" content-type="image/png">{inline}</binary>
</FictionBook>""",
        encoding="utf-8",
    )


def _write_epub(path: Path, language: str) -> None:
    title_index = 0 if language == "en" else 1
    text_index = 2 if language == "en" else 3
    book = epub.EpubBook()
    book.set_identifier(f"test-{language}")
    book.set_title("End-to-end test")
    book.set_language(language)
    book.set_cover("cover.png", PIXEL_PNG)

    inline_image = epub.EpubImage(
        uid="inline-image", file_name="Images/inline.png",
        media_type="image/png", content=PIXEL_PNG,
    )
    book.add_item(inline_image)

    chapter_items = []
    toc = []
    for index, chapter in enumerate(CHAPTERS, 1):
        title, body = chapter[title_index], chapter[text_index]
        chapter_id = f"ch{index}"
        image = (
            '<p><img src="../Images/inline.png" alt="test image"/></p>'
            if index == 1 and language == "en" else ""
        )
        item = epub.EpubHtml(
            uid=chapter_id, file_name=f"Text/ch{index}.xhtml",
            title=title, lang=language,
        )
        item.content = (
            f'<html><body><h1 id="{chapter_id}">{title}</h1>'
            f"<p>{body}</p>{image}</body></html>"
        )
        book.add_item(item)
        chapter_items.append(item)
        toc.append(epub.Link(
            f"Text/ch{index}.xhtml#{chapter_id}", title, chapter_id,
        ))

    book.toc = tuple(toc)
    book.spine = chapter_items
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(path), book)


def _stub_qwen(monkeypatch) -> None:
    """Prevent model loading and return deterministic equal similarities."""
    torch_stub = types.ModuleType("torch")
    torch_stub.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False),
    )
    model_stub = types.ModuleType("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, *_args, **_kwargs):
            pytest.fail("the test embedding cache should bypass model inference")

    model_stub.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "sentence_transformers", model_stub)

    def cached_embeddings(_encode, texts, *_args, **_kwargs):
        return np.ones((len(texts), 8), dtype=np.float32), 0, 0.0

    monkeypatch.setattr(builder, "embed_with_cache", cached_embeddings)


def _write_input(path: Path, language: str) -> None:
    if path.suffix.lower() == ".fb2":
        _write_fb2(path, language)
    else:
        _write_epub(path, language)


def _read_chapter_docs(path: Path):
    book = epub.read_epub(str(path))
    docs = {
        item.get_name(): item
        for item in book.get_items_of_type(ITEM_DOCUMENT)
        if item.get_name().startswith("chap_")
    }
    spine_names = {
        item.get_id(): item.get_name() for item in book.get_items()
    }
    chapter_order = [
        spine_names[item_id]
        for item_id, _linear in book.spine
        if item_id in spine_names and spine_names[item_id].startswith("chap_")
    ]
    return book, docs, chapter_order


@pytest.mark.parametrize(
    ("en_ext", "ru_ext"),
    [(".fb2", ".fb2"), (".fb2", ".epub"),
     (".epub", ".fb2"), (".epub", ".epub")],
    ids=["fb2-fb2", "fb2-epub", "epub-fb2", "epub-epub"],
)
def test_full_converter_cli_succeeds_for_all_input_formats(
    tmp_path, monkeypatch, capsys, en_ext: str, ru_ext: str,
) -> None:
    _stub_qwen(monkeypatch)
    en_path = tmp_path / f"english{en_ext}"
    ru_path = tmp_path / f"russian{ru_ext}"
    output_path = tmp_path / "output" / "bilingual.epub"
    _write_input(en_path, "en")
    _write_input(ru_path, "ru")

    result = builder.main([
        "--en", str(en_path), "--ru", str(ru_path),
        "--output", str(output_path), "--all",
    ])

    log = capsys.readouterr().out
    assert result == 0
    assert output_path.is_file()
    assert "VERDICT: OK" in log
    assert "Preflight: OK; continuing." in log
    assert "Successfully converted:" in log
    assert "[FAIL]" not in log

    generated, docs, chapter_order = _read_chapter_docs(output_path)
    assert len(docs) == 2
    assert chapter_order == ["chap_1.xhtml", "chap_2.xhtml"]

    for index, (en_title, ru_title, en_text, ru_text) in enumerate(CHAPTERS, 1):
        soup = BeautifulSoup(
            docs[f"chap_{index}.xhtml"].get_content(), "lxml-xml",
        )
        en_blocks = [
            " ".join(node.get_text().split()) for node in soup.select(".en")
        ]
        ru_blocks = [
            " ".join(node.get_text().split()) for node in soup.select(".ru")
        ]
        assert en_blocks[0] == en_title
        assert ru_blocks[0] == ru_title
        assert en_blocks[1] == en_text
        assert ru_blocks[1] == ru_text

    names = {item.get_name() for item in generated.get_items()}
    first_doc = BeautifulSoup(
        docs["chap_1.xhtml"].get_content(), "lxml-xml",
    )
    image_src = first_doc.find("img")["src"]
    assert image_src in names
    assert generated.get_item_with_href(image_src).get_content() == PIXEL_PNG

    cover_item = generated.get_item_with_id("cover-img")
    assert cover_item is not None
    assert cover_item.get_content() == PIXEL_PNG
    cover_doc = generated.get_item_with_id("cover")
    assert cover_doc is not None
    cover_soup = BeautifulSoup(cover_doc.get_content(), "lxml-xml")
    cover_src = cover_soup.find("img")["src"]
    cover_asset = generated.get_item_with_href(cover_src)
    assert cover_asset is not None
    assert cover_asset.get_content() == PIXEL_PNG


def test_post_generation_verification_error_is_not_reported_as_success(
    tmp_path, monkeypatch, capsys,
) -> None:
    _stub_qwen(monkeypatch)
    en_path = tmp_path / "english.fb2"
    ru_path = tmp_path / "russian.fb2"
    output_path = tmp_path / "bilingual.epub"
    _write_fb2(en_path, "en")
    _write_fb2(ru_path, "ru")
    monkeypatch.setattr(
        builder, "_verify", lambda *_args: [("forced failure", False)],
    )

    result = builder.main([
        "--en", str(en_path), "--ru", str(ru_path),
        "--output", str(output_path), "--all",
    ])

    log = capsys.readouterr().out
    assert output_path.is_file()
    assert result != 0
    assert "[FAIL] forced failure" in log
    assert "Successfully converted" not in log
    assert "Conversion failed: EPUB verification did not pass." in log
