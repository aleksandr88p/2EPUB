"""Full converter CLI-path tests using tiny real EPUB/FB2 books.

The parsers, preflight checks, alignment algorithms, EPUB writer and
post-write verifier run for real. Only model construction and embedding
inference are stubbed.
"""

from __future__ import annotations

import base64
import sys
import types
from zipfile import ZipFile
from pathlib import Path

import numpy as np
import pytest
from bs4 import BeautifulSoup
from ebooklib import epub, ITEM_DOCUMENT

from bookparse.cli import main as bookparse_main
from bookparse.languages import LANGUAGE_NAMES
from experiments import build_preview_epub as builder

PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/"
    "x8AAwMCAO+/p5sAAAAASUVORK5CYII="
)
BOOK_CONTENT = {
    "en": [
        ("Arrival", "The traveler arrived. Rain covered the road."),
        ("Night", "The fire burned. Stars filled the sky."),
    ],
    "ru": [
        ("Прибытие", "Путник прибыл. Дождь залил дорогу."),
        ("Ночь", "Костёр горел. Звёзды заполнили небо."),
    ],
    "es": [
        ("Llegada", "El viajero llegó. La lluvia cubrió el camino."),
        ("Noche", "El fuego ardió. Las estrellas llenaron el cielo."),
    ],
}
LANGUAGE_TAGS = {"en": "eng", "ru": "rus", "es": "spa"}
_DEFAULT_METADATA = object()


def _write_fb2(
    path: Path,
    language: str,
    *,
    include_images: bool = True,
    metadata_lang: str | None | object = _DEFAULT_METADATA,
) -> None:
    metadata_lang = (
        LANGUAGE_TAGS[language]
        if metadata_lang is _DEFAULT_METADATA
        else metadata_lang
    )
    sections = []
    for index, (title, body) in enumerate(BOOK_CONTENT[language], 1):
        image = (
            '<image l:href="#inline"/>'
            if index == 1 and include_images else ""
        )
        sections.append(
            f"<section><title><p>{title}</p></title>"
            f"<p>{body}</p>{image}</section>"
        )
    cover = base64.b64encode(PIXEL_PNG).decode("ascii")
    inline = base64.b64encode(PIXEL_PNG).decode("ascii")
    language_element = (
        f"<lang>{metadata_lang}</lang>" if metadata_lang else ""
    )
    path.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0"
             xmlns:l="http://www.w3.org/1999/xlink">
  <description><title-info>
    <book-title>End-to-end test</book-title>{language_element}
    <coverpage><image l:href="#cover"/></coverpage>
  </title-info></description>
  <body>{"".join(sections)}</body>
  <binary id="cover" content-type="image/png">{cover}</binary>
  <binary id="inline" content-type="image/png">{inline}</binary>
</FictionBook>""",
        encoding="utf-8",
    )


def _write_epub(
    path: Path,
    language: str,
    *,
    include_images: bool = True,
    metadata_lang: str | None | object = _DEFAULT_METADATA,
) -> None:
    metadata_lang = (
        LANGUAGE_TAGS[language]
        if metadata_lang is _DEFAULT_METADATA
        else metadata_lang
    )
    book = epub.EpubBook()
    book.set_identifier(f"test-{language}")
    book.set_title("End-to-end test")
    if metadata_lang:
        book.set_language(str(metadata_lang))
    book.set_cover("cover.png", PIXEL_PNG)

    inline_image = epub.EpubImage(
        uid="inline-image", file_name="Images/inline.png",
        media_type="image/png", content=PIXEL_PNG,
    )
    book.add_item(inline_image)

    chapter_items = []
    toc = []
    for index, (title, body) in enumerate(BOOK_CONTENT[language], 1):
        chapter_id = f"ch{index}"
        image = (
            '<p><img src="../Images/inline.png" alt="test image"/></p>'
            if index == 1 and include_images else ""
        )
        item = epub.EpubHtml(
            uid=chapter_id, file_name=f"Text/ch{index}.xhtml",
            title=title, lang=str(metadata_lang) if metadata_lang else None,
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


def _write_input(
    path: Path,
    language: str,
    *,
    include_images: bool = True,
    metadata_lang: str | None | object = _DEFAULT_METADATA,
) -> None:
    if path.suffix.lower() == ".fb2":
        _write_fb2(
            path, language, include_images=include_images,
            metadata_lang=metadata_lang,
        )
    else:
        _write_epub(
            path, language, include_images=include_images,
            metadata_lang=metadata_lang,
        )


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
    _write_input(en_path, "en", include_images=True)
    _write_input(ru_path, "ru", include_images=False)

    result = bookparse_main([
        "build", str(en_path), str(ru_path),
        "--output", str(output_path),
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

    for index, ((en_title, en_text), (ru_title, ru_text)) in enumerate(
        zip(BOOK_CONTENT["en"], BOOK_CONTENT["ru"]), 1,
    ):
        soup = BeautifulSoup(
            docs[f"chap_{index}.xhtml"].get_content(), "lxml-xml",
        )
        en_blocks = [
            " ".join(node.get_text().split())
            for node in soup.select(".source")
        ]
        ru_blocks = [
            " ".join(node.get_text().split())
            for node in soup.select(".translation")
        ]
        assert en_blocks[0] == en_title
        assert ru_blocks[0] == ru_title
        assert en_blocks[1] == en_text
        assert ru_blocks[1] == ru_text
        xhtml_lang = soup.find("html").get("lang")
        assert xhtml_lang == "en"
        translated = soup.select(".translation")
        assert all(node.get("lang") == "ru" for node in translated)

    assert [v[0] for v in generated.get_metadata("DC", "language")] == [
        "en", "ru",
    ]
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


@pytest.mark.parametrize(
    ("source_lang", "translation_lang", "source_ext", "translation_ext"),
    [
        ("en", "ru", ".fb2", ".epub"),
        ("ru", "en", ".epub", ".fb2"),
        ("en", "es", ".fb2", ".epub"),
        ("es", "en", ".epub", ".fb2"),
        ("ru", "es", ".fb2", ".epub"),
        ("es", "ru", ".epub", ".fb2"),
    ],
    ids=["en-ru", "ru-en", "en-es", "es-en", "ru-es", "es-ru"],
)
def test_full_converter_supports_all_directed_language_pairs(
    tmp_path, monkeypatch, capsys,
    source_lang: str, translation_lang: str,
    source_ext: str, translation_ext: str,
) -> None:
    _stub_qwen(monkeypatch)
    source_path = tmp_path / f"source{source_ext}"
    translation_path = tmp_path / f"translation{translation_ext}"
    output_path = tmp_path / "out" / "bilingual.epub"
    _write_input(source_path, source_lang, include_images=True)
    _write_input(translation_path, translation_lang, include_images=False)

    result = bookparse_main([
        "build", str(source_path), str(translation_path),
        "--output", str(output_path),
    ])

    log = capsys.readouterr().out
    assert result == 0
    assert output_path.is_file()
    assert LANGUAGE_NAMES[source_lang].lower() in log.lower()
    assert "Successfully converted:" in log

    generated, docs, order = _read_chapter_docs(output_path)
    assert order == ["chap_1.xhtml", "chap_2.xhtml"]
    assert [
        value[0] for value in generated.get_metadata("DC", "language")
    ] == [source_lang, translation_lang]
    for index, (source_chapter, translation_chapter) in enumerate(
        zip(BOOK_CONTENT[source_lang], BOOK_CONTENT[translation_lang]), 1,
    ):
        soup = BeautifulSoup(
            docs[f"chap_{index}.xhtml"].get_content(), "lxml-xml",
        )
        source_blocks = [
            " ".join(node.get_text().split())
            for node in soup.select(".source")
        ]
        translation_blocks = [
            " ".join(node.get_text().split())
            for node in soup.select(".translation")
        ]
        assert source_blocks == list(source_chapter)
        assert translation_blocks == list(translation_chapter)
        with ZipFile(output_path) as archive:
            chapter_file = next(
                name for name in archive.namelist()
                if name.endswith(f"chap_{index}.xhtml")
            )
            stored_soup = BeautifulSoup(
                archive.read(chapter_file), "lxml-xml",
            )
        assert stored_soup.find("html").get("lang") == source_lang
        assert all(
            node.get("lang") == translation_lang
            for node in stored_soup.select(".translation")
        )

    first_chapter = BeautifulSoup(
        docs["chap_1.xhtml"].get_content(), "lxml-xml",
    )
    image_src = first_chapter.find("img")["src"]
    image_item = generated.get_item_with_href(image_src)
    assert image_item is not None
    assert image_item.get_content() == PIXEL_PNG
    cover_item = generated.get_item_with_id("cover-img")
    assert cover_item is not None and cover_item.get_content() == PIXEL_PNG


def test_language_overrides_replace_bad_or_missing_metadata(
    tmp_path, monkeypatch, capsys,
) -> None:
    _stub_qwen(monkeypatch)
    source_path = tmp_path / "spanish.fb2"
    translation_path = tmp_path / "english.epub"
    output_path = tmp_path / "override.epub"
    _write_fb2(source_path, "es", metadata_lang="fr")
    _write_epub(translation_path, "en", metadata_lang=None)

    result = bookparse_main([
        "build", str(source_path), str(translation_path),
        "--source-lang", "spa", "--translation-lang", "eng",
        "--output", str(output_path),
    ])

    assert result == 0
    assert "Successfully converted:" in capsys.readouterr().out
    generated = epub.read_epub(str(output_path))
    assert [
        value[0] for value in generated.get_metadata("DC", "language")
    ] == ["es", "en"]


def test_missing_language_metadata_keeps_legacy_en_ru_default(
    tmp_path, monkeypatch, capsys,
) -> None:
    _stub_qwen(monkeypatch)
    source_path = tmp_path / "source.fb2"
    translation_path = tmp_path / "translation.fb2"
    output_path = tmp_path / "legacy-default.epub"
    _write_fb2(source_path, "en", metadata_lang=None)
    _write_fb2(translation_path, "ru", metadata_lang=None)

    result = bookparse_main([
        "build", str(source_path), str(translation_path),
        "--output", str(output_path),
    ])

    assert result == 0
    assert "Successfully converted:" in capsys.readouterr().out
    generated = epub.read_epub(str(output_path))
    assert [
        value[0] for value in generated.get_metadata("DC", "language")
    ] == ["en", "ru"]


def test_unsupported_metadata_fails_before_generation(tmp_path) -> None:
    source_path = tmp_path / "unsupported.fb2"
    translation_path = tmp_path / "translation.fb2"
    output_path = tmp_path / "must-not-exist.epub"
    _write_fb2(source_path, "es", metadata_lang="fr")
    _write_fb2(translation_path, "ru")

    with pytest.raises(SystemExit, match="unsupported language 'fr'"):
        bookparse_main([
            "build", str(source_path), str(translation_path),
            "--output", str(output_path),
        ])
    assert not output_path.exists()


def test_unsupported_language_override_fails_clearly(tmp_path) -> None:
    source_path = tmp_path / "source.fb2"
    translation_path = tmp_path / "translation.fb2"
    output_path = tmp_path / "must-not-exist.epub"
    _write_fb2(source_path, "en")
    _write_fb2(translation_path, "ru")

    with pytest.raises(SystemExit, match="unsupported language 'fr'"):
        bookparse_main([
            "build", str(source_path), str(translation_path),
            "--source-lang", "fr", "--output", str(output_path),
        ])
    assert not output_path.exists()


def test_post_generation_verification_error_is_not_reported_as_success(
    tmp_path, monkeypatch, capsys,
) -> None:
    _stub_qwen(monkeypatch)
    en_path = tmp_path / "english.fb2"
    ru_path = tmp_path / "russian.fb2"
    output_path = tmp_path / "bilingual.epub"
    _write_fb2(en_path, "en")
    _write_fb2(ru_path, "ru", include_images=False)
    monkeypatch.setattr(
        builder, "_verify", lambda *_args: [("forced failure", False)],
    )

    result = bookparse_main([
        "build", str(en_path), str(ru_path),
        "--output", str(output_path),
    ])

    log = capsys.readouterr().out
    assert output_path.is_file()
    assert result != 0
    assert "[FAIL] forced failure" in log
    assert "Successfully converted" not in log
    assert "Conversion failed: EPUB verification did not pass." in log
