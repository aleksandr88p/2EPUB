"""FB2 (FictionBook) structure extraction via BeautifulSoup XML parsing.

Each <section> becomes a chapter; nested sections become nested chapters
(level reflects depth). All <body> elements are processed, including
auxiliary bodies such as "notes" or "comments". Titles, epigraphs,
poems, citations, images and empty lines are preserved as typed
paragraphs so no text is lost.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from bs4 import BeautifulSoup, Tag

from .models import Book, Chapter
from .epub_parser import _ChapterBuilder

Block = tuple[str, str, str | None, list[dict[str, str | None]]]
# (type, text, source_id, images)

_INNER_P_TYPES = {"title": "heading", "epigraph": "epigraph", "cite": "cite", "poem": "verse"}


def _text(el: Tag) -> str:
    return " ".join(el.get_text().split())


def _image_href(el: Tag) -> str | None:
    return el.get("l:href") or el.get("xlink:href") or el.get("href")


def _blocks(el: Tag, context: tuple[str, ...]) -> Iterator[Block]:
    name = el.name
    if name == "title" or name in {"epigraph", "cite", "poem", "stanza"}:
        for child in el.children:
            if isinstance(child, Tag):
                yield from _blocks(child, context + (name,))
    elif name == "subtitle":
        yield ("subtitle", _text(el), el.get("id"), [])
    elif name == "v":
        yield ("verse", _text(el), el.get("id"), [])
    elif name == "p":
        p_type = _INNER_P_TYPES.get(context[-1], "paragraph") if context else "paragraph"
        yield (p_type, _text(el), el.get("id"), [])
    elif name == "text-author":
        yield ("text-author", _text(el), el.get("id"), [])
    elif name == "image":
        yield ("image", "", el.get("id"), [{"src": _image_href(el), "alt": None, "id": el.get("id")}])
    elif name == "empty-line":
        yield ("empty-line", "", el.get("id"), [])
    elif name == "table":
        yield ("table", _text(el), el.get("id"), [])
    elif name == "section":
        raise ValueError("nested <section> must be handled by the caller")
    else:
        text = _text(el)
        if text or el.find("image"):
            yield ("paragraph", text, el.get("id"), [])


def _title_text(section: Tag) -> str | None:
    title = section.find("title", recursive=False)
    if title is None:
        return None
    return _text(title) or None


def _consume_section(
    section: Tag, level: int, body_name: str,
    book: Book, builder: _ChapterBuilder,
) -> None:
    chapter = builder.new_chapter(
        title=_title_text(section),
        level=level,
        source_id=section.get("id"),
    )
    chapter.body = body_name
    for child in section.children:
        if not isinstance(child, Tag):
            continue
        if child.name == "section":
            _consume_section(child, level + 1, body_name, book, builder)
            continue
        for p_type, text, source_id, images in _blocks(child, ()):
            builder.add_paragraph(chapter, p_type, text, source_id=source_id, images=images)


def _metadata(soup: BeautifulSoup) -> dict[str, str]:
    info = soup.find("title-info")
    if info is None:
        return {}
    author = info.find("author")
    author_name = " ".join(
        filter(None, [
            author.find("first-name").get_text(strip=True) if author and author.find("first-name") else "",
            author.find("last-name").get_text(strip=True) if author and author.find("last-name") else "",
        ])
    )
    title = info.find("book-title")
    lang = info.find("lang")
    return {
        "title": title.get_text(strip=True) if title else "",
        "author": author_name,
        "language": lang.get_text(strip=True) if lang else "",
    }


def parse_fb2(path: str | Path, lang: str | None = None) -> Book:
    path = Path(path)
    soup = BeautifulSoup(path.read_bytes(), "lxml-xml")

    meta = _metadata(soup)
    lang = lang or meta.get("language") or None
    book = Book(source_file=str(path), format="fb2", metadata=meta)
    builder = _ChapterBuilder(book, lang)

    for body in soup.find_all("body"):
        body_name = body.get("name") or "main"
        chapter: Chapter | None = None
        for child in body.children:
            if not isinstance(child, Tag):
                continue
            if child.name == "section":
                _consume_section(child, 1, body_name, book, builder)
                chapter = None
                continue
            # Stray content at body level (e.g. book <title>) goes into
            # an implicit chapter so nothing is lost.
            if chapter is None:
                chapter = builder.new_chapter(None, 1, None)
                chapter.body = body_name
            for p_type, text, source_id, images in _blocks(child, ()):
                builder.add_paragraph(chapter, p_type, text, source_id=source_id, images=images)

    return book
