"""EPUB structure extraction via EbookLib + BeautifulSoup.

Chapters are delimited by the book's TOC (NCX navPoints / EPUB3 nav),
which usually point to anchors inside XHTML files — one spine file can
contain several chapters. If the book has no usable TOC, top-level
headings are used as chapter boundaries; as a last resort each spine
file becomes a chapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

from bs4 import BeautifulSoup, NavigableString, Tag
from ebooklib import epub

from .models import Book, Chapter, Paragraph, Sentence
from .segment import split_sentences

LEAF_BLOCK_TAGS = {
    "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "blockquote", "pre", "li", "dt", "dd", "td", "th",
    "figcaption", "caption", "address",
}
CONTAINER_TAGS = {
    "body", "div", "section", "article", "aside", "main", "header",
    "footer", "nav", "form", "fieldset", "ul", "ol", "dl", "table",
    "tbody", "thead", "tfoot", "tr", "figure",
}
HEADING_TAGS = {"h1", "h2", "h3"}
SKIP_TAGS = {"script", "style"}

TYPE_MAP = {
    "blockquote": "quote",
    "li": "list-item",
    "pre": "preformatted",
}

TocMarker = tuple[str, str, str, int]  # (file, fragment, title, level)


def _flatten_toc(toc: Iterable) -> list[TocMarker]:
    """Flatten ebooklib's nested TOC into (file, fragment, title, level)."""
    markers: list[TocMarker] = []

    def add(node: object, level: int) -> None:
        href = getattr(node, "href", "") or ""
        title = (getattr(node, "title", "") or "").strip()
        if not href:
            return
        file_name, _, fragment = href.partition("#")
        markers.append((file_name, fragment, title, level))

    def walk(items: Iterable, level: int) -> None:
        for item in items:
            if isinstance(item, tuple):
                add(item[0], level)
                walk(item[1], level + 1)
            else:
                add(item, level)

    walk(toc, 1)
    return markers


def _iter_blocks(parent: Tag) -> Iterator[Tag | NavigableString]:
    """Yield block-level content elements in document order."""
    for child in parent.children:
        if isinstance(child, NavigableString):
            if child.strip():
                yield child
            continue
        if not isinstance(child, Tag):
            continue
        name = child.name.lower()
        if name in SKIP_TAGS:
            continue
        if name in LEAF_BLOCK_TAGS and not child.find(LEAF_BLOCK_TAGS):
            yield child
        elif name in CONTAINER_TAGS or child.find(LEAF_BLOCK_TAGS):
            yield from _iter_blocks(child)
        elif child.get_text(strip=True) or child.find("img"):
            yield child


def _block_text(el: Tag) -> str:
    for br in el.find_all("br"):
        br.replace_with(" ")
    return " ".join(el.get_text().split())


def _block_source_id(el: Tag) -> str | None:
    if el.get("id"):
        return el.get("id")
    anchor = el.find("a", id=True)
    return anchor.get("id") if anchor else None


def _norm(text: str) -> str:
    return " ".join(text.split()).strip().lower()


class _ChapterBuilder:
    """Accumulates paragraphs and assigns sequential ids."""

    def __init__(self, book: Book, lang: str | None) -> None:
        self.book = book
        self.lang = lang
        self._para_n = 0
        self._sent_n = 0

    def new_chapter(self, title: str | None, level: int, source_id: str | None) -> Chapter:
        chapter = Chapter(
            id=f"ch-{len(self.book.chapters) + 1:03d}",
            order=len(self.book.chapters),
            title=title,
            level=level,
            source_id=source_id,
        )
        self.book.chapters.append(chapter)
        return chapter

    def add_paragraph(
        self,
        chapter: Chapter,
        el_type: str,
        text: str,
        source_id: str | None = None,
        images: list[dict[str, str | None]] | None = None,
    ) -> Paragraph:
        self._para_n += 1
        para = Paragraph(
            id=f"p-{self._para_n:05d}",
            order=self._para_n - 1,
            type=el_type,
            text=text,
            source_id=source_id,
            images=images or [],
        )
        if text:
            for i, sent_text in enumerate(split_sentences(text, self.lang)):
                self._sent_n += 1
                para.sentences.append(
                    Sentence(id=f"s-{self._sent_n:06d}", order=self._sent_n - 1, text=sent_text)
                )
        chapter.paragraphs.append(para)
        return para


def _metadata(book: epub.EpubBook) -> dict[str, str]:
    def first(field: str) -> str:
        values = book.get_metadata("DC", field)
        return values[0][0] if values else ""

    return {
        "title": first("title"),
        "author": first("creator"),
        "language": first("language"),
        "identifier": first("identifier"),
    }


def parse_epub(path: str | Path, lang: str | None = None) -> Book:
    path = Path(path)
    ebook = epub.read_epub(str(path))

    meta = _metadata(ebook)
    lang = lang or meta.get("language") or None
    book = Book(source_file=str(path), format="epub", metadata=meta)
    builder = _ChapterBuilder(book, lang)

    spine_docs: list[tuple[str, Tag]] = []
    for idref, linear in ebook.spine:
        if linear == "no":
            continue
        item = ebook.get_item_with_id(idref)
        if item is None or item.media_type != "application/xhtml+xml":
            continue
        soup = BeautifulSoup(item.get_content(), "lxml-xml")
        body = soup.find("body")
        if body is not None:
            spine_docs.append((item.get_name(), body))

    markers = _flatten_toc(ebook.toc)
    anchor_markers: dict[str, dict[str, tuple[str, int]]] = {}
    file_markers: dict[str, tuple[str, int]] = {}
    for file_name, fragment, title, level in markers:
        if fragment:
            anchor_markers.setdefault(file_name, {})[fragment] = (title, level)
        else:
            file_markers.setdefault(file_name, (title, level))

    has_headings = any(
        body.find(HEADING_TAGS) is not None for _, body in spine_docs
    )
    use_toc = bool(markers)

    chapter: Chapter | None = None
    for file_name, body in spine_docs:
        first_block = True
        for block in _iter_blocks(body):
            anchors: list[str] = []
            if isinstance(block, Tag):
                if block.get("id"):
                    anchors.append(block["id"])
                anchors.extend(a["id"] for a in block.find_all("a", id=True))

            start = None
            if use_toc:
                file_anchors = anchor_markers.get(file_name, {})
                for a in anchors:
                    if a in file_anchors:
                        start = (*file_anchors[a], f"{file_name}#{a}")
                        break
                if first_block and file_name in file_markers:
                    title, level = file_markers[file_name]
                    start = (title, level, file_name)
            elif has_headings:
                if isinstance(block, Tag) and block.name.lower() in HEADING_TAGS:
                    start = (_block_text(block), int(block.name[1]), None)
            elif first_block:
                start = (file_name, 1, file_name)
            first_block = False

            if start is not None:
                title, level, source_id = start
                chapter = builder.new_chapter(title or None, level, source_id)
            elif chapter is None:
                # Content before the first chapter marker.
                chapter = builder.new_chapter(None, 1, file_name)

            if isinstance(block, NavigableString):
                builder.add_paragraph(chapter, "paragraph", " ".join(str(block).split()))
                continue

            name = block.name.lower()
            text = _block_text(block)
            images = [
                {"src": img.get("src"), "alt": img.get("alt"), "id": img.get("id")}
                for img in block.find_all("img")
            ]
            if name in HEADING_TAGS or (start is not None and _norm(text) == _norm(title or "")):
                el_type = "heading"
            elif not text and images:
                el_type = "image"
            else:
                el_type = TYPE_MAP.get(name, "paragraph")
            builder.add_paragraph(
                chapter, el_type, text,
                source_id=_block_source_id(block), images=images,
            )

    return book
