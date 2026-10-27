"""Data model for extracted book structure."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .segment import count_words


@dataclass
class Sentence:
    id: str
    order: int
    text: str


@dataclass
class Paragraph:
    id: str
    order: int
    # heading | paragraph | epigraph | cite | subtitle | verse |
    # text-author | image | empty-line | quote | list-item | table | ...
    type: str
    text: str
    # Original identifier from the source file (element id / anchor), if any.
    source_id: str | None = None
    images: list[dict[str, str | None]] = field(default_factory=list)
    sentences: list[Sentence] = field(default_factory=list)


@dataclass
class Chapter:
    id: str
    order: int
    title: str | None
    level: int = 1
    # "file.xhtml#anchor" for EPUB, section id for FB2.
    source_id: str | None = None
    # FB2 may contain extra bodies such as "notes" or "comments".
    body: str = "main"
    paragraphs: list[Paragraph] = field(default_factory=list)


@dataclass
class Book:
    source_file: str
    format: str  # "epub" | "fb2"
    metadata: dict[str, str]
    chapters: list[Chapter] = field(default_factory=list)

    def stats(self) -> dict[str, int]:
        paragraphs = [p for ch in self.chapters for p in ch.paragraphs]
        return {
            "chapters": len(self.chapters),
            "paragraphs": len(paragraphs),
            "sentences": sum(len(p.sentences) for p in paragraphs),
            "words": sum(count_words(p.text) for p in paragraphs),
        }

    def to_dict(self) -> dict:
        data = asdict(self)
        data["stats"] = self.stats()
        return data
