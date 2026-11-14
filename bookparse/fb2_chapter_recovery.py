"""Conservative recovery of chapter structure from a single FB2 section."""

from __future__ import annotations

from dataclasses import replace
from math import ceil
from statistics import median
from typing import Any

from .models import Book, Chapter, Paragraph
from .segment import count_words

_MIN_PARAGRAPHS = 24
_MIN_REPEATED_HEADINGS = 4
_MIN_SPAN_RATIO = 0.25
_MIN_CONFIDENCE = 0.80
_COMPACT_TYPES = {"paragraph", "heading", "subtitle"}


def _is_compact(paragraph: Paragraph, max_words: int) -> bool:
    return (
        paragraph.type in _COMPACT_TYPES
        and bool(paragraph.text.strip())
        and count_words(paragraph.text) <= max_words
        and len(paragraph.text) <= 120
    )


def _gap_before(paragraphs: list[Paragraph], index: int) -> int:
    index -= 1
    count = 0
    while index >= 0 and paragraphs[index].type == "empty-line":
        count += 1
        index -= 1
    return count


def _next_paragraph(
    paragraphs: list[Paragraph], index: int,
) -> tuple[int | None, int]:
    index += 1
    gap = 0
    while index < len(paragraphs) and paragraphs[index].type == "empty-line":
        gap += 1
        index += 1
    return (index if index < len(paragraphs) else None), gap


def _short_run_after(
    paragraphs: list[Paragraph],
    index: int,
    max_words: int,
    min_prose_words: int,
) -> tuple[int, int | None]:
    """Count compact blocks before prose resumes after a heading pair."""
    index, _ = _next_paragraph(paragraphs, index)
    short_count = 0
    while index is not None:
        paragraph = paragraphs[index]
        if paragraph.type == "image" and not paragraph.text.strip():
            index, _ = _next_paragraph(paragraphs, index)
            continue
        if _is_compact(paragraph, max_words):
            short_count += 1
            if short_count > 2:
                return short_count, None
            index, _ = _next_paragraph(paragraphs, index)
            continue
        if count_words(paragraph.text) >= min_prose_words:
            return short_count, index
        return short_count, None
    return short_count, None


def _chapter_candidates(
    paragraphs: list[Paragraph],
    max_words: int,
    min_prose_words: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for start, first in enumerate(paragraphs):
        if not _is_compact(first, max_words) or _gap_before(paragraphs, start) not in (1, 2):
            continue
        second_index, inner_gap = _next_paragraph(paragraphs, start)
        if (
            inner_gap != 1
            or second_index is None
            or not _is_compact(paragraphs[second_index], max_words)
        ):
            continue
        _, trailing_gap = _next_paragraph(paragraphs, second_index)
        if trailing_gap not in (1, 2):
            continue
        short_count, prose_index = _short_run_after(
            paragraphs, second_index, max_words, min_prose_words,
        )
        if short_count > 1 or prose_index is None:
            continue
        candidates.append({
            "start": start,
            "title_end": second_index,
            "title": f"{first.text.strip()} — "
                     f"{paragraphs[second_index].text.strip()}",
            "kind": "chapter",
            "level": 1,
            "prose_index": prose_index,
            "immediate_prose": short_count == 0,
        })
    return candidates


def _part_candidates(
    paragraphs: list[Paragraph],
    max_words: int,
    min_prose_words: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for start, first in enumerate(paragraphs):
        if (
            not _is_compact(first, max_words)
            or _gap_before(paragraphs, start) < 4
        ):
            continue
        second_index, inner_gap = _next_paragraph(paragraphs, start)
        if (
            inner_gap != 1
            or second_index is None
            or not _is_compact(paragraphs[second_index], max_words)
        ):
            continue
        short_count, prose_index = _short_run_after(
            paragraphs, second_index, max_words, min_prose_words,
        )
        # A repeated, isolated title block followed by a repeated chapter
        # heading pattern is stronger evidence of a part than its wording.
        if short_count != 2 or prose_index is None:
            continue
        candidates.append({
            "start": start,
            "title_end": second_index,
            "title": f"{first.text.strip()} — "
                     f"{paragraphs[second_index].text.strip()}",
            "kind": "part",
            "level": 1,
            "prose_index": prose_index,
            "immediate_prose": False,
        })
    return candidates


def _tail_section_start(
    paragraphs: list[Paragraph],
    last_heading_start: int,
    max_words: int,
) -> int | None:
    """Recognize a clearly separated, short heading in the book's tail."""
    tail_start = max(last_heading_start + 1, int(len(paragraphs) * 0.9))
    for index in range(tail_start, len(paragraphs)):
        paragraph = paragraphs[index]
        if not _is_compact(paragraph, max_words) or _gap_before(paragraphs, index) < 2:
            continue
        next_index, trailing_gap = _next_paragraph(paragraphs, index)
        if next_index is None or trailing_gap < 2:
            continue
        following_text = [
            item for item in paragraphs[next_index:]
            if item.text.strip()
        ]
        if len(following_text) >= 2:
            return index
    return None


def _chapterize(
    book: Book,
    original: Chapter,
    headings: list[dict[str, Any]],
    backmatter_start: int | None,
) -> None:
    paragraphs = original.paragraphs
    heading_positions = {
        position
        for heading in headings
        for position in (heading["start"], heading["title_end"])
    }
    if backmatter_start is not None:
        heading_positions.add(backmatter_start)

    def copy_range(start: int, end: int) -> list[Paragraph]:
        return [
            replace(paragraph, type="heading")
            if index in heading_positions else paragraph
            for index, paragraph in enumerate(paragraphs[start:end], start)
        ]

    recovered: list[Chapter] = []
    if headings[0]["start"] > 0:
        recovered.append(Chapter(
            id="", order=0, title="Front matter", level=1,
            body="front-matter", paragraphs=copy_range(0, headings[0]["start"]),
        ))

    for position, heading in enumerate(headings):
        next_start = (
            headings[position + 1]["start"]
            if position + 1 < len(headings)
            else backmatter_start if backmatter_start is not None
            else len(paragraphs)
        )
        if next_start <= heading["start"]:
            continue
        recovered.append(Chapter(
            id="", order=0, title=heading["title"],
            level=heading["level"], body="main",
            source_id=original.source_id,
            paragraphs=copy_range(heading["start"], next_start),
        ))

    if backmatter_start is not None:
        recovered.append(Chapter(
            id="", order=0,
            title=paragraphs[backmatter_start].text.strip(),
            level=1, body="back-matter",
            paragraphs=copy_range(backmatter_start, len(paragraphs)),
        ))

    replaced: list[Chapter] = []
    for chapter in book.chapters:
        if chapter is original:
            replaced.extend(recovered)
        else:
            replaced.append(chapter)
    for order, chapter in enumerate(replaced):
        chapter.id = f"ch-{order + 1:03d}"
        chapter.order = order
    book.chapters = replaced


def recover_single_section(book: Book) -> dict[str, Any] | None:
    """Recover repeated structural headings only when normal FB2 chapters are absent."""
    main = [chapter for chapter in book.chapters if chapter.body == "main"]
    if (
        book.format != "fb2"
        or len(main) != 1
        or len(main[0].paragraphs) < _MIN_PARAGRAPHS
    ):
        return None

    original = main[0]
    paragraphs = original.paragraphs
    paragraph_words = [
        count_words(paragraph.text)
        for paragraph in paragraphs
        if paragraph.text.strip()
    ]
    typical_words = median(paragraph_words) if paragraph_words else 0
    max_heading_words = min(15, max(6, ceil(typical_words * 0.75)))
    min_prose_words = max(max_heading_words + 1, ceil(typical_words * 0.75))

    candidate_parts = _part_candidates(
        paragraphs, max_heading_words, min_prose_words,
    )
    chapters = _chapter_candidates(
        paragraphs, max_heading_words, min_prose_words,
    )
    parts = [
        part for part in candidate_parts
        if any(
            part["title_end"] < chapter["start"] < part["prose_index"]
            for chapter in chapters
        )
    ]
    if len(chapters) < _MIN_REPEATED_HEADINGS:
        parts = []
    part_positions = {
        position
        for part in parts
        for position in (part["start"], part["title_end"])
    }
    chapters = [
        candidate for candidate in chapters
        if candidate["start"] not in part_positions
        and candidate["title_end"] not in part_positions
    ]

    headings = sorted([*parts, *chapters], key=lambda item: item["start"])
    span_ratio = (
        (headings[-1]["start"] - headings[0]["start"]) / max(len(paragraphs) - 1, 1)
        if headings else 0.0
    )
    repeat_score = min(1.0, len(chapters) / _MIN_REPEATED_HEADINGS)
    span_score = min(1.0, span_ratio / _MIN_SPAN_RATIO)
    part_score = min(1.0, len(parts) / 2) if parts else 0.5
    confidence = round(
        0.60 * repeat_score + 0.25 * span_score + 0.15 * part_score, 3,
    )
    for heading in headings:
        heading["confidence"] = round(
            0.65
            + 0.20 * repeat_score
            + 0.10 * span_score
            + (0.05 if heading["immediate_prose"] else 0.025),
            3,
        )

    candidate_summary = [
        {
            "paragraph_order": paragraphs[item["start"]].order,
            "kind": item["kind"],
            "title": item["title"],
            "level": item["level"],
            "confidence": item["confidence"],
        }
        for item in headings
    ]
    diagnostics: dict[str, Any] = {
        "status": "insufficient_evidence",
        "candidate_count": len(headings),
        "chapter_count": len(chapters),
        "part_count": len(parts),
        "confidence": confidence,
        "span_ratio": round(span_ratio, 3),
        "candidates": candidate_summary,
    }
    if (
        len(chapters) < _MIN_REPEATED_HEADINGS
        or span_ratio < _MIN_SPAN_RATIO
        or confidence < _MIN_CONFIDENCE
    ):
        book.chapter_recovery = diagnostics
        return diagnostics

    # Keep children beneath the most recent confidently repeated part.
    if parts:
        part_starts = [part["start"] for part in parts]
        for heading in chapters:
            if any(start < heading["start"] for start in part_starts):
                heading["level"] = 2
    headings.sort(key=lambda item: item["start"])
    backmatter_start = _tail_section_start(
        paragraphs, headings[-1]["start"], max_heading_words,
    )
    _chapterize(book, original, headings, backmatter_start)

    diagnostics.update({
        "status": "recovered",
        "recovered_chapters": len(chapters),
        "recovered_parts": len(parts),
        "service_sections": int(headings[0]["start"] > 0)
        + int(backmatter_start is not None),
    })
    book.chapter_recovery = diagnostics
    return diagnostics
