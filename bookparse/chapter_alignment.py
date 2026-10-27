"""Chapter-level alignment between English and Russian books.

Monotonic 1:1 alignment + skips via dynamic programming
(Needleman-Wunsch style). The pair score combines relative text
volume (dominant signal), position in the book and nesting level.
Chapter merges (1:N) are intentionally not implemented yet.

Only *titled main-body* chapters take part in the alignment — they
are the plot structure. Untitled front-matter sections and auxiliary
bodies (notes, comments, ...) are kept separately in the result as
service sections. Nothing is dropped or ignored.
"""

from __future__ import annotations

from typing import Any

from .models import Book, Chapter
from .segment import count_words

# Pair-score weights.
_W_VOLUME = 2.0
_W_POSITION = 1.0
_W_LEVEL = 0.5

# Doubtful-pair thresholds (same scale as preflight checks).
_DOUBT_VOLUME_RANGE = (0.55, 1.5)
_DOUBT_VOLUME_MIN_DIFF = 100   # words
_DOUBT_POSITION = 0.06         # relative position drift


def _words(ch: Chapter) -> int:
    return sum(count_words(p.text) for p in ch.paragraphs)


def _plot_chapters(book: Book) -> list[Chapter]:
    """Titled main-body chapters in reading order."""
    return [c for c in book.chapters if c.body == "main" and c.title]


def _entry(ch: Chapter, service_reason: str | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "chapter_id": ch.id,
        "title": ch.title,
        "order": ch.order,
        "level": ch.level,
        "body": ch.body,
        "words": _words(ch),
    }
    if service_reason:
        entry["service_reason"] = service_reason
    return entry


def _service_chapters(book: Book) -> list[dict[str, Any]]:
    """Non-plot sections kept aside but not ignored."""
    out = []
    for ch in book.chapters:
        if ch.body != "main":
            out.append(_entry(ch, service_reason=f"auxiliary-body:{ch.body}"))
        elif not ch.title:
            out.append(_entry(ch, service_reason="untitled-front-matter"))
    return out


def _positions(n: int) -> list[float]:
    """Relative positions 0..1 of n chapters."""
    return [i / (n - 1) if n > 1 else 0.5 for i in range(n)]


def _skip_cost(words: int) -> float:
    """Cost of leaving a chapter unmatched (larger chapters hurt more)."""
    return 1.0 + 0.5 * min(words / 3000, 1.0)


def _pair_score(
    en_w: int, ru_w: int, en_pos: float, ru_pos: float,
    en_lvl: int, ru_lvl: int,
) -> float:
    volume = min(en_w, ru_w) / max(en_w, ru_w) if max(en_w, ru_w) else 1.0
    position = 1.0 - abs(en_pos - ru_pos)
    level = 1.0 if en_lvl == ru_lvl else 0.5
    return _W_VOLUME * volume + _W_POSITION * position + _W_LEVEL * level


def _dp_align(
    en_ch: list[Chapter], ru_ch: list[Chapter],
) -> list[tuple[str, int | None, int | None, float]]:
    """Needleman-Wunsch alignment.

    Returns ordered ops: ("pair", en_idx, ru_idx, score),
    ("skip_en", en_idx, None, cost), ("skip_ru", None, ru_idx, cost).
    """
    n, m = len(en_ch), len(ru_ch)
    en_w = [_words(c) for c in en_ch]
    ru_w = [_words(c) for c in ru_ch]
    en_pos, ru_pos = _positions(n), _positions(m)

    d = [[0.0] * (m + 1) for _ in range(n + 1)]
    # trace: "p" pair, "e" skip en, "r" skip ru
    trace = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = d[i - 1][0] - _skip_cost(en_w[i - 1])
        trace[i][0] = "e"
    for j in range(1, m + 1):
        d[0][j] = d[0][j - 1] - _skip_cost(ru_w[j - 1])
        trace[0][j] = "r"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            pair = d[i - 1][j - 1] + _pair_score(
                en_w[i - 1], ru_w[j - 1],
                en_pos[i - 1], ru_pos[j - 1],
                en_ch[i - 1].level, ru_ch[j - 1].level,
            )
            skip_en = d[i - 1][j] - _skip_cost(en_w[i - 1])
            skip_ru = d[i][j - 1] - _skip_cost(ru_w[j - 1])
            best = max(pair, skip_en, skip_ru)
            d[i][j] = best
            trace[i][j] = "p" if best == pair else "e" if best == skip_en else "r"

    ops: list[tuple[str, int | None, int | None, float]] = []
    i, j = n, m
    while i > 0 or j > 0:
        step = trace[i][j]
        if step == "p" and i > 0 and j > 0:
            score = _pair_score(
                en_w[i - 1], ru_w[j - 1],
                en_pos[i - 1], ru_pos[j - 1],
                en_ch[i - 1].level, ru_ch[j - 1].level,
            )
            ops.append(("pair", i - 1, j - 1, score))
            i, j = i - 1, j - 1
        elif step == "e" and i > 0:
            ops.append(("skip_en", i - 1, None, _skip_cost(en_w[i - 1])))
            i -= 1
        else:
            ops.append(("skip_ru", None, j - 1, _skip_cost(ru_w[j - 1])))
            j -= 1
    ops.reverse()
    return ops


def _assess_pair(
    en_w: int, ru_w: int, en_pos: float, ru_pos: float,
    en_lvl: int, ru_lvl: int,
) -> tuple[str, list[str]]:
    """Quality of a chosen pair: "confident" or "doubtful" + reasons."""
    reasons = []
    ratio = ru_w / en_w if en_w else None
    if abs(ru_w - en_w) >= _DOUBT_VOLUME_MIN_DIFF and (
        ratio is None
        or not _DOUBT_VOLUME_RANGE[0] <= ratio <= _DOUBT_VOLUME_RANGE[1]
    ):
        reasons.append("volume-mismatch")
    if en_lvl != ru_lvl:
        reasons.append("level-mismatch")
    if abs(en_pos - ru_pos) > _DOUBT_POSITION:
        reasons.append("position-drift")
    return ("doubtful" if reasons else "confident"), reasons


def align_chapters(en: Book, ru: Book) -> dict[str, Any]:
    """Align plot chapters of two parsed books. Returns a JSON-ready dict."""
    en_ch, ru_ch = _plot_chapters(en), _plot_chapters(ru)
    en_pos, ru_pos = _positions(len(en_ch)), _positions(len(ru_ch))

    pairs: list[dict[str, Any]] = []
    unmatched_en: list[dict[str, Any]] = []
    unmatched_ru: list[dict[str, Any]] = []

    for op, i, j, value in _dp_align(en_ch, ru_ch):
        if op == "pair":
            a, b = en_ch[i], ru_ch[j]
            we, wr = _words(a), _words(b)
            quality, reasons = _assess_pair(
                we, wr, en_pos[i], ru_pos[j], a.level, b.level,
            )
            pairs.append({
                "en_chapter_id": a.id,
                "ru_chapter_id": b.id,
                "en_title": a.title,
                "ru_title": b.title,
                "en_order": a.order,
                "ru_order": b.order,
                "level": {"en": a.level, "ru": b.level},
                "words": {"en": we, "ru": wr},
                "volume_ratio": round(wr / we, 3) if we else None,
                "score": round(value, 3),
                "quality": quality,
                "reasons": reasons,
            })
        elif op == "skip_en":
            unmatched_en.append(_entry(en_ch[i]))
        else:
            unmatched_ru.append(_entry(ru_ch[j]))

    confident = sum(1 for p in pairs if p["quality"] == "confident")
    total_en_words = sum(_words(c) for c in en_ch)
    total_ru_words = sum(_words(c) for c in ru_ch)
    paired_en_words = sum(p["words"]["en"] for p in pairs)
    paired_ru_words = sum(p["words"]["ru"] for p in pairs)

    return {
        "english_source": en.source_file,
        "russian_source": ru.source_file,
        "pairs": pairs,
        "unmatched": {"english": unmatched_en, "russian": unmatched_ru},
        "service": {
            "english": _service_chapters(en),
            "russian": _service_chapters(ru),
        },
        "quality": {
            "pairs_total": len(pairs),
            "confident": confident,
            "doubtful": len(pairs) - confident,
            "unmatched_english": len(unmatched_en),
            "unmatched_russian": len(unmatched_ru),
            "word_coverage_en": round(paired_en_words / total_en_words, 4)
                if total_en_words else 1.0,
            "word_coverage_ru": round(paired_ru_words / total_ru_words, 4)
                if total_ru_words else 1.0,
        },
    }


def format_report(result: dict[str, Any]) -> str:
    """Human-readable summary of an alignment result."""
    q = result["quality"]
    lines = [
        "CHAPTER ALIGNMENT REPORT",
        f"English: {result['english_source']}",
        f"Russian: {result['russian_source']}",
        "",
        f"Paired chapters:      {q['pairs_total']} "
        f"(confident {q['confident']}, doubtful {q['doubtful']})",
        f"Unmatched EN chapters: {q['unmatched_english']}",
        f"Unmatched RU chapters: {q['unmatched_russian']}",
        f"Word coverage: EN {q['word_coverage_en']:.2%}, "
        f"RU {q['word_coverage_ru']:.2%}",
        f"Service sections: EN {len(result['service']['english'])}, "
        f"RU {len(result['service']['russian'])}",
    ]

    doubtful = [p for p in result["pairs"] if p["quality"] == "doubtful"]
    if doubtful:
        lines += ["", "Doubtful pairs:"]
        for p in doubtful:
            lines.append(
                f"  {p['en_chapter_id']} {p['en_title']!r} ↔ "
                f"{p['ru_chapter_id']} {p['ru_title']!r} "
                f"[{', '.join(p['reasons'])}]"
            )

    for side, key in (("English", "english"), ("Russian", "russian")):
        un = result["unmatched"][key]
        if un:
            lines.append(f"Unmatched {side} chapters:")
            lines += [f"  {c['chapter_id']} {c['title']!r} ({c['words']} w)"
                      for c in un]
    return "\n".join(lines)
