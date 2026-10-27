"""Preflight validation of parsed book structures.

Runs simple heuristics on the parsed Book objects (no ML):

Per source:
  * empty / near-empty chapters
  * paragraphs ending mid-sentence (PDF page-split artifacts)
  * page-number / navigation leftovers, broken-encoding characters
  * duplicated text blocks (running heads, repeated content)

Between the English and Russian books:
  * chapter counts and hierarchy (only titled chapters of the main
    body are treated as "plot" chapters; notes/auxiliary bodies and
    untitled front matter are handled separately)
  * positional pairing of chapters by volume, with a small offset
    search to tolerate unpaired front/back matter and a regional
    check for mid-book chapter insertions/skips

Levels: INFO — notable but harmless; WARNING — review recommended;
ERROR — the source is probably unusable for alignment as-is.
All checks are heuristic; messages note possible false positives.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .models import Book, Chapter
from .segment import count_words

INFO, WARNING, ERROR = "INFO", "WARNING", "ERROR"

# Paragraph types that carry layout/metadata rather than running text.
SERVICE_TYPES = {
    "heading", "subtitle", "image", "empty-line", "text-author",
    "epigraph", "cite", "verse",
}

_TERMINAL_RE = re.compile(r"[.!?…\"»’)\]—–-]\s*$")
_PAGE_ARTIFACT_RE = re.compile(r"^[«\s]*[\^_~<>]+[»\s]*$|^\d{1,4}$")
_REPLACEMENT_CHAR = "�"

_MID_SPLIT_ERROR_RATIO = 0.15   # >15% of paragraphs split mid-sentence → broken source
_DUP_MIN_LEN = 40               # duplicated blocks shorter than this are ignored
_DIVIDER_MAX_WORDS = 25         # titled chapter with fewer words is likely a divider
_VOLUME_RATIO_RANGE = (0.55, 1.5)
_VOLUME_MIN_DIFF = 100          # ignore ratio flags on tiny chapters
_MAX_OFFSET = 4                 # chapter-pairing offset search window


@dataclass
class Issue:
    level: str
    side: str  # "english" | "russian" | "both"
    code: str
    message: str
    details: list[str] = field(default_factory=list)


def _text_paragraphs(book: Book) -> list:
    return [
        p for ch in book.chapters for p in ch.paragraphs
        if p.type not in SERVICE_TYPES
    ]


def _chapter_words(chapter: Chapter) -> int:
    return sum(count_words(p.text) for p in chapter.paragraphs)


def _titled_main(book: Book) -> list[Chapter]:
    return [c for c in book.chapters if c.body == "main" and c.title]


def _aux_chapters(book: Book) -> list[Chapter]:
    return [c for c in book.chapters if c.body != "main"]


def _untitled_main(book: Book) -> list[Chapter]:
    return [c for c in book.chapters if c.body == "main" and not c.title]


def check_book(book: Book, side: str) -> list[Issue]:
    """Structural checks for one parsed book."""
    issues: list[Issue] = []
    paras = _text_paragraphs(book)

    for ch in book.chapters:
        words = _chapter_words(ch)
        has_images = any(p.images for p in ch.paragraphs)
        if words == 0 and not has_images:
            issues.append(Issue(
                WARNING if ch.body == "main" else INFO,
                side, "empty-chapter",
                f"Chapter {ch.id} ({ch.title!r}, body={ch.body!r}) "
                "contains no text.",
            ))
        elif (
            ch.body == "main" and ch.title and words <= _DIVIDER_MAX_WORDS
        ):
            issues.append(Issue(
                INFO, side, "divider-chapter",
                f"Chapter {ch.id} ({ch.title!r}) has only {words} words — "
                "probably a part divider.",
            ))

    # Compare neighbours only within one chapter: a paragraph ending
    # without terminal punctuation at a chapter boundary is normal.
    splits = []
    for ch in book.chapters:
        ch_paras = [p for p in ch.paragraphs if p.type not in SERVICE_TYPES]
        for prev, nxt in zip(ch_paras, ch_paras[1:]):
            if (
                prev.text.strip() and nxt.text.strip()
                and not _TERMINAL_RE.search(prev.text.strip())
                and nxt.text.strip()[0].islower()
            ):
                splits.append((prev, nxt))
    if splits:
        ratio = len(splits) / max(len(paras), 1)
        issues.append(Issue(
            ERROR if ratio > _MID_SPLIT_ERROR_RATIO else WARNING,
            side, "mid-sentence-splits",
            f"{len(splits)} paragraphs end mid-sentence "
            f"({ratio:.0%} of text paragraphs) — typical sign of a "
            "PDF conversion where pages cut the text flow. "
            "A few splits can be legitimate (false positive possible).",
            [
                f"{a.id}: …{a.text.strip()[-45:]!r}  →  {b.text.strip()[:45]!r}"
                for a, b in splits[:5]
            ],
        ))

    artifacts = [p for p in paras if _PAGE_ARTIFACT_RE.match(p.text.strip())]
    if artifacts:
        issues.append(Issue(
            WARNING, side, "page-artifacts",
            f"{len(artifacts)} paragraphs look like page numbers or "
            "navigation leftovers (false positive possible: standalone "
            "numbers can be legitimate text).",
            [f"{p.id}: {p.text[:60]!r}" for p in artifacts[:5]],
        ))

    bad_chars = [p for p in paras if _REPLACEMENT_CHAR in p.text]
    if bad_chars:
        issues.append(Issue(
            WARNING, side, "encoding-artifacts",
            f"{len(bad_chars)} paragraphs contain U+FFFD replacement "
            "characters — the file may have been decoded with a "
            "wrong encoding.",
            [f"{p.id}: {p.text[:60]!r}" for p in bad_chars[:5]],
        ))

    counter = Counter(
        " ".join(p.text.lower().split())
        for p in paras if len(p.text) >= _DUP_MIN_LEN
    )
    dups = sorted(
        ((t, n) for t, n in counter.items() if n > 1),
        key=lambda x: -x[1],
    )
    if dups:
        issues.append(Issue(
            WARNING, side, "duplicated-blocks",
            f"{len(dups)} distinct text blocks (≥{_DUP_MIN_LEN} chars) "
            "appear more than once — possible running heads or "
            "duplicated content (false positive possible for "
            "deliberate repetitions).",
            [f"×{n}: {t[:70]!r}" for t, n in dups[:5]],
        ))

    if not issues:
        issues.append(Issue(INFO, side, "clean",
                            "No structural problems detected."))
    return issues


def _pair_volumes(en_ch: Chapter, ru_ch: Chapter) -> tuple[int, int, float | None]:
    we, wr = _chapter_words(en_ch), _chapter_words(ru_ch)
    ratio = wr / we if we else None
    return we, wr, ratio


def _volume_flag(we: int, wr: int, ratio: float | None) -> bool:
    if ratio is None:
        return wr > _VOLUME_MIN_DIFF
    return (
        abs(wr - we) >= _VOLUME_MIN_DIFF
        and not _VOLUME_RATIO_RANGE[0] <= ratio <= _VOLUME_RATIO_RANGE[1]
    )


def _best_offset(
    en_ch: list[Chapter], ru_ch: list[Chapter],
) -> tuple[int, list[tuple[int, int]]]:
    """Find the EN index offset that best aligns chapters with RU ones.

    Returns (offset, [(en_idx, ru_idx), ...]). Tolerates unpaired
    front/back matter chapters instead of shifting the whole book.
    """
    def score(shift: int) -> tuple[int, int, int]:
        pairs = []
        flagged = 0
        for ri, r in enumerate(ru_ch):
            ei = ri + shift
            if 0 <= ei < len(en_ch):
                pairs.append((ei, ri))
                we, wr, ratio = _pair_volumes(en_ch[ei], r)
                flagged += _volume_flag(we, wr, ratio)
        unpaired = len(en_ch) + len(ru_ch) - 2 * len(pairs)
        # Unpaired chapters cost more than flagged pairs: a volume
        # mismatch still counts as a plausible pairing.
        # Tie-break: smallest absolute shift.
        return (unpaired * 10 + flagged, abs(shift), shift)

    best = min(range(-_MAX_OFFSET, _MAX_OFFSET + 1), key=score)
    pairs = [
        (i + best, i)
        for i in range(len(ru_ch))
        if 0 <= i + best < len(en_ch)
    ]
    return best, pairs


def _regional_shifts(
    en_ch: list[Chapter],
    ru_ch: list[Chapter],
    global_shift: int,
    segments: int = 3,
) -> tuple[list[int], list[int]]:
    """Best pairing offset per book segment.

    A single global offset cannot express a chapter inserted or dropped
    in the middle of a book: the pairing before it needs shift N, after
    it shift N+1. Comparing per-segment best offsets detects this.
    Returns (shifts, boundaries) — boundaries are RU chapter indices
    where the preferred offset changes.
    """
    n = len(ru_ch)
    bounds = [round(i * n / segments) for i in range(segments + 1)]

    def best_in_range(lo: int, hi: int) -> int:
        def key(shift: int) -> tuple[int, int, int]:
            flagged = pairs = 0
            for ri in range(lo, hi):
                ei = ri + shift
                if 0 <= ei < len(en_ch):
                    pairs += 1
                    we, wr, ratio = _pair_volumes(en_ch[ei], ru_ch[ri])
                    flagged += _volume_flag(we, wr, ratio)
            unpaired = (hi - lo) - pairs
            # Prefer the global shift on ties so ambiguous regions
            # (uniform chapter sizes) do not produce false positives.
            return (unpaired * 10 + flagged, abs(shift - global_shift), abs(shift))

        return min(range(-_MAX_OFFSET, _MAX_OFFSET + 1), key=key)

    shifts = [
        best_in_range(lo, hi)
        for lo, hi in zip(bounds, bounds[1:])
        if hi > lo
    ]
    boundaries = [
        bounds[i]
        for i in range(1, len(shifts))
        if shifts[i] != shifts[i - 1]
    ]
    return shifts, boundaries


def compare_books(en: Book, ru: Book) -> list[Issue]:
    """Cross-book structural comparison (EN vs RU)."""
    issues: list[Issue] = []
    en_ch, ru_ch = _titled_main(en), _titled_main(ru)

    issues.append(Issue(
        INFO, "both", "chapter-inventory",
        f"Titled main-body chapters: EN={len(en_ch)}, RU={len(ru_ch)}. "
        f"Auxiliary sections (notes etc.): EN={len(_aux_chapters(en))}, "
        f"RU={len(_aux_chapters(ru))}. Untitled front-matter sections: "
        f"EN={len(_untitled_main(en))}, RU={len(_untitled_main(ru))}.",
    ))

    if not en_ch or not ru_ch:
        issues.append(Issue(
            ERROR, "both", "no-chapters",
            "One side has no titled main-body chapters — chapter-level "
            "alignment is impossible.",
        ))
        return issues

    if len(en_ch) != len(ru_ch):
        issues.append(Issue(
            WARNING, "both", "chapter-count",
            f"Chapter count differs: EN={len(en_ch)}, RU={len(ru_ch)}. "
            "Check for extra front/back matter or missing chapters.",
        ))

    en_levels = Counter(c.level for c in en_ch)
    ru_levels = Counter(c.level for c in ru_ch)
    if en_levels != ru_levels:
        issues.append(Issue(
            INFO, "both", "hierarchy",
            f"Chapter hierarchy differs: EN {dict(en_levels)} vs "
            f"RU {dict(ru_levels)} (levels → counts). Common when one "
            "edition groups chapters under parts and the other does not.",
        ))

    offset, pairs = _best_offset(en_ch, ru_ch)
    paired_en = {ei for ei, _ in pairs}
    paired_ru = {ri for _, ri in pairs}
    unpaired_en = [en_ch[i] for i in range(len(en_ch)) if i not in paired_en]
    unpaired_ru = [ru_ch[i] for i in range(len(ru_ch)) if i not in paired_ru]

    regional, boundaries = _regional_shifts(en_ch, ru_ch, offset)
    local_shift_suspected = len(set(regional)) > 1

    if local_shift_suspected:
        issues.append(Issue(
            WARNING, "both", "local-shift-suspected",
            f"Best chapter-pairing offset differs between book regions "
            f"({regional} for successive segments vs global {offset:+d}) "
            "— a chapter may be inserted or dropped mid-book, so "
            "positional pairing after that point is unreliable. "
            "Heuristic check; verify manually.",
            [
                f"offset change near RU chapter {ru_ch[b].id} "
                f"{ru_ch[b].title!r} (index {b})"
                for b in boundaries
            ],
        ))

    if offset or unpaired_en or unpaired_ru:
        details = [
            f"{side} unpaired: " + ", ".join(
                f"{c.id} {c.title!r}" for c in chs
            )
            for side, chs in (("EN", unpaired_en), ("RU", unpaired_ru))
            if chs
        ]
        issues.append(Issue(
            WARNING, "both", "unpaired-chapters",
            f"Best chapter alignment uses offset {offset:+d}; "
            f"{len(unpaired_en)} EN and {len(unpaired_ru)} RU chapters "
            "have no positional counterpart — likely extra front/back "
            "matter (verify).",
            details,
        ))

    flagged = []
    for ei, ri in pairs:
        a, b = en_ch[ei], ru_ch[ri]
        we, wr, ratio = _pair_volumes(a, b)
        if _volume_flag(we, wr, ratio):
            flagged.append((ei + 1, a, b, we, wr, ratio))
    if flagged:
        issues.append(Issue(
            WARNING, "both", "volume-mismatch",
            f"{len(flagged)} of {len(pairs)} paired chapters differ "
            "strongly in volume (ratio outside "
            f"{_VOLUME_RATIO_RANGE[0]}–{_VOLUME_RATIO_RANGE[1]}, "
            f">={_VOLUME_MIN_DIFF} words) — check for missing or "
            "duplicated text. Languages differ in verbosity, so "
            "moderate differences are normal (false positives possible).",
            [
                f"#{i} EN {a.title!r} ({we} w) ↔ RU {b.title!r} "
                f"({wr} w), ratio {ratio:.2f}"
                for i, a, b, we, wr, ratio in flagged[:8]
            ],
        ))
    elif not local_shift_suspected:
        issues.append(Issue(
            INFO, "both", "volume-ok",
            f"All {len(pairs)} paired chapters have comparable volume "
            f"(RU/EN ratio within {_VOLUME_RATIO_RANGE[0]}–"
            f"{_VOLUME_RATIO_RANGE[1]}).",
        ))

    en_words = sum(_chapter_words(c) for c in en.chapters)
    ru_words = sum(_chapter_words(c) for c in ru.chapters)
    overall = ru_words / en_words if en_words else 0
    issues.append(Issue(
        INFO, "both", "total-volume",
        f"Total volume: EN={en_words} words, RU={ru_words} words "
        f"(ratio {overall:.2f}). Differences in word/paragraph counts "
        "between languages are expected and are not an error.",
    ))
    return issues


def _source_summary(book: Book) -> str:
    s = book.stats()
    return (f"{book.source_file} ({book.format} | {s['chapters']} chapters | "
            f"{s['paragraphs']} paragraphs | {s['sentences']} sentences | "
            f"{s['words']} words)")


def build_report(en: Book, ru: Book) -> tuple[str, list[Issue]]:
    """Full text report. Returns (report_text, all_issues)."""
    issues = (
        check_book(en, "english")
        + check_book(ru, "russian")
        + compare_books(en, ru)
    )

    def section(title: str, side: str) -> list[str]:
        lines = [f"--- {title} ---"]
        for iss in issues:
            if iss.side != side:
                continue
            lines.append(f"[{iss.level}] {iss.code}: {iss.message}")
            lines.extend(f"    {d}" for d in iss.details)
        if len(lines) == 1:
            lines.append("    (no issues)")
        return lines

    lines = [
        "=" * 66,
        "PREFLIGHT REPORT",
        "=" * 66,
        f"English: {_source_summary(en)}",
        f"Russian: {_source_summary(ru)}",
        "",
        *section("ENGLISH SOURCE", "english"),
        "",
        *section("RUSSIAN SOURCE", "russian"),
        "",
        *section("CROSS-BOOK COMPARISON", "both"),
        "",
        "=" * 66,
        f"VERDICT: {verdict(issues)}",
        "=" * 66,
    ]
    return "\n".join(lines), issues


def verdict(issues: list[Issue]) -> str:
    errors = [i for i in issues if i.level == ERROR]
    warnings = [i for i in issues if i.level == WARNING]
    if errors:
        sides = sorted({i.side for i in errors})
        return (
            "STOP — critical structural problems in "
            f"{', '.join(sides)} source(s): "
            + "; ".join(i.code for i in errors)
            + ". Check or replace the affected file before alignment."
        )
    if warnings:
        return (
            f"PROCEED WITH CAUTION — {len(warnings)} warning(s). "
            "Review the report; the books are probably usable but "
            "the flagged points may reduce alignment quality."
        )
    return "OK — no structural problems detected; sources look ready for alignment."
