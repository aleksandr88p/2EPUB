"""Tests for preflight validation (bookparse.checks)."""

from __future__ import annotations

from bookparse.checks import (
    ERROR,
    INFO,
    WARNING,
    build_report,
    check_book,
    compare_books,
    verdict,
)
from bookparse.models import Book, Chapter, Paragraph


def _para(text: str, p_type: str = "paragraph") -> Paragraph:
    return Paragraph(id="p", order=0, type=p_type, text=text)


def _chapter(title: str | None, texts: list[str], level: int = 1,
             body: str = "main", order: int = 0) -> Chapter:
    return Chapter(
        id=f"ch-{order:03d}", order=order, title=title, level=level,
        body=body, paragraphs=[_para(t) for t in texts],
    )


def _book(chapters: list[Chapter], fmt: str = "fb2") -> Book:
    for i, c in enumerate(chapters):
        c.id = f"ch-{i + 1:03d}"
        c.order = i
    return Book(source_file="test", format=fmt, metadata={}, chapters=chapters)


def _healthy_book(prefix: str, n: int = 3, words: int = 60) -> Book:
    return _book([
        _chapter(f"{prefix} chapter {i}",
                 [" ".join(f"{prefix}w{i}x{w}" for w in range(words)) + "."],
                 order=i)
        for i in range(n)
    ])


def _codes(issues) -> set[str]:
    return {i.code for i in issues}


# --- per-source checks ------------------------------------------------------

def test_empty_chapter_warns() -> None:
    book = _book([_chapter("Empty", []), _chapter("Full", ["Some text here."])])
    issues = check_book(book, "english")
    empty = [i for i in issues if i.code == "empty-chapter"]
    assert empty and empty[0].level == WARNING


def test_mid_sentence_splits_error() -> None:
    # Simulate PDF page-splitting: half the paragraphs end mid-sentence.
    texts = []
    for i in range(20):
        if i % 2 == 0:
            texts.append("this paragraph ends without any punctuation at all and")
        else:
            texts.append("continues in lowercase on the next page fragment.")
    book = _book([_chapter("Broken", texts)])
    issues = check_book(book, "english")
    splits = [i for i in issues if i.code == "mid-sentence-splits"]
    assert splits and splits[0].level == ERROR


def test_mid_sentence_check_respects_chapter_boundaries() -> None:
    # Last paragraph of ch-1 ends without punctuation and the first
    # paragraph of ch-2 starts lowercase — but they are in different
    # chapters, so this is NOT a split.
    book = _book([
        _chapter("One", ["This chapter ends abruptly without punctuation and"], order=0),
        _chapter("Two", ["continues here. Plus some more text."], order=1),
    ])
    issues = check_book(book, "english")
    assert "mid-sentence-splits" not in _codes(issues)


def test_few_mid_sentence_splits_is_warning() -> None:
    texts = ["A normal sentence."] * 20 + ["a suspicious ending without punct and",
                                          "continues here."]
    book = _book([_chapter("Ok", texts)])
    issues = check_book(book, "english")
    splits = [i for i in issues if i.code == "mid-sentence-splits"]
    assert splits and splits[0].level == WARNING


def test_page_artifacts_and_duplicates_warn() -> None:
    block = "This is a fairly long repeated paragraph of running text " * 2
    texts = [block, "« ^ »", "123", block, "Normal text."]
    book = _book([_chapter("Ch", texts)])
    issues = check_book(book, "english")
    codes = _codes(issues)
    assert "page-artifacts" in codes
    assert "duplicated-blocks" in codes


def test_clean_book_is_clean() -> None:
    book = _healthy_book("en")
    issues = check_book(book, "english")
    assert _codes(issues) == {"clean"}


def test_notes_body_chapters_not_flagged_as_empty() -> None:
    # A notes body with real text is auxiliary, not a plot chapter.
    book = _book([
        _chapter("Story", ["Real text."] * 5, body="main", order=0),
        _chapter(None, [], body="notes", order=1),
    ])
    issues = check_book(book, "english")
    empty = [i for i in issues if i.code == "empty-chapter"]
    assert empty and empty[0].level == INFO  # notes → INFO, not WARNING


# --- cross-book comparison --------------------------------------------------

def test_chapter_count_mismatch_warns() -> None:
    en = _healthy_book("en", n=4)
    ru = _healthy_book("ru", n=3)
    issues = compare_books(en, ru)
    assert "chapter-count" in _codes(issues)


def test_volume_mismatch_flagged() -> None:
    en = _book([
        _chapter("A", ["word " * 500], order=0),
        _chapter("B", ["word " * 500], order=1),
    ])
    ru = _book([
        _chapter("А", ["word " * 450], order=0),
        _chapter("Б", ["word " * 60], order=1),  # suspiciously thin
    ])
    issues = compare_books(en, ru)
    flagged = [i for i in issues if i.code == "volume-mismatch"]
    assert flagged and "Б" in flagged[0].details[0]


def test_volume_difference_is_not_error() -> None:
    # 30% smaller Russian text overall is normal for translation.
    en = _book([_chapter("A", ["word " * 500], order=0)])
    ru = _book([_chapter("А", ["word " * 350], order=0)])
    issues = compare_books(en, ru)
    assert not any(i.level == ERROR for i in issues)
    assert "total-volume" in _codes(issues)


def test_offset_alignment_finds_extra_front_chapter() -> None:
    # EN has an extra "Contents" chapter at the front.
    en = _book([_chapter("Contents", ["tiny toc."], order=0)]
               + [_chapter(f"Ch{i}", ["word " * 300], order=i + 1)
                  for i in range(4)])
    ru = _book([_chapter(f"Гл{i}", ["word " * 260], order=i)
                for i in range(4)])
    issues = compare_books(en, ru)
    codes = _codes(issues)
    assert "unpaired-chapters" in codes
    assert not any(i.code == "volume-mismatch" for i in issues)


def test_mid_book_insertion_detected() -> None:
    # Same chapter sequence, but EN has an extra chapter in the middle.
    # Chapter sizes alternate so a shifted pairing is visibly wrong.
    sizes = [500, 3000, 500, 3000, 500, 3000]
    ru = _book([
        _chapter(f"Гл{i}", ["w " * s], order=i)
        for i, s in enumerate(sizes)
    ])
    en_sizes = sizes[:2] + [500] + sizes[2:]  # insertion at index 2
    en = _book([
        _chapter(f"Ch{i}", ["w " * s], order=i)
        for i, s in enumerate(en_sizes)
    ])
    issues = compare_books(en, ru)
    local = [i for i in issues if i.code == "local-shift-suspected"]
    assert local and local[0].level == WARNING
    # The confident "everything paired fine" message must not appear.
    assert "volume-ok" not in _codes(issues)


def test_consistent_shift_is_not_flagged_as_local() -> None:
    # Extra chapter only at the end: global pairing stays consistent,
    # no mid-book shift should be suspected.
    sizes = [500, 3000, 500, 3000, 500, 3000]
    ru = _book([
        _chapter(f"Гл{i}", ["w " * s], order=i)
        for i, s in enumerate(sizes)
    ])
    en = _book([
        _chapter(f"Ch{i}", ["w " * s], order=i)
        for i, s in enumerate(sizes + [500])  # extra last chapter
    ])
    issues = compare_books(en, ru)
    assert "local-shift-suspected" not in _codes(issues)


def test_no_titled_chapters_is_error() -> None:
    en = _book([_chapter(None, ["text."])])
    ru = _healthy_book("ru")
    issues = compare_books(en, ru)
    assert any(i.level == ERROR for i in issues)


# --- report / verdict --------------------------------------------------------

def test_verdict_levels() -> None:
    ok = check_book(_healthy_book("en"), "english")
    assert verdict(ok + [i for i in compare_books(_healthy_book("a"),
                                                _healthy_book("b"))
                       if i.level != INFO]) == (
        "OK — no structural problems detected; sources look ready for alignment."
    )


def test_build_report_contains_sections() -> None:
    report, issues = build_report(_healthy_book("en"), _healthy_book("ru"))
    assert "PREFLIGHT REPORT" in report
    assert "ENGLISH SOURCE" in report
    assert "VERDICT" in report
    assert isinstance(issues, list)
