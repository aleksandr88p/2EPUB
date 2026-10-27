"""Tests for DP chapter alignment (bookparse.chapter_alignment)."""

from __future__ import annotations

from bookparse.chapter_alignment import align_chapters, format_report
from bookparse.models import Book, Chapter, Paragraph


def _chapter(title: str | None, words: int, order: int,
             level: int = 1, body: str = "main") -> Chapter:
    return Chapter(
        id=f"ch-{order:03d}", order=order, title=title, level=level,
        body=body, paragraphs=[
            Paragraph(id="p", order=0, type="paragraph", text="w " * words)
        ],
    )


def _book(titles: list[str | None], sizes: list[int],
          levels: list[int] | None = None) -> Book:
    levels = levels or [1] * len(titles)
    chapters = [
        _chapter(t, s, i, levels[i]) for i, (t, s) in enumerate(zip(titles, sizes))
    ]
    return Book(source_file="test", format="fb2", metadata={}, chapters=chapters)


def _pair_titles(result):
    return [(p["en_title"], p["ru_title"]) for p in result["pairs"]]


# --- scenarios ---------------------------------------------------------------

def test_identical_books_all_confident() -> None:
    sizes = [500, 3000, 1500, 800]
    en = _book(["A", "B", "C", "D"], sizes)
    ru = _book(["А", "Б", "В", "Г"], sizes)
    result = align_chapters(en, ru)

    q = result["quality"]
    assert q["pairs_total"] == 4
    assert q["confident"] == 4 and q["doubtful"] == 0
    assert q["unmatched_english"] == 0 and q["unmatched_russian"] == 0


def test_extra_chapter_at_start() -> None:
    en = _book(["Front", "A", "B", "C"], [200, 500, 3000, 800])
    ru = _book(["А", "Б", "В"], [500, 3000, 800])
    result = align_chapters(en, ru)

    assert _pair_titles(result) == [("A", "А"), ("B", "Б"), ("C", "В")]
    assert [c["title"] for c in result["unmatched"]["english"]] == ["Front"]


def test_extra_chapter_in_middle() -> None:
    sizes = [500, 3000, 500, 3000]
    en = _book(["A", "B", "Inserted", "C", "D"], [500, 3000, 400, 500, 3000])
    ru = _book(["А", "Б", "В", "Г"], sizes)
    result = align_chapters(en, ru)

    assert _pair_titles(result) == [
        ("A", "А"), ("B", "Б"), ("C", "В"), ("D", "Г"),
    ]
    assert [c["title"] for c in result["unmatched"]["english"]] == ["Inserted"]


def test_extra_chapter_at_end() -> None:
    en = _book(["A", "B", "C", "Back"], [500, 3000, 800, 150])
    ru = _book(["А", "Б", "В"], [500, 3000, 800])
    result = align_chapters(en, ru)

    assert _pair_titles(result) == [("A", "А"), ("B", "Б"), ("C", "В")]
    assert [c["title"] for c in result["unmatched"]["english"]] == ["Back"]


def test_missing_chapter_on_russian_side() -> None:
    en = _book(["A", "B", "C", "D"], [500, 3000, 1500, 800])
    ru = _book(["А", "Б", "Г"], [500, 3000, 800])
    result = align_chapters(en, ru)

    assert [c["title"] for c in result["unmatched"]["english"]] == ["C"]
    assert _pair_titles(result) == [("A", "А"), ("B", "Б"), ("D", "Г")]


def test_missing_chapter_on_english_side() -> None:
    en = _book(["A", "B", "D"], [500, 3000, 800])
    ru = _book(["А", "Б", "В", "Г"], [500, 3000, 1500, 800])
    result = align_chapters(en, ru)

    assert [c["title"] for c in result["unmatched"]["russian"]] == ["В"]


def test_volume_mismatch_marks_doubtful() -> None:
    en = _book(["A", "B"], [500, 3000])
    ru = _book(["А", "Б"], [200, 1200])  # same order, 2.5x smaller
    result = align_chapters(en, ru)

    assert len(result["pairs"]) == 2  # order preserved anyway
    assert all(p["quality"] == "doubtful" for p in result["pairs"])
    assert all("volume-mismatch" in p["reasons"] for p in result["pairs"])


def test_service_sections_kept_separate() -> None:
    en = _book(["A"], [500])
    ru = Book(source_file="t", format="fb2", metadata={}, chapters=[
        _chapter("А", 500, 0),
        _chapter(None, 60, 1),                    # untitled front matter
        _chapter("Заметки", 40, 2, body="notes"),  # notes body
    ])
    result = align_chapters(en, ru)

    assert result["quality"]["pairs_total"] == 1
    ru_service = result["service"]["russian"]
    assert len(ru_service) == 2
    assert {c["service_reason"].split(":")[0] for c in ru_service} == {
        "untitled-front-matter", "auxiliary-body",
    }


def test_level_mismatch_reason() -> None:
    en = _book(["A"], [500], levels=[1])
    ru = _book(["А"], [500], levels=[2])
    result = align_chapters(en, ru)

    pair = result["pairs"][0]
    assert pair["quality"] == "doubtful"
    assert "level-mismatch" in pair["reasons"]


def test_report_and_json_structure() -> None:
    en = _book(["A", "Extra"], [500, 100])
    ru = _book(["А"], [500])
    result = align_chapters(en, ru)

    for key in ("english_source", "russian_source", "pairs",
                    "unmatched", "service", "quality"):
        assert key in result
    pair = result["pairs"][0]
    for key in ("en_chapter_id", "ru_chapter_id", "en_title", "ru_title",
                "words", "volume_ratio", "score", "quality", "reasons"):
        assert key in pair
    report = format_report(result)
    assert "CHAPTER ALIGNMENT REPORT" in report
    assert "Unmatched EN chapters: 1" in report
