"""Tests for bilingual paragraph reconstruction
(bookparse.paragraph_reconstruction).

Hand-built AlignOp lists — no ML, no parsing.
"""

from __future__ import annotations

import pytest

from bookparse.paragraph_reconstruction import rebuild_chapter
from bookparse.sentence_alignment import AlignOp


def _op(kind: str, en: tuple[int, ...], ru: tuple[int, ...]) -> AlignOp:
    return AlignOp(
        kind=kind, en_idx=en, ru_idx=ru, score=0.7,
        en_text="", ru_text="", quality="confident",
    )


def _ru_list(paras) -> list[str]:
    return [s for p in paras for s in p.ru_sentences]


def test_basic_1to1_distribution() -> None:
    paras = rebuild_chapter(
        [["E0", "E1"], ["E2"]],
        ["R0", "R1", "R2"],
        [_op("1:1", (0,), (0,)), _op("1:1", (1,), (1,)),
         _op("1:1", (2,), (2,))],
    )
    assert [p.ru_sentences for p in paras] == [["R0", "R1"], ["R2"]]
    assert [p.en_sentences for p in paras] == [["E0", "E1"], ["E2"]]
    assert all(p.type == "paragraph" for p in paras)


def test_1to2_attaches_both_ru() -> None:
    paras = rebuild_chapter(
        [["E0"]],
        ["R0", "R1"],
        [_op("1:2", (0,), (0, 1))],
    )
    assert paras[0].ru_sentences == ["R0", "R1"]


def test_2to1_same_paragraph() -> None:
    paras = rebuild_chapter(
        [["E0", "E1"], ["E2"]],
        ["R0", "R1"],
        [_op("2:1", (0, 1), (0,)), _op("1:1", (2,), (1,))],
    )
    assert [p.ru_sentences for p in paras] == [["R0"], ["R1"]]


def test_2to1_across_boundary_goes_to_first() -> None:
    # The merged translation lands in the FIRST EN paragraph; the
    # second paragraph keeps its EN text but gets no RU.
    paras = rebuild_chapter(
        [["E0"], ["E1"], ["E2"]],
        ["R0", "R1"],
        [_op("2:1", (0, 1), (0,)), _op("1:1", (2,), (1,))],
    )
    assert [p.ru_sentences for p in paras] == [["R0"], [], ["R1"]]


def test_skip_en_keeps_text_without_ru() -> None:
    paras = rebuild_chapter(
        [["E0"], ["E1"]],
        ["R0"],
        [_op("skip_en", (0,), ()), _op("1:1", (1,), (0,))],
    )
    assert paras[0].en_sentences == ["E0"]
    assert paras[0].ru_sentences == []
    assert paras[1].ru_sentences == ["R0"]


def test_skip_ru_attaches_to_next_match_in_order() -> None:
    # Translator insertion before the next sentence's translation:
    # lands in the NEXT matched paragraph, before its own RU.
    paras = rebuild_chapter(
        [["E0"], ["E1"]],
        ["R0", "R1", "R2"],
        [_op("1:1", (0,), (0,)), _op("skip_ru", (), (1,)),
         _op("1:1", (1,), (2,))],
    )
    assert paras[0].ru_sentences == ["R0"]
    assert paras[1].ru_sentences == ["R1", "R2"]


def test_skip_ru_leading_attaches_to_first_match() -> None:
    paras = rebuild_chapter(
        [["E0"], ["E1"]],
        ["R0", "R1", "R2"],
        [_op("skip_ru", (), (0,)), _op("1:1", (0,), (1,)),
         _op("1:1", (1,), (2,))],
    )
    assert paras[0].ru_sentences == ["R0", "R1"]
    assert paras[1].ru_sentences == ["R2"]


def test_skip_ru_trailing_attaches_to_last_match() -> None:
    paras = rebuild_chapter(
        [["E0"], ["E1"]],
        ["R0", "R1", "R2"],
        [_op("1:1", (0,), (0,)), _op("1:1", (1,), (1,)),
         _op("skip_ru", (), (2,))],
    )
    assert paras[1].ru_sentences == ["R1", "R2"]


def test_consecutive_skip_ru_stay_together() -> None:
    paras = rebuild_chapter(
        [["E0"]],
        ["R0", "R1", "R2"],
        [_op("1:1", (0,), (0,)), _op("skip_ru", (), (1,)),
         _op("skip_ru", (), (2,))],
    )
    assert paras[0].ru_sentences == ["R0", "R1", "R2"]


def test_nothing_lost_or_duplicated() -> None:
    # Mixed scenario: every RU index appears exactly once, EN
    # sentences stay in their original paragraphs.
    en_paras = [["E0"], ["E1", "E2"], ["E3"], ["E4"]]
    ru = ["R0", "R1", "R2", "R3", "R4"]
    ops = [
        _op("1:1", (0,), (0,)),
        _op("skip_ru", (), (1,)),      # insertion -> para of next match
        _op("2:1", (1, 2), (2,)),      # merge inside para 1
        _op("skip_en", (3,), ()),
        _op("1:2", (4,), (3, 4)),
    ]
    paras = rebuild_chapter(en_paras, ru, ops)
    assert sorted(_ru_list(paras)) == ru
    assert len(_ru_list(paras)) == len(ru)
    assert [p.en_sentences for p in paras] == en_paras
    assert [p.ru_sentences for p in paras] == [
        ["R0"], ["R1", "R2"], [], ["R3", "R4"],
    ]


def test_para_types_and_indices_propagated() -> None:
    paras = rebuild_chapter(
        [["E0"], ["E1"]],
        ["R0", "R1"],
        [_op("1:1", (0,), (0,)), _op("1:1", (1,), (1,))],
        para_types=["heading", "epigraph"],
    )
    assert [(p.index, p.type) for p in paras] == [
        (0, "heading"), (1, "epigraph"),
    ]


def test_empty_paragraph_preserved() -> None:
    # e.g. an image or empty-line block between paragraphs
    paras = rebuild_chapter(
        [["E0"], [], ["E1"]],
        ["R0", "R1"],
        [_op("1:1", (0,), (0,)), _op("1:1", (1,), (1,))],
        para_types=["paragraph", "image", "paragraph"],
    )
    assert paras[1].type == "image"
    assert paras[1].en_sentences == [] and paras[1].ru_sentences == []


def test_skip_ru_with_no_match_at_all() -> None:
    # Degenerate chapter: RU text exists but nothing matched — it is
    # still kept, in paragraph 0, instead of being lost.
    paras = rebuild_chapter(
        [["E0"]],
        ["R0"],
        [_op("skip_en", (0,), ()), _op("skip_ru", (), (0,))],
    )
    assert paras[0].ru_sentences == ["R0"]


def test_incomplete_coverage_raises() -> None:
    with pytest.raises(ValueError, match="cover"):
        rebuild_chapter(
            [["E0", "E1"]],
            ["R0"],
            [_op("1:1", (0,), (0,))],   # E1 uncovered
        )
