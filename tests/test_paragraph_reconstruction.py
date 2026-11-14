"""Tests for bilingual paragraph reconstruction
(bookparse.paragraph_reconstruction).

Hand-built AlignOp lists — no ML, no parsing.
"""

from __future__ import annotations

import pytest

from bookparse.paragraph_reconstruction import rebuild_chapter
from bookparse.segment import count_words
from bookparse.sentence_alignment import AlignOp


def _op(kind: str, en: tuple[int, ...], ru: tuple[int, ...]) -> AlignOp:
    return AlignOp(
        kind=kind, en_idx=en, ru_idx=ru, score=0.7,
        en_text="", ru_text="", quality="confident",
    )


def _ru_list(paras) -> list[str]:
    return [s for p in paras for s in p.ru_sentences]


def _sentence_words(prefix: str, count: int = 10) -> str:
    return " ".join(
        [f"{prefix}{i}" for i in range(count - 1)]
        + [f"{prefix}{count - 1}."]
    )


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


def test_no_en_paragraphs_with_ru_text_raises() -> None:
    # A chapter whose EN side has no paragraphs cannot hold RU
    # sentences — fail loudly instead of crashing with IndexError.
    with pytest.raises(ValueError, match="no paragraphs"):
        rebuild_chapter([], ["R0"], [_op("skip_ru", (), (0,))])


def test_incomplete_coverage_raises() -> None:
    with pytest.raises(ValueError, match="cover"):
        rebuild_chapter(
            [["E0", "E1"]],
            ["R0"],
            [_op("1:1", (0,), (0,))],   # E1 uncovered
        )


def test_short_paragraphs_split_long_paragraph_and_keep_default_unchanged() -> None:
    en = [_sentence_words(f"e{i}_") for i in range(8)]
    ru = [f"R{i}." for i in range(8)]
    ops = [_op("1:1", (i,), (i,)) for i in range(8)]

    legacy = rebuild_chapter([en], ru, ops)
    short = rebuild_chapter([en], ru, ops, short_paragraphs=True)

    assert len(legacy) == 1
    assert [sum(count_words(s) for s in p.en_sentences) for p in short] == [
        40, 40,
    ]
    assert [p.en_sentences for p in short] == [en[:4], en[4:]]
    assert [p.ru_sentences for p in short] == [ru[:4], ru[4:]]
    assert [p.index for p in short] == [0, 0]
    assert [p.include_images for p in short] == [True, False]


def test_short_paragraphs_leave_short_paragraph_intact() -> None:
    en = [_sentence_words(f"e{i}_") for i in range(3)]
    ru = [f"R{i}." for i in range(3)]
    ops = [_op("1:1", (i,), (i,)) for i in range(3)]

    short = rebuild_chapter([en], ru, ops, short_paragraphs=True)

    assert len(short) == 1
    assert short[0].en_sentences == en
    assert short[0].ru_sentences == ru


def test_short_paragraphs_leave_oversized_alignment_op_intact() -> None:
    en = [_sentence_words("e_", 45)]
    ru = ["Перевод."]
    ops = [_op("1:1", (0,), (0,))]

    short = rebuild_chapter([en], ru, ops, short_paragraphs=True)

    assert len(short) == 1
    assert short[0].en_sentences == en
    assert short[0].ru_sentences == ru


def test_short_paragraphs_split_at_safe_boundary_below_target_minimum() -> None:
    en = [_sentence_words("e0_", 20), _sentence_words("e1_", 25)]
    ru = ["R0.", "R1."]
    ops = [_op("1:1", (0,), (0,)), _op("1:1", (1,), (1,))]

    short = rebuild_chapter([en], ru, ops, short_paragraphs=True)

    assert [sum(count_words(s) for s in p.en_sentences) for p in short] == [
        20, 25,
    ]


def test_short_paragraphs_rebalance_tiny_tail_to_nearest_boundaries() -> None:
    en = [_sentence_words(f"e{i}_") for i in range(9)]
    ru = [f"R{i}." for i in range(9)]
    ops = [_op("1:1", (i,), (i,)) for i in range(9)]

    short = rebuild_chapter([en], ru, ops, short_paragraphs=True)

    assert [
        sum(count_words(sentence) for sentence in paragraph.en_sentences)
        for paragraph in short
    ] == [30, 30, 30]


def test_short_paragraphs_partition_400_words_without_breaking_ops() -> None:
    operation_sizes = [10, 20, 30, 50, 60, 10, 20, 30, 50, 60, 10, 20, 30]
    en: list[str] = []
    ru: list[str] = []
    ops: list[AlignOp] = []
    en_index = 0
    ru_index = 0

    for op_index, size in enumerate(operation_sizes):
        if size in {20, 60}:
            sentence_size = size // 2
            en_group = (
                _sentence_words(f"e{op_index}a_", sentence_size),
                _sentence_words(f"e{op_index}b_", sentence_size),
            )
            kind = "2:1"
            ru_group = (f"r{op_index}.",)
        elif size == 50:
            en_group = (_sentence_words(f"e{op_index}_", size),)
            kind = "1:2"
            ru_group = (f"r{op_index}a.", f"r{op_index}b.")
        else:
            en_group = (_sentence_words(f"e{op_index}_", size),)
            kind = "1:1"
            ru_group = (
                _sentence_words(f"r{op_index}_", 70)
                if op_index == 0 else f"r{op_index}.",
            )

        en.extend(en_group)
        ru.extend(ru_group)
        ops.append(_op(
            kind,
            tuple(range(en_index, en_index + len(en_group))),
            tuple(range(ru_index, ru_index + len(ru_group))),
        ))
        en_index += len(en_group)
        ru_index += len(ru_group)

    short = rebuild_chapter([en], ru, ops, short_paragraphs=True)
    block_sizes = [
        sum(count_words(sentence) for sentence in paragraph.en_sentences)
        for paragraph in short
    ]
    en_owner = {
        sentence: block_index
        for block_index, paragraph in enumerate(short)
        for sentence in paragraph.en_sentences
    }
    ru_owner = {
        sentence: block_index
        for block_index, paragraph in enumerate(short)
        for sentence in paragraph.ru_sentences
    }

    assert sum(block_sizes) == 400
    assert block_sizes == [30, 30, 50, 60, 30, 30, 50, 60, 30, 30]
    assert not any(100 <= size <= 400 for size in block_sizes)
    assert [sentence for paragraph in short for sentence in paragraph.en_sentences] == en
    assert [sentence for paragraph in short for sentence in paragraph.ru_sentences] == ru
    for op in ops:
        en_blocks = {en_owner[en[i]] for i in op.en_idx}
        ru_blocks = {ru_owner[ru[i]] for i in op.ru_idx}
        assert len(en_blocks) == len(ru_blocks) == 1
        assert en_blocks == ru_blocks
    for block_index, size in enumerate(block_sizes):
        if size > 40:
            assert sum(
                any(en_owner[en[i]] == block_index for i in op.en_idx)
                for op in ops
            ) == 1


def test_short_paragraphs_keep_group_ops_indivisible() -> None:
    en = [_sentence_words(f"e{i}_") for i in range(8)]
    ru = [f"R{i}." for i in range(8)]
    ops = [
        _op("1:1", (0,), (0,)),
        _op("1:2", (1,), (1, 2)),
        _op("2:1", (2, 3), (3,)),
        *[_op("1:1", (i,), (i,)) for i in range(4, 8)],
    ]

    short = rebuild_chapter([en], ru, ops, short_paragraphs=True)

    assert [p.en_sentences for p in short] == [en[:4], en[4:]]
    assert [p.ru_sentences for p in short] == [ru[:4], ru[4:]]


def test_short_paragraphs_preserve_skip_text_and_order() -> None:
    en = [_sentence_words(f"e{i}_") for i in range(8)]
    ru = [f"R{i}." for i in range(10)]
    ops = [
        _op("1:1", (0,), (0,)),
        _op("skip_ru", (), (1,)),
        _op("1:1", (1,), (2,)),
        _op("skip_en", (2,), ()),
        _op("1:1", (3,), (3,)),
        _op("1:1", (4,), (4,)),
        _op("1:1", (5,), (5,)),
        _op("skip_ru", (), (6,)),
        _op("1:1", (6,), (7,)),
        _op("1:1", (7,), (8,)),
        _op("skip_ru", (), (9,)),
    ]

    short = rebuild_chapter([en], ru, ops, short_paragraphs=True)

    assert [s for p in short for s in p.en_sentences] == en
    assert [s for p in short for s in p.ru_sentences] == ru
    assert [p.ru_sentences for p in short] == [
        ru[:4], ru[4:],
    ]


def test_short_paragraphs_do_not_split_cross_paragraph_group() -> None:
    en = [_sentence_words(f"e{i}_") for i in range(12)]
    ru = [f"R{i}." for i in range(11)]
    ops = [
        *[_op("1:1", (i,), (i,)) for i in range(5)],
        _op("2:1", (5, 6), (5,)),
        *[_op("1:1", (i,), (i - 1,)) for i in range(7, 12)],
    ]

    short = rebuild_chapter(
        [en[:6], en[6:]], ru, ops, short_paragraphs=True,
    )

    assert len(short) == 2
    assert [p.en_sentences for p in short] == [en[:6], en[6:]]
    assert [sum(count_words(s) for s in p.en_sentences) for p in short] == [
        60, 60,
    ]
    assert [s for p in short for s in p.en_sentences] == en
    assert [s for p in short for s in p.ru_sentences] == ru


def test_short_paragraphs_do_not_split_heading_blocks() -> None:
    en = [_sentence_words(f"e{i}_") for i in range(8)]
    ru = [f"R{i}." for i in range(8)]
    ops = [_op("1:1", (i,), (i,)) for i in range(8)]

    short = rebuild_chapter(
        [en], ru, ops, para_types=["heading"], short_paragraphs=True,
    )

    assert len(short) == 1
    assert short[0].type == "heading"
    assert short[0].en_sentences == en
