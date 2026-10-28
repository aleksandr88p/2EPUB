"""Tests for the sentence DP alignment (bookparse.sentence_alignment).

Uses hand-built similarity matrices and stub group scorers — no ML.
"""

from __future__ import annotations

import numpy as np
import pytest

from bookparse.sentence_alignment import align_sentences, summarize_ops


def _sim(n: int, m: int, fill: float = 0.1) -> np.ndarray:
    return np.full((n, m), fill)


def _ops_index_pairs(ops):
    return [(o.en_idx, o.ru_idx) for o in ops]


def _assert_monotonic_and_complete(ops, n: int, m: int) -> None:
    """Every index used exactly once, order strictly preserved."""
    en_used = [i for op in ops for i in op.en_idx]
    ru_used = [j for op in ops for j in op.ru_idx]
    assert sorted(en_used) == list(range(n))
    assert sorted(ru_used) == list(range(m))
    assert en_used == sorted(en_used) and ru_used == sorted(ru_used)


def test_1to1_alignment() -> None:
    en = ["a1", "b1", "c1"]
    ru = ["a2", "b2", "c2"]
    sim = _sim(3, 3)
    np.fill_diagonal(sim, 0.9)

    ops = align_sentences(en, ru, sim)

    assert _ops_index_pairs(ops) == [
        ((0,), (0,)), ((1,), (1,)), ((2,), (2,)),
    ]
    assert all(o.kind == "1:1" and o.quality == "confident" for o in ops)
    _assert_monotonic_and_complete(ops, 3, 3)


def test_1to2_merge() -> None:
    # EN[1] was translated by two RU sentences ru[1]+ru[2].
    en = ["a", "big one", "c"]
    ru = ["a2", "part one", "part two", "c2"]
    sim = _sim(3, 4)
    sim[0][0] = sim[2][3] = 0.9
    sim[1][1] = sim[1][2] = 0.3           # individually weak
    scores = {(1, (1, 2)): 0.85}          # joined texts: strong match

    def scorer(ei, rj):
        return scores.get((ei[0], rj), 0.0)

    ops = align_sentences(en, ru, sim, group_scorer=scorer)

    assert [o.kind for o in ops] == ["1:1", "1:2", "1:1"]
    assert ops[1].en_idx == (1,) and ops[1].ru_idx == (1, 2)
    assert ops[1].ru_text == "part one part two"
    _assert_monotonic_and_complete(ops, 3, 4)


def test_2to1_merge() -> None:
    # Two EN sentences were merged into one RU sentence.
    en = ["part one", "part two", "c"]
    ru = ["joined ab", "c2"]
    sim = _sim(3, 2)
    sim[2][1] = 0.9
    sim[0][0] = sim[1][0] = 0.3
    scores = {((0, 1), (0,)): 0.85}

    def scorer(ei, rj):
        return scores.get((ei, rj), 0.0)

    ops = align_sentences(en, ru, sim, group_scorer=scorer)

    assert [o.kind for o in ops] == ["2:1", "1:1"]
    assert ops[0].en_idx == (0, 1) and ops[0].ru_idx == (0,)
    assert ops[0].en_text == "part one part two"
    _assert_monotonic_and_complete(ops, 3, 2)


def test_group_score_uses_joined_not_max() -> None:
    # If the scorer returned max(individual sims), a 1:2 with one strong
    # and one unrelated member would look good. Our scorer measures the
    # JOINED text, so the merge must lose to 1:1 + skip here.
    en = ["a", "b"]
    ru = ["a2", "unrelated"]
    sim = _sim(2, 2)
    sim[0][0] = 0.9
    sim[0][1] = 0.7          # a vs unrelated — decent but wrong
    sim[1][0] = 0.1

    # scorer: the joined RU text is dilute → low group similarity
    ops = align_sentences(
        en, ru, sim,
        group_scorer=lambda ei, rj: 0.25,
        bandwidth=15,
    )
    kinds = [o.kind for o in ops]
    assert "1:2" not in kinds and "2:1" not in kinds
    _assert_monotonic_and_complete(ops, 2, 2)


def test_decoy_backward_jump_is_rejected() -> None:
    # High similarity off the diagonal must not reorder the alignment.
    en = ["a", "b", "c"]
    ru = ["a2", "b2", "c2", "x", "y", "decoy"]
    sim = _sim(3, 6)
    sim[0][0] = 0.85
    sim[1][1] = sim[2][2] = 0.9
    sim[0][5] = 0.99          # decoy: en[0] likes ru[5]

    ops = align_sentences(en, ru, sim)

    assert _ops_index_pairs(ops)[:3] == [
        ((0,), (0,)), ((1,), (1,)), ((2,), (2,)),
    ]
    assert all(o.kind == "skip_ru" for o in ops[3:])
    _assert_monotonic_and_complete(ops, 3, 6)


def test_extra_sentence_stays_unmatched() -> None:
    en = ["a", "noise", "b"]
    ru = ["a2", "b2"]
    sim = _sim(3, 2, fill=0.15)
    sim[0][0] = sim[2][1] = 0.9

    ops = align_sentences(en, ru, sim)

    assert [o.kind for o in ops] == ["1:1", "skip_en", "1:1"]
    assert ops[1].quality == "unmatched" and ops[1].en_idx == (1,)
    _assert_monotonic_and_complete(ops, 3, 2)


def test_missing_ru_sentence_is_skipped() -> None:
    en = ["a", "b"]
    ru = ["a2", "untranslated", "b2"]
    sim = _sim(2, 3, fill=0.15)
    sim[0][0] = sim[1][2] = 0.9

    ops = align_sentences(en, ru, sim)

    assert [o.kind for o in ops] == ["1:1", "skip_ru", "1:1"]
    assert ops[1].ru_idx == (1,)
    _assert_monotonic_and_complete(ops, 2, 3)


def test_empty_side() -> None:
    ops = align_sentences(["a", "b"], [], np.zeros((2, 0)))
    assert [o.kind for o in ops] == ["skip_en", "skip_en"]

    ops = align_sentences([], ["x"], np.zeros((0, 1)))
    assert [o.kind for o in ops] == ["skip_ru"]


def test_sim_matrix_shape_checked() -> None:
    with pytest.raises(ValueError, match="shape"):
        align_sentences(["a"], ["b"], np.zeros((2, 2)))


def test_group_scorer_called_only_within_bandwidth() -> None:
    calls: list[tuple] = []

    def scorer(ei, rj):
        calls.append((ei, rj))
        return 0.0

    en = [f"e{i}" for i in range(10)]
    ru = [f"r{i}" for i in range(10)]
    sim = _sim(10, 10)
    np.fill_diagonal(sim, 0.9)

    align_sentences(en, ru, sim, group_scorer=scorer, bandwidth=2)
    assert calls                       # used, but...
    # only near-diagonal group candidates were scored
    for (ei, rj) in calls:
        assert abs(ei[-1] - rj[-1]) <= 2


def test_summarize_ops() -> None:
    en = ["a", "noise", "b"]
    ru = ["a2", "b2"]
    sim = _sim(3, 2, fill=0.15)
    sim[0][0] = sim[2][1] = 0.9
    ops = align_sentences(en, ru, sim)
    counts = summarize_ops(ops)
    assert counts["1:1"] == 2 and counts["skip_en"] == 1
    assert counts["confident"] == 2
