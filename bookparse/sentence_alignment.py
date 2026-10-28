"""Sentence-level monotonic alignment via dynamic programming.

Pure DP prototype: takes two sentence lists and a precomputed pairwise
similarity matrix — no ML model is loaded inside this module.

Transitions: 1:1, 1:2, 2:1 plus skips on either side. Group scores come
from a `group_scorer` callback that measures semantic similarity of the
*joined* texts of the group — not the max of individual sentence sims.
In production that callback joins the sentences and calls the embedding
model (through the embedding cache); in tests it can be a stub.

Costs (minimized):
    1:1 pair        1 - sim[i][j]
    1:2 / 2:1       2 * (1 - group_sim) + group_penalty
    skip            skip_cost (per unmatched sentence)

The group cost is scaled by the number of covered sentences
(max(len_en, len_r) = 2 for the supported groups). A per-operation
"1 - sim" would undercharge groups: they cover more sentences per
constant and tend to absorb stray sentences whose true match exists
elsewhere. With the scaled cost a group wins over "1:1 + skip" only
when its joined similarity is genuinely higher than the single-sentence
match minus a margin — real merges still win, spurious ones do not.

`bandwidth` restricts group transitions to cells near the diagonal —
the only place where merged translations are plausible — which also
bounds the number of (potentially expensive) group_scorer calls.

Order is strictly preserved; every sentence appears in exactly one
operation — as a pair, a group member, or an explicit skip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

# (en_indices, ru_indices) -> cosine similarity of the joined texts
GroupScorer = Callable[[tuple[int, ...], tuple[int, ...]], float]


@dataclass(frozen=True)
class AlignOp:
    kind: str                     # "1:1" | "1:2" | "2:1" | "skip_en" | "skip_ru"
    en_idx: tuple[int, ...]
    ru_idx: tuple[int, ...]
    score: float                  # similarity for matches, -skip_cost for skips
    en_text: str
    ru_text: str
    quality: str                  # "confident" | "doubtful" | "unmatched"


def _join(texts: Sequence[str], idx: tuple[int, ...]) -> str:
    return " ".join(texts[i] for i in idx)


def align_sentences(
    en_sentences: Sequence[str],
    ru_sentences: Sequence[str],
    sim_matrix: Sequence[Sequence[float]] | np.ndarray,
    group_scorer: GroupScorer | None = None,
    *,
    skip_cost: float = 0.35,
    group_penalty: float = 0.05,
    bandwidth: int = 15,
    confident_min: float = 0.55,
) -> list[AlignOp]:
    """Align two sentence sequences monotonically.

    Returns an ordered list of AlignOp. Every EN index and every RU
    index appears in exactly one op: pairs ("1:1", "1:2", "2:1") or
    explicit skips ("skip_en", "skip_ru").

    `group_scorer(en_idx, ru_idx)` must return the cosine similarity of
    the joined group texts; it is only queried for transitions within
    `bandwidth` of the diagonal and only when non-None.
    """
    n, m = len(en_sentences), len(ru_sentences)
    sim = np.asarray(sim_matrix, dtype=float)
    if sim.shape != (n, m):
        raise ValueError(
            f"sim_matrix shape {sim.shape} does not match "
            f"({n}, {m}) sentences"
        )
    if sim.size and not np.isfinite(sim).all():
        raise ValueError("sim_matrix contains non-finite values")
    if skip_cost <= 0:
        raise ValueError("skip_cost must be positive")
    if group_penalty < 0:
        raise ValueError("group_penalty must be >= 0")
    if not isinstance(bandwidth, int) or bandwidth < 0:
        raise ValueError("bandwidth must be a non-negative integer")

    INF = float("inf")
    d = np.full((n + 1, m + 1), INF)
    trace: list[list[tuple | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    d[0][0] = 0.0
    for i in range(1, n + 1):
        d[i][0] = d[i - 1][0] + skip_cost
        trace[i][0] = ("skip_en",)
    for j in range(1, m + 1):
        d[0][j] = d[0][j - 1] + skip_cost
        trace[0][j] = ("skip_ru",)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Candidate transitions, best first on ties: pairing is
            # preferred over skips, 1:1 over groups.
            cand: list[tuple[float, tuple]] = [
                (d[i - 1][j - 1] + (1.0 - sim[i - 1][j - 1]), ("1:1",))
            ]
            if group_scorer is not None and abs(i - j) <= bandwidth:
                # Group cost scales with the number of covered
                # sentences: max(len_en, len_r) = 2.
                if j >= 2:
                    g = group_scorer((i - 1,), (j - 2, j - 1))
                    cand.append((
                        d[i - 1][j - 2] + 2.0 * (1.0 - g) + group_penalty,
                        ("1:2", g),
                    ))
                if i >= 2:
                    g = group_scorer((i - 2, i - 1), (j - 1,))
                    cand.append((
                        d[i - 2][j - 1] + 2.0 * (1.0 - g) + group_penalty,
                        ("2:1", g),
                    ))
            cand.append((d[i - 1][j] + skip_cost, ("skip_en",)))
            cand.append((d[i][j - 1] + skip_cost, ("skip_ru",)))

            d[i][j], trace[i][j] = min(cand, key=lambda c: c[0])

    ops: list[AlignOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        step = trace[i][j]
        assert step is not None  # guaranteed by initialization
        kind = step[0]
        if kind == "1:1":
            score = float(sim[i - 1][j - 1])
            en_idx, ru_idx = (i - 1,), (j - 1,)
            i, j = i - 1, j - 1
        elif kind == "1:2":
            score = step[1]
            en_idx, ru_idx = (i - 1,), (j - 2, j - 1)
            i, j = i - 1, j - 2
        elif kind == "2:1":
            score = step[1]
            en_idx, ru_idx = (i - 2, i - 1), (j - 1,)
            i, j = i - 2, j - 1
        elif kind == "skip_en":
            ops.append(AlignOp(
                kind="skip_en", en_idx=(i - 1,), ru_idx=(),
                score=-skip_cost,
                en_text=en_sentences[i - 1], ru_text="",
                quality="unmatched",
            ))
            i -= 1
            continue
        else:  # skip_ru
            ops.append(AlignOp(
                kind="skip_ru", en_idx=(), ru_idx=(j - 1,),
                score=-skip_cost,
                en_text="", ru_text=ru_sentences[j - 1],
                quality="unmatched",
            ))
            j -= 1
            continue

        ops.append(AlignOp(
            kind=kind, en_idx=en_idx, ru_idx=ru_idx, score=score,
            en_text=_join(en_sentences, en_idx),
            ru_text=_join(ru_sentences, ru_idx),
            quality="confident" if score >= confident_min else "doubtful",
        ))

    ops.reverse()
    return ops


def summarize_ops(ops: Sequence[AlignOp]) -> dict[str, int]:
    """Small tally useful for reports and tests."""
    counts: dict[str, int] = {}
    for op in ops:
        counts[op.kind] = counts.get(op.kind, 0) + 1
        if op.kind not in ("skip_en", "skip_ru"):
            counts[op.quality] = counts.get(op.quality, 0) + 1
    return counts
