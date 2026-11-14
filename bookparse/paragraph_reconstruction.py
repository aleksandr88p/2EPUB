"""Rebuild bilingual paragraphs from sentence-alignment ops.

The EN paragraph structure is authoritative: each output paragraph
mirrors one source EN paragraph — same order, same sentences. RU
sentences are attached to paragraphs according to the alignment ops.

Deterministic attachment rules (part of the contract):

* matched ops ("1:1", "1:2", "2:1"): RU sentences go to the paragraph
  of the op's FIRST EN index. A 2:1 merge spanning an EN paragraph
  boundary puts the whole merged translation in the earlier paragraph.
* "skip_ru" (RU-only): attached to the paragraph of the NEXT op that
  covers both sides — translator insertions usually precede the
  sentence they decorate. Trailing skip_ru ops (no later match) fall
  back to the paragraph of the last matched op.
* "skip_en": the EN paragraph keeps its text and gets no translation.

RU order inside each output paragraph always follows the source RU
stream. The function is chapter-scoped — pass ops produced for one
aligned chapter pair and chapters can never mix. Nothing is dropped
or duplicated: a ValueError is raised unless the ops cover every EN
and RU index exactly once, in order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .segment import count_words
from .sentence_alignment import AlignOp

_SHORT_PARAGRAPH_MIN_WORDS = 35
_SHORT_PARAGRAPH_MAX_WORDS = 40
_UNSPLITTABLE_TYPES = {
    "empty-line", "heading", "image", "preformatted", "subtitle", "table",
    "verse",
}


@dataclass
class BilingualParagraph:
    """One source paragraph or fragment plus its aligned translation."""

    index: int                    # EN paragraph index within the chapter
    type: str                     # block type from the EN source
    en_sentences: list[str] = field(default_factory=list)
    ru_sentences: list[str] = field(default_factory=list)
    # Kinds of ops that contributed RU, in order (quality reports).
    ru_op_kinds: list[str] = field(default_factory=list)
    include_images: bool = True


def rebuild_chapter(
    en_paragraphs: Sequence[Sequence[str]],
    ru_sentences: Sequence[str],
    ops: Sequence[AlignOp],
    *,
    para_types: Sequence[str] | None = None,
    short_paragraphs: bool = False,
) -> list[BilingualParagraph]:
    """Distribute RU sentences over EN paragraphs via alignment ops.

    `en_paragraphs` — EN sentences grouped by paragraph, in order;
    their flattening is the sentence stream the ops were computed on.
    `para_types` optionally carries the source block type per
    paragraph (defaults to "paragraph").
    `short_paragraphs` splits eligible blocks between alignment ops.
    """
    if para_types is None:
        para_types = ["paragraph"] * len(en_paragraphs)
    elif len(para_types) != len(en_paragraphs):
        raise ValueError("para_types length must match en_paragraphs")

    if not en_paragraphs and ru_sentences:
        raise ValueError(
            "EN chapter has no paragraphs to hold "
            f"{len(ru_sentences)} RU sentence(s)"
        )

    sent_para = [
        pi for pi, para in enumerate(en_paragraphs) for _ in para
    ]
    n, m = len(sent_para), len(ru_sentences)

    en_used = [i for op in ops for i in op.en_idx]
    ru_used = [j for op in ops for j in op.ru_idx]
    if (
        en_used != sorted(en_used)
        or ru_used != sorted(ru_used)
        or sorted(en_used) != list(range(n))
        or sorted(ru_used) != list(range(m))
    ):
        raise ValueError(
            "ops must cover every EN and RU index exactly once, in order"
        )

    # Paragraph owning each matched op: paragraph of its FIRST EN
    # index; None for ops without EN (skip_ru).
    op_para: list[int | None] = [
        sent_para[op.en_idx[0]] if op.en_idx else None for op in ops
    ]

    # For each position, the paragraph of the next RU-covering op —
    # the lookahead anchor for skip_ru.
    next_match: list[int | None] = [None] * len(ops)
    nxt: int | None = None
    for k in range(len(ops) - 1, -1, -1):
        next_match[k] = nxt
        if ops[k].ru_idx and op_para[k] is not None:
            nxt = op_para[k]

    ru_owner: list[list[int]] = [[] for _ in en_paragraphs]
    op_kinds: list[list[str]] = [[] for _ in en_paragraphs]
    last_para: int | None = None
    for k, op in enumerate(ops):
        if not op.ru_idx:
            continue
        target = op_para[k] if op_para[k] is not None else next_match[k]
        if target is None:  # trailing skip_ru (or no matches at all)
            target = last_para if last_para is not None else 0
        ru_owner[target].extend(op.ru_idx)
        op_kinds[target].append(op.kind)
        if op.en_idx:
            last_para = target

    rebuilt = [
        BilingualParagraph(
            index=pi,
            type=para_types[pi],
            en_sentences=list(para),
            ru_sentences=[ru_sentences[j] for j in ru_owner[pi]],
            ru_op_kinds=op_kinds[pi],
        )
        for pi, para in enumerate(en_paragraphs)
    ]
    if short_paragraphs:
        return _split_rebuilt_paragraphs(
            rebuilt, en_paragraphs, ru_sentences, ops,
        )
    return rebuilt


def _partition_ops(
    op_indices: Sequence[int],
    ops: Sequence[AlignOp],
    en_sentences: Sequence[str],
) -> list[list[int]]:
    weights = {
        op_index: count_words(" ".join(
            en_sentences[i] for i in ops[op_index].en_idx
        ))
        for op_index in op_indices
    }
    groups: list[list[int]] = []
    current: list[int] = []

    for op_index in op_indices:
        if weights[op_index] > _SHORT_PARAGRAPH_MAX_WORDS:
            groups.extend(_partition_op_run(current, weights))
            current = []
            groups.append([op_index])
        else:
            current.append(op_index)

    groups.extend(_partition_op_run(current, weights))
    return groups or [list(op_indices)]


def _partition_op_run(
    op_indices: Sequence[int],
    weights: dict[int, int],
) -> list[list[int]]:
    if not op_indices:
        return []

    target = (
        _SHORT_PARAGRAPH_MIN_WORDS + _SHORT_PARAGRAPH_MAX_WORDS
    ) / 2
    count = len(op_indices)
    costs = [0.0] * (count + 1)
    block_counts = [0] * (count + 1)
    ends = [-1] * count

    for start in range(count - 1, -1, -1):
        total_words = 0
        best_score = (float("inf"), count + 1, 0)
        for end in range(start, count):
            total_words += weights[op_indices[end]]
            if total_words > _SHORT_PARAGRAPH_MAX_WORDS:
                break
            next_start = end + 1
            score = (
                (total_words - target) ** 2 + costs[next_start],
                1 + block_counts[next_start],
                -total_words,
            )
            if score < best_score:
                best_score = score
                ends[start] = end

        costs[start] = best_score[0]
        block_counts[start] = best_score[1]

    groups = []
    start = 0
    while start < count:
        end = ends[start]
        groups.append(list(op_indices[start:end + 1]))
        start = end + 1
    return groups


def _split_rebuilt_paragraphs(
    rebuilt: list[BilingualParagraph],
    en_paragraphs: Sequence[Sequence[str]],
    ru_sentences: Sequence[str],
    ops: Sequence[AlignOp],
) -> list[BilingualParagraph]:
    en_sentences = [sentence for para in en_paragraphs for sentence in para]
    sent_para = [
        pi for pi, para in enumerate(en_paragraphs) for _ in para
    ]
    op_para = [
        sent_para[op.en_idx[0]] if op.en_idx else None for op in ops
    ]
    ops_by_para: list[list[int]] = [[] for _ in en_paragraphs]
    crossing_paras: set[int] = set()
    for op_index, op in enumerate(ops):
        if not op.en_idx:
            continue
        touched = {sent_para[i] for i in op.en_idx}
        owner = op_para[op_index]
        ops_by_para[owner].append(op_index)
        if len(touched) > 1:
            crossing_paras.update(touched)

    fragments_by_para: list[list[BilingualParagraph]] = []
    was_split: list[bool] = []
    op_chunk = [0] * len(ops)
    for pi, paragraph in enumerate(rebuilt):
        op_indices = ops_by_para[pi]
        if (
            not paragraph.en_sentences
            or paragraph.type in _UNSPLITTABLE_TYPES
            or pi in crossing_paras
        ):
            groups = [op_indices]
        else:
            groups = _partition_ops(op_indices, ops, en_sentences)

        if len(groups) <= 1:
            fragments = [paragraph]
            if groups:
                for op_index in groups[0]:
                    op_chunk[op_index] = 0
            fragments_by_para.append(fragments)
            was_split.append(False)
            continue

        fragments = []
        for chunk_index, group in enumerate(groups):
            en_indices = sorted(
                en_index
                for op_index in group
                for en_index in ops[op_index].en_idx
                if sent_para[en_index] == pi
            )
            fragments.append(BilingualParagraph(
                index=paragraph.index,
                type=paragraph.type,
                en_sentences=[en_sentences[i] for i in en_indices],
                include_images=chunk_index == 0,
            ))
            for op_index in group:
                op_chunk[op_index] = chunk_index
        fragments_by_para.append(fragments)
        was_split.append(True)

    next_match_op: list[int | None] = [None] * len(ops)
    next_match: int | None = None
    for op_index in range(len(ops) - 1, -1, -1):
        next_match_op[op_index] = next_match
        if ops[op_index].ru_idx and op_para[op_index] is not None:
            next_match = op_index
    last_match_op = next(
        (
            op_index
            for op_index in range(len(ops) - 1, -1, -1)
            if ops[op_index].ru_idx and op_para[op_index] is not None
        ),
        None,
    )

    for op_index, op in enumerate(ops):
        if not op.ru_idx:
            continue
        if op_para[op_index] is not None:
            target_para = op_para[op_index]
            target_chunk = op_chunk[op_index]
        else:
            target_op = next_match_op[op_index]
            if target_op is None:
                target_op = last_match_op
            if target_op is None:
                target_para, target_chunk = 0, 0
            else:
                target_para = op_para[target_op]
                target_chunk = op_chunk[target_op]

        if was_split[target_para]:
            target = fragments_by_para[target_para][target_chunk]
            target.ru_sentences.extend(ru_sentences[i] for i in op.ru_idx)
            target.ru_op_kinds.append(op.kind)

    return [
        fragment
        for fragments in fragments_by_para
        for fragment in fragments
    ]
