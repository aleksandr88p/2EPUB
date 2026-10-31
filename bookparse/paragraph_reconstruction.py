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

from .sentence_alignment import AlignOp


@dataclass
class BilingualParagraph:
    """One EN paragraph plus the RU sentences aligned to it."""

    index: int                    # EN paragraph index within the chapter
    type: str                     # block type from the EN source
    en_sentences: list[str] = field(default_factory=list)
    ru_sentences: list[str] = field(default_factory=list)
    # Kinds of ops that contributed RU, in order (quality reports).
    ru_op_kinds: list[str] = field(default_factory=list)


def rebuild_chapter(
    en_paragraphs: Sequence[Sequence[str]],
    ru_sentences: Sequence[str],
    ops: Sequence[AlignOp],
    *,
    para_types: Sequence[str] | None = None,
) -> list[BilingualParagraph]:
    """Distribute RU sentences over EN paragraphs via alignment ops.

    `en_paragraphs` — EN sentences grouped by paragraph, in order;
    their flattening is the sentence stream the ops were computed on.
    `para_types` optionally carries the source block type per
    paragraph (defaults to "paragraph").
    """
    if para_types is None:
        para_types = ["paragraph"] * len(en_paragraphs)
    elif len(para_types) != len(en_paragraphs):
        raise ValueError("para_types length must match en_paragraphs")

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

    return [
        BilingualParagraph(
            index=pi,
            type=para_types[pi],
            en_sentences=list(para),
            ru_sentences=[ru_sentences[j] for j in ru_owner[pi]],
            ru_op_kinds=op_kinds[pi],
        )
        for pi, para in enumerate(en_paragraphs)
    ]
