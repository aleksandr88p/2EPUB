"""Extended diagnostic: DP sentence alignment on real book text.

Builds a continuous EN/RU sentence stream from the first aligned
narrative chapters ('The End' ↔ 'Конец' and onward), takes the first
EN_WINDOW / RU_WINDOW sentences, embeds them via the shared embedding
cache, runs the DP aligner with a joined-text group scorer and writes
a readable report to output/.

The group scorer embeds the joined text of each adjacent sentence pair
(single batch through the embedding cache), so group similarity matches
what a production pipeline would compute — the same quantity the DP
cost model was designed around.

Trailing RU-only ops after the last covered EN sentence are marked as
the boundary region and excluded from quality stats — they are outside
the selected EN fragment, not alignment errors.

Run:  venv/bin/python experiments/sentence_alignment_diagnostic.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bookparse.chapter_alignment import align_chapters  # noqa: E402
from bookparse.embedding_cache import embed_with_cache  # noqa: E402
from bookparse.fb2_parser import parse_fb2  # noqa: E402
from bookparse.sentence_alignment import (  # noqa: E402
    AlignOp,
    align_sentences,
)

EN_PATH = PROJECT_ROOT / "books" / "blade itslef.fb2"
RU_PATH = PROJECT_ROOT / "books" / "1 Krov i zhelezo.fb2"
MODELS_DIR = PROJECT_ROOT / "cache" / "models"
EMB_DIR = PROJECT_ROOT / "cache" / "embeddings"
OUT_DIR = PROJECT_ROOT / "output"

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
ENCODE_PARAMS: dict = {"prompt": None, "normalize_embeddings": True}

EN_WINDOW = 120
RU_WINDOW = 160


def _chapter_sentences(chapter) -> list[str]:
    return [s.text for p in chapter.paragraphs for s in p.sentences]


def _stream(pairs, en_book, ru_book):
    """Concatenate sentences of consecutive paired chapters."""
    en_sents: list[str] = []
    ru_sents: list[str] = []
    chapters: list[tuple[str, str]] = []
    for p in pairs:
        ec = en_book.chapters[p["en_order"]]
        rc = ru_book.chapters[p["ru_order"]]
        chapters.append((f"{ec.id} {ec.title!r}", f"{rc.id} {rc.title!r}"))
        en_sents += _chapter_sentences(ec)
        ru_sents += _chapter_sentences(rc)
        if len(en_sents) >= EN_WINDOW and len(ru_sents) >= RU_WINDOW:
            break
    return en_sents[:EN_WINDOW], ru_sents[:RU_WINDOW], chapters


def _normed(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def _fmt_idx(idx: tuple[int, ...], prefix: str) -> str:
    return "[" + ",".join(f"{prefix}{i}" for i in idx) + "]" if idx else "[]"


def main() -> int:
    en_book = parse_fb2(EN_PATH)
    ru_book = parse_fb2(RU_PATH)
    alignment = align_chapters(en_book, ru_book)
    en_sents, ru_sents, chapters = _stream(
        alignment["pairs"], en_book, ru_book,
    )
    print(f"Chapters used: {[t for t, _ in chapters]}")
    print(f"Window: EN {len(en_sents)}, RU {len(ru_sents)}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    t0 = time.perf_counter()
    model = SentenceTransformer(
        MODEL_NAME, device=device, cache_folder=str(MODELS_DIR),
    )
    print(f"Model load: {time.perf_counter() - t0:.2f} s ({device})")

    def encode(batch: list[str]) -> np.ndarray:
        kwargs = {k: v for k, v in ENCODE_PARAMS.items() if v is not None}
        return model.encode(batch, convert_to_numpy=True, **kwargs)

    texts = en_sents + ru_sents
    vectors, n_new, embed_s = embed_with_cache(
        encode, texts, EMB_DIR, MODEL_NAME, ENCODE_PARAMS,
    )
    print(f"Embeddings: {embed_s:.2f} s for {n_new} new texts "
          f"({len(texts) - n_new} cached)")

    # Joined texts for every adjacent pair the DP may group — one batch
    # through the same cache, so repeated runs cost nothing.
    en_joins = [" ".join(en_sents[i:i + 2])
                for i in range(len(en_sents) - 1)]
    ru_joins = [" ".join(ru_sents[j:j + 2])
                for j in range(len(ru_sents) - 1)]
    join_vecs, n_new_j, embed_s_j = embed_with_cache(
        encode, en_joins + ru_joins, EMB_DIR, MODEL_NAME, ENCODE_PARAMS,
    )
    print(f"Join embeddings: {embed_s_j:.2f} s for {n_new_j} new texts "
          f"({len(en_joins) + len(ru_joins) - n_new_j} cached)")

    E, R = vectors[: len(en_sents)], vectors[len(en_sents):]
    EJ = join_vecs[: len(en_joins)]
    RJ = join_vecs[len(en_joins):]
    sim = E @ R.T

    def group_scorer(ei: tuple[int, ...], rj: tuple[int, ...]) -> float:
        a = _normed(E[ei[0]] if len(ei) == 1 else EJ[ei[0]])
        b = _normed(R[rj[0]] if len(rj) == 1 else RJ[rj[0]])
        return float(a @ b)

    ops = align_sentences(en_sents, ru_sents, sim, group_scorer=group_scorer)

    # --- coverage verification --------------------------------------
    en_used = [i for op in ops for i in op.en_idx]
    ru_used = [j for op in ops for j in op.ru_idx]
    coverage_ok = (
        sorted(en_used) == list(range(len(en_sents)))
        and sorted(ru_used) == list(range(len(ru_sents)))
        and en_used == sorted(en_used)
        and ru_used == sorted(ru_used)
    )

    # --- boundary region: ops after the last covered EN sentence ----
    last_en_op = max(
        k for k, op in enumerate(ops) if op.en_idx
    )
    boundary_start = last_en_op + 1
    core_ops, boundary_ops = ops[:boundary_start], ops[boundary_start:]

    def is_boundary(op: AlignOp) -> bool:
        return op in boundary_ops

    lines: list[str] = [
        "SENTENCE ALIGNMENT DIAGNOSTIC (DP prototype)",
        f"Model: {MODEL_NAME}  device: {device}  params: {ENCODE_PARAMS}",
        "Group scorer: joined-text embeddings (via cache)",
        f"Chapters: {' + '.join(t for t, _ in chapters)} "
        f"<-> {' + '.join(t for _, t in chapters)}",
        f"Window: EN {len(en_sents)} sentences, RU {len(ru_sents)} sentences",
        f"Coverage check (each index exactly once, order preserved): "
        f"{'OK' if coverage_ok else 'FAILED'}",
        "",
        "=== OPERATIONS (in order; [B] = boundary region) ===",
    ]
    for k, op in enumerate(ops):
        flag = " [B]" if k >= boundary_start else ""
        lines.append(
            f"{k:3d}{flag:4s} {op.kind:<8s} {op.score:+.3f} "
            f"{op.quality:<9s} "
            f"{_fmt_idx(op.en_idx, 'E'):<14s} -> {_fmt_idx(op.ru_idx, 'R'):<16s}"
        )
        if op.en_text:
            lines.append(f"      EN: {op.en_text}")
        if op.ru_text:
            lines.append(f"      RU: {op.ru_text}")
    lines += ["", f"--- boundary region starts at op #{boundary_start} ---",
              "(RU-only ops beyond the selected EN fragment; not errors)"]

    core = [o for o in ops if not is_boundary(o)]
    counts: dict[str, int] = {}
    for o in core:
        counts[o.kind] = counts.get(o.kind, 0) + 1
        if o.quality != "unmatched":
            counts[o.quality] = counts.get(o.quality, 0) + 1

    def section(title: str, sel: list[AlignOp]) -> None:
        lines.append("")
        lines.append(f"=== {title} ({len(sel)}) ===")
        for o in sel:
            lines.append(
                f"{o.kind:<8s} {o.score:+.3f} "
                f"{_fmt_idx(o.en_idx, 'E')} -> {_fmt_idx(o.ru_idx, 'R')}"
            )
            if o.en_text:
                lines.append(f"   EN: {o.en_text}")
            if o.ru_text:
                lines.append(f"   RU: {o.ru_text}")

    lines += ["", "=== STATISTICS (core region, boundary excluded) ==="]
    lines += [f"  {k}: {v}" for k, v in sorted(counts.items())]
    lines.append(f"  boundary ops (excluded): {len(boundary_ops)}")

    section("UNMATCHED (core region)",
            [o for o in core if o.quality == "unmatched"])
    section("DOUBTFUL (core region)",
            [o for o in core if o.quality == "doubtful"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "sentence_alignment_diagnostic.txt"
    out.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nReport: {out}")
    print("Core ops:", counts, f"boundary={len(boundary_ops)}")
    print(f"Coverage: {'OK' if coverage_ok else 'FAILED'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
