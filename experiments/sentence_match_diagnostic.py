"""Diagnostic experiment: EN/RU sentence similarity in one chapter.

Takes the first aligned narrative chapter pair of both FB2 books,
embeds the first N sentences of each side with Qwen3-Embedding-0.6B
(via the shared embedding cache) and prints the top-K Russian
candidates per English sentence to a readable report in output/.

No correctness conclusions: high cosine similarity does not guarantee
a correct translation pair — the report is for manual review.

Run:  venv/bin/python experiments/sentence_match_diagnostic.py
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

EN_PATH = PROJECT_ROOT / "books" / "blade itslef.fb2"
RU_PATH = PROJECT_ROOT / "books" / "1 Krov i zhelezo.fb2"
MODELS_DIR = PROJECT_ROOT / "cache" / "models"
EMB_DIR = PROJECT_ROOT / "cache" / "embeddings"
OUT_DIR = PROJECT_ROOT / "output"

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
# Must match the encode settings used for all cached embeddings.
ENCODE_PARAMS: dict = {"prompt": None, "normalize_embeddings": True}

N_SENTENCES = 40
TOP_K = 5


def _chapter_sentences(chapter) -> list[str]:
    return [s.text for p in chapter.paragraphs for s in p.sentences]


def main() -> int:
    en_book = parse_fb2(EN_PATH)
    ru_book = parse_fb2(RU_PATH)

    alignment = align_chapters(en_book, ru_book)
    pair = alignment["pairs"][0]
    en_ch = en_book.chapters[pair["en_order"]]
    ru_ch = ru_book.chapters[pair["ru_order"]]
    print(f"Chapter pair: EN {en_ch.id} {en_ch.title!r} "
          f"↔ RU {ru_ch.id} {ru_ch.title!r}")

    en_sents = _chapter_sentences(en_ch)[:N_SENTENCES]
    ru_sents = _chapter_sentences(ru_ch)[:N_SENTENCES]
    print(f"Sentences taken: EN {len(en_sents)}, RU {len(ru_sents)}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}   Model: {MODEL_NAME}")
    t0 = time.perf_counter()
    model = SentenceTransformer(
        MODEL_NAME, device=device, cache_folder=str(MODELS_DIR),
    )
    print(f"Model load: {time.perf_counter() - t0:.2f} s")

    def encode(batch: list[str]) -> np.ndarray:
        kwargs = {k: v for k, v in ENCODE_PARAMS.items() if v is not None}
        return model.encode(batch, convert_to_numpy=True, **kwargs)

    all_texts = en_sents + ru_sents
    vectors, n_new, embed_s = embed_with_cache(
        encode, all_texts, EMB_DIR, MODEL_NAME, ENCODE_PARAMS,
    )
    print(f"Embeddings: {embed_s:.2f} s for {n_new} new texts "
          f"({len(all_texts) - n_new} from cache)")

    en_vecs, ru_vecs = vectors[: len(en_sents)], vectors[len(en_sents):]
    sims = en_vecs @ ru_vecs.T  # vectors are unit-normalized

    lines: list[str] = [
        "SENTENCE MATCHING DIAGNOSTIC",
        f"Model: {MODEL_NAME}  device: {device}  params: {ENCODE_PARAMS}",
        f"EN {en_ch.id} {en_ch.title!r}  ↔  RU {ru_ch.id} {ru_ch.title!r}",
        "NOTE: similarity only — a high score does not guarantee "
        "a correct translation.",
        "",
        "=== ENGLISH SENTENCES (original order) ===",
    ]
    lines += [f"E{i + 1:02d} {s}" for i, s in enumerate(en_sents)]
    lines += ["", "=== RUSSIAN SENTENCES (original order) ==="]
    lines += [f"R{i + 1:02d} {s}" for i, s in enumerate(ru_sents)]
    lines += ["", f"=== TOP-{TOP_K} RUSSIAN CANDIDATES PER ENGLISH SENTENCE ==="]

    top1_sims: list[float] = []
    top1_deltas: list[int] = []
    for i, en_text in enumerate(en_sents):
        order = np.argsort(-sims[i])[:TOP_K]
        top1_sims.append(float(sims[i][order[0]]))
        top1_deltas.append(int(order[0]) - i)
        lines.append("")
        lines.append(f"E{i + 1:02d} {en_text}")
        for rank, j in enumerate(order, 1):
            lines.append(
                f"   {rank}) sim={sims[i][j]:.4f}  R{j + 1:02d}  {ru_sents[j]}"
            )

    lines += [
        "",
        "=== SIMILARITY STATISTICS (descriptive only) ===",
        f"top-1 sim: mean={np.mean(top1_sims):.4f} "
        f"min={min(top1_sims):.4f} max={max(top1_sims):.4f}",
        "top-1 index delta (RU_idx - EN_idx): "
        + ", ".join(f"{d:+d}" for d in top1_deltas),
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "sentence_match_diagnostic.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to: {out}")
    print(f"top-1 sim mean: {np.mean(top1_sims):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
