"""Preview: bilingual paragraphs rebuilt from real alignment.

Takes the first aligned chapter pair ('The End' <-> 'Конец'), runs the
frozen DP aligner with the joined-text group scorer (all embeddings
served from the cache), rebuilds bilingual paragraphs and writes the
first PREVIEW_N of them to output/paragraph_preview.txt.

Run:  venv/bin/python experiments/paragraph_preview.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bookparse.chapter_alignment import align_chapters  # noqa: E402
from bookparse.embedding_cache import embed_with_cache  # noqa: E402
from bookparse.fb2_parser import parse_fb2  # noqa: E402
from bookparse.paragraph_reconstruction import (  # noqa: E402
    rebuild_chapter,
)
from bookparse.sentence_alignment import align_sentences  # noqa: E402

EN_PATH = PROJECT_ROOT / "books" / "blade itslef.fb2"
RU_PATH = PROJECT_ROOT / "books" / "1 Krov i zhelezo.fb2"
MODELS_DIR = PROJECT_ROOT / "cache" / "models"
EMB_DIR = PROJECT_ROOT / "cache" / "embeddings"
OUT_DIR = PROJECT_ROOT / "output"

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
ENCODE_PARAMS: dict = {"prompt": None, "normalize_embeddings": True}

PREVIEW_N = 10


def _normed(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def main() -> int:
    en_book = parse_fb2(EN_PATH)
    ru_book = parse_fb2(RU_PATH)
    pair = align_chapters(en_book, ru_book)["pairs"][0]
    en_ch = en_book.chapters[pair["en_order"]]
    ru_ch = ru_book.chapters[pair["ru_order"]]

    en_paragraphs = [[s.text for s in p.sentences] for p in en_ch.paragraphs]
    para_types = [p.type for p in en_ch.paragraphs]
    en_sents = [s for p in en_paragraphs for s in p]
    ru_sents = [s.text for p in ru_ch.paragraphs for s in p.sentences]
    print(f"Chapter pair: {en_ch.id} {en_ch.title!r} <-> "
          f"{ru_ch.id} {ru_ch.title!r}")
    print(f"EN paragraphs: {len(en_paragraphs)} ({len(en_sents)} sents), "
          f"RU sents: {len(ru_sents)}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(
        MODEL_NAME, device=device, cache_folder=str(MODELS_DIR)
    )

    def encode(batch: list[str]) -> np.ndarray:
        kwargs = {k: v for k, v in ENCODE_PARAMS.items() if v is not None}
        return model.encode(batch, convert_to_numpy=True, **kwargs)

    vectors, n_new, _ = embed_with_cache(
        encode, en_sents + ru_sents, EMB_DIR, MODEL_NAME, ENCODE_PARAMS,
    )
    en_joins = [" ".join(en_sents[i:i + 2])
                for i in range(len(en_sents) - 1)]
    ru_joins = [" ".join(ru_sents[j:j + 2])
                for j in range(len(ru_sents) - 1)]
    join_vecs, n_new_j, _ = embed_with_cache(
        encode, en_joins + ru_joins, EMB_DIR, MODEL_NAME, ENCODE_PARAMS,
    )
    print(f"Embeddings: {n_new} new, joins: {n_new_j} new")

    E, R = vectors[: len(en_sents)], vectors[len(en_sents):]
    EJ = join_vecs[: len(en_joins)]
    RJ = join_vecs[len(en_joins):]
    sim = E @ R.T

    def group_scorer(ei: tuple[int, ...], rj: tuple[int, ...]) -> float:
        a = _normed(E[ei[0]] if len(ei) == 1 else EJ[ei[0]])
        b = _normed(R[rj[0]] if len(rj) == 1 else RJ[rj[0]])
        return float(a @ b)

    ops = align_sentences(en_sents, ru_sents, sim, group_scorer=group_scorer)
    paras = rebuild_chapter(en_paragraphs, ru_sents, ops,
                            para_types=para_types)

    # --- integrity: nothing lost, nothing duplicated -----------------
    rebuilt_ru = [s for p in paras for s in p.ru_sentences]
    rebuilt_en = [s for p in paras for s in p.en_sentences]
    integrity = rebuilt_ru == ru_sents and rebuilt_en == en_sents
    n_bare = sum(1 for p in paras if p.en_sentences and not p.ru_sentences)

    lines = [
        "BILINGUAL PARAGRAPH PREVIEW (rebuild_chapter on real alignment)",
        f"Chapter pair: {en_ch.id} {en_ch.title!r} "
        f"<-> {ru_ch.id} {ru_ch.title!r}",
        f"EN paragraphs: {len(en_paragraphs)} "
        f"({len(en_sents)} sents), RU sents: {len(ru_sents)}",
        f"Integrity (RU/EN preserved, order kept): "
        f"{'OK' if integrity else 'FAILED'}",
        f"EN paragraphs without RU: {n_bare}",
        "",
    ]
    for p in paras[:PREVIEW_N]:
        lines.append(f"=== p{p.index} ({p.type}) ops: "
                     f"{','.join(p.ru_op_kinds) or '-'} ===")
        for s in p.en_sentences:
            lines.append(f"EN> {s}")
        if p.ru_sentences:
            for s in p.ru_sentences:
                lines.append(f"RU> {s}")
        else:
            lines.append("RU> (no aligned RU)")
        lines.append("")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "paragraph_preview.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport: {out}")
    print(f"Integrity: {'OK' if integrity else 'FAILED'}, "
          f"bare EN paragraphs: {n_bare}")
    return 0 if integrity else 1


if __name__ == "__main__":
    sys.exit(main())
