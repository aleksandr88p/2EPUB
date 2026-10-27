"""Experiment: Qwen3-Embedding-0.6B on EN/RU sentence pairs.

Runs on Apple Silicon via PyTorch MPS. The model is downloaded from
Hugging Face on first run (into cache/models) and works locally after
that. Computed embeddings are cached in cache/embeddings/ keyed by
model name + text, so reruns skip inference for known sentences.

Run:  venv/bin/python experiments/test_embeddings.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bookparse.embedding_cache import embed_with_cache  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "cache" / "models"
EMB_DIR = PROJECT_ROOT / "cache" / "embeddings"

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
# Every encode setting that can change the produced vector goes here —
# it becomes part of the embedding cache key.
ENCODE_PARAMS: dict = {
    "prompt": None,                 # no instruction prefix
    "normalize_embeddings": True,   # unit-norm vectors
}

# (kind, english, russian)
PAIRS: list[tuple[str, str, str]] = [
    # correct translations (from the actual books)
    ("correct",
     "Logen plunged through the trees, bare feet slipping and sliding on the wet earth.",
     "Логен пробирался между деревьями, его босые ноги оскальзывались и проезжали по влажной земле."),
    ("correct",
     "'Shit!' Logen swore.",
     "— Дерьмо! — выругался Логен."),
    ("correct",
     "He considered that he'd earned a little luck.",
     "Однако Логен считал, что заслужил немного везения."),
    ("correct",
     "The huge, scarred man had been with him until a moment before, he was sure, but there wasn't any sign of him now.",
     "Огромный зарубцованный мужчина был с ним ещё минуту назад, он был уверен, но теперь тот исчез без следа."),
    # similar meaning, different wording
    ("paraphrase",
     "The end is coming.",
     "Конец уже близко."),
    ("paraphrase",
     "A shadow fell across his face.",
     "На лицо ему упала тень."),
    # wrong / unrelated
    ("wrong",
     "Logen plunged through the trees, bare feet slipping on the wet earth.",
     "Джезаль любил карты и не любил рано вставать."),
    ("wrong",
     "'Shit!' Logen swore.",
     "Стены были побелены и выглядели чистыми."),
    ("wrong",
     "The end is coming.",
     "Он открыл дверь и увидел за столом трёх практиков."),
]


def _encode_fn(model: SentenceTransformer):
    def encode(batch: list[str]) -> np.ndarray:
        kwargs = {k: v for k, v in ENCODE_PARAMS.items() if v is not None}
        return model.encode(batch, convert_to_numpy=True, **kwargs)
    return encode


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else 0.0


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1e6


def main() -> int:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}   Model: {MODEL_NAME}")
    print(f"Encode params (part of cache key): {ENCODE_PARAMS}")
    print(f"Model cache: {MODELS_DIR}   Embedding cache: {EMB_DIR}")

    rss0 = _rss_mb()
    t0 = time.perf_counter()
    model = SentenceTransformer(
        MODEL_NAME, device=device, cache_folder=str(MODELS_DIR),
    )
    load_s = time.perf_counter() - t0
    rss_model = _rss_mb()

    texts = sorted({t for pair in PAIRS for t in pair[1:]})
    vectors, n_computed, embed_s = embed_with_cache(
        _encode_fn(model), texts, EMB_DIR, MODEL_NAME, ENCODE_PARAMS,
    )
    by_text = dict(zip(texts, vectors))
    rss_after = _rss_mb()

    mps_alloc = (
        torch.mps.current_allocated_memory() / 1e6
        if device == "mps" else 0.0
    )
    print(f"\nModel load:        {load_s:7.2f} s")
    print(f"Embeddings:        {embed_s:7.2f} s for {n_computed} new texts "
          f"({len(texts) - n_computed} from cache)")
    print(f"Process RSS:       {rss0:7.0f} MB before -> {rss_model:.0f} MB "
          f"after load -> {rss_after:.0f} MB after encode")
    if device == "mps":
        print(f"MPS allocated:     {mps_alloc:7.0f} MB")

    print("\n#    kind        sim      English -> Russian")
    print("-" * 100)
    for i, (kind, en, ru) in enumerate(PAIRS, 1):
        sim = _cosine(by_text[en], by_text[ru])
        en_s = (en[:44] + "…") if len(en) > 45 else en
        ru_s = (ru[:40] + "…") if len(ru) > 41 else ru
        print(f"{i:<4} {kind:<11} {sim:.4f}   {en_s}  ->  {ru_s}")

    by_kind: dict[str, list[float]] = {}
    for kind, en, ru in PAIRS:
        by_kind.setdefault(kind, []).append(_cosine(by_text[en], by_text[ru]))
    print("\nAverages by pair kind:")
    for kind, sims in by_kind.items():
        print(f"  {kind:<11} mean={np.mean(sims):.4f}  "
              f"min={min(sims):.4f}  max={max(sims):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
