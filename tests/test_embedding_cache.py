"""Tests for the embedding cache (bookparse.embedding_cache).

Uses a fake encoder — no ML model is loaded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bookparse.embedding_cache import cache_key, embed_with_cache

MODEL = "Qwen/Qwen3-Embedding-0.6B"


def make_encoder(calls: list[list[str]]):
    """Deterministic fake encoder that records its batches."""
    def encode(batch: list[str]) -> np.ndarray:
        calls.append(list(batch))
        return np.stack([
            np.full(8, (abs(hash(t)) % 1000) / 1000.0, dtype=np.float32)
            for t in batch
        ])
    return encode


def test_second_run_serves_from_cache(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    enc = make_encoder(calls)
    texts = ["alpha", "beta", "gamma"]

    v1, n1, _ = embed_with_cache(enc, texts, tmp_path, MODEL, {})
    assert n1 == 3 and len(calls) == 1

    v2, n2, _ = embed_with_cache(enc, texts, tmp_path, MODEL, {})
    assert n2 == 0 and len(calls) == 1       # encoder not called again
    np.testing.assert_array_equal(v1, v2)     # identical vectors


def test_params_change_invalidates_cache(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    enc = make_encoder(calls)
    texts = ["alpha"]

    embed_with_cache(enc, texts, tmp_path, MODEL,
                     {"normalize_embeddings": True, "prompt": None})
    v2, n2, _ = embed_with_cache(
        enc, texts, tmp_path, MODEL,
        {"normalize_embeddings": False, "prompt": None},
    )
    assert n2 == 1                            # recomputed, no stale hit
    v3, n3, _ = embed_with_cache(
        enc, texts, tmp_path, MODEL,
        {"normalize_embeddings": False, "prompt": None},
    )
    assert n3 == 0
    np.testing.assert_array_equal(v2, v3)


def test_prompt_and_model_are_part_of_key(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    enc = make_encoder(calls)
    texts = ["alpha"]

    _, n1, _ = embed_with_cache(enc, texts, tmp_path, MODEL, {"prompt": None})
    _, n2, _ = embed_with_cache(enc, texts, tmp_path, MODEL,
                                {"prompt": "Represent this query: "})
    _, n3, _ = embed_with_cache(enc, texts, tmp_path, "other-model",
                                {"prompt": None})
    assert (n1, n2, n3) == (1, 1, 1)


def test_cache_key_is_stable_and_distinct() -> None:
    k1 = cache_key(MODEL, "a", {"normalize_embeddings": True})
    k2 = cache_key(MODEL, "a", {"normalize_embeddings": True})
    assert k1 == k2
    assert k1 != cache_key(MODEL, "a", {"normalize_embeddings": False})
    assert k1 != cache_key(MODEL, "b", {"normalize_embeddings": True})
    assert k1 != cache_key("other", "a", {"normalize_embeddings": True})


def test_duplicate_texts_encoded_once(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    enc = make_encoder(calls)
    vectors, n, _ = embed_with_cache(
        enc, ["alpha", "alpha", "beta"], tmp_path, MODEL, {},
    )
    assert n == 2
    assert vectors.shape == (3, 8)
    np.testing.assert_array_equal(vectors[0], vectors[1])
