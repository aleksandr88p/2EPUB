"""Tests for the embedding cache (bookparse.embedding_cache).

Uses a fake encoder — no ML model is loaded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

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


def test_corrupted_cache_file_is_recomputed(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    enc = make_encoder(calls)
    v1, n1, _ = embed_with_cache(enc, ["alpha"], tmp_path, MODEL, {})
    assert n1 == 1

    # Corrupt the cached .npy — next run must recompute, not crash.
    (tmp_path / f"{cache_key(MODEL, 'alpha', {})}.npy").write_bytes(
        b"not a valid npy file"
    )
    v2, n2, _ = embed_with_cache(enc, ["alpha"], tmp_path, MODEL, {})
    assert n2 == 1
    np.testing.assert_array_equal(v1, v2)

    # And the repaired entry serves subsequent runs.
    _, n3, _ = embed_with_cache(enc, ["alpha"], tmp_path, MODEL, {})
    assert n3 == 0


def test_encoder_wrong_row_count_raises(tmp_path: Path) -> None:
    def bad_encode(batch: list[str]) -> np.ndarray:
        return np.zeros((len(batch) - 1, 8), dtype=np.float32)

    with pytest.raises(ValueError, match="returned .* vectors for .* texts"):
        embed_with_cache(bad_encode, ["a", "b", "c"], tmp_path, MODEL, {})


def test_encoder_nan_raises(tmp_path: Path) -> None:
    def nan_encode(batch: list[str]) -> np.ndarray:
        out = np.ones((len(batch), 8), dtype=np.float32)
        out[0, 0] = np.nan
        return out

    with pytest.raises(ValueError, match="non-finite"):
        embed_with_cache(nan_encode, ["a"], tmp_path, MODEL, {})
    # Nothing was cached — retry with a healthy encoder works.
    _, n, _ = embed_with_cache(
        make_encoder([]), ["a"], tmp_path, MODEL, {},
    )
    assert n == 1


def test_encoder_bad_shape_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="2D numeric array"):
        embed_with_cache(
            lambda b: np.array([1.0, 2.0, 3.0]),  # 1D instead of 2D
            ["a"], tmp_path, MODEL, {},
        )
