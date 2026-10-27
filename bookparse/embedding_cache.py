"""Disk cache for sentence embeddings.

Vectors are stored as .npy files keyed by a hash of
(model name, vector-affecting encode params, text). Encode params that
influence the vector — e.g. `prompt`/`prompt_name` (instruction) or
`normalize_embeddings` — are part of the key, so changing them yields
new cache entries instead of stale hits.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np


def cache_key(model_name: str, text: str, params: dict | None = None) -> str:
    payload = json.dumps(
        {"model": model_name, "params": params or {}, "text": text},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _save_atomic(path: Path, vec: np.ndarray) -> None:
    """Write .npy via a temp file + os.replace so readers never see a
    partially written cache entry."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, vec)
    os.replace(tmp, path)


def _encode_checked(
    encode_fn: Callable[[list[str]], np.ndarray], batch: list[str],
) -> np.ndarray:
    """Run encode_fn and validate its output before it is cached."""
    arr = np.asarray(encode_fn(batch))
    if arr.ndim != 2 or not np.issubdtype(arr.dtype, np.number):
        raise ValueError(
            f"encode_fn must return a 2D numeric array "
            f"(n_texts, dim); got shape={arr.shape}, dtype={arr.dtype}"
        )
    if arr.shape[0] != len(batch):
        raise ValueError(
            f"encode_fn returned {arr.shape[0]} vectors for "
            f"{len(batch)} texts"
        )
    if arr.shape[1] == 0:
        raise ValueError("encode_fn returned vectors with zero dimension")
    if not np.isfinite(arr).all():
        raise ValueError(
            "encode_fn returned non-finite values (NaN or inf)"
        )
    return arr.astype(np.float32)


def embed_with_cache(
    encode_fn: Callable[[list[str]], np.ndarray],
    texts: list[str],
    cache_dir: str | Path,
    model_name: str,
    params: dict | None = None,
) -> tuple[np.ndarray, int, float]:
    """Embed texts, reusing cached vectors where possible.

    `encode_fn` takes a batch of texts and returns a (n, dim) array.
    `params` must contain every encode setting that can change the
    produced vector — it becomes part of the cache key.

    Returns (vectors in input order, n_newly_computed, compute_seconds).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    keys = [cache_key(model_name, t, params) for t in texts]
    vectors: dict[str, np.ndarray] = {}
    missing: list[tuple[str, str]] = []
    seen: set[str] = set()
    for key, text in zip(keys, texts):
        if key in seen:
            continue
        seen.add(key)
        f = cache_dir / f"{key}.npy"
        if f.is_file():
            try:
                vectors[key] = np.load(f)
                continue
            except (OSError, ValueError):
                pass  # corrupted/unreadable cache entry → recompute
        missing.append((key, text))

    t0 = time.perf_counter()
    if missing:
        new = _encode_checked(encode_fn, [t for _, t in missing])
        for (key, _), vec in zip(missing, new):
            _save_atomic(cache_dir / f"{key}.npy", vec)
            vectors[key] = vec
    elapsed = time.perf_counter() - t0
    return np.stack([vectors[k] for k in keys]), len(missing), elapsed
