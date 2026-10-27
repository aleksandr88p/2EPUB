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
    for key, text in zip(keys, texts):
        if key in vectors:
            continue
        f = cache_dir / f"{key}.npy"
        if f.is_file():
            vectors[key] = np.load(f)
        else:
            vectors[key] = np.empty(0)  # placeholder to dedupe `missing`
            missing.append((key, text))

    t0 = time.perf_counter()
    if missing:
        new = encode_fn([t for _, t in missing])
        for (key, _), vec in zip(missing, new):
            vec = np.asarray(vec, dtype=np.float32)
            np.save(cache_dir / f"{key}.npy", vec)
            vectors[key] = vec
    elapsed = time.perf_counter() - t0
    return np.stack([vectors[k] for k in keys]), len(missing), elapsed
