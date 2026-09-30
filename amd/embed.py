"""Embeddings — turn text into vectors via Hugging Face.

We use `sentence-transformers/all-MiniLM-L6-v2`:
  - 384-dimensional vectors
  - Fast, small (~90 MB), good quality for retrieval
  - Free tier on Hugging Face Inference API

The HF endpoint has quirks we handle here:
  - Cold start: model may need to load (503). We send `wait_for_model`.
  - Rate limits: 429 with `x-wait-for-model` header. We retry with backoff.
  - Batch limits: keep batches under ~32 to stay reliable.
"""
from __future__ import annotations

import os
import time
from typing import Iterable

import httpx
import numpy as np

MODEL_ID = "BAAI/bge-small-en-v1.5"
HF_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"
EMBED_DIM = 384

_BATCH_SIZE = 16
_MAX_RETRIES = 5
_TIMEOUT = 60.0


class EmbedError(RuntimeError):
    pass


def _auth_header() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EmbedError("HF_TOKEN is not set. Did you fill in .env?")
    return {"Authorization": f"Bearer {token}"}


def _post_batch(batch: list[str]) -> list[list[float]]:
    """Send one batch, retrying on 503 (cold start) and 429 (rate limit)."""
    payload = {"inputs": batch, "options": {"wait_for_model": True}}
    delay = 2.0
    last_body = ""
    for attempt in range(_MAX_RETRIES):
        try:
            r = httpx.post(
                HF_URL,
                headers=_auth_header(),
                json=payload,
                timeout=_TIMEOUT,
            )
        except httpx.HTTPError as e:
            raise EmbedError(f"HF request failed: {e}") from e

        if r.status_code == 200:
            return r.json()

        if r.status_code in (503, 429):
            last_body = r.text[:200]
            time.sleep(delay)
            delay = min(delay * 2, 20.0)
            continue

        raise EmbedError(f"HF {r.status_code}: {r.text[:300]}")

    raise EmbedError(
        f"HF gave up after {_MAX_RETRIES} retries. Last body: {last_body}"
    )


def embed_texts(texts: Iterable[str]) -> np.ndarray:
    """Embed a list of strings. Returns (N, 384) float32 array."""
    items = [t for t in texts if t and t.strip()]
    if not items:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    out: list[list[float]] = []
    for i in range(0, len(items), _BATCH_SIZE):
        batch = items[i : i + _BATCH_SIZE]
        vecs = _post_batch(batch)
        out.extend(vecs)

    arr = np.asarray(out, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != EMBED_DIM:
        raise EmbedError(
            f"unexpected embedding shape {arr.shape}; expected (N, {EMBED_DIM})"
        )
    return arr


def embed_query(text: str) -> np.ndarray:
    """Embed a single string. Returns (384,) float32 array."""
    arr = embed_texts([text])
    if arr.shape[0] != 1:
        raise EmbedError("expected exactly one embedding")
    return arr[0]
