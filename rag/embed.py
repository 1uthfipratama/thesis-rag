"""Embedding via fastembed (ONNX on CPU, no PyTorch).

fastembed's bge-small-en-v1.5 is the quantised ONNX export (Qdrant/...-onnx-Q,
67 MB): small enough for a free CPU Space, slightly less exact than fp32.

fastembed's query_embed() does NOT add BGE's retrieval instruction (checked in
fastembed/text/text_embedding_base.py: it just calls embed()). BGE v1.5 works
without it, but the model card recommends it for short-query -> passage search,
so it's prepended to queries only (settings.query_instruction; "" disables it,
which Phase 6 can compare).
"""

from collections.abc import Iterable
from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding

from rag.config import settings


@lru_cache(maxsize=1)
def model() -> TextEmbedding:
    return TextEmbedding(settings.embed_model)


def _normalise(v: np.ndarray) -> np.ndarray:
    # Unit vectors: cosine distance == 1 - dot product, and the index stores them as-is.
    return (v / np.linalg.norm(v, axis=-1, keepdims=True)).astype(np.float32)


def embed_passages(texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
    return _normalise(np.array(list(model().passage_embed(list(texts), batch_size=batch_size))))


def embed_query(query: str) -> np.ndarray:
    return _embed_query(settings.query_instruction + query).copy()


@lru_cache(maxsize=2048)
def _embed_query(text: str) -> np.ndarray:
    # Cached: evaluation runs the same questions through several configurations.
    return _normalise(np.array(next(iter(model().query_embed(text)))))


def dim() -> int:
    return int(embed_query("dimension probe").shape[-1])
