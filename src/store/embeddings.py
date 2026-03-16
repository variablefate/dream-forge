"""
Local embedding generation using all-MiniLM-L6-v2.

~91MB model, CPU-only, 384-dim vectors, truncates at 256 word pieces.
"""

from functools import lru_cache
from typing import Optional

import numpy as np


@lru_cache(maxsize=1)
def _get_model():
    """Lazy-load the embedding model. Cached after first call."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


def embed_text(text: str) -> list[float]:
    """Embed a single text string. Returns 384-dim float vector."""
    model = _get_model()
    vec = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
    return vec.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts. Returns list of 384-dim vectors."""
    model = _get_model()
    vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return vecs.tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_np = np.array(a)
    b_np = np.array(b)
    dot = np.dot(a_np, b_np)
    norm = np.linalg.norm(a_np) * np.linalg.norm(b_np)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def embed_optional(text: Optional[str]) -> Optional[list[float]]:
    """Embed text if not None, return None otherwise."""
    if text is None or text.strip() == "":
        return None
    return embed_text(text)
