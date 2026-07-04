"""Lazy embedding model access shared by routing and semantic caching."""

from __future__ import annotations
from fastembed import TextEmbedding
from app.core.settings import settings
_embedder:TextEmbedding | None=None

def get_embedder() -> TextEmbedding:
    """Load the local embedding model once per process."""
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _embedder
def embed_text(text:str)->list[float]:
    """Return one 384-dimensional embedding for downstream reuse."""
    vecs = list(get_embedder().embed([text]))
    return vecs[0].tolist()
