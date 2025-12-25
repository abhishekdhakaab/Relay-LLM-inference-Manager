from __future__ import annotations
from fastembed import TextEmbedding
from app.core.settings import settings
_embedder:TextEmbedding | None=None


def get_embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _embedder

def embed_text(text:str)->list[float]:
    vecs = list(get_embedder().embed([text]))
    return vecs[0].tolist()
