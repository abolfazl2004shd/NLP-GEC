from sentence_transformers import SentenceTransformer
import numpy as np
from .config import settings


_model: SentenceTransformer | None = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


def embed_text(text: str) -> np.ndarray:
    model = get_model()
    emb = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    return emb[0]


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_model()
    embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embs
