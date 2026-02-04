import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from .embeddings import embed_text, embed_texts

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False


class VectorStore:
    """Vector store that uses FAISS if available, otherwise falls back to numpy brute-force.

    Stores items (meta) aligned with embeddings. Provides save/load to disk.
    """

    def __init__(self, path: str | None = None):
        self.path = path
        self.embeddings: np.ndarray | None = None
        self.items: List[Dict[str, Any]] = []
        self.meta: Dict[str, Any] = {}
        self._index = None

    def _build_index(self):
        if self.embeddings is None:
            return
        if _HAS_FAISS:
            dim = self.embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(self.embeddings)
            self._index.add(self.embeddings)
        else:
            self._index = None

    def add(self, texts: List[str], metas: List[Dict[str, Any]]):
        embs = embed_texts(texts)
        if self.embeddings is None:
            self.embeddings = embs
        else:
            self.embeddings = np.vstack([self.embeddings, embs])
        self.items.extend(metas)
        self._build_index()
        if self.path:
            self.save(self.path)

    def query(self, text: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        if (self.embeddings is None or len(self.items) == 0):
            return []
        q = embed_text(text)
        if _HAS_FAISS and self._index is not None:
            vec = q.reshape(1, -1).astype('float32')
            faiss.normalize_L2(vec)
            D, I = self._index.search(vec, top_k)
            results = []
            for idx, score in zip(I[0], D[0]):
                if idx < 0:
                    continue
                results.append((self.items[int(idx)], float(score)))
            return results

        embs = self.embeddings
        sims = embs @ q
        idx = np.argsort(-sims)[:top_k]
        return [(self.items[int(i)], float(sims[int(i)])) for i in idx]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # include optional metadata
        meta_str = json.dumps(self.meta) if getattr(self, "meta", None) is not None else json.dumps({})
        np.savez_compressed(path, embeddings=self.embeddings, items=json.dumps(self.items), meta=meta_str)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        data = np.load(path, allow_pickle=True)
        self.embeddings = data["embeddings"]
        self.items = json.loads(str(data["items"].tolist()))
        # load optional metadata if present
        try:
            meta_raw = data["meta"]
            self.meta = json.loads(str(meta_raw.tolist()))
        except Exception:
            self.meta = {}
        self._build_index()
