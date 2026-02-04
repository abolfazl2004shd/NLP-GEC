from typing import Optional
from .vector_store import VectorStore
from .config import settings
from .models import CorrectionResponse


class SemanticCache:
    def __init__(self, path: str | None = None, threshold: float | None = None):
        self.path = path
        self.store = VectorStore(path)
        self.threshold = threshold or settings.CACHE_THRESHOLD
        self.hits = 0
        self.misses = 0

    def load(self, path: str):
        self.store.load(path)

    def query(self, text: str) -> Optional[CorrectionResponse]:
        results = self.store.query(text, top_k=1)
        if not results:
            self.misses += 1
            return None
        item, sim = results[0]
        if sim >= self.threshold:
            self.hits += 1
            return CorrectionResponse(**item["value"])
        self.misses += 1
        return None

    def upsert(self, text: str, response: CorrectionResponse):
        meta = {"value": response.dict()}
        self.store.add([text], [meta])
        if self.path:
            self.store.save(self.path)

    def metrics(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {"hits": self.hits, "misses": self.misses, "hit_rate": hit_rate}
