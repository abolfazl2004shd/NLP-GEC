from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from .models import CorrectionRequest, CorrectionResponse
from .vector_store import VectorStore
from .cache import SemanticCache
from .prompt_builder import build_prompt
from .llm_client import call_llm
from .config import settings

app = FastAPI(title="GEC RAG+CoT Service")

support_store = VectorStore(path="./data/support_index.npz")
support_store.load("./data/support_index.npz")

cache = SemanticCache(path="./data/cache_index.npz")
cache.load("./data/cache_index.npz")


@app.post("/correct", response_model=CorrectionResponse)
def correct(req: CorrectionRequest):
    hit = cache.query(req.input)
    if hit:
        return hit

    retrieved = [m for m, s in support_store.query(req.input, top_k=settings.TOP_K)]

    prompt = build_prompt(req.input, retrieved, top_k=settings.TOP_K)

    out = call_llm(prompt)
    try:
        response = CorrectionResponse(
            input=req.input,
            reasoning=out.get("reasoning", ""),
            correction=out.get("correction", ""),
            error_type=out.get("error_type"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    cache.upsert(req.input, response)

    return response
