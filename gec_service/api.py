from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from .models import CorrectionRequest, CorrectionResponse
from .vector_store import VectorStore
from .cache import SemanticCache
from .prompt_builder import build_prompt
from .llm_client import call_llm, call_llm_async
from .config import settings
from .logger import logger
import asyncio


app = FastAPI(title="GEC RAG+CoT Service")

support_store = VectorStore(path="./data/support_index.npz")
support_store.load("./data/support_index.npz")

cache = SemanticCache(path="./data/cache_index.npz")
cache.load("./data/cache_index.npz")


@app.post("/correct", response_model=CorrectionResponse)
async def correct(req: CorrectionRequest):
    # check semantic cache first
    hit = cache.query(req.input)
    if hit:
        logger.info("cache hit for input")
        return hit

    # determine retrieval usage and k
    use_retrieval = getattr(req, "use_retrieval", True) and settings.RETRIEVAL_ENABLED
    top_k = req.top_k or settings.TOP_K

    # retrieve few-shot examples (or empty list if disabled)
    if use_retrieval and top_k > 0:
        retrieved = [m for m, s in support_store.query(req.input, top_k=top_k)]
    else:
        retrieved = []

    prompt = build_prompt(req.input, retrieved, top_k=top_k)

    # call LLM asynchronously; fall back to sync if async not supported
    out = None
    try:
        out = await call_llm_async(prompt)
    except Exception as e:
        logger.exception("Async LLM failed, falling back to sync: %s", e)
        out = call_llm(prompt)

    if not out or not out.get("correction"):
        # LLM failed to produce valid output; if we have a close cache item, return it
        fallback = cache.query(req.input)
        if fallback:
            logger.warning("LLM failed; returning cached fallback")
            return fallback
        raise HTTPException(status_code=502, detail="LLM failed to produce valid output")

    try:
        response = CorrectionResponse(
            input=req.input,
            reasoning=out.get("reasoning", ""),
            correction=out.get("correction", ""),
            error_type=out.get("error_type"),
        )
    except Exception as e:
        logger.exception("Failed to build CorrectionResponse: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    # update cache asynchronously (do not block response)
    try:
        cache.upsert(req.input, response)
    except Exception:
        logger.exception("Failed to upsert into cache")

    return response


@app.get("/metrics")
def metrics():
    return {"cache": cache.metrics(), "support_count": len(support_store.items) if support_store.items else 0}
