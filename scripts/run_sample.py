"""Run a mocked end-to-end sample against the FastAPI app.

This patches the async LLM call with a deterministic mock so we can exercise
retrieval, caching, and the JSON output flow without network or keys.
"""
import sys
from pathlib import Path
# ensure repo root is on sys.path so `gec_service` imports work when running from scripts/
repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Provide a minimal dummy `fastapi` module if it's not installed, so importing
# `gec_service.api` in this offline test doesn't fail.
try:
    import fastapi  # type: ignore
except ModuleNotFoundError:
    import types
    fm = types.ModuleType("fastapi")
    class FastAPI:
        def __init__(self, *a, **kw):
            pass
        def post(self, *a, **kw):
            def deco(f):
                return f
            return deco
        def get(self, *a, **kw):
            def deco(f):
                return f
            return deco
    class HTTPException(Exception):
        pass
    fm.FastAPI = FastAPI
    fm.HTTPException = HTTPException
    sys.modules["fastapi"] = fm
try:
    import pydantic  # type: ignore
except ModuleNotFoundError:
    import types, os
    pm = types.ModuleType("pydantic")
    class ValidationError(Exception):
        pass
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def dict(self):
            return self.__dict__
    class BaseSettings(BaseModel):
        def __init__(self, **kwargs):
            cls = self.__class__
            # copy uppercase class attrs as defaults and override from env
            for name, val in cls.__dict__.items():
                if name.isupper():
                    envv = os.environ.get(name)
                    setattr(self, name, type(val)(envv) if (envv is not None and not isinstance(val, str) and isinstance(val, (int, float))) else (envv if envv is not None else val))
            for k, v in kwargs.items():
                setattr(self, k, v)
    pm.ValidationError = ValidationError
    pm.BaseModel = BaseModel
    pm.BaseSettings = BaseSettings
    sys.modules['pydantic'] = pm
# provide a minimal sentence_transformers stub if missing
try:
    import sentence_transformers  # type: ignore
except ModuleNotFoundError:
    import types
    import numpy as _np
    st = types.ModuleType("sentence_transformers")
    class SentenceTransformer:
        def __init__(self, model_name=None):
            self.dim = 384
        def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
            if isinstance(texts, str):
                texts = [texts]
            embs = []
            for t in texts:
                v = _np.full((self.dim,), float((abs(hash(t)) % 100) / 100.0), dtype=_np.float32)
                embs.append(v)
            return _np.stack(embs, axis=0)
    st.SentenceTransformer = SentenceTransformer
    sys.modules['sentence_transformers'] = st
 # provide a minimal openai stub to avoid import errors during module import
try:
    import openai  # type: ignore
except ModuleNotFoundError:
    import types
    om = types.ModuleType("openai")
    class ChatCompletion:
        @staticmethod
        def create(*args, **kwargs):
            return {"choices": [{"message": {"content": "{}"}}]}
        @staticmethod
        async def acreate(*args, **kwargs):
            return {"choices": [{"message": {"content": "{}"}}]}
    om.ChatCompletion = ChatCompletion
    sys.modules['openai'] = om

import gec_service.api as api_module
from gec_service.models import CorrectionRequest
import asyncio
import json


async def _mock_llm(prompt: str, max_tokens: int = 256):
    # This mock returns a valid CorrectionResponse-like dict
    return {
        "input": "She go to school yesterday.",
        "reasoning": "The verb 'go' should be in past tense to match 'yesterday'.",
        "correction": "She went to school yesterday.",
        "error_type": "VT",
    }


def main():
    # patch the api module's async LLM to our mock
    api_module.call_llm_async = _mock_llm

    payload = {"input": "She go to school yesterday."}
    print("CALL api.correct with ->", payload)

    req = CorrectionRequest(input=payload["input"])
    res = asyncio.run(api_module.correct(req))
    # res is a pydantic model; convert to dict
    print(json.dumps(res.dict(), indent=2, ensure_ascii=False))

    m = api_module.metrics()
    print("/metrics ->", json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
