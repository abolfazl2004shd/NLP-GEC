"""Utility to precompute embeddings for a support set JSONL file.

Expect input JSONL where each line is: {"input":..., "reasoning":..., "correction":..., "error_type":...}
"""
import json
from gec_service.vector_store import VectorStore
from gec_service.config import settings


def build_index(input_path: str, out_path: str):
    texts = []
    metas = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj.get("input", ""))
            metas.append({"value": obj})
    store = VectorStore(out_path)
    # record which embedding model was used to create this index
    try:
        store.meta["embedding_model"] = settings.EMBEDDING_MODEL
    except Exception:
        store.meta = {"embedding_model": getattr(settings, "EMBEDDING_MODEL", None)}
    store.add(texts, metas)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", required=True)
    p.add_argument("--out", dest="outfile", required=True)
    args = p.parse_args()
    build_index(args.infile, args.outfile)
