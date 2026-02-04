"""Simple retrieval sanity checks using a quick index (npz with embeddings/items/meta).

Usage:
python scripts/retrieval_sanity.py --index data/support_index.npz --query "She go to school yesterday." --topk 3
"""
from pathlib import Path
import numpy as np
import json
import argparse


def load_index(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    data = np.load(str(p), allow_pickle=True)
    embs = data["embeddings"]
    items = json.loads(str(data["items"].tolist()))
    meta = {}
    try:
        meta = json.loads(str(data["meta"].tolist()))
    except Exception:
        meta = {}
    return embs, items, meta


def pseudo_embed(text: str, dim: int):
    v = float(abs(hash(text)) % 100) / 100.0
    vec = np.full((dim,), v, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def query_index(index_path: str, query: str, topk: int = 5):
    embs, items, meta = load_index(index_path)
    dim = embs.shape[1]
    q = pseudo_embed(query, dim)
    # assume stored embs are normalized
    sims = embs @ q
    idx = np.argsort(-sims)[:topk]
    results = [(int(i), float(sims[int(i)]), items[int(i)]) for i in idx]
    return results, meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--topk", type=int, default=5)
    args = p.parse_args()

    res, meta = query_index(args.index, args.query, topk=args.topk)
    print("index meta:", meta)
    for i, score, item in res:
        print(f"rank {i} score={score:.6f} input={item.get('value', {}).get('input')}")

if __name__ == "__main__":
    main()
