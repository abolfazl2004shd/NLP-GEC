"""Build a quick index using a deterministic hash-based pseudo-embedder.

This helps run retrieval sanity checks without real embedding dependencies.
"""
from pathlib import Path
import json
import numpy as np
import argparse


def pseudo_embed(text: str, dim: int = 384) -> np.ndarray:
    v = float(abs(hash(text)) % 100) / 100.0
    vec = np.full((dim,), v, dtype=np.float32)
    # normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def read_jsonl(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))
    return items


def build(input_jsonl: str, out_path: str, dim: int = 384, model_name: str = "pseudo-hash-embedder"):
    p = Path(input_jsonl)
    items = read_jsonl(p)
    texts = [it["input"] for it in items]
    embs = np.stack([pseudo_embed(t, dim=dim) for t in texts], axis=0)
    metas = [{"value": it} for it in items]
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    # save embeddings, items, meta
    meta = {"embedding_model": model_name}
    np.savez_compressed(str(outp), embeddings=embs, items=json.dumps(metas, ensure_ascii=False), meta=json.dumps(meta))
    print(f"Wrote quick index with {len(items)} items -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", required=True)
    p.add_argument("--out", dest="out", required=True)
    p.add_argument("--dim", type=int, default=384)
    args = p.parse_args()
    build(args.infile, args.out, dim=args.dim)
