"""Checks embedding model consistency across config and stored indexes.

It will:
 - inspect `gec_service.config.settings.EMBEDDING_MODEL`
 - load support and cache index files if present and report their saved `meta.embedding_model`
 - check embedding dimensionality compatibility if embeddings present
 - warn on mismatches or missing metadata
"""
from pathlib import Path
import json
import sys
import os
import numpy as np
from pathlib import Path as _P

# Prefer environment override, fallback to the common default from config.py
CONFIG_EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def inspect_index(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"exists": False}
    try:
        data = np.load(str(p), allow_pickle=True)
    except Exception as e:
        return {"exists": True, "error": str(e)}
    info = {"exists": True}
    try:
        items_json = data["items"]
        items = json.loads(str(items_json.tolist()))
        info["n_items"] = len(items)
    except Exception:
        info["n_items"] = None
    try:
        meta_raw = data["meta"]
        meta = json.loads(str(meta_raw.tolist()))
        info["meta"] = meta
    except Exception:
        info["meta"] = {}
    try:
        emb = data["embeddings"]
        info["emb_dim"] = int(emb.shape[1])
    except Exception:
        info["emb_dim"] = None
    return info


def main():
    print("Embedding model (env/default):", CONFIG_EMBEDDING_MODEL)
    support_path = "./data/support_index.npz"
    cache_path = "./data/cache_index.npz"

    print("Inspecting support index:")
    s_info = inspect_index(support_path)
    print(json.dumps(s_info, indent=2))

    print("Inspecting cache index:")
    c_info = inspect_index(cache_path)
    print(json.dumps(c_info, indent=2))

    # compare model names
    s_model = s_info.get("meta", {}).get("embedding_model") if s_info.get("meta") else None
    c_model = c_info.get("meta", {}).get("embedding_model") if c_info.get("meta") else None

    if s_model and s_model != CONFIG_EMBEDDING_MODEL:
        print(f"WARNING: support index built with model '{s_model}' but config/env specifies '{CONFIG_EMBEDDING_MODEL}'")
    if c_model and c_model != CONFIG_EMBEDDING_MODEL:
        print(f"WARNING: cache index built with model '{c_model}' but config/env specifies '{CONFIG_EMBEDDING_MODEL}'")

    # check embedding dim compatibility (if embeddings present)
    dims = [info.get("emb_dim") for info in (s_info, c_info) if info.get("emb_dim")]
    dims = [d for d in dims if d is not None]
    if dims:
        if len(set(dims)) > 1:
            print("WARNING: embedding dims across indexes differ:", dims)
        else:
            print("Embedding dim consistent across indexes:", dims[0])
    else:
        print("No embeddings found in indexes to check dimension.")
    # Embedding normalization audit
    for name, info in (("support", s_info), ("cache", c_info)):
        emb_dim = info.get("emb_dim")
        if emb_dim:
            try:
                data = np.load(name.replace("support", support_path).replace("cache", cache_path), allow_pickle=True)
                emb = data["embeddings"]
                norms = np.linalg.norm(emb, axis=1)
                print(f"{name} embeddings norms: mean={norms.mean():.6f}, std={norms.std():.6f}, min={norms.min():.6f}, max={norms.max():.6f}")
                tol = 1e-3
                if np.all(np.abs(norms - 1.0) < tol):
                    print(f"{name} embeddings appear L2-normalized (tol={tol}).")
                else:
                    print(f"WARNING: {name} embeddings not normalized (count outside tol): {(np.abs(norms - 1.0) >= tol).sum()} / {len(norms)}")
            except Exception as e:
                print(f"Could not load embeddings for {name}: {e}")


if __name__ == "__main__":
    main()
