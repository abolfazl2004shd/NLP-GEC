"""Dataset validation and splitting utilities for support and eval sets.

Usage examples:

# Validate a JSONL support file
python scripts/validate_datasets.py --validate support.jsonl

# Split JSONL into support/eval ensuring no overlap
python scripts/validate_datasets.py --split all_examples.jsonl --out-support data/support.jsonl --out-eval data/eval.jsonl --eval-frac 0.1

The script enforces a normalized schema with fields: input, correction, error_spans (optional list), error_type (optional), metadata (optional dict).
"""
from __future__ import annotations
import json
from typing import Dict, Any, Iterable, List, Tuple, Optional
from pathlib import Path
import random
import argparse

REQUIRED_FIELDS = ["input", "correction"]
NORMAL_FIELDS = ["input", "correction", "error_spans", "error_type", "reasoning", "metadata"]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i} of {path}: {e}")
            items.append(obj)
    return items


def normalize_item(obj: Dict[str, Any]) -> Dict[str, Any]:
    norm: Dict[str, Any] = {}
    for k in NORMAL_FIELDS:
        if k in obj:
            norm[k] = obj[k]
    # ensure required
    for k in REQUIRED_FIELDS:
        if k not in norm:
            # try common alternatives
            if k == "input" and "original" in obj:
                norm["input"] = obj["original"]
            elif k == "correction" and "corrected" in obj:
                norm["correction"] = obj["corrected"]
            else:
                raise ValueError(f"Missing required field '{k}' in item: {obj}")
    # fill missing optional fields
    if "error_spans" not in norm:
        norm["error_spans"] = []
    if "error_type" not in norm:
        norm["error_type"] = None
    if "reasoning" not in norm:
        norm["reasoning"] = ""
    if "metadata" not in norm:
        norm["metadata"] = {}
    return norm


def validate_items(items: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    normalized: List[Dict[str, Any]] = []
    errors: List[str] = []
    for i, obj in enumerate(items, 1):
        try:
            norm = normalize_item(obj)
            # basic content checks
            if not isinstance(norm["input"], str) or not norm["input"].strip():
                raise ValueError("Empty or non-string 'input'")
            if not isinstance(norm["correction"], str):
                raise ValueError("'correction' must be a string")
            normalized.append(norm)
        except Exception as e:
            errors.append(f"Item {i}: {e}")
    return normalized, errors


def ensure_no_overlap(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> List[str]:
    # report any exact-input overlaps
    set_a = set(x["input"].strip() for x in a)
    set_b = set(x["input"].strip() for x in b)
    overlap = sorted(list(set_a.intersection(set_b)))
    return overlap


def split_items(items: List[Dict[str, Any]], eval_frac: float = 0.1, seed: Optional[int] = 1337) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    random.seed(seed)
    items_copy = items.copy()
    random.shuffle(items_copy)
    n = len(items_copy)
    n_eval = max(1, int(n * eval_frac))
    eval_set = items_copy[:n_eval]
    support_set = items_copy[n_eval:]
    return support_set, eval_set


def write_jsonl(items: Iterable[Dict[str, Any]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--validate", help="validate a jsonl file", required=False)
    p.add_argument("--split", help="split a jsonl file into support/eval", required=False)
    p.add_argument("--normalize", help="normalize a jsonl file and write output", required=False)
    p.add_argument("--out-support", help="output path for support set (jsonl)", required=False)
    p.add_argument("--out-eval", help="output path for eval set (jsonl)", required=False)
    p.add_argument("--eval-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    if args.validate:
        path = Path(args.validate)
        items = read_jsonl(path)
        norm, errors = validate_items(items)
        print(f"Read {len(items)} items; normalized to {len(norm)}; errors: {len(errors)}")
        for e in errors[:10]:
            print("ERR:", e)
        if errors:
            return
        # quick schema sample
        print("Sample normalized item:")
        print(json.dumps(norm[0], ensure_ascii=False, indent=2))
        return

    if args.normalize:
        path = Path(args.normalize)
        items = read_jsonl(path)
        norm, errors = validate_items(items)
        print(f"Read {len(items)} items; normalized to {len(norm)}; errors: {len(errors)}")
        if errors:
            print("Validation errors; aborting normalize")
            for e in errors[:20]:
                print(e)
            return
        out_path = Path(args.out_support) if args.out_support else path.with_name(path.stem + "_normalized.jsonl")
        write_jsonl(norm, out_path)
        print(f"Wrote normalized JSONL -> {out_path}")
        return

    if args.split:
        path = Path(args.split)
        items = read_jsonl(path)
        norm, errors = validate_items(items)
        if errors:
            print(f"Validation errors ({len(errors)}). Aborting split.")
            for e in errors[:20]:
                print(e)
            return
        support, eval_set = split_items(norm, eval_frac=args.eval_frac, seed=args.seed)
        # ensure no overlap
        overlap = ensure_no_overlap(support, eval_set)
        if overlap:
            print(f"Overlap detected ({len(overlap)} items) — this should not happen")
            for o in overlap[:10]:
                print("Overlap input:", o)
            return
        if args.out_support:
            write_jsonl(support, Path(args.out_support))
            print(f"Wrote support set {len(support)} -> {args.out_support}")
        if args.out_eval:
            write_jsonl(eval_set, Path(args.out_eval))
            print(f"Wrote eval set {len(eval_set)} -> {args.out_eval}")
        return

    p.print_help()


if __name__ == "__main__":
    main()
