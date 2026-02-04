"""Helpers to convert common GEC formats into support JSONL for indexing.

Expect inputs:
- CoNLL/CoNLL2014 M2 files -> parse to jsonl where each line: {"input":..., "reasoning":"", "correction":..., "error_type":...}
- BEA-style TSV/JSON -> similar conversion

This script doesn't fetch datasets. Place original files locally and run conversions.
"""
import json
from typing import List


def parse_m2_to_jsonl(m2_path: str, out_path: str):
    # Very small M2 parser: groups sentences and applies first correction
    with open(m2_path, "r", encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as out:
        sent = None
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("S "):
                sent = line[2:]
            elif line.startswith("A ") and sent:
                # A lines: start end|||error_type|||correction|||... -> take correction
                parts = line.split("|||")
                if len(parts) >= 3:
                    corr = parts[2].strip()
                    item = {"input": sent, "reasoning": "", "correction": corr, "error_type": None}
                    out.write(json.dumps(item, ensure_ascii=False) + "\n")
            elif line.strip() == "":
                sent = None


def tsv_to_jsonl(tsv_path: str, out_path: str, input_col: int = 0, correction_col: int = 1):
    with open(tsv_path, "r", encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as out:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(input_col, correction_col):
                continue
            item = {"input": parts[input_col], "reasoning": "", "correction": parts[correction_col], "error_type": None}
            out.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--m2", help="path to m2 file to convert", required=False)
    p.add_argument("--tsv", help="path to tsv file to convert", required=False)
    p.add_argument("--out", help="output jsonl file", required=True)
    args = p.parse_args()
    if args.m2:
        parse_m2_to_jsonl(args.m2, args.out)
    elif args.tsv:
        tsv_to_jsonl(args.tsv, args.out)
    else:
        raise SystemExit("Provide --m2 or --tsv and --out")
