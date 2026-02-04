"""Wrapper to run an external M2 scorer against system outputs.

This script expects the M2 scorer to be available on the PATH or provide a path to it.
Input files: gold.m2 and system.out (tab separated or json -> convert as needed).
"""
import subprocess
import tempfile
import os
from typing import Optional


def run_m2_scorer(m2_scorer_path: str, gold_path: str, sys_path: str) -> str:
    if not os.path.exists(m2_scorer_path):
        raise FileNotFoundError("M2 scorer not found at provided path")
    cmd = [m2_scorer_path, gold_path, sys_path]
    out = subprocess.run(cmd, capture_output=True, text=True)
    return out.stdout + out.stderr


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--m2", required=True, help="path to m2 scorer binary/script")
    p.add_argument("--gold", required=True)
    p.add_argument("--sys", required=True)
    args = p.parse_args()
    print(run_m2_scorer(args.m2, args.gold, args.sys))
