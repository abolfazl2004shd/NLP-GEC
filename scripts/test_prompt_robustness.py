"""Run prompt robustness scenarios against `build_prompt`.

Checks:
- empty retrieved examples
- very short input
- very long input
- many retrieved examples
- truncation by max_chars
"""
from pathlib import Path
import sys
from pathlib import Path as _P
# ensure repo root on sys.path
repo_root = str(_P(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import json
from gec_service.prompt_builder import build_prompt


def case_empty_retrieved():
    p = build_prompt("Hello.", [], top_k=5)
    assert "Reference Examples" not in p
    print("empty_retrieved OK")


def case_short_input():
    p = build_prompt("Hi.", [], top_k=5)
    assert "Input: Hi." in p
    print("short_input OK")


def case_long_input():
    long_in = " ".join(["word"] * 2000)
    p = build_prompt(long_in, [], top_k=5)
    assert "Task:" in p
    print("long_input OK; length=", len(p))


def case_many_retrieved():
    retrieved = []
    for i in range(20):
        retrieved.append({"value": {"input": f"Ex {i}", "reasoning": "r", "correction": "c", "error_type": "VT"}})
    p = build_prompt("Test sentence.", retrieved, top_k=5)
    # should include only up to top_k examples
    assert p.count("Example Input:") == 5
    print("many_retrieved OK")


def case_truncation_max_chars():
    # emulate truncation externally by limiting the prompt string length
    retrieved = []
    for i in range(10):
        retrieved.append({"value": {"input": f"Long example text {i} " + ("x"*200), "reasoning": "r", "correction": "c", "error_type": "VT"}})
    max_chars = 2000
    p2 = build_prompt("Short query.", retrieved, top_k=10, max_chars=max_chars)
    assert len(p2) <= max_chars
    print("truncation enforced OK; truncated_len=", len(p2))


def main():
    case_empty_retrieved()
    case_short_input()
    case_long_input()
    case_many_retrieved()
    case_truncation_max_chars()

if __name__ == "__main__":
    main()
