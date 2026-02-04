"""Evaluation helpers: wrapper for computing Precision, Recall, F0.5 given counts.

This module provides scaffolding for integrating an external M2 scorer.
"""
from typing import Tuple


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision + recall == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def precision_recall_from_counts(tp: int, fp: int, fn: int) -> Tuple[float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return p, r
