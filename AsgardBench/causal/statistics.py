"""Unit-clustered paired inference for CausalPilot confirmatory analyses."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from AsgardBench.causal.metrics import _causal_unit, _triplet_rows


def unit_metric(rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
    """Average nested triplet observations within each causal-state unit."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for item in _triplet_rows(rows):
        if metric not in item:
            continue
        buckets[str(item["causal_state_unit"])].append(float(item[metric]))
    return {unit: float(np.mean(values)) for unit, values in buckets.items()}


def exact_sign_pvalue(wins: int, losses: int) -> float:
    """Two-sided exact sign test; ties are excluded."""
    n = wins + losses
    if n == 0:
        return 1.0
    smaller = min(wins, losses)
    tail = sum(math.comb(n, k) for k in range(smaller + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def paired_unit_contrast(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    metric: str,
    *,
    iterations: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute right-minus-left effect using paired causal-state units."""
    left = unit_metric(left_rows, metric)
    right = unit_metric(right_rows, metric)
    units = sorted(set(left) & set(right))
    differences = np.asarray([right[unit] - left[unit] for unit in units], dtype=float)
    wins = int(np.sum(differences > 0))
    losses = int(np.sum(differences < 0))
    ties = int(np.sum(differences == 0))
    if not units:
        return {
            "metric": metric, "n_units": 0, "difference": None,
            "bootstrap_95_ci": [None, None], "wins": 0, "losses": 0,
            "ties": 0, "sign_test_p": None,
        }
    rng = np.random.default_rng(seed)
    draws = rng.choice(differences, size=(iterations, len(differences)), replace=True)
    means = draws.mean(axis=1)
    return {
        "metric": metric,
        "n_units": len(units),
        "difference": float(differences.mean()),
        "bootstrap_95_ci": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "sign_test_p": exact_sign_pvalue(wins, losses),
        "cluster": "causal_state_unit",
    }


def holm_adjust(pvalues: Iterable[float | None]) -> list[float | None]:
    """Holm step-down adjusted p-values, preserving input order."""
    values = list(pvalues)
    valid = [(index, value) for index, value in enumerate(values) if value is not None]
    ordered = sorted(valid, key=lambda item: float(item[1]))
    adjusted: dict[int, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (index, value) in enumerate(ordered):
        running = max(running, (count - rank) * float(value))
        adjusted[index] = min(1.0, running)
    return [adjusted.get(index) for index in range(len(values))]


def filter_condition(
    rows: list[dict[str, Any]], model: str, method: str
) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if row.get("model") == model and row.get("method") == method
    ]
