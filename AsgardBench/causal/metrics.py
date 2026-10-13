from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Iterable

import numpy as np

CORE_VARIANTS = ("anchor", "causal", "nuisance")
VIEWPOINT_PAIRS = tuple((name, f"{name}_viewpoint") for name in CORE_VARIANTS)


def _causal_unit(row: dict[str, Any]) -> str:
    triplet_id = str(row["triplet_id"])
    return str(row.get("causal_state_unit") or triplet_id.split("__render_", 1)[0])


def _mean(values: Iterable[bool | float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def _triplet_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[row["triplet_id"]][row["variant"]] = row

    triplets = []
    for triplet_id, variants in grouped.items():
        if not set(CORE_VARIANTS).issubset(variants):
            continue
        anchor = variants["anchor"]
        causal = variants["causal"]
        nuisance = variants["nuisance"]
        anchor_prediction = anchor.get("prediction")
        causal_prediction = causal.get("prediction")
        nuisance_prediction = nuisance.get("prediction")
        triplets.append(
            {
                "triplet_id": triplet_id,
                "causal_state_unit": _causal_unit(anchor),
                "ordinary": _mean(
                    variants[name].get("correct", False) for name in CORE_VARIANTS
                ),
                "causal_response": bool(
                    anchor.get("correct")
                    and causal.get("correct")
                    and anchor_prediction != causal_prediction
                ),
                "nuisance_invariance": bool(
                    anchor_prediction is not None
                    and anchor_prediction == nuisance_prediction
                ),
                "nuisance_pair_correct": bool(
                    anchor.get("correct")
                    and nuisance.get("correct")
                    and anchor_prediction == nuisance_prediction
                ),
                "grounded_triplet": bool(
                    anchor.get("correct")
                    and causal.get("correct")
                    and nuisance.get("correct")
                    and anchor_prediction == nuisance_prediction
                    and anchor_prediction != causal_prediction
                ),
                "prior_persistence": bool(
                    anchor_prediction is not None
                    and anchor_prediction == causal_prediction
                ),
            }
        )
        current = triplets[-1]
        if all(
            left in variants and right in variants for left, right in VIEWPOINT_PAIRS
        ):
            invariant = []
            pair_correct = []
            for left, right in VIEWPOINT_PAIRS:
                left_row, right_row = variants[left], variants[right]
                same = bool(
                    left_row.get("prediction") is not None
                    and left_row.get("prediction") == right_row.get("prediction")
                )
                invariant.append(same)
                pair_correct.append(
                    bool(left_row.get("correct") and right_row.get("correct") and same)
                )
            viewpoint_anchor = variants["anchor_viewpoint"]
            viewpoint_causal = variants["causal_viewpoint"]
            current.update(
                {
                    "viewpoint_invariance": _mean(invariant),
                    "viewpoint_strict_invariance": all(invariant),
                    "viewpoint_pair_accuracy": _mean(pair_correct),
                    "viewpoint_causal_response": bool(
                        viewpoint_anchor.get("correct")
                        and viewpoint_causal.get("correct")
                        and viewpoint_anchor.get("prediction")
                        != viewpoint_causal.get("prediction")
                    ),
                    # Equal weight for the lighting and viewpoint nuisance families.
                    "nuisance_invariance_average": _mean(
                        [current["nuisance_invariance"], _mean(invariant)]
                    ),
                    "grounded_v3": bool(
                        current["grounded_triplet"]
                        and all(pair_correct)
                        and viewpoint_anchor.get("prediction")
                        != viewpoint_causal.get("prediction")
                    ),
                }
            )
    return triplets


def _point_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    triplets = _triplet_rows(rows)
    by_variant = {
        variant: [row for row in rows if row.get("variant") == variant]
        for variant in (*CORE_VARIANTS, *(right for _, right in VIEWPOINT_PAIRS))
    }
    core_rows = [row for row in rows if row.get("variant") in CORE_VARIANTS]
    viewpoint_rows = [
        row for row in rows if str(row.get("variant", "")).endswith("_viewpoint")
    ]
    viewpoint_triplets = [t for t in triplets if "viewpoint_invariance" in t]
    result = {
        # Preserve the v1/v2 definition: ordinary accuracy is over the core triplet.
        "ordinary_next_action_accuracy": _mean(
            r.get("correct", False) for r in core_rows
        ),
        "all_variant_next_action_accuracy": _mean(
            r.get("correct", False) for r in rows
        ),
        "anchor_next_action_accuracy": _mean(
            r.get("correct", False) for r in by_variant["anchor"]
        ),
        "causal_next_action_accuracy": _mean(
            r.get("correct", False) for r in by_variant["causal"]
        ),
        "nuisance_next_action_accuracy": _mean(
            r.get("correct", False) for r in by_variant["nuisance"]
        ),
        "causal_response_accuracy": _mean(t["causal_response"] for t in triplets),
        "nuisance_invariance": _mean(t["nuisance_invariance"] for t in triplets),
        "nuisance_pair_accuracy": _mean(t["nuisance_pair_correct"] for t in triplets),
        "grounded_triplet_accuracy": _mean(t["grounded_triplet"] for t in triplets),
        "prior_persistence_rate": _mean(t["prior_persistence"] for t in triplets),
        "viewpoint_next_action_accuracy": _mean(
            r.get("correct", False) for r in viewpoint_rows
        ),
        "viewpoint_invariance": _mean(
            t["viewpoint_invariance"] for t in viewpoint_triplets
        ),
        "viewpoint_strict_invariance": _mean(
            t["viewpoint_strict_invariance"] for t in viewpoint_triplets
        ),
        "viewpoint_pair_accuracy": _mean(
            t["viewpoint_pair_accuracy"] for t in viewpoint_triplets
        ),
        "viewpoint_causal_response_accuracy": _mean(
            t["viewpoint_causal_response"] for t in viewpoint_triplets
        ),
        "nuisance_invariance_average": _mean(
            t["nuisance_invariance_average"] for t in viewpoint_triplets
        ),
        "grounded_v3_accuracy": _mean(t["grounded_v3"] for t in viewpoint_triplets),
        "n_variants": len(rows),
        "n_triplets": len(triplets),
        "n_causal_state_units": len({item["causal_state_unit"] for item in triplets}),
        "n_viewpoint_triplets": len(viewpoint_triplets),
    }
    # Additive submission-level nuisance diagnostics.  The frozen lighting,
    # viewpoint, grounded-triplet, and grounded-v3 definitions above remain
    # byte-for-byte compatible with the pilot; new nuisance families do not
    # change their denominators.
    available_variants = {str(row.get("variant")) for row in rows}
    for family in ("viewpoint", "material"):
        pairs = [
            (base, f"{base}_{family}")
            for base in CORE_VARIANTS
            if base in available_variants and f"{base}_{family}" in available_variants
        ]
        invariance: list[bool] = []
        pair_accuracy: list[bool] = []
        by_triplet: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in rows:
            by_triplet[str(row["triplet_id"])][str(row["variant"])] = row
        for variants in by_triplet.values():
            for left, right in pairs:
                if left not in variants or right not in variants:
                    continue
                same = bool(
                    variants[left].get("prediction") is not None
                    and variants[left].get("prediction")
                    == variants[right].get("prediction")
                )
                invariance.append(same)
                pair_accuracy.append(
                    bool(
                        same
                        and variants[left].get("correct")
                        and variants[right].get("correct")
                    )
                )
        result[f"{family}_all_pair_invariance"] = _mean(invariance)
        result[f"{family}_all_pair_accuracy"] = _mean(pair_accuracy)
        result[f"n_{family}_pairs"] = len(invariance)
    return result


def _bootstrap(
    rows: list[dict[str, Any]],
    iterations: int,
    seed: int,
) -> dict[str, list[float]]:
    by_triplet: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_triplet[row["triplet_id"]].append(row)
    by_cluster: dict[str, list[str]] = defaultdict(list)
    for triplet_id, triplet_rows in by_triplet.items():
        cluster = _causal_unit(triplet_rows[0])
        by_cluster[cluster].append(triplet_id)
    cluster_ids = sorted(by_cluster)
    if not cluster_ids or iterations <= 0:
        return {}

    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(iterations):
        sampled_ids = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        sampled_rows = []
        for sample_number, cluster_id in enumerate(sampled_ids):
            for triplet_id in by_cluster[str(cluster_id)]:
                for row in by_triplet[triplet_id]:
                    copied = dict(row)
                    copied["triplet_id"] = (
                        f"{triplet_id}__cluster_bootstrap_{sample_number}"
                    )
                    copied["causal_state_unit"] = (
                        f"{cluster_id}__cluster_bootstrap_{sample_number}"
                    )
                    sampled_rows.append(copied)
        for name, value in _point_metrics(sampled_rows).items():
            if name.startswith("n_") or np.isnan(value):
                continue
            samples[name].append(value)
    return {
        name: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
        for name, values in samples.items()
    }


def compute_metrics(
    rows: list[dict[str, Any]],
    bootstrap_iterations: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute estimates and causal-state-clustered bootstrap intervals."""
    point = _point_metrics(rows)
    return {
        "metrics": point,
        "bootstrap_95_ci": _bootstrap(rows, bootstrap_iterations, seed),
    }


def grouped_metrics(
    rows: list[dict[str, Any]],
    group_fields: tuple[str, ...] = (
        "model",
        "method",
        "predicate",
        "task_type",
        "decision_object",
        "scene",
        "replica_profile",
        "certification_tier",
    ),
    bootstrap_iterations: int = 1000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Return overall/model/method/predicate/task-type slices."""
    group_fields = tuple(
        field for field in group_fields if rows and all(field in row for row in rows)
    )
    outputs = []
    specifications: list[
        tuple[str, Callable[[dict[str, Any]], str], tuple[str, ...]]
    ] = [("overall", lambda _: "all", ())]
    specifications.extend(
        (field, lambda row, field=field: str(row[field]), (field,))
        for field in group_fields
    )
    if "model" in group_fields and "method" in group_fields:
        specifications.append(
            (
                "model_method",
                lambda row: f"{row['model']}::{row['method']}",
                ("model", "method"),
            )
        )
        for slice_field in group_fields:
            if slice_field in {"model", "method"}:
                continue
            if slice_field in group_fields:
                specifications.append(
                    (
                        f"model_method_{slice_field}",
                        lambda row, slice_field=slice_field: "::".join(
                            (
                                str(row["model"]),
                                str(row["method"]),
                                str(row[slice_field]),
                            )
                        ),
                        ("model", "method", slice_field),
                    )
                )

    for group_name, key_fn, fixed_fields in specifications:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            copied = dict(row)
            # A triplet is an experimental unit within a model+method condition.
            # When aggregating over either dimension, keep those repeated
            # conditions distinct instead of overwriting equal triplet IDs.
            varying_condition = [
                str(row[field])
                for field in ("model", "method")
                if field not in fixed_fields and field in row
            ]
            copied["triplet_id"] = "::".join(
                [*varying_condition, str(row["triplet_id"])]
            )
            buckets[key_fn(row)].append(copied)
        for group_value, bucket in sorted(buckets.items()):
            result = compute_metrics(bucket, bootstrap_iterations, seed)
            outputs.append(
                {
                    "group": group_name,
                    "value": group_value,
                    **result,
                }
            )
    return outputs
