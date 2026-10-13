"""Strict material/texture nuisance augmentation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from AsgardBench.causal.generator import (
    InterventionValidationError,
    TripletGenerator,
    _diff,
    _image_rms,
    _load_plan,
)
from AsgardBench.causal.viewpoint import BASE_VARIANTS, _candidate_for


def augment_materials(
    dataset_path: Path,
    retry_rejected: bool = False,
    shard_index: int = 0,
    shard_count: int = 1,
    triplet_ids: set[str] | None = None,
) -> dict[str, Any]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Require 0 <= shard_index < shard_count")
    root = dataset_path.parent
    manifest = json.loads(dataset_path.read_text(encoding="utf-8"))
    generator = TripletGenerator(root)
    rejected = manifest.setdefault("material_rejected", [])
    if retry_rejected:
        rejected.clear()
    rejected_keys = {(item["triplet_id"], item["variant"]) for item in rejected}
    dirty_by_source = {
        str(item["source_plan"]): item
        for item in manifest["triplets"]
        if item["predicate"] == "isDirty"
        and item["target"]["predicate_object_type"] != "Mirror"
    }

    def persist() -> None:
        temporary = dataset_path.with_name(f"{dataset_path.name}.tmp")
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(dataset_path)
        accepted = sum(
            name.endswith("_material")
            for item in manifest["triplets"]
            for name in item["variants"]
        )
        values = [
            variant["validation"]["material"]["image_rms"]
            for item in manifest["triplets"]
            for name, variant in item["variants"].items()
            if name.endswith("_material")
            and "material" in variant.get("validation", {})
        ]
        report = {
            "nuisance_family": "material",
            "triplets": len(manifest["triplets"]),
            "accepted_variants": accepted,
            "complete_triplets": sum(
                all(f"{base}_material" in item["variants"] for base in BASE_VARIANTS)
                for item in manifest["triplets"]
            ),
            "rejected_candidates": len(rejected),
            "predicate_distribution_complete": dict(
                sorted(
                    Counter(
                        item["predicate"]
                        for item in manifest["triplets"]
                        if all(
                            f"{base}_material" in item["variants"]
                            for base in BASE_VARIANTS
                        )
                    ).items()
                )
            ),
            "image_rms_range": [min(values), max(values)] if values else None,
            "contract": {
                "task_symbolic_state_differences": 0,
                "acceptable_action_set_unchanged": True,
                "target_bbox_area_ratio_minimum": 0.01,
                "minimum_image_rms": 1.0,
                "mechanism": "AI2-THOR RandomizeMaterials",
            },
            "rejected_candidates_detail": rejected,
        }
        (root / "material_validation_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

    for index, triplet in enumerate(manifest["triplets"]):
        if triplet_ids is not None and triplet["triplet_id"] not in triplet_ids:
            continue
        if index % shard_count != shard_index:
            continue
        candidate = _candidate_for(triplet, dirty_by_source)
        for base in BASE_VARIANTS:
            name = f"{base}_material"
            if (
                name in triplet["variants"]
                or (triplet["triplet_id"], name) in rejected_keys
            ):
                continue
            try:
                variant, history, target = generator._build_variant(
                    _load_plan(candidate.plan_path),
                    candidate,
                    triplet["triplet_id"],
                    name,
                    root / "triplets" / triplet["triplet_id"],
                )
                value = variant.to_dict()
                if history != triplet["history"] or target != triplet["target"]:
                    raise InterventionValidationError(
                        "Material reconstruction changed history or target"
                    )
                differences = _diff(
                    triplet["variants"][base]["symbolic_state"],
                    value["symbolic_state"],
                )
                if differences:
                    raise InterventionValidationError(
                        f"Material nuisance changed symbolic state: {differences}"
                    )
                if (
                    value["acceptable_actions"]
                    != triplet["variants"][base]["acceptable_actions"]
                ):
                    raise InterventionValidationError(
                        "Material nuisance changed acceptable action set"
                    )
                rms = _image_rms(
                    root / triplet["variants"][base]["image"], root / value["image"]
                )
                if rms < 1.0:
                    raise InterventionValidationError(
                        f"Material image change too small (RMS={rms:.3f})"
                    )
                value["validation"]["material"] = {
                    "symbolic_state_differences": differences,
                    "action_set_unchanged": True,
                    "image_rms": rms,
                }
                triplet["variants"][name] = value
                print(f"ACCEPT {triplet['triplet_id']} {name}", flush=True)
            except Exception as error:
                rejected.append(
                    {
                        "triplet_id": triplet["triplet_id"],
                        "variant": name,
                        "predicate": triplet["predicate"],
                        "object": triplet["target"]["predicate_object_type"],
                        "scene": triplet["scene"],
                        "generation_stage": "material_nuisance",
                        "rejection_reason": type(error).__name__,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                rejected_keys.add((triplet["triplet_id"], name))
                print(f"REJECT {triplet['triplet_id']} {name}: {error}", flush=True)
            persist()
    persist()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Add certified material nuisances")
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/causal_states.json")
    )
    parser.add_argument("--retry-rejected", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--triplet-id",
        action="append",
        help="Restrict augmentation to selected certified triplets (repeatable)",
    )
    args = parser.parse_args()
    manifest = augment_materials(
        args.dataset,
        args.retry_rejected,
        args.shard_index,
        args.shard_count,
        set(args.triplet_id) if args.triplet_id else None,
    )
    complete = sum(
        all(f"{base}_material" in item["variants"] for base in BASE_VARIANTS)
        for item in manifest["triplets"]
    )
    print(f"Complete material triplets: {complete}")


if __name__ == "__main__":
    main()
