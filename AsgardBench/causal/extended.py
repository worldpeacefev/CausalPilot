"""Generate the documented submission-only extension tier.

The original pilot contract is deliberately untouched.  This module retries a
small, predeclared subset of its rejected candidates under two scientifically
interpretable relaxations:

* an articulated openable may change its own root pose when it opens; and
* a visible target may occupy 0.5% of the image (0.15% for cooked food).

Non-target predicates, relations, inventory, nuisance state, action-branch
separation, and a measurable target-local visual signal remain mandatory.
Cooked-state candidates additionally require target-local RMS >= 15 after a
visual-only ambiguity audit; model correctness is never used for retention.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageStat

from AsgardBench.causal.generator import (
    Candidate,
    TripletGenerator,
    _candidate_triplet_id,
    _image_rms,
    _json_default,
    discover_candidates,
)

MINIMUM_TARGET_RMS = {"isCooked": 15.0}


def _minimum_target_rms(predicate: str) -> float:
    return MINIMUM_TARGET_RMS.get(predicate, 1.0)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=_json_default), encoding="utf-8"
    )
    temporary.replace(path)


def _crop_rms(root: Path, anchor: dict[str, Any], causal: dict[str, Any]) -> float:
    left = Image.open(root / anchor["image"]).convert("RGB")
    right = Image.open(root / causal["image"]).convert("RGB")
    boxes = (
        anchor["validation"]["predicate_bbox"],
        causal["validation"]["predicate_bbox"],
    )
    x1 = max(0, min(int(box[0]) for box in boxes))
    y1 = max(0, min(int(box[1]) for box in boxes))
    x2 = min(left.width, max(int(box[2]) for box in boxes))
    y2 = min(left.height, max(int(box[3]) for box in boxes))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    difference = ImageChops.difference(
        left.crop((x1, y1, x2, y2)), right.crop((x1, y1, x2, y2))
    )
    stat = ImageStat.Stat(difference)
    return math.sqrt(sum(value * value for value in stat.rms) / len(stat.rms))


def _policy(candidate: Candidate) -> dict[str, Any] | None:
    if candidate.predicate == "isOpen" and candidate.decision_type in {
        "Drawer",
        "Microwave",
    }:
        return {
            "min_bbox_area_ratio": 0.01,
            "allow_target_articulation": True,
            "certification_tier": "extended-target-articulation-v1",
        }
    if candidate.predicate == "isDirty":
        return {
            "min_bbox_area_ratio": 0.005,
            "allow_target_articulation": False,
            "certification_tier": "extended-small-visible-v1",
        }
    if candidate.predicate == "isCooked":
        return {
            "min_bbox_area_ratio": 0.0015,
            "allow_target_articulation": False,
            "certification_tier": "extended-small-visible-v1",
        }
    return None


def _eligible(candidate: Candidate, strict_rejections: list[dict[str, Any]]) -> bool:
    matching = [
        item
        for item in strict_rejections
        if item.get("predicate") == candidate.predicate
        and item.get("source_plan") == str(candidate.plan_path)
    ]
    errors = "\n".join(str(item.get("error", "")) for item in matching)
    if candidate.predicate == "isOpen":
        return candidate.decision_type in {"Drawer", "Microwave"} and (
            "changed more than the target predicate" in errors
        )
    if candidate.predicate in {"isDirty", "isCooked"}:
        return "too small in frame" in errors
    return False


def generate_extended(
    dataset_path: Path,
    source_plans: Path,
    *,
    limit: int | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
    review_existing_only: bool = False,
) -> dict[str, Any]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Require 0 <= shard_index < shard_count")
    root = dataset_path.parent
    manifest = json.loads(dataset_path.read_text(encoding="utf-8"))
    strict_rejections = [
        *manifest.get("rejected", []),
        *manifest.get("strict_rejections_superseded", []),
    ]
    candidates = (
        []
        if review_existing_only
        else [
            candidate
            for values in discover_candidates(source_plans).values()
            for candidate in values
            if _policy(candidate) is not None
            and _eligible(candidate, strict_rejections)
        ]
    )
    candidates.sort(
        key=lambda item: (
            item.predicate,
            str(item.decision_type),
            str(item.plan_path),
            str(item.target_name),
        )
    )
    candidates = [
        item
        for index, item in enumerate(candidates)
        if index % shard_count == shard_index
    ]
    if limit is not None:
        candidates = candidates[:limit]

    existing_ids = {item["triplet_id"] for item in manifest["triplets"]}
    rejected = manifest.setdefault("extended_rejected", [])
    # A missing display is an execution-environment failure, not evidence that
    # a scientific candidate violates the certification contract.
    rejected[:] = [
        item
        for item in rejected
        if "No valid X display found" not in str(item.get("error", ""))
    ]
    rejected_keys = {
        (item.get("predicate"), item.get("source_plan"), item.get("target_name"))
        for item in rejected
    }

    for candidate in candidates:
        triplet_id = _candidate_triplet_id(candidate)
        key = (candidate.predicate, str(candidate.plan_path), candidate.target_name)
        if triplet_id in existing_ids or key in rejected_keys:
            continue
        policy = _policy(candidate)
        assert policy is not None
        generator = TripletGenerator(root, **policy)
        try:
            record = generator.generate(candidate)
            value = record.to_dict()
            anchor, causal = value["variants"]["anchor"], value["variants"]["causal"]
            global_rms = _image_rms(root / anchor["image"], root / causal["image"])
            target_rms = _crop_rms(root, anchor, causal)
            minimum_target_rms = _minimum_target_rms(candidate.predicate)
            if global_rms < 0.1 or target_rms < minimum_target_rms:
                raise ValueError(
                    "Causal visual signal too small: "
                    f"global_rms={global_rms:.3f}, target_rms={target_rms:.3f}, "
                    f"required_target_rms={minimum_target_rms:.3f}"
                )
            value["validation"]["causal_visual_signal"] = {
                "global_image_rms": global_rms,
                "target_union_bbox_rms": target_rms,
                "minimum_global_image_rms": 0.1,
                "minimum_target_union_bbox_rms": minimum_target_rms,
                "passed": True,
            }
            _atomic_json(root / "triplets" / triplet_id / "triplet.json", value)
            manifest["triplets"].append(value)
            existing_ids.add(triplet_id)

            superseded = []
            retained = []
            for item in manifest.get("rejected", []):
                if item.get("predicate") == candidate.predicate and item.get(
                    "source_plan"
                ) == str(candidate.plan_path):
                    moved = copy.deepcopy(item)
                    moved["final_disposition"] = "accepted_under_extended_contract"
                    moved["accepted_triplet_id"] = triplet_id
                    superseded.append(moved)
                else:
                    retained.append(item)
            manifest["rejected"] = retained
            manifest.setdefault("strict_rejections_superseded", []).extend(superseded)
            print(f"ACCEPT EXTENDED {triplet_id}", flush=True)
        except Exception as error:
            rejected.append(
                {
                    "triplet_id": triplet_id,
                    "predicate": candidate.predicate,
                    "source_plan": str(candidate.plan_path),
                    "target_name": candidate.target_name,
                    "object": candidate.decision_type or candidate.target_type,
                    "target_object": candidate.target_type,
                    "generation_stage": "extended_certification",
                    "rejection_reason": type(error).__name__,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            rejected_keys.add(key)
            print(f"REJECT EXTENDED {triplet_id}: {error}", flush=True)
        manifest["triplets"].sort(key=lambda item: item["triplet_id"])
        _atomic_json(dataset_path, manifest)

    # Apply the current visual-only contract to resumable artifacts generated
    # before this gate was introduced. Keep their files for rejection audit.
    retained = []
    for item in manifest["triplets"]:
        minimum_target_rms = _minimum_target_rms(item["predicate"])
        signal = item.get("validation", {}).get("causal_visual_signal", {})
        target_rms = float(signal.get("target_union_bbox_rms", float("inf")))
        if (
            item.get("certification_tier", "pilot-v3-strict").startswith("extended-")
            and signal
        ):
            signal["minimum_target_union_bbox_rms"] = minimum_target_rms
            signal["passed"] = bool(
                float(signal.get("global_image_rms", 0.0))
                >= float(signal.get("minimum_global_image_rms", 0.1))
                and target_rms >= minimum_target_rms
            )
        if not (
            item.get("certification_tier", "pilot-v3-strict").startswith("extended-")
            and target_rms < minimum_target_rms
        ):
            retained.append(item)
            continue
        rejection = {
            "triplet_id": item["triplet_id"],
            "predicate": item["predicate"],
            "source_plan": item["source_plan"],
            "target_name": item["target"].get("predicate_object_name"),
            "object": item["target"]["predicate_object_type"],
            "target_object": item["target"]["predicate_object_type"],
            "scene": item["scene"],
            "generation_stage": "extended_visual_review",
            "rejection_reason": "target_visual_signal_below_semantic_gate",
            "error": (
                f"Target-local RMS {target_rms:.3f} is below the visual-only "
                f"semantic gate {minimum_target_rms:.3f}"
            ),
        }
        rejection_key = (
            rejection["predicate"],
            rejection["source_plan"],
            rejection["target_name"],
        )
        if rejection_key not in {
            (
                value.get("predicate"),
                value.get("source_plan"),
                value.get("target_name"),
            )
            for value in rejected
        }:
            rejected.append(rejection)
        for strict_rejection in manifest.get("strict_rejections_superseded", []):
            if strict_rejection.get("accepted_triplet_id") == item["triplet_id"]:
                strict_rejection["final_disposition"] = (
                    "rejected_under_extended_visual_review"
                )
        print(f"REJECT VISUAL REVIEW {item['triplet_id']}", flush=True)
    manifest["triplets"] = retained
    _atomic_json(dataset_path, manifest)

    report = {
        "contract_version": "submission-tiered-v1",
        "accepted_by_tier": dict(
            sorted(
                Counter(
                    item.get("certification_tier", "pilot-v3-strict")
                    for item in manifest["triplets"]
                ).items()
            )
        ),
        "extended_rejections": len(rejected),
        "relaxations": {
            "extended-target-articulation-v1": {
                "allowed": "root position/rotation of the manipulated openable only",
                "forbidden": "non-target state, relation, inventory, or action drift",
                "minimum_bbox_area_ratio": 0.01,
            },
            "extended-small-visible-v1": {
                "allowed": "lower target-area gate with target-local pixel certificate",
                "minimum_bbox_area_ratio_dirty": 0.005,
                "minimum_bbox_area_ratio_cooked": 0.0015,
                "minimum_target_union_bbox_rms_default": 1.0,
                "minimum_target_union_bbox_rms_cooked": 15.0,
                "selection_independent_of_model_results": True,
            },
        },
    }
    _atomic_json(root / "extended_validation_report.json", report)
    return manifest


def merge_shards(shard_paths: list[Path], destination: Path) -> dict[str, Any]:
    """Merge disjoint candidate shards without treating replicas as new units."""
    manifest = json.loads(destination.read_text(encoding="utf-8"))
    by_id = {item["triplet_id"]: item for item in manifest["triplets"]}
    accepted_sources: set[tuple[str, str]] = set()
    for shard_path in shard_paths:
        shard = json.loads(shard_path.read_text(encoding="utf-8"))
        for item in shard["triplets"]:
            if item["triplet_id"] not in by_id:
                manifest["triplets"].append(item)
                by_id[item["triplet_id"]] = item
            if item.get("certification_tier", "pilot-v3-strict").startswith(
                "extended-"
            ):
                accepted_sources.add((item["predicate"], item["source_plan"]))
        for field in ("extended_rejected", "strict_rejections_superseded"):
            existing = manifest.setdefault(field, [])
            known = {json.dumps(item, sort_keys=True) for item in existing}
            for item in shard.get(field, []):
                key = json.dumps(item, sort_keys=True)
                if key not in known:
                    existing.append(item)
                    known.add(key)
    manifest["rejected"] = [
        item
        for item in manifest.get("rejected", [])
        if (item.get("predicate"), item.get("source_plan")) not in accepted_sources
    ]
    manifest["triplets"].sort(key=lambda item: item["triplet_id"])
    _atomic_json(destination, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the documented extension tier"
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/causal_states.json")
    )
    parser.add_argument(
        "--source-plans", type=Path, default=Path("Generated/magt_benchmark")
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--review-existing-only",
        action="store_true",
        help="Apply current offline visual gates without launching AI2-THOR",
    )
    parser.add_argument("--merge-shards", nargs="+", type=Path)
    args = parser.parse_args()
    if args.merge_shards:
        manifest = merge_shards(args.merge_shards, args.dataset)
        print(f"Merged {len(manifest['triplets'])} causal state units")
        return
    manifest = generate_extended(
        args.dataset,
        args.source_plans,
        limit=args.limit,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        review_existing_only=args.review_existing_only,
    )
    print(f"Dataset now contains {len(manifest['triplets'])} causal state units")


if __name__ == "__main__":
    main()
