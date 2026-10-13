from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from AsgardBench.causal.generator import (
    Candidate,
    InterventionValidationError,
    TripletGenerator,
    _diff,
    _image_rms,
    _load_plan,
)

BASE_VARIANTS = ("anchor", "causal", "nuisance")


def _candidate_for(
    triplet: dict[str, Any],
    dirty_by_source: dict[str, dict[str, Any]],
) -> Candidate:
    source = Path(triplet["source_plan"])
    predicate = triplet["predicate"]
    target = triplet["target"]
    if predicate == "isOpen":
        if target["predicate_object_type"] == "Box":
            return Candidate(
                predicate,
                source,
                target_type=target.get("dependent_object", "RemoteControl"),
                decision_type="Box",
                generation_stage="remote_container_checkpoint",
            )
        return Candidate(
            predicate,
            source,
            target_type=target["held_object"],
            decision_type=target["predicate_object_type"],
            generation_stage=target.get("checkpoint_family", "candidate_discovery"),
        )
    if predicate == "isDirty":
        return Candidate(
            predicate,
            source,
            target_type=target["predicate_object_type"],
            target_name=target["predicate_object_name"],
            decision_type=target["predicate_object_type"],
            generation_stage=target.get("checkpoint_family", "candidate_discovery"),
        )
    if predicate == "isCooked":
        return Candidate(
            predicate,
            source,
            target_type=target["predicate_object_type"],
            decision_type=target["predicate_object_type"],
            generation_stage=target.get("checkpoint_family", "cooked_state_checkpoint"),
        )
    if target["predicate_object_type"] == "Television":
        return Candidate(
            predicate,
            source,
            target_type="Television",
            decision_type="Television",
        )

    dirty = dirty_by_source.get(str(source))
    if dirty is None:
        raise InterventionValidationError(
            f"No accepted dirty checkpoint found for Faucet source {source}"
        )
    return Candidate(
        predicate,
        source,
        target_type=target["dependent_object"],
        target_name=dirty["target"]["predicate_object_name"],
        decision_type="Faucet",
        checkpoint_pose=dirty["variants"]["anchor"]["symbolic_state"]["agentPose"],
        generation_stage=target.get("checkpoint_family", "candidate_discovery"),
    )


def _angle_delta(left: float, right: float) -> float:
    return abs((right - left + 180.0) % 360.0 - 180.0)


def _normalize_replayed_snapshot(
    snapshot: dict[str, Any], held_names: set[str]
) -> dict[str, Any]:
    """Compare task state while recording cross-replay geometry separately."""
    result = copy.deepcopy(snapshot)
    result.pop("agentPose", None)
    for state in result["objects"].values():
        state.pop("position", None)
        state.pop("rotation", None)
    return result


def _validate_viewpoint(
    root: Path,
    triplet: dict[str, Any],
    base_name: str,
    viewpoint: dict[str, Any],
) -> dict[str, Any]:
    base = triplet["variants"][base_name]
    if base["acceptable_actions"] != viewpoint["acceptable_actions"]:
        raise InterventionValidationError(
            f"{base_name} viewpoint changed the acceptable action set"
        )

    base_held = {
        name
        for name, state in base["symbolic_state"]["objects"].items()
        if state.get("isPickedUp") is True
    }
    viewpoint_held = {
        name
        for name, state in viewpoint["symbolic_state"]["objects"].items()
        if state.get("isPickedUp") is True
    }
    if base_held != viewpoint_held:
        raise InterventionValidationError(
            f"Reconstruction changed held object: {base_held} -> {viewpoint_held}"
        )
    raw_symbolic_diffs = _diff(base["symbolic_state"], viewpoint["symbolic_state"])
    symbolic_diffs = _diff(
        _normalize_replayed_snapshot(base["symbolic_state"], base_held),
        _normalize_replayed_snapshot(viewpoint["symbolic_state"], base_held),
    )
    if symbolic_diffs:
        raise InterventionValidationError(
            f"{base_name} viewpoint changed symbolic state: {symbolic_diffs}"
        )

    integrity = viewpoint["validation"].get("viewpoint_integrity") or {}
    if not integrity.get("state_frozen"):
        raise InterventionValidationError(
            "Missing same-event camera-only state-integrity certificate"
        )
    for field in (
        "object_metadata_differences",
        "agent_metadata_differences",
        "inventory_metadata_differences",
    ):
        if integrity.get(field):
            raise InterventionValidationError(
                f"Camera action changed {field}: {integrity[field]}"
            )

    camera = viewpoint["validation"].get("viewpoint_camera") or {}
    source_position = camera.get("source_position", {})
    position = camera.get("position", {})
    distance = math.sqrt(
        sum(
            (float(position[axis]) - float(source_position[axis])) ** 2
            for axis in ("x", "y", "z")
        )
    )
    yaw_delta = _angle_delta(float(camera["source_yaw"]), float(camera["yaw"]))
    pitch_delta = abs(float(camera["pitch"]) - float(camera["source_pitch"]))
    if not (0.0049 <= distance <= 0.051):
        raise InterventionValidationError(
            f"Unexpected camera translation magnitude: {distance:.4f}m"
        )
    if not (0.49 <= yaw_delta <= 5.01 and 0.49 <= pitch_delta <= 3.01):
        raise InterventionValidationError(
            f"Unexpected camera rotation delta: yaw={yaw_delta:.3f}, "
            f"pitch={pitch_delta:.3f}"
        )

    bbox_area = float(viewpoint["validation"]["predicate_bbox_area_ratio"])
    if bbox_area < 0.01:
        raise InterventionValidationError(
            f"Viewpoint target bbox fell below 1%: {bbox_area:.4f}"
        )
    if not all(viewpoint["validation"]["visible"].values()):
        raise InterventionValidationError("A required decision object is not visible")

    source_image = viewpoint["validation"].get("viewpoint_source_image")
    source_metadata = viewpoint["validation"].get("viewpoint_source_metadata")
    if not source_image or not source_metadata:
        raise InterventionValidationError(
            "Missing paired egocentric source image/metadata"
        )
    rms = _image_rms(root / source_image, root / viewpoint["image"])
    if rms < 1.0:
        raise InterventionValidationError(
            f"Viewpoint image change too small (RMS={rms:.3f})"
        )
    return {
        "base_variant": base_name,
        "symbolic_metadata_differences": symbolic_diffs,
        "cross_replay_geometry_differences": raw_symbolic_diffs,
        "paired_egocentric_source_image": source_image,
        "paired_egocentric_source_metadata": source_metadata,
        "same_event_object_metadata_differences": integrity[
            "object_metadata_differences"
        ],
        "same_event_agent_metadata_differences": integrity[
            "agent_metadata_differences"
        ],
        "same_event_inventory_metadata_differences": integrity[
            "inventory_metadata_differences"
        ],
        "camera_translation_meters": distance,
        "camera_yaw_delta_degrees": yaw_delta,
        "camera_pitch_delta_degrees": pitch_delta,
        "predicate_bbox_area_ratio": bbox_area,
        "image_rms": rms,
        "target_visible": True,
        "action_set_unchanged": True,
        "goal_unchanged": True,
    }


def _report(manifest: dict[str, Any]) -> dict[str, Any]:
    accepted = []
    complete = []
    for triplet in manifest["triplets"]:
        found = 0
        for base_name in BASE_VARIANTS:
            name = f"{base_name}_viewpoint"
            variant = triplet["variants"].get(name)
            if variant is None:
                continue
            validation = variant.get("validation", {}).get("viewpoint")
            if validation is None:
                continue
            accepted.append((triplet, name, validation))
            found += 1
        if found == len(BASE_VARIANTS):
            complete.append(triplet)

    rms_values = [item[2]["image_rms"] for item in accepted]
    bbox_values = [item[2]["predicate_bbox_area_ratio"] for item in accepted]
    return {
        "dataset_version": "v3",
        "triplets": len(manifest["triplets"]),
        "complete_viewpoint_triplets": len(complete),
        "accepted_viewpoint_variants": len(accepted),
        "rejected_viewpoint_candidates": len(manifest.get("viewpoint_rejected", [])),
        "predicate_distribution_complete_triplets": dict(
            sorted(Counter(item["predicate"] for item in complete).items())
        ),
        "base_variant_distribution": dict(
            sorted(
                Counter(item[1].removesuffix("_viewpoint") for item in accepted).items()
            )
        ),
        "image_rms_range": ([min(rms_values), max(rms_values)] if rms_values else None),
        "bbox_area_ratio_range": (
            [min(bbox_values), max(bbox_values)] if bbox_values else None
        ),
        "rejected_candidates_detail": manifest.get("viewpoint_rejected", []),
        "contract": {
            "symbolic_metadata_differences": 0,
            "cross_replay_exception": "continuous geometry and camera pose",
            "paired_egocentric_and_third_party_observations": True,
            "same_event_object_metadata_differences": 0,
            "same_event_agent_metadata_differences": 0,
            "same_event_inventory_metadata_differences": 0,
            "camera_mechanism": "AI2-THOR third-party camera",
            "camera_translation_meters": [0.005, 0.05],
            "camera_yaw_delta_degrees": [0.5, 5.0],
            "camera_pitch_delta_degrees": [0.5, 3.0],
            "predeclared_perturbation_candidates": 10,
            "target_bbox_area_ratio_minimum": 0.01,
            "minimum_image_rms": 1.0,
            "acceptable_action_set_unchanged": True,
            "goal_unchanged": True,
        },
    }


def _clean_v2_and_create_v3(source: Path, destination: Path) -> None:
    manifest = json.loads(source.read_text(encoding="utf-8"))
    manifest.pop("viewpoint_rejected", None)
    # Run inexpensive open/toggle checkpoints first so failures in every
    # predicate family surface early; mirror reconstruction is much slower.
    ordered_triplets = sorted(
        manifest["triplets"],
        key=lambda item: (
            (
                2
                if item["target"]["predicate_object_type"] == "Mirror"
                else 1 if item["predicate"] == "isDirty" else 0
            ),
            item["triplet_id"],
        ),
    )
    for triplet in ordered_triplets:
        for name in list(triplet["variants"]):
            if name.endswith("_viewpoint"):
                del triplet["variants"][name]
        triplet["validation"].pop("viewpoint_nuisance", None)
    source.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if destination != source:
        shutil.copyfile(source, destination)


def augment(
    dataset_path: Path,
    retry_rejected: bool = False,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Require 0 <= shard_index < shard_count")
    root = dataset_path.parent
    manifest = json.loads(dataset_path.read_text(encoding="utf-8"))
    generator = TripletGenerator(root)
    rejected = manifest.setdefault("viewpoint_rejected", [])
    if retry_rejected:
        rejected.clear()
    rejected_keys = {(item["triplet_id"], item.get("variant")) for item in rejected}
    dirty_by_source = {
        str(item["source_plan"]): item
        for item in manifest["triplets"]
        if item["predicate"] == "isDirty"
        and item["target"]["predicate_object_type"] != "Mirror"
    }

    def persist() -> None:
        dataset_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        report_name = (
            "viewpoint_validation_report.json"
            if dataset_path.stem == "dataset_v3"
            else f"{dataset_path.stem}_viewpoint_validation_report.json"
        )
        (root / report_name).write_text(
            json.dumps(_report(manifest), indent=2), encoding="utf-8"
        )

    selected_ids = {
        item["triplet_id"]
        for index, item in enumerate(manifest["triplets"])
        if index % shard_count == shard_index
    }
    ordered_triplets = sorted(
        (item for item in manifest["triplets"] if item["triplet_id"] in selected_ids),
        key=lambda item: (
            (
                2
                if item["target"]["predicate_object_type"] == "Mirror"
                else 1 if item["predicate"] == "isDirty" else 0
            ),
            item["triplet_id"],
        ),
    )
    for triplet in ordered_triplets:
        triplet_id = triplet["triplet_id"]
        candidate = _candidate_for(triplet, dirty_by_source)
        for base_name in BASE_VARIANTS:
            variant_name = f"{base_name}_viewpoint"
            if variant_name in triplet["variants"]:
                continue
            if (triplet_id, variant_name) in rejected_keys:
                continue
            try:
                variant, history, target = generator._build_variant(
                    _load_plan(candidate.plan_path),
                    candidate,
                    triplet_id,
                    variant_name,
                    root / "triplets" / triplet_id,
                )
                if history != triplet["history"] or target != triplet["target"]:
                    raise InterventionValidationError(
                        "Viewpoint reconstruction changed history or target identity"
                    )
                variant_dict = variant.to_dict()
                validation = _validate_viewpoint(root, triplet, base_name, variant_dict)
                variant_dict["validation"]["viewpoint"] = validation
                triplet["variants"][variant_name] = variant_dict
                print(f"ACCEPT {triplet_id} {variant_name}", flush=True)
            except Exception as error:
                rejected.append(
                    {
                        "triplet_id": triplet_id,
                        "variant": variant_name,
                        "predicate": triplet["predicate"],
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                rejected_keys.add((triplet_id, variant_name))
                print(f"REJECT {triplet_id} {variant_name}: {error}", flush=True)
            persist()
    persist()
    return manifest


def merge_shards(shard_paths: list[Path], destination: Path) -> dict[str, Any]:
    if not shard_paths:
        raise ValueError("At least one shard path is required")
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in shard_paths]
    merged = copy.deepcopy(manifests[0])
    by_id = {item["triplet_id"]: item for item in merged["triplets"]}
    rejected = []
    for manifest in manifests:
        rejected.extend(manifest.get("viewpoint_rejected", []))
        for incoming in manifest["triplets"]:
            target = by_id[incoming["triplet_id"]]
            for name, variant in incoming["variants"].items():
                if name.endswith("_viewpoint"):
                    target["variants"][name] = variant
    merged["viewpoint_rejected"] = rejected
    retained = []
    excluded = []
    rejected_by_triplet: dict[str, list[dict[str, Any]]] = {}
    for item in rejected:
        rejected_by_triplet.setdefault(item["triplet_id"], []).append(item)
    for triplet in merged["triplets"]:
        missing = [
            f"{name}_viewpoint"
            for name in BASE_VARIANTS
            if f"{name}_viewpoint" not in triplet["variants"]
        ]
        if missing:
            excluded.append(
                {
                    "triplet_id": triplet["triplet_id"],
                    "predicate": triplet["predicate"],
                    "missing_variants": missing,
                    "rejections": rejected_by_triplet.get(triplet["triplet_id"], []),
                }
            )
        else:
            retained.append(triplet)
    merged["triplets"] = retained
    merged["viewpoint_excluded_triplets"] = excluded
    destination.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    report = _report(merged)
    (destination.parent / "viewpoint_validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add strictly validated camera-only viewpoint nuisances"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("CausalPilot/data_v2/dataset_v2.json"),
        help="Clean v2 source manifest",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("CausalPilot/data_v2/dataset_v3.json"),
        help="v3 destination manifest (resumed when it already exists)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Clean prototype viewpoint entries from v2 and start v3 afresh",
    )
    parser.add_argument(
        "--retry-rejected",
        action="store_true",
        help="Retry previously rejected variants after validator changes",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--merge-shards",
        nargs="+",
        type=Path,
        help="Merge completed shard manifests into --dataset and exit",
    )
    args = parser.parse_args()
    if args.merge_shards:
        manifest = merge_shards(args.merge_shards, args.dataset)
        report = _report(manifest)
        print(
            f"Merged {report['accepted_viewpoint_variants']} viewpoint variants "
            f"across {report['complete_viewpoint_triplets']} complete triplets; "
            f"rejected {report['rejected_viewpoint_candidates']}",
            flush=True,
        )
        return
    if args.fresh or not args.dataset.exists():
        _clean_v2_and_create_v3(args.source, args.dataset)
    manifest = augment(
        args.dataset,
        retry_rejected=args.retry_rejected,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    report = _report(manifest)
    print(
        f"Generated {report['accepted_viewpoint_variants']} viewpoint variants "
        f"across {report['complete_viewpoint_triplets']} complete triplets; "
        f"rejected {report['rejected_viewpoint_candidates']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
