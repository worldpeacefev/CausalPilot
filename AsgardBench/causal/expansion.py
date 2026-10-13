"""Submission-level packaging, validation, and scientific audits.

This module does not redefine a causal sample.  It applies the pilot contract
to a larger manifest, emits the required accepted/rejected projections, and
compares the result with the frozen v3 reference distribution.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from AsgardBench.causal.generator import SYMBOLIC_KEYS, _diff, _rejection_category

CORE_VARIANTS = ("anchor", "causal", "nuisance")
VISUAL_PREDICATES = {"isDirty", "isOpen", "isToggled", "isCooked"}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _canonical_action(choice: dict[str, Any]) -> str:
    return f"{choice['action'].strip().lower()} {(choice.get('object') or 'none').strip().lower()}"


def _hist(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _scene_family(scene: str) -> str:
    suffix = scene.removeprefix("FloorPlan")
    try:
        number = int(suffix)
    except ValueError:
        return "unknown"
    return {
        0: "kitchen",
        2: "living_room",
        3: "bedroom",
        4: "bathroom",
    }.get(number // 100, "unknown")


def _semantic_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Remove replay-only continuous geometry, preserving task state."""
    result = copy.deepcopy(snapshot)
    result.pop("agentPose", None)
    for state in result.get("objects", {}).values():
        state.pop("position", None)
        state.pop("rotation", None)
    return result


def _contract_for(triplet: dict[str, Any]) -> dict[str, Any]:
    tier = triplet.get("certification_tier", "pilot-v3-strict")
    saved = triplet.get("validation", {}).get("certification_contract", {})
    defaults = {
        "pilot-v3-strict": (0.01, False),
        "extended-target-articulation-v1": (0.01, True),
        "extended-small-visible-v1": (
            0.0015 if triplet.get("predicate") == "isCooked" else 0.005,
            False,
        ),
    }
    minimum, articulation = defaults.get(tier, (0.01, False))
    return {
        "tier": tier,
        "minimum_bbox_area_ratio": float(saved.get("minimum_bbox_area_ratio", minimum)),
        "target_local_articulation_allowed": bool(
            saved.get("target_local_articulation_allowed", articulation)
        ),
    }


def _metadata_objects(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["objectId"]: item for item in metadata.get("objects", [])}


def _relation_errors(metadata: dict[str, Any]) -> list[str]:
    """Check bidirectional containment and inventory consistency."""
    objects = _metadata_objects(metadata)
    errors: list[str] = []
    for object_id, item in objects.items():
        for parent_id in item.get("parentReceptacles") or []:
            parent = objects.get(parent_id)
            if parent is None:
                errors.append(f"missing_parent:{object_id}:{parent_id}")
            elif object_id not in (parent.get("receptacleObjectIds") or []):
                errors.append(f"parent_missing_child:{parent_id}:{object_id}")
        for child_id in item.get("receptacleObjectIds") or []:
            child = objects.get(child_id)
            if child is None:
                errors.append(f"missing_child:{object_id}:{child_id}")
            elif object_id not in (child.get("parentReceptacles") or []):
                errors.append(f"child_missing_parent:{child_id}:{object_id}")

    inventory_ids = {item["objectId"] for item in metadata.get("inventoryObjects", [])}
    picked_ids = {
        object_id
        for object_id, item in objects.items()
        if item.get("isPickedUp") is True
    }
    if inventory_ids != picked_ids:
        errors.append(
            "inventory_pickup_mismatch:"
            f"inventory={sorted(inventory_ids)}:picked={sorted(picked_ids)}"
        )
    for object_id in inventory_ids:
        if objects[object_id].get("parentReceptacles"):
            errors.append(f"held_object_has_parent:{object_id}")
    return errors


def _interface_actions(triplet: dict[str, Any]) -> set[str]:
    return {
        item.split()[0].strip().lower().replace("put", "put", 1)
        for item in triplet.get("action_interface", [])
    }


def _normalize_rejection(
    item: dict[str, Any], triplets: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    result = copy.deepcopy(item)
    triplet = triplets.get(str(result.get("triplet_id")))
    if triplet:
        result.setdefault("predicate", triplet["predicate"])
        result.setdefault("object", triplet["target"]["predicate_object_type"])
        result.setdefault("scene", triplet["scene"])
    source = result.get("source_plan")
    if source and Path(source).exists():
        try:
            plan = json.loads(Path(source).read_text(encoding="utf-8"))
            result.setdefault("scene", plan.get("scene"))
            result.setdefault("task_type", str(plan.get("name", "")).split("__", 1)[0])
        except (OSError, json.JSONDecodeError):
            pass
    error = str(result.get("error", ""))
    if not result.get("object"):
        match = re.search(r"objects\.([A-Za-z]+)_", error)
        if match:
            result["object"] = match.group(1)
        elif result.get("predicate") == "isToggled":
            result["object"] = "Faucet_or_Television"
        else:
            result["object"] = "unknown"
    if not result.get("scene"):
        match = re.search(r"FloorPlan\d+", str(source or result.get("triplet_id", "")))
        result["scene"] = match.group(0) if match else "unknown"
    result.setdefault("generation_stage", "pilot_generation")
    result.setdefault("rejection_reason", _rejection_category(error))
    return result


def validate_triplet(root: Path, triplet: dict[str, Any]) -> dict[str, Any]:
    """Revalidate a saved triplet without changing the pilot definition."""
    errors: list[dict[str, str]] = []

    def fail(check: str, detail: str, stage: str = "offline_validation") -> None:
        errors.append({"check": check, "detail": detail, "generation_stage": stage})

    missing = [
        name for name in CORE_VARIANTS if name not in triplet.get("variants", {})
    ]
    if missing:
        fail("required_variants", f"missing={missing}")
        return {"passed": False, "errors": errors}

    variants = triplet["variants"]
    target_name = triplet["target"]["predicate_object_name"]
    predicate = triplet["predicate"]
    contract = _contract_for(triplet)
    expected_path = f"objects.{target_name}.{predicate}"
    causal_diffs = _diff(
        variants["anchor"]["symbolic_state"],
        variants["causal"]["symbolic_state"],
    )
    nuisance_diffs = _diff(
        variants["anchor"]["symbolic_state"],
        variants["nuisance"]["symbolic_state"],
    )
    allowed_causal_diffs = {expected_path}
    if contract["target_local_articulation_allowed"] and predicate == "isOpen":
        target_prefix = f"objects.{target_name}."
        allowed_causal_diffs.update(
            path
            for path in causal_diffs
            if path.startswith(f"{target_prefix}position.")
            or path.startswith(f"{target_prefix}rotation.")
        )
    if set(causal_diffs) != allowed_causal_diffs:
        fail("non_target_symbolic_control", f"causal_differences={causal_diffs}")
    if nuisance_diffs:
        fail("nuisance_symbolic_control", f"differences={nuisance_diffs}")

    anchor_value = variants["anchor"].get("validation", {}).get("predicate_value")
    causal_value = variants["causal"].get("validation", {}).get("predicate_value")
    if not isinstance(anchor_value, bool) or causal_value is not (not anchor_value):
        fail(
            "target_predicate_changed",
            f"anchor={anchor_value!r},causal={causal_value!r}",
        )

    action_sets: dict[str, set[str]] = {}
    valid_interface = _interface_actions(triplet)
    for name, variant in variants.items():
        image = root / variant["image"]
        metadata_path = root / variant["metadata"]
        if not image.exists():
            fail("artifact_exists", f"missing_image:{variant['image']}")
        if not metadata_path.exists():
            fail("artifact_exists", f"missing_metadata:{variant['metadata']}")
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for detail in _relation_errors(metadata):
            fail("relations_inventory_consistency", f"{name}:{detail}")

        validation = variant.get("validation", {})
        visible = validation.get("visible", {})
        if not visible or not all(visible.values()):
            fail("object_visibility", f"{name}:{visible}")
        bbox_ratio = validation.get("predicate_bbox_area_ratio")
        if (
            not isinstance(bbox_ratio, (int, float))
            or bbox_ratio < contract["minimum_bbox_area_ratio"]
        ):
            fail("object_visibility", f"{name}:bbox_ratio={bbox_ratio}")

        choices = variant.get("acceptable_actions", [])
        if not choices:
            fail("action_validity", f"{name}:empty_acceptable_set")
        canonical = {_canonical_action(choice) for choice in choices}
        action_sets[name] = canonical
        object_types = {item["objectType"] for item in metadata.get("objects", [])}
        for choice in choices:
            action = choice["action"].strip().lower()
            obj = choice.get("object")
            if action not in valid_interface:
                fail("action_validity", f"{name}:action_not_in_interface:{action}")
            if action != "done" and (not obj or obj not in object_types):
                fail("action_validity", f"{name}:unknown_object_type:{obj}")

    if action_sets.get("anchor", set()) & action_sets.get("causal", set()):
        fail("causal_action_separation", "anchor_and_causal_sets_overlap")
    if action_sets.get("anchor") != action_sets.get("nuisance"):
        fail("nuisance_invariance_validity", "lighting_action_set_changed")

    if contract["tier"].startswith("extended-"):
        signal = triplet.get("validation", {}).get("causal_visual_signal", {})
        if (
            not signal.get("passed")
            or float(signal.get("target_union_bbox_rms", 0.0)) < 1.0
        ):
            fail("causal_visual_manifestation", f"certificate={signal}")

    nuisance_pairs = [("anchor", "nuisance", "lighting")]
    for suffix in ("viewpoint", "material"):
        for base in CORE_VARIANTS:
            name = f"{base}_{suffix}"
            if name in variants:
                nuisance_pairs.append((base, name, suffix))
                if action_sets.get(base) != action_sets.get(name):
                    fail(
                        "nuisance_invariance_validity",
                        f"{base}->{name}:action_set_changed",
                    )
                if _diff(
                    _semantic_snapshot(variants[base]["symbolic_state"]),
                    _semantic_snapshot(variants[name]["symbolic_state"]),
                ):
                    fail(
                        "nuisance_invariance_validity",
                        f"{base}->{name}:task_state_changed",
                    )

    checks = {
        "target_predicate_changed": not any(
            item["check"] == "target_predicate_changed" for item in errors
        ),
        "non_target_symbolic_states_controlled": not any(
            item["check"] == "non_target_symbolic_control" for item in errors
        ),
        "relations_and_inventory_consistent": not any(
            item["check"] == "relations_inventory_consistency" for item in errors
        ),
        "object_visible": not any(
            item["check"] == "object_visibility" for item in errors
        ),
        "actions_valid": not any(item["check"] == "action_validity" for item in errors),
        "causal_action_sets_disjoint": not any(
            item["check"] == "causal_action_separation" for item in errors
        ),
        "nuisance_action_branch_invariant": not any(
            item["check"] == "nuisance_invariance_validity" for item in errors
        ),
        "causal_visual_manifestation_for_extended_tier": not any(
            item["check"] == "causal_visual_manifestation" for item in errors
        ),
    }
    return {
        "passed": not errors,
        "checks": checks,
        "causal_symbolic_differences": causal_diffs,
        "allowed_causal_symbolic_differences": sorted(allowed_causal_diffs),
        "certification_contract": contract,
        "nuisance_symbolic_differences": nuisance_diffs,
        "nuisance_pairs": [
            {"reference": left, "variant": right, "family": family}
            for left, right, family in nuisance_pairs
        ],
        "errors": errors,
    }


def upgrade_symbolic_snapshots(root: Path, manifest: dict[str, Any]) -> int:
    """Backfill newly controlled fields from immutable saved metadata."""
    changed = 0
    for triplet in manifest["triplets"]:
        for variant in triplet["variants"].values():
            metadata_path = root / variant["metadata"]
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            by_name = {item["name"]: item for item in metadata["objects"]}
            for name, state in variant["symbolic_state"]["objects"].items():
                source = by_name[name]
                for key in SYMBOLIC_KEYS:
                    if key in source and key not in state:
                        value = copy.deepcopy(source[key])
                        if key in {
                            "parentReceptacles",
                            "receptacleObjectIds",
                        } and isinstance(value, list):
                            value = sorted(value)
                        state[key] = value
                        changed += 1
    return changed


def coverage_report(manifest: dict[str, Any]) -> dict[str, Any]:
    triplets = manifest["triplets"]
    nuisance_values = []
    action_values = []
    for triplet in triplets:
        names = set(triplet["variants"])
        nuisance_values.append("lighting")
        if all(f"{base}_viewpoint" in names for base in CORE_VARIANTS):
            nuisance_values.append("viewpoint")
        if all(f"{base}_material" in names for base in CORE_VARIANTS):
            nuisance_values.append("material")
        for variant in ("anchor", "causal"):
            action_values.extend(
                _canonical_action(choice)
                for choice in triplet["variants"][variant]["acceptable_actions"]
            )
    report = {
        "total_triplets": len(triplets),
        "independent_causal_state_units": len(
            {item.get("causal_state_unit", item["triplet_id"]) for item in triplets}
        ),
        "total_observations": sum(len(item["variants"]) for item in triplets),
        "certification_tier_histogram": _hist(
            item.get("certification_tier", "pilot-v3-strict") for item in triplets
        ),
        "predicate_histogram": _hist(item["predicate"] for item in triplets),
        "object_histogram": _hist(
            item["target"]["predicate_object_type"] for item in triplets
        ),
        "held_object_histogram": _hist(
            str(item["target"].get("held_object", "none")) for item in triplets
        ),
        "scene_histogram": _hist(item["scene"] for item in triplets),
        "scene_family_histogram": _hist(
            _scene_family(item["scene"]) for item in triplets
        ),
        "task_histogram": _hist(item["task_type"] for item in triplets),
        "nuisance_histogram": _hist(nuisance_values),
        "action_histogram": _hist(action_values),
        "checkpoint_history_length_histogram": _hist(
            str(len(item["history"])) for item in triplets
        ),
    }
    concentration = {}
    for field in (
        "predicate_histogram",
        "object_histogram",
        "scene_histogram",
        "task_histogram",
    ):
        distribution = _distribution(report[field])
        category, share = max(
            distribution.items(), key=lambda item: item[1], default=(None, 0.0)
        )
        concentration[field.removesuffix("_histogram")] = {
            "largest_category": category,
            "share": share,
            "flagged": share
            > (0.5 if field in {"predicate_histogram", "task_histogram"} else 0.35),
        }
    report["concentration_audit"] = concentration
    return report


def _distribution(histogram: dict[str, int]) -> dict[str, float]:
    total = sum(histogram.values())
    return {key: value / total for key, value in histogram.items()} if total else {}


def _js_divergence(left: dict[str, int], right: dict[str, int]) -> float:
    p, q = _distribution(left), _distribution(right)
    keys = set(p) | set(q)
    midpoint = {key: (p.get(key, 0.0) + q.get(key, 0.0)) / 2 for key in keys}

    def kl(source: dict[str, float]) -> float:
        return sum(
            value * math.log2(value / midpoint[key])
            for key, value in source.items()
            if value > 0 and midpoint[key] > 0
        )

    return (kl(p) + kl(q)) / 2


def compare_distributions(
    pilot_manifest: dict[str, Any], final_manifest: dict[str, Any]
) -> dict[str, Any]:
    pilot = coverage_report(pilot_manifest)
    final = coverage_report(final_manifest)
    fields = (
        "predicate_histogram",
        "object_histogram",
        "scene_family_histogram",
        "task_histogram",
        "nuisance_histogram",
        "action_histogram",
    )
    comparisons = {}
    shifts = []
    for field in fields:
        left, right = pilot[field], final[field]
        p, q = _distribution(left), _distribution(right)
        deltas = {
            key: q.get(key, 0.0) - p.get(key, 0.0) for key in sorted(set(p) | set(q))
        }
        divergence = _js_divergence(left, right)
        comparisons[field] = {
            "pilot": left,
            "final": right,
            "proportion_delta": deltas,
            "jensen_shannon_divergence_bits": divergence,
        }
        for key, delta in deltas.items():
            if abs(delta) >= 0.15:
                shifts.append({"dimension": field, "category": key, "delta": delta})

    return {
        "pilot_triplets": pilot["total_triplets"],
        "final_triplets": final["total_triplets"],
        "growth_factor": (
            final["total_triplets"] / pilot["total_triplets"]
            if pilot["total_triplets"]
            else None
        ),
        "comparisons": comparisons,
        "preserved_trends": {
            "causal_and_nuisance_observations_retained": all(
                set(CORE_VARIANTS).issubset(item["variants"])
                for item in final_manifest["triplets"]
            ),
            "pilot_predicates_retained": set(pilot["predicate_histogram"]).issubset(
                final["predicate_histogram"]
            ),
            "no_single_predicate_majority": max(
                _distribution(final["predicate_histogram"]).values(), default=0
            )
            <= 0.5,
        },
        "unexpected_shifts": shifts,
        "possible_benchmark_bias": [
            "A category shifted by at least 15 percentage points." for _ in shifts[:1]
        ],
    }


def predicate_feasibility_report(manifest: dict[str, Any]) -> dict[str, Any]:
    accepted = Counter(item["predicate"] for item in manifest["triplets"])
    accepted_by_tier = Counter(
        item.get("certification_tier", "pilot-v3-strict")
        for item in manifest["triplets"]
    )
    rejected: dict[str, list[dict[str, Any]]] = {}
    for item in manifest.get("rejected", []):
        rejected.setdefault(str(item.get("predicate", "unknown")), []).append(item)
    return {
        "accepted_predicates": dict(sorted(accepted.items())),
        "accepted_by_certification_tier": dict(sorted(accepted_by_tier.items())),
        "candidate_rejections": {
            key: {
                "count": len(values),
                "reason_histogram": _hist(
                    str(
                        item.get("rejection_reason")
                        or _rejection_category(item.get("error", ""))
                    )
                    for item in values
                ),
            }
            for key, values in sorted(rejected.items())
        },
        "design_screening": {
            "isOpen_Box": {
                "accepted": False,
                "reason": (
                    "Representative TV checkpoint changed Box and contained-remote "
                    "root geometry in addition to isOpen; remaining equivalent "
                    "scenes were pruned after the structural family-level failure."
                ),
            },
            "isOpen_Microwave": {
                "accepted": False,
                "reason": (
                    "All six cooking candidates changed articulated Microwave "
                    "geometry beyond the Boolean predicate."
                ),
            },
            "isFilledWithLiquid": {
                "accepted": False,
                "reason": (
                    "Changing filledness also changes fillLiquid; coffee-machine "
                    "checkpoints additionally trigger device state, so the pilot's "
                    "single-symbolic-variable contract cannot be retained."
                ),
            },
            "isSliced": {
                "accepted": False,
                "reason": (
                    "Slicing replaces one object with multiple slice instances and "
                    "changes topology/geometry, not one certified Boolean field."
                ),
            },
            "isCooked": {
                "accepted": accepted.get("isCooked", 0) > 0,
                "reason": (
                    "Whole-Potato targets occupied only 0.19-0.38% of the image; "
                    "the representative BreadSliced target occupied 0.44%. All "
                    "were below the unchanged 1% visibility gate, so equivalent "
                    "Bread scenes were pruned and the predicate was not retained."
                    if accepted.get("isCooked", 0) == 0
                    else "Retained in the explicitly reported small-visible extension tier only when CookObject changes isCooked alone, target-local RMS certifies a visual manifestation, and cook versus pickup actions are disjoint."
                ),
            },
        },
        "selection_rule": (
            "A predicate is retained only when simulator modification, visible "
            "manifestation, action-branch separation, and automatic certification "
            "all succeed."
        ),
        "rejection_counting_note": (
            "Candidate rejection histograms retain historical strict-contract "
            "failures even when a candidate is later accepted under a named "
            "extended contract. This preserves the audit trail."
        ),
    }


def _overall_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("action_summary", value) if isinstance(value, dict) else value
    if not isinstance(rows, list):
        return None
    row = next((item for item in rows if item.get("group") == "overall"), None)
    return row.get("metrics") if row else None


def diagnostic_reference_report(repository_root: Path) -> dict[str, Any] | None:
    table_path = repository_root / "CausalPilot/paper_v3/main_table.json"
    predicate_path = repository_root / "CausalPilot/diagnosis_v3/predicate_summary.json"
    if not table_path.exists() or not predicate_path.exists():
        return None
    table = json.loads(table_path.read_text(encoding="utf-8"))
    conditions = {row["condition"]: row for row in table}
    direct = conditions.get("gpt-5.4 Direct")
    no_image = conditions.get("gpt-5.4 No-Image")
    oracle = conditions.get("gpt-5.4 Oracle-State")
    predicate_rows = json.loads(predicate_path.read_text(encoding="utf-8"))
    predicate = next(item for item in predicate_rows if item["group"] == "overall")
    scene = _overall_metrics(
        repository_root / "CausalPilot/scene_description_v1/summary.json"
    )
    boundary = _overall_metrics(
        repository_root / "CausalPilot/boundary_probe_v1/summary.json"
    )
    bgcp = _overall_metrics(repository_root / "CausalPilot/bgcp_v1/summary.json")
    if not all((direct, no_image, oracle, scene, boundary, bgcp)):
        return None
    return {
        "scope": "frozen_27_triplet_reference_distribution",
        "sample_selection_dependency": False,
        "direct": direct,
        "no_image": no_image,
        "oracle_state": oracle,
        "predicate_probe": predicate,
        "full_scene_description": scene,
        "boundary_probe": boundary,
        "bgcp": bgcp,
        "scientific_pattern_checks": {
            "direct_below_oracle_causal_response": direct["causal_response_accuracy"]
            < oracle["causal_response_accuracy"],
            "no_image_exposes_action_prior": no_image["prior_persistence_rate"] == 1.0,
            "scene_description_not_sufficient": scene["grounded_v3_accuracy"] < 1.0,
            "boundary_representation_beats_full_description": bgcp[
                "grounded_v3_accuracy"
            ]
            > scene["grounded_v3_accuracy"],
            "predicate_perception_imperfect": predicate["predicate_accuracy"] < 1.0,
        },
        "final_expansion_status": (
            "The two newly retained hard core triplets were not sent to an external "
            "model during generation. Re-run experiments/run_matrix.py to estimate "
            "final-set effects; generation remains independent of model behavior."
        ),
    }


def _paper_story_markdown(
    manifest: dict[str, Any],
    validations: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
) -> str:
    coverage = coverage_report(manifest)
    failed = [key for key, value in validations.items() if not value["passed"]]
    impossible = [
        item["triplet_id"]
        for item in manifest["triplets"]
        if item["predicate"] not in VISUAL_PREDICATES
        or min(
            float(variant.get("validation", {}).get("predicate_bbox_area_ratio", 0))
            for variant in item["variants"].values()
        )
        < _contract_for(item)["minimum_bbox_area_ratio"]
    ]
    prior_pair_hist = _hist(
        " -> ".join(
            (
                next(
                    iter(
                        _canonical_action(x)
                        for x in item["variants"]["anchor"]["acceptable_actions"]
                    )
                ),
                next(
                    iter(
                        _canonical_action(x)
                        for x in item["variants"]["causal"]["acceptable_actions"]
                    )
                ),
            )
        )
        for item in manifest["triplets"]
    )
    max_pair_share = max(prior_pair_hist.values(), default=0) / max(
        coverage["total_triplets"], 1
    )
    strengthened = (
        not failed
        and not impossible
        and coverage["total_triplets"] > comparison["pilot_triplets"]
        and len(coverage["scene_histogram"]) >= 10
        and len(coverage["object_histogram"]) >= 5
    )
    verdict = "Yes" if strengthened else "Not yet"
    return f"""# Paper Story Audit

## Verdict

**{verdict}.** The expansion is judged on certified causal-grounding coverage, not raw sample count.

## Coverage Audit

- Total certified triplets: {coverage['total_triplets']}
- Certification tiers: {json.dumps(coverage['certification_tier_histogram'], sort_keys=True)}
- Predicates: {json.dumps(coverage['predicate_histogram'], sort_keys=True)}
- Decision objects: {json.dumps(coverage['object_histogram'], sort_keys=True)}
- Scenes: {len(coverage['scene_histogram'])} across {json.dumps(coverage['scene_family_histogram'], sort_keys=True)}
- Tasks: {json.dumps(coverage['task_histogram'], sort_keys=True)}
- Nuisances: {json.dumps(coverage['nuisance_histogram'], sort_keys=True)}

## Distribution Drift Audit

- Pilot triplets: {comparison['pilot_triplets']}; final triplets: {comparison['final_triplets']}.
- Preserved trends: {json.dumps(comparison['preserved_trends'], sort_keys=True)}
- Shifts requiring interpretation: {json.dumps(comparison['unexpected_shifts'], sort_keys=True)}
- No category is retained merely to reach a target count.

## Paper Story Questions

1. **Does the benchmark still test causal grounding?** {'Yes' if not failed else 'No: validation failures remain'}. Every retained pair must change only the target predicate and must have disjoint acceptable action sets.
2. **Does it still support visual decision-boundary diagnosis?** {'Yes' if not impossible else 'No: visually uncertifiable cases remain'}. Retained predicates have simulator-rendered manifestations and visible targets.
3. **Are any samples only object-recognition tests?** No certified sample is accepted on object identity alone; correctness changes across two states of the same object identity.
4. **Can action priors dominate?** Yes, by design this is a measured failure mode. The largest action-transition template occupies {max_pair_share:.1%}; no-image evaluation is required to quantify it.
5. **Is visual evidence impossible to infer?** {'No retained cases were flagged.' if not impossible else 'Flagged: ' + ', '.join(impossible)}

## Quality-Control Outcome

- Offline validation failures among retained samples: {len(failed)}.
- Invalid categories are rejected with stage, predicate, object, and scene metadata.
- Model accuracy is never used as a sample-retention criterion.

## Experimental Diagnosis Status

The frozen pilot results already establish Direct < Oracle-State, 100% no-image prior persistence, and stronger targeted boundary/state conditioning than unrestricted description. Expanded-set model runs are intentionally separate from generation and must use the scripts in `experiments/`; if those patterns weaken, the response is to audit generation and distribution drift, not to relabel outcomes.
"""


def audit_dataset(
    dataset_path: Path,
    pilot_path: Path,
    reject_invalid: bool = False,
    upgrade_snapshots: bool = False,
) -> dict[str, Any]:
    manifest = json.loads(dataset_path.read_text(encoding="utf-8"))
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    root = dataset_path.parent
    if upgrade_snapshots:
        upgrade_symbolic_snapshots(root, manifest)
        _write_json(dataset_path, manifest)
    validations = {
        item["triplet_id"]: validate_triplet(root, item)
        for item in manifest["triplets"]
    }
    invalid_ids = {key for key, value in validations.items() if not value["passed"]}
    by_id = {item["triplet_id"]: item for item in manifest["triplets"]}
    raw_rejected = [
        *manifest.get("rejected", []),
        *manifest.get("extended_rejected", []),
        *manifest.get("strict_rejections_superseded", []),
        *manifest.get("viewpoint_rejected", []),
        *manifest.get("material_rejected", []),
        *(
            item
            for item in pilot.get("viewpoint_rejected", [])
            if item.get("triplet_id") in by_id
            and item not in manifest.get("viewpoint_rejected", [])
        ),
    ]
    rejected = [_normalize_rejection(item, by_id) for item in raw_rejected]
    for item in manifest["triplets"]:
        if item["triplet_id"] not in invalid_ids:
            continue
        first = validations[item["triplet_id"]]["errors"][0]
        rejected.append(
            {
                "triplet_id": item["triplet_id"],
                "predicate": item["predicate"],
                "object": item["target"]["predicate_object_type"],
                "scene": item["scene"],
                "generation_stage": first["generation_stage"],
                "rejection_reason": first["check"],
                "error": first["detail"],
            }
        )
    if reject_invalid and invalid_ids:
        manifest["triplets"] = [
            item
            for item in manifest["triplets"]
            if item["triplet_id"] not in invalid_ids
        ]
        manifest["rejected"] = rejected
        _write_json(dataset_path, manifest)

    accepted = [
        item for item in manifest["triplets"] if item["triplet_id"] not in invalid_ids
    ]
    accepted_manifest = {**manifest, "triplets": accepted, "rejected": rejected}
    report = {
        "contract_version": "submission-tiered-v1",
        "accepted_triplets": len(accepted),
        "rejected_candidates": len(rejected),
        "accepted_by_tier": _hist(
            item.get("certification_tier", "pilot-v3-strict") for item in accepted
        ),
        "superseded_strict_rejections": sum(
            item.get("final_disposition") == "accepted_under_extended_contract"
            for item in rejected
        ),
        "retained_validation_failures": 0 if reject_invalid else len(invalid_ids),
        "required_checks": [
            "target_predicate_changed",
            "non_target_symbolic_states_controlled",
            "relations_and_inventory_consistent",
            "object_visible",
            "actions_valid",
            "causal_action_sets_disjoint",
            "nuisance_action_branch_invariant",
            "causal_visual_manifestation_for_extended_tier",
        ],
        "per_triplet": validations,
        "rejected_reason_distribution": _hist(
            str(
                item.get("rejection_reason")
                or _rejection_category(item.get("error", ""))
            )
            for item in rejected
        ),
        "rejected_candidates_detail": rejected,
    }
    comparison = compare_distributions(pilot, accepted_manifest)
    _write_json(root / "accepted_samples.json", accepted)
    _write_json(root / "rejected_samples.json", rejected)
    _write_json(root / "validation_report.json", report)
    _write_json(root / "coverage_report.json", coverage_report(accepted_manifest))
    _write_json(root / "pilot_vs_final_report.json", comparison)
    _write_json(
        root / "predicate_feasibility_report.json",
        predicate_feasibility_report(accepted_manifest),
    )
    repository_root = Path(__file__).resolve().parents[2]
    diagnostic = diagnostic_reference_report(repository_root)
    if diagnostic is not None:
        _write_json(root / "diagnostic_audit.json", diagnostic)
    for family in ("viewpoint", "material"):
        complete = [
            item
            for item in accepted
            if all(f"{base}_{family}" in item["variants"] for base in CORE_VARIANTS)
        ]
        _write_json(
            root / f"dataset_{family}_complete.json",
            {**accepted_manifest, "triplets": complete},
        )
    (root / "paper_story_audit.md").write_text(
        _paper_story_markdown(accepted_manifest, validations, comparison),
        encoding="utf-8",
    )
    return report


def bootstrap_from_pilot(pilot_path: Path, output_dir: Path) -> Path:
    """Copy only referenced pilot artifacts into a resumable final directory."""
    manifest = json.loads(pilot_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    source_root = pilot_path.parent
    for triplet in manifest["triplets"]:
        source = source_root / "triplets" / triplet["triplet_id"]
        destination = output_dir / "triplets" / triplet["triplet_id"]
        if destination.exists():
            continue
        shutil.copytree(source, destination)
        # Keep the per-triplet record aligned with the augmented manifest.
        _write_json(destination / "triplet.json", triplet)
    destination_manifest = output_dir / "dataset.json"
    copied = copy.deepcopy(manifest)
    copied["format_version"] = 3
    copied["reference_distribution"] = str(pilot_path)
    copied["generator"] = "AsgardBench.causal.generator+expansion"
    _write_json(destination_manifest, copied)
    return destination_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package and audit the expanded causal benchmark"
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("CausalPilot/data_final/dataset.json")
    )
    parser.add_argument(
        "--pilot",
        type=Path,
        default=Path("CausalPilot/data_v2/dataset_v3.json"),
    )
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--reject-invalid", action="store_true")
    parser.add_argument("--upgrade-symbolic-snapshots", action="store_true")
    args = parser.parse_args()
    if args.bootstrap:
        bootstrap_from_pilot(args.pilot, args.dataset.parent)
    report = audit_dataset(
        args.dataset,
        args.pilot,
        args.reject_invalid,
        args.upgrade_symbolic_snapshots,
    )
    print(
        f"Accepted {report['accepted_triplets']} triplets; "
        f"rejected evidence records: {report['rejected_candidates']}"
    )


if __name__ == "__main__":
    main()
