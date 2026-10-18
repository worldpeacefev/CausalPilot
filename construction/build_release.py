"""Preserve the archived release and assemble scenario/type/scene shards.

All manifest paths in the release are relative to data/, independent of shard
depth. Run from the repository root. Repeated builds reuse existing images.
"""

import argparse
import copy
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

from PIL import Image

from AsgardBench.causal.render_replicas import LARGE_PROFILES, _crop_rms, _rms
from construction.common import ASPECTS, ROLES, diff, sha256, write_json
from construction.nuisance import EXTRA_PROFILES


def load(path):
    return json.loads(Path(path).read_text())


def group_path(row):
    return (
        Path("scenarios") / ASPECTS[row["predicate"]] / row["predicate"] / row["scene"]
    )


def normalize_existing(row, data, state=False):
    result = copy.deepcopy(row)
    unit = row["causal_state_unit"]
    profile = row.get("replica_profile", "simulator_lighting")
    for role, variant in result["variants"].items():
        if role not in ROLES:
            continue
        old = data / variant["image"]
        new = group_path(row) / unit / profile / f"{role}.png"
        destination = data / new
        if old.resolve() != destination.resolve():
            destination.parent.mkdir(parents=True, exist_ok=True)
            if old.exists():
                if destination.exists():
                    if sha256(old) != sha256(destination):
                        raise ValueError(f"Image collision: {destination}")
                else:
                    old.rename(destination)
            elif not destination.exists():
                raise FileNotFoundError(old)
        variant["image"] = new.as_posix()
    return result


def compact(row, data):
    result = copy.deepcopy(row)
    for role in ROLES:
        v = result["variants"][role]
        if "symbolic_state" in v:
            state = v.pop("symbolic_state")
            import hashlib

            content = json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
            name = hashlib.sha256(content).hexdigest()
            relative = f"../simulation/states/{name}.json"
            destination = data / relative
            if not destination.exists():
                write_json(destination, state)
            v["symbolic_state_path"] = relative
    result["scenario_aspect"] = ASPECTS[row["predicate"]]
    result["physical_scene_type"] = room_type(row["scene"])
    result["release_partition"] = (
        "paper_archive"
        if not row["causal_state_unit"].startswith("extension__")
        and row.get("release_partition") != "nuisance_extension"
        else row.get("release_partition", "predicate_extension")
    )
    result.setdefault("review_status", "paper_claim_not_independently_reverified")
    return result


def align_contract(row):
    """Correct historical target-articulation tier labels with retained provenance."""
    result = copy.deepcopy(row)
    variants = result["variants"]
    changes = diff(
        variants["anchor"]["symbolic_state"], variants["causal"]["symbolic_state"]
    )
    target = result["target"]["predicate_object_name"]
    expected = f"objects.{target}.{result['predicate']}"
    extra = set(changes) - {expected}
    if (
        result["predicate"] == "isOpen"
        and extra
        and all(
            p.startswith((f"objects.{target}.position.", f"objects.{target}.rotation."))
            for p in extra
        )
    ):
        if not all(
            variants[r]["validation"]["predicate_bbox_area_ratio"] >= 0.01
            for r in ROLES
        ):
            raise ValueError("Articulation tier requires bbox >=1%")
        result["original_certification_tier"] = result["certification_tier"]
        result["certification_tier"] = "extended-target-articulation-v1"
        result["normalization_notes"] = (
            "Tier aligned with manuscript Table 4: only target's own root pose accompanies isOpen. Archived compatibility manifest retains original tier."
        )
        result["validation"]["causal_symbolic_differences"] = changes
        result["validation"]["allowed_causal_symbolic_differences"] = changes
    return result


def room_type(scene):
    n = int(scene.removeprefix("FloorPlan"))
    return (
        "kitchen"
        if n < 200
        else "living_room"
        if n < 300
        else "bedroom"
        if n < 400
        else "bathroom"
    )


def import_unit(row, source, data):
    result = copy.deepcopy(row)
    unit = row["causal_state_unit"]
    profile = row.get("replica_profile", "simulator_lighting")
    for role, v in result["variants"].items():
        relative = group_path(row) / unit / profile / f"{role}.png"
        destination = data / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / v["image"], destination)
        v["image"] = relative.as_posix()
        metadata_source = source / v["metadata"]
        meta = (
            f"../simulation/metadata/{row['predicate']}/{sha256(metadata_source)}.json"
        )
        destination = data / meta
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(metadata_source, destination)
        v["metadata"] = meta
    suffix = "" if profile == "simulator_lighting" else f"__{profile}"
    recipe = f"simulation/recipes/{unit}{suffix}.json"
    (data.parent / recipe).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / row["source_plan"], data.parent / recipe)
    result["source_plan"] = recipe
    result["replica_profile"] = profile
    result["triplet_id"] = unit + "__render_" + profile
    result["release_partition"] = "predicate_extension"
    result.setdefault(
        "independence_role",
        "primary_simulator_render"
        if profile == "simulator_lighting"
        else "correlated_simulator_replica",
    )
    return result


def render(row, profile, transforms, data):
    light, strong, parameters = transforms
    unit = row["causal_state_unit"]
    output = group_path(row) / unit / profile
    cache = data / output / "construction.json"
    source_hashes = {
        r: sha256(data / row["variants"][r]["image"]) for r in ("anchor", "causal")
    }

    def transformed_box(box, role):
        if profile != "digital_view_scale":
            return box
        with Image.open(data / row["variants"]["anchor"]["image"]) as source:
            width, height = source.size
        scale = (
            parameters["nuisance_scale"]
            if role == "nuisance"
            else parameters["context_scale"]
        )
        new_width, new_height = round(width * scale), round(height * scale)
        dx, dy = (width - new_width) // 2, (height - new_height) // 2
        return [
            int(box[0] * new_width / width + dx),
            int(box[1] * new_height / height + dy),
            int(box[2] * new_width / width + dx),
            int(box[3] * new_height / height + dy),
        ]

    if profile == "digital_view_scale":
        with Image.open(data / row["variants"]["anchor"]["image"]) as source:
            image_area = source.width * source.height
        tier = row["certification_tier"]
        minimum = (
            (0.0015 if row["predicate"] == "isCooked" else 0.005)
            if tier == "extended-small-visible-v1"
            else 0.0015
            if tier == "extension-intervention-trace-v1"
            else 0.01
        )
        for role in ROLES:
            b = transformed_box(
                row["variants"]["anchor" if role == "nuisance" else role]["validation"][
                    "predicate_bbox"
                ],
                role,
            )
            if (b[2] - b[0]) * (b[3] - b[1]) / image_area < minimum:
                return None
    cached = load(cache) if cache.exists() else None
    if cached and (
        cached["source_image_hashes"] != source_hashes
        or cached["parameters"] != parameters
    ):
        raise ValueError(f"Stale nuisance construction: {cache}; use a fresh output")
    outputs_exist = all((data / output / f"{r}.png").exists() for r in ROLES)
    if cached and outputs_exist:
        images = None
        causal_rms, nuisance_rms = (
            cached["causal_target_union_bbox_rms"],
            cached["nuisance_image_rms"],
        )
    else:
        original = {
            r: Image.open(data / row["variants"][r]["image"]).convert("RGB")
            for r in ("anchor", "causal")
        }
        images = {
            "anchor": light(original["anchor"]),
            "causal": light(original["causal"]),
            "nuisance": strong(original["anchor"]),
        }
        boxes = [
            transformed_box(row["variants"][r]["validation"]["predicate_bbox"], r)
            for r in ("anchor", "causal")
        ]
        causal_rms = _crop_rms(images["anchor"], images["causal"], boxes)
        nuisance_rms = _rms(images["anchor"], images["nuisance"])
    if causal_rms < 1 or nuisance_rms < 1:
        return None
    result = copy.deepcopy(row)
    result["triplet_id"] = unit + "__render_" + profile
    result["replica_profile"] = profile
    result["independence_role"] = "correlated_render_replica"
    result["review_status"] = "automated_checks_only"
    result["release_partition"] = (
        "nuisance_extension"
        if row.get("release_partition") != "predicate_extension"
        else "predicate_extension"
    )
    for role in ROLES:
        v = copy.deepcopy(row["variants"]["anchor" if role == "nuisance" else role])
        relative = output / f"{role}.png"
        destination = data / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if images is not None and not destination.exists():
            images[role].save(destination)
        v["name"], v["image"] = role, relative.as_posix()
        if profile == "digital_view_scale":
            v["validation"]["source_predicate_bbox"] = v["validation"]["predicate_bbox"]
            v["validation"]["predicate_bbox"] = transformed_box(
                v["validation"]["predicate_bbox"], role
            )
            b = v["validation"]["predicate_bbox"]
            with Image.open(destination) as im:
                v["validation"]["predicate_bbox_area_ratio"] = (
                    (b[2] - b[0]) * (b[3] - b[1]) / (im.width * im.height)
                )
        v["validation"]["render_replica"] = {
            "profile": profile,
            "parameters": parameters,
            "same_transform_for_anchor_and_causal": True,
            "symbolic_state_changed": False,
            "acceptable_action_branch_changed": False,
            "causal_target_union_bbox_rms": causal_rms,
            "nuisance_image_rms": nuisance_rms,
        }
        result["variants"][role] = v
    write_json(
        cache,
        {
            "profile": profile,
            "parameters": parameters,
            "source_image_hashes": source_hashes,
            "causal_target_union_bbox_rms": causal_rms,
            "nuisance_image_rms": nuisance_rms,
        },
    )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--legacy-benchmark", type=Path, required=True)
    p.add_argument("--legacy-states", type=Path, required=True)
    p.add_argument("--extension", type=Path, nargs="+")
    p.add_argument("--additional-triplets", type=Path, nargs="+")
    p.add_argument("--paper", type=Path, required=True)
    args = p.parse_args()
    data = args.root / "data"
    legacy = load(args.legacy_benchmark)["triplets"]
    legacy_states = load(args.legacy_states)["triplets"]
    if len(legacy) != 996 or len(legacy_states) != 71:
        raise ValueError("Archived release must retain all 996 triplets and 71 units")
    identity_path = data / "paper_identity.json"
    if not identity_path.exists():
        identity = []
        for row in legacy:
            identity.append(
                {
                    "triplet_id": row["triplet_id"],
                    "causal_state_unit": row["causal_state_unit"],
                    "variants": {
                        r: {
                            "original_image": row["variants"][r]["image"],
                            "sha256": sha256(data / row["variants"][r]["image"]),
                            "acceptable_actions": row["variants"][r][
                                "acceptable_actions"
                            ],
                        }
                        for r in ROLES
                    },
                }
            )
        write_json(
            identity_path,
            {
                "source_commit": "4946240",
                "paper_sha256": sha256(args.paper),
                "source_benchmark_sha256": sha256(args.legacy_benchmark),
                "source_states_sha256": sha256(args.legacy_states),
                "triplets": identity,
            },
        )
    existing = [normalize_existing(r, data) for r in legacy]
    states = [normalize_existing(r, data, state=True) for r in legacy_states]
    write_json(
        data / "benchmark.json",
        {
            "format_version": 1,
            "release_scope": "archived_996_triplets",
            "triplets": existing,
        },
    )
    write_json(
        data / "causal_states.json",
        {"format_version": 1, "release_scope": "archived_71_units", "triplets": states},
    )
    existing = [align_contract(r) for r in existing]
    states = [align_contract(r) for r in states]
    new_units = [
        import_unit(r, source.parent, data)
        for source in (args.extension or [])
        for r in load(source)["triplets"]
    ]
    new_units = list({r["causal_state_unit"]: r for r in new_units}.values())
    primary_signal = {}
    for base in [*states, *new_units]:
        images = {
            r: Image.open(data / base["variants"][r]["image"]).convert("RGB")
            for r in ("anchor", "causal")
        }
        boxes = [
            base["variants"][r]["validation"]["predicate_bbox"]
            for r in ("anchor", "causal")
        ]
        rms = _crop_rms(images["anchor"], images["causal"], boxes)
        primary_signal[base["causal_state_unit"]] = {
            "causal_target_union_bbox_rms": rms,
            "quality_flags": ["weak_primary_causal_visual_signal"] if rms < 1 else [],
            "interpretation": "Primary image cue strength; archived strict tier has no RMS>=1 requirement. Flags do not remove archived data.",
        }
    rows = list(existing)
    rejected = Counter()
    for index, base in enumerate([*states, *new_units]):
        if base in new_units:
            rows.append(base)
        profiles = (
            {**LARGE_PROFILES, **EXTRA_PROFILES}
            if base in new_units
            else EXTRA_PROFILES
        )
        for name, transform in profiles.items():
            item = render(base, name, transform, data)
            if item:
                rows.append(item)
            else:
                rejected[name] += 1
        print(f"rendered {index + 1}/{len(states) + len(new_units)}", flush=True)
    unit_ids = {r["causal_state_unit"] for r in rows}
    for source in args.additional_triplets or []:
        for record in load(source)["triplets"]:
            if record["causal_state_unit"] not in unit_ids:
                raise ValueError(
                    "Additional nuisance triplet must reference an existing causal-state unit"
                )
            rows.append(import_unit(record, source.parent, data))
    for row in rows:
        row["primary_visual_signal"] = primary_signal[row["causal_state_unit"]]
    ids = [r["triplet_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate triplet ID")
    grouped = defaultdict(list)
    for row in rows:
        grouped[group_path(row)].append(compact(row, data))
    catalog = []
    for folder, group in sorted(grouped.items()):
        path = data / folder / "triplets.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(
                json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
                for r in sorted(group, key=lambda r: r["triplet_id"])
            )
        )
        catalog.append(
            {
                "manifest": path.relative_to(data).as_posix(),
                "scenario_aspect": group[0]["scenario_aspect"],
                "predicate": group[0]["predicate"],
                "scene": group[0]["scene"],
                "physical_scene_type": group[0]["physical_scene_type"],
                "triplets": len(group),
                "causal_state_units": len({r["causal_state_unit"] for r in group}),
            }
        )
        unit_groups = defaultdict(list)
        for row in group:
            unit_groups[row["causal_state_unit"]].append(row)
        text = f"# {group[0]['scene']} / {group[0]['predicate']}\n\n{len(group)} triplets from {len(unit_groups)} causal-state units.\n\n"
        text += "All paths in [triplets.jsonl](triplets.jsonl) resolve relative to top-level `data/`.\n\n"
        text += "| Causal-state unit | Task | Profiles | Tier |\n| --- | --- | ---: | --- |\n"
        for unit, items in sorted(unit_groups.items()):
            text += f"| [{unit}]({quote(unit, safe='')}/) | {items[0]['task_type']} | {len(items)} | {items[0]['certification_tier']} |\n"
        (data / folder / "README.md").write_text(text)
    all_units = {r["causal_state_unit"]: r for r in rows}
    counts = {
        "causal_state_units": len(all_units),
        "triplets": len(rows),
        "observations": len(rows) * 3,
        "physical_scenes": len({r["scene"] for r in rows}),
        "weak_primary_visual_signal_units": sum(
            bool(v["quality_flags"]) for v in primary_signal.values()
        ),
        "states_by_predicate": dict(
            sorted(Counter(r["predicate"] for r in all_units.values()).items())
        ),
        "triplets_by_predicate": dict(
            sorted(Counter(r["predicate"] for r in rows).items())
        ),
        "triplets_by_profile": dict(
            sorted(
                Counter(
                    r.get("replica_profile", "simulator_lighting") for r in rows
                ).items()
            )
        ),
        "states_by_certification_tier": dict(
            sorted(Counter(r["certification_tier"] for r in all_units.values()).items())
        ),
        "triplets_by_partition": dict(
            sorted(Counter(compact_partition(r) for r in rows).items())
        ),
        "paper_archive": {
            "causal_state_units": 71,
            "triplets": 996,
            "observations": 2988,
            "implemented_predicates": sorted({r["predicate"] for r in legacy}),
        },
    }
    write_json(data / "counts.json", counts)
    write_json(
        data / "catalog.json",
        {
            "format_version": 2,
            "release_version": "2026.10.19",
            "path_base": "data",
            "taxonomy": ASPECTS,
            "counts": counts,
            "shards": catalog,
        },
    )
    for predicate, aspect in ASPECTS.items():
        selected = [s for s in catalog if s["predicate"] == predicate]
        folder = data / "scenarios" / aspect / predicate
        folder.mkdir(parents=True, exist_ok=True)
        text = f"# {predicate}\n\nSemantic scenario: `{aspect}`.\n\n| Physical scene | Room | Units | Triplets |\n| --- | --- | ---: | ---: |\n"
        for shard in selected:
            text += f"| [{shard['scene']}]({shard['scene']}/) | {shard['physical_scene_type']} | {shard['causal_state_units']} | {shard['triplets']} |\n"
        (folder / "README.md").write_text(text)
    for aspect in sorted(set(ASPECTS.values())):
        text = (
            f"# {aspect}\n\n| Predicate | Units | Triplets |\n| --- | ---: | ---: |\n"
        )
        for pred in [p for p, a in ASPECTS.items() if a == aspect]:
            text += f"| [{pred}]({pred}/) | {counts['states_by_predicate'].get(pred, 0)} | {counts['triplets_by_predicate'].get(pred, 0)} |\n"
        (data / "scenarios" / aspect / "README.md").write_text(text)
    text = f"# Classified benchmark\n\n{counts['causal_state_units']} units, {counts['triplets']} triplets, {counts['observations']} observations across {counts['physical_scenes']} physical scenes.\n\n"
    text += "| Predicate | Archived units | All units | All triplets |\n| --- | ---: | ---: | ---: |\n"
    archived_counts = Counter(r["predicate"] for r in legacy_states)
    for pred, aspect in ASPECTS.items():
        text += f"| [{pred}](scenarios/{aspect}/{pred}/) | {archived_counts[pred]} | {counts['states_by_predicate'].get(pred, 0)} | {counts['triplets_by_predicate'].get(pred, 0)} |\n"
    text += "\nThe archived 996 triplets remain intact. Extension tiers and review status are explicit; trace-derived consumables are excluded by the default loader. See [the dataset card](../docs/dataset_card.md).\n"
    (data / "README.md").write_text(text)
    # Group by physical scene so replicas and same-scene checkpoints cannot cross splits.
    import hashlib

    scenes = sorted(
        {r["scene"] for r in rows},
        key=lambda s: hashlib.sha256(
            ("CausalPilot-20261019:" + s).encode()
        ).hexdigest(),
    )
    size = max(1, len(scenes) // 10)
    assignments = {
        s: "holdout" if i < size else "validation" if i < 2 * size else "development"
        for i, s in enumerate(scenes)
    }
    write_json(
        data / "splits.json",
        {
            "seed": "CausalPilot-20261019",
            "grouping": "physical_scene",
            "purpose": "optional construction/review partitions; not the paper evaluation split",
            "scene_assignments": assignments,
            "unit_assignments": {
                u: assignments[r["scene"]] for u, r in sorted(all_units.items())
            },
        },
    )
    write_json(
        data / "unit_index.json",
        {
            "units": [
                {
                    "causal_state_unit": u,
                    "predicate": r["predicate"],
                    "scene": r["scene"],
                    "certification_tier": r["certification_tier"],
                    "manifest": (group_path(r) / "triplets.jsonl").as_posix(),
                }
                for u, r in sorted(all_units.items())
            ]
        },
    )
    print(json.dumps(counts, indent=2))


def compact_partition(row):
    return (
        "predicate_extension"
        if row["causal_state_unit"].startswith("extension__")
        else row.get("release_partition", "paper_archive")
    )


if __name__ == "__main__":
    main()
