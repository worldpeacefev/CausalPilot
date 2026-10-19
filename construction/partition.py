"""Build disjoint paper-candidate / extended views from a reviewed ID selection.

An eligibility file is external review input, not an inference/evaluation result.
It must contain eligible_triplet_ids and agent_primary_visual_review_units.
This assembler never certifies a record just because a quota needs filling.
"""

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

from construction.common import ASPECTS, write_json
from construction.review import load_reviews, verify_binding


def counts(rows):
    # Source/screened rows precede bulk replicas. A bulk replica's weaker
    # output contract must not overwrite the originating unit's source tier.
    units = {}
    for row in rows:
        units.setdefault(row["causal_state_unit"], row)
    return {
        "causal_state_units": len(units),
        "triplets": len(rows),
        "observations": 3 * len(rows),
        "physical_scenes": len({r["scene"] for r in rows}),
        "states_by_predicate": dict(
            sorted(Counter(r["predicate"] for r in units.values()).items())
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
            sorted(Counter(r["certification_tier"] for r in units.values()).items())
        ),
        "triplets_by_partition": dict(
            sorted(Counter(r["release_partition"] for r in rows).items())
        ),
        "triplets_by_certification_tier": dict(
            sorted(Counter(r["certification_tier"] for r in rows).items())
        ),
    }


def read_rows(data, catalog):
    for shard in catalog["shards"]:
        with (data / shard["manifest"]).open() as stream:
            yield from (json.loads(line) for line in stream)


def write_view(data, name, rows, bulk_shards=()):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["scenario_aspect"], row["predicate"], row["scene"])].append(row)
    shards = []
    for (aspect, predicate, scene), group in sorted(groups.items()):
        path = Path(name) / "scenarios" / aspect / predicate / scene / "triplets.jsonl"
        target = data / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "".join(
                json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
                for r in sorted(group, key=lambda r: r["triplet_id"])
            )
        )
        shards.append(
            {
                "manifest": path.as_posix(),
                "scenario_aspect": aspect,
                "predicate": predicate,
                "scene": scene,
                "physical_scene_type": group[0]["physical_scene_type"],
                "triplets": len(group),
                "causal_state_units": len({r["causal_state_unit"] for r in group}),
            }
        )
    shards.extend(bulk_shards)
    # Drop obsolete collection views after membership changes. Shared source
    # assets and bulk images are retained; a rejected unit must not survive as
    # an unlisted but apparently approved paper manifest.
    view_root = data / name / "scenarios"
    keep = {(data / s["manifest"]).resolve() for s in shards}
    active_dirs = {
        parent
        for path in keep
        for parent in path.parents
        if parent.is_relative_to(view_root.resolve())
    }
    for old in view_root.rglob("triplets.jsonl"):
        if old.resolve() not in keep:
            old.unlink()
    for old in view_root.rglob("README.md"):
        if old.parent.resolve() not in active_dirs:
            old.unlink()
    for folder in sorted(
        (p for p in view_root.rglob("*") if p.is_dir()),
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        if not any(folder.iterdir()):
            folder.rmdir()
    # Every collection can be browsed independently by scenario/type/scene.
    for aspect in sorted({s["scenario_aspect"] for s in shards}):
        aspect_path = data / name / "scenarios" / aspect
        aspect_path.mkdir(parents=True, exist_ok=True)
        (aspect_path / "README.md").write_text(
            f"# {aspect}\n\n"
            + "\n".join(
                f"- [{p}]({p}/)"
                for p in sorted(
                    {s["predicate"] for s in shards if s["scenario_aspect"] == aspect}
                )
            )
            + "\n"
        )
        for predicate in sorted(
            {s["predicate"] for s in shards if s["scenario_aspect"] == aspect}
        ):
            pred_shards = [s for s in shards if s["predicate"] == predicate]
            pred_path = aspect_path / predicate
            pred_path.mkdir(parents=True, exist_ok=True)
            (pred_path / "README.md").write_text(
                f"# {predicate}\n\n"
                + "\n".join(
                    f"- [{scene}]({scene}/)"
                    for scene in sorted({s["scene"] for s in pred_shards})
                )
                + "\n"
            )
            for scene in sorted({s["scene"] for s in pred_shards}):
                scene_path = pred_path / scene
                manifests = [s for s in pred_shards if s["scene"] == scene]
                (scene_path / "README.md").write_text(
                    f"# {scene} / {predicate}\n\nAll record paths resolve relative to top-level `data/`.\n\n"
                    + "\n".join(
                        f"- [{Path(s['manifest']).relative_to(scene_path.relative_to(data))}]({quote(Path(s['manifest']).relative_to(scene_path.relative_to(data)).as_posix(), safe='/')}) ({s['triplets']} triplets)"
                        for s in manifests
                    )
                    + "\n"
                )
    return shards


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--eligibility", type=Path, required=True)
    parser.add_argument(
        "--visual-review",
        type=Path,
        help="Curated proxy review JSONL under data/; defaults to the distributed review if present",
    )
    args = parser.parse_args()
    data = args.root / "data"
    source = data / "source_catalog.json"
    if not source.exists():
        source.write_bytes((data / "catalog.json").read_bytes())
    source_catalog = json.loads(source.read_text())
    original = list(read_rows(data, source_catalog))
    visual_path = args.visual_review or data / "review/proxy_visual_review.jsonl"
    if args.visual_review and not visual_path.is_file():
        raise FileNotFoundError(visual_path)
    reviews = load_reviews(visual_path) if visual_path.exists() else {}
    if visual_path.exists() and not reviews:
        raise ValueError("An existing visual review file must not be empty")
    visual_ref = (
        visual_path.resolve().relative_to(data.resolve()).as_posix()
        if reviews
        else None
    )
    source_ids = {r["triplet_id"] for r in original}
    if set(reviews) - source_ids:
        raise ValueError("Visual review references unknown source triplets")
    for row in original:
        if row["triplet_id"] in reviews:
            verify_binding(row, reviews[row["triplet_id"]], data)
    selection = json.loads(args.eligibility.read_text())
    eligible = set(selection["eligible_triplet_ids"])
    reviewed = set(selection["agent_primary_visual_review_units"])
    pool = [
        r
        for r in original
        if r["triplet_id"] in eligible
        and r["causal_state_unit"] in reviewed
        and (not reviews or reviews.get(r["triplet_id"], {}).get("verdict") == "pass")
    ]
    if eligible - {r["triplet_id"] for r in original}:
        raise ValueError("Eligibility references unknown triplets")
    # Require an eligible primary for each selected unit; cap units at paper scale.
    primaries = {
        r["causal_state_unit"]: r
        for r in pool
        if r.get("replica_profile", "simulator_lighting") == "simulator_lighting"
    }
    unit_ids = set(
        sorted(primaries, key=lambda u: (u.startswith("extension__"), u))[:71]
    )
    paper = [primaries[u] for u in sorted(unit_ids)]
    primary_ids = {r["triplet_id"] for r in paper}
    remaining = [
        r
        for r in pool
        if r["causal_state_unit"] in unit_ids and r["triplet_id"] not in primary_ids
    ]
    # Preserve as much archived population as controls allow; never consult models.
    remaining.sort(
        key=lambda r: (
            r["release_partition"] != "paper_archive",
            r.get("replica_profile", ""),
            r["causal_state_unit"],
        )
    )
    paper += remaining[: max(0, 996 - len(paper))]
    paper_ids = {r["triplet_id"] for r in paper}
    extended = [copy.deepcopy(r) for r in original if r["triplet_id"] not in paper_ids]
    paper = copy.deepcopy(paper)
    for r in paper:
        r["collection"] = "paper"
        r["review_status"] = (
            "automated_controls_and_agent_triplet_visual_review"
            if reviews
            else "automated_controls_and_agent_primary_visual_review"
        )
        r["paper_control_eligible"] = True
        r["human_review_of_this_triplet_verified"] = False
    for r in extended:
        r["collection"] = "extended"
    for r in paper + extended:
        review = reviews.get(r["triplet_id"])
        if review:
            r["proxy_visual_review"] = {
                "record_manifest": visual_ref,
                "verdict": review["verdict"],
                "reason_codes": review["reason_codes"],
                "reviewer_kind": "agent",
                "actual_human_review": False,
            }
            if r["collection"] == "extended":
                r["source_review_status"] = r["review_status"]
                r["review_status"] = "agent_visual_review_" + review["verdict"]
    bulk_catalog = json.loads((data / "extended/bulk_catalog.json").read_text())
    bulk = list(read_rows(data, bulk_catalog))
    all_rows = paper + extended + bulk
    ids = [r["triplet_id"] for r in all_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Collections contain duplicate triplet IDs")
    paper_counts, extended_counts = counts(paper), counts(extended + bulk)
    missing = sorted(set(ASPECTS) - {r["predicate"] for r in paper})
    requirements = {
        "reference": "all-finish.pdf, Table 4 and Appendix A.2",
        "paper_sha256": json.loads((data / "paper_identity.json").read_text())[
            "paper_sha256"
        ],
        "required": {
            "causal_state_units": 71,
            "triplets": 996,
            "observations": 2988,
            "predicates": sorted(ASPECTS),
        },
        "actual": paper_counts,
        "membership": {
            "historical_triplets_selected": sum(
                r["release_partition"] == "paper_archive" for r in paper
            ),
            "additional_triplets_selected": sum(
                r["release_partition"] != "paper_archive" for r in paper
            ),
            "historical_triplets_preserved_across_full_release": 996,
        },
        "fully_conforms_to_paper": False,
        "status": "incomplete_paper_requirements",
        "missing_predicates": missing,
        "unit_shortfall": max(0, 71 - paper_counts["causal_state_units"]),
        "triplet_shortfall": max(0, 996 - paper_counts["triplets"]),
        "review": {
            "reviewer": "Codex agent",
            "date": "2026-10-19",
            "all_selected_triplets": "raw metadata, exact symbolic controls, action sets, target visibility, bbox thresholds and image signals checked",
            "primary_triplets": "agent visual review of every selected causal-state unit primary",
            "human_review_of_every_generated_triplet_verified": False,
            "selection_uses_tested_model_outputs": False,
        },
        "blockers": {
            "isFilledWithLiquid": "Native filling also changes fillLiquid; Table 4 Boolean-only rule does not authorize this extra change.",
            "isSliced": "Native slicing creates descendants and changes reciprocal containment; original three tiers do not authorize topology changes.",
            "isUsedUp": "AI2-THOR 5.0.0 reports constant false through raw metadata accessor; trace-derived extension is not raw predicate certification.",
            "manual_review": "Paper requires manual review of every generated triplet. Agent primary visual review and automated per-triplet checks are recorded honestly; human review is not claimed.",
        },
        "excluded_legacy_unit": {
            "causal_state_unit": "istoggled__turn_on_tv__Confirmatory_FloorPlan224_C1",
            "reason": "Side view of television hides on/off display; weak primary causal visual signal. Retained in extended/archive only.",
        },
        "interpretation": "A reviewed candidate collection at paper scale where possible, not the original experimental cohort or a claim of seven-predicate compliance.",
    }
    if reviews:
        rejected = [
            r
            for r in original
            if reviews.get(r["triplet_id"], {}).get("verdict") != "pass"
            and r["triplet_id"] in reviews
        ]
        primary_rejections = [
            r["causal_state_unit"]
            for r in rejected
            if r.get("replica_profile", "simulator_lighting") == "simulator_lighting"
        ]
        requirements["status"] = "proxy_review_complete_predicate_coverage_incomplete"
        requirements["review"].update(
            {
                "mode": "single_agent_simulating_manual_review_at_user_request",
                "proxy_visual_review_complete": True,
                "all_selected_triplets_visually_accepted": len(paper),
                "proxy_reviewed_candidate_triplets": len(reviews),
                "proxy_review_verdicts": dict(
                    Counter(r["verdict"] for r in reviews.values())
                ),
                "record_manifest": visual_ref,
                "inspection": "Every candidate shown in matched A/C/N contact sheets: target crops for the original cohort, full frames for fresh television candidates. All faucet candidates additionally shown as full frames; flagged cases re-inspected using full frames and native target crops. Context and action labels checked for each originating unit.",
                "independent_human_reviewer_count": 0,
            }
        )
        requirements["blockers"].pop("manual_review")
        requirements["blockers"]["human_review_provenance"] = (
            "The user requested a simulated manual process. Full triplet proxy review is complete; this is agent review, not evidence of actual human review in the manuscript."
        )
        requirements["visual_exclusions"] = {
            "triplets": len(rejected),
            "primary_rejected_units": primary_rejections,
            "reason_counts": dict(
                Counter(
                    code
                    for row in rejected
                    for code in reviews[row["triplet_id"]]["reason_codes"]
                )
            ),
            "preservation": "Excluded observations remain in extended/archive; no archived data removed.",
        }
    write_json(data / "paper/requirements.json", requirements)
    paper_shards = write_view(data, "paper", paper)
    extended_shards = write_view(data, "extended", extended, bulk_catalog["shards"])
    for name, c, shards in (
        ("paper", paper_counts, paper_shards),
        ("extended", extended_counts, extended_shards),
    ):
        write_json(
            data / name / "catalog.json",
            {
                "format_version": 2,
                "path_base": "data",
                "collection": name,
                "taxonomy": ASPECTS,
                "counts": c,
                "shards": shards,
                "fully_conforms_to_paper": False,
            },
        )
    full_counts = counts(all_rows)
    full_counts["weak_primary_visual_signal_units"] = len(
        {
            r["causal_state_unit"]
            for r in original
            if r.get("primary_visual_signal", {}).get("quality_flags")
        }
    )
    full_counts["collections"] = {"paper": paper_counts, "extended": extended_counts}
    full_counts["paper_archive"] = source_catalog["counts"]["paper_archive"]
    write_json(data / "counts.json", full_counts)
    write_json(
        data / "catalog.json",
        {
            "format_version": 2,
            "release_version": "2026.10.19-split",
            "path_base": "data",
            "taxonomy": ASPECTS,
            "counts": full_counts,
            "collection_catalogs": {
                "paper": "paper/catalog.json",
                "extended": "extended/catalog.json",
            },
            "shards": paper_shards + extended_shards,
        },
    )
    index = json.loads((data / "unit_index.json").read_text())
    paper_unit_counts = Counter(r["causal_state_unit"] for r in paper)
    extended_unit_counts = Counter(r["causal_state_unit"] for r in extended + bulk)
    for unit in index["units"]:
        uid = unit["causal_state_unit"]
        unit["collection_triplets"] = {
            "paper": paper_unit_counts[uid],
            "extended": extended_unit_counts[uid],
        }
    write_json(data / "unit_index.json", index)
    lines = [
        "# Explore the collections",
        "",
        "Browse matched decision checkpoints by semantic aspect, predicate, and FloorPlan.",
        "All observation paths resolve relative to the top-level `data/` directory.",
        "",
        "| Collection | Units | Triplets | Observations | Predicates |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, c in (("paper", paper_counts), ("extended", extended_counts)):
        label = "Curated core (`paper`)" if name == "paper" else "Expanded (`extended`)"
        lines.append(
            f"| [{label}]({name}/) | {c['causal_state_units']:,} | {c['triplets']:,} | {c['observations']:,} | {len(c['states_by_predicate'])} |"
        )
    lines += [
        "",
        "Triplet IDs are disjoint; units overlap because replicas retain their source-unit identity.",
        (
            "Every selected core triplet has passing agent visual-review provenance."
            if reviews
            else "Core records retain their per-record review status; see the collection requirements for review coverage."
        ),
        f"The combined release contains **{full_counts['causal_state_units']:,} source units / {full_counts['triplets']:,} triplets / {full_counts['observations']:,} observations**.",
        "",
        "[Dataset card](../docs/dataset_card.md) · [Schema and metrics](../docs/schema.md) ·",
        "[Core requirements](paper/requirements.json) · [Curation records](review/)",
        "",
        "The preserved [reference archive](benchmark.json) contains the original 996-triplet cohort.",
        "Use the collection catalogs for the curated and expanded views, with collection-specific",
        "evidence and review contracts described in the dataset card.",
    ]
    (data / "README.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "paper": paper_counts,
                "extended": extended_counts,
                "missing_predicates": missing,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
