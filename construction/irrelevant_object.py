"""Certify a decision-preserving, task-irrelevant empty-container change."""

import argparse
import copy
import json
from pathlib import Path

from PIL import Image

from AsgardBench.causal.render_replicas import _rms
from construction.common import certification_diff, diff, write_json
from construction.expand_simulator import BUILD, bbox, prepare, step


def main():
    from ai2thor.controller import Controller
    from ai2thor.platform import Linux64

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset.parent
    rows = json.loads(args.dataset.read_text())["triplets"]
    row = next(
        r
        for r in rows
        if r["predicate"] == "isFilledWithLiquid"
        and r["target"]["predicate_object_type"] == "Mug"
    )
    recipe = json.loads((root / row["source_plan"]).read_text())
    anchor = json.loads((root / row["variants"]["anchor"]["metadata"]).read_text())
    poses = recipe.get("object_poses") or [
        {"objectName": o["name"], "position": o["position"], "rotation": o["rotation"]}
        for o in anchor["objects"]
        if o["pickupable"] or o["moveable"]
    ]
    # Empty visible drawers cannot move contained task-relevant objects.
    candidates = [
        o
        for o in anchor["objects"]
        if o["objectType"] == "Drawer"
        and o["visible"]
        and not o["isOpen"]
        and not o.get("receptacleObjectIds")
    ]
    controller = Controller(
        scene=row["scene"],
        width=512,
        height=512,
        quality="High",
        platform=Linux64,
        renderInstanceSegmentation=True,
        visibilityDistance=2,
        fieldOfView=60,
        snapToGrid=False,
        server_timeout=90,
        commit_id=BUILD,
    )
    tolerance = {"position": 1e-5, "rotation": 1e-3}
    accepted = []
    try:
        for distractor in candidates:
            target, _, _ = prepare(
                controller,
                row["scene"],
                "Mug",
                row["predicate"],
                name=row["target"]["predicate_object_name"],
                pose=recipe["inspection_pose"],
                object_poses=poses,
            )
            before = copy.deepcopy(controller.last_event.metadata)
            # Reconstructed Anchor must match the released one first.
            from construction.common import snapshot

            baseline_diffs = certification_diff(
                snapshot(anchor), snapshot(before), tolerance
            )
            if baseline_diffs:
                print(
                    "REJECT",
                    distractor["name"],
                    "reconstruction",
                    baseline_diffs,
                    flush=True,
                )
                continue
            event = step(
                controller,
                "OpenObject",
                objectId=distractor["objectId"],
                openness=1,
                forceAction=True,
            )
            after = copy.deepcopy(event.metadata)
            changes = certification_diff(snapshot(anchor), snapshot(after), tolerance)
            prefix = f"objects.{distractor['name']}."
            if f"{prefix}isOpen" not in changes or any(
                p != f"{prefix}isOpen"
                and not p.startswith((prefix + "position.", prefix + "rotation."))
                for p in changes
            ):
                print(
                    "REJECT",
                    distractor["name"],
                    "uncontrolled changes",
                    changes,
                    flush=True,
                )
                continue
            post_target = next(
                o for o in after["objects"] if o["name"] == target["name"]
            )
            if (
                not post_target["visible"]
                or post_target[row["predicate"]]
                != row["variants"]["anchor"]["validation"]["predicate_value"]
            ):
                print(
                    "REJECT", distractor["name"], "target visibility/state", flush=True
                )
                continue
            box = bbox(event, [target["objectId"]])
            if (box[2] - box[0]) * (box[3] - box[1]) / (512 * 512) < 0.01:
                print("REJECT", distractor["name"], "target box too small", flush=True)
                continue
            image = Image.fromarray(event.frame)
            with Image.open(root / row["variants"]["anchor"]["image"]) as source:
                rms = _rms(source, image)
            if rms < 1:
                print(
                    "REJECT",
                    distractor["name"],
                    "insufficient image difference",
                    rms,
                    flush=True,
                )
                continue
            unit = row["causal_state_unit"]
            folder = args.output / unit
            folder.mkdir(parents=True, exist_ok=True)
            result = copy.deepcopy(row)
            result["triplet_id"] = unit + "__render_simulator_irrelevant_object"
            result["replica_profile"] = "simulator_irrelevant_object"
            result["review_status"] = "automated_checks_only"
            result["validation"]["numeric_geometry_tolerance"] = tolerance
            result["validation"]["nuisance_allowed_symbolic_differences"] = changes
            result["validation"]["nuisance_symbolic_differences"] = changes
            result["validation"]["nuisance_image_rms"] = rms
            result["validation"]["task_relevant_state_unchanged"] = True
            result["validation"]["irrelevant_object"] = {
                "name": distractor["name"],
                "objectType": "Drawer",
                "empty_before_intervention": True,
                "not_in_instruction_or_action_history": True,
                "raw_symbolic_differences": diff(snapshot(anchor), snapshot(after)),
            }
            for role in ("anchor", "causal"):
                import shutil

                for field, suffix in (("image", "png"), ("metadata", "metadata.json")):
                    path = f"{unit}/{role}.{suffix}"
                    shutil.copy2(
                        root / row["variants"][role][field], args.output / path
                    )
                    result["variants"][role][field] = path
            v = copy.deepcopy(result["variants"]["anchor"])
            v.update(
                name="nuisance",
                image=f"{unit}/nuisance.png",
                metadata=f"{unit}/nuisance.metadata.json",
                symbolic_state=snapshot(after),
            )
            v["validation"]["predicate_bbox"] = box
            v["validation"]["predicate_bbox_area_ratio"] = (
                (box[2] - box[0]) * (box[3] - box[1]) / (512 * 512)
            )
            result["variants"]["nuisance"] = v
            image.save(folder / "nuisance.png")
            write_json(folder / "nuisance.metadata.json", after)
            recipe["irrelevant_nuisance_intervention"] = {
                "action": "OpenObject",
                "objectId": distractor["objectId"],
                "openness": 1,
                "forceAction": True,
            }
            recipe["object_poses"] = poses
            recipe["nuisance_contract"] = (
                "Only this empty, task-irrelevant drawer's open state and own articulation may differ."
            )
            result["source_plan"] = f"{unit}/recipe.json"
            write_json(folder / "recipe.json", recipe)
            accepted.append(result)
            print("ACCEPT", result["triplet_id"], changes, flush=True)
            break
    finally:
        controller.stop()
    write_json(
        args.output / "dataset.json", {"format_version": 2, "triplets": accepted}
    )
    if not accepted:
        raise RuntimeError("No task-irrelevant change met the complete contract")


if __name__ == "__main__":
    main()
