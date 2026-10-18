"""Fresh scene reconstructions for missing predicate families.

The extension explicitly records target topology, liquid-content coupling,
and AI2-THOR's unimplemented isUsedUp metadata accessor. Raw metadata is never
rewritten. Use-up labels are trace-derived and kept in a separate tier.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from construction.common import ROLES, certification_diff, diff, snapshot, write_json

BUILD = "f0825767cd50d69f666c7f282e54abfe58f1e917"
INTERFACE = [
    "clean OBJECT",
    "close OBJECT",
    "cook OBJECT",
    "empty OBJECT",
    "drink OBJECT",
    "find OBJECT",
    "open OBJECT",
    "pickup OBJECT",
    "put DESTINATION_RECEPTACLE",
    "slice OBJECT",
    "spray OBJECT",
    "toggle_on OBJECT",
    "toggle_off OBJECT",
    "done none",
]


def step(controller, action, **kwargs):
    event = controller.step(action=action, **kwargs)
    if not event.metadata["lastActionSuccess"]:
        raise ValueError(f"{action}: {event.metadata['errorMessage']}")
    return event


def object_by_name(controller, name):
    return next(
        o for o in controller.last_event.metadata["objects"] if o["name"] == name
    )


def bbox(event, ids):
    boxes = [
        event.instance_detections2D[i] for i in ids if i in event.instance_detections2D
    ]
    if not boxes:
        raise ValueError("No target instance pixels")
    return [
        int(min(b[0] for b in boxes)),
        int(min(b[1] for b in boxes)),
        int(max(b[2] for b in boxes)),
        int(max(b[3] for b in boxes)),
    ]


def choose_pose(controller, target):
    center = target["axisAlignedBoundingBox"]["center"]
    positions = step(controller, "GetReachablePositions").metadata["actionReturn"]

    def distance(p):
        return math.hypot(p["x"] - center["x"], p["z"] - center["z"])

    positions = sorted(
        (p for p in positions if 0.35 < distance(p) < 1.4),
        key=lambda p: (abs(distance(p) - 0.50), p["x"], p["z"]),
    )
    for p in positions[:32]:
        yaw = math.degrees(math.atan2(center["x"] - p["x"], center["z"] - p["z"])) % 360
        horizon = min(
            60,
            max(
                -30, math.degrees(math.atan2(p["y"] + 0.675 - center["y"], distance(p)))
            ),
        )
        pose = {
            "position": p,
            "rotation": {"x": 0, "y": yaw, "z": 0},
            "horizon": horizon,
            "standing": True,
        }
        e = controller.step(action="TeleportFull", **pose)
        if not e.metadata["lastActionSuccess"]:
            continue
        obj = object_by_name(controller, target["name"])
        try:
            b = bbox(e, [target["objectId"]])
        except ValueError:
            continue
        if (
            obj["visible"]
            and (b[2] - b[0]) * (b[3] - b[1]) / (e.frame.shape[0] * e.frame.shape[1])
            >= 0.01
        ):
            return pose
    raise ValueError("No reachable inspection pose with visible target box >=1%")


def prepare(controller, scene, typ, predicate, name=None, pose=None, object_poses=None):
    controller.reset(scene=scene)
    step(controller, "PausePhysicsAutoSim")
    if object_poses is None:
        object_poses = [
            {
                "objectName": o["name"],
                "position": copy.deepcopy(o["position"]),
                "rotation": copy.deepcopy(o["rotation"]),
            }
            for o in controller.last_event.metadata["objects"]
            if o["pickupable"] or o["moveable"]
        ]
    step(controller, "SetObjectPoses", objectPoses=object_poses)
    objects = controller.last_event.metadata["objects"]
    if predicate == "isUsedUp" and name is None:
        usable = [
            o
            for o in objects
            if o["objectType"] == typ and "Used_Up" not in o["assetId"]
        ]
        if len(usable) != 1:
            raise ValueError(
                f"Requires exactly one non-exhausted {typ} asset for an unambiguous collect/finish branch"
            )
        name = usable[0]["name"]
    target = (
        next((o for o in objects if o["name"] == name), None)
        if name
        else next((o for o in objects if o["objectType"] == typ), None)
    )
    if target is None:
        raise ValueError(f"Missing {typ}")
    if predicate == "isSliced":
        tool = next((o for o in objects if o["objectType"] == "Knife"), None)
        if tool is None:
            raise ValueError("No slicing tool")
        step(controller, "PickupObject", objectId=tool["objectId"], forceAction=True)
    if predicate == "isFilledWithLiquid" and target["isFilledWithLiquid"]:
        step(
            controller,
            "EmptyLiquidFromObject",
            objectId=target["objectId"],
            forceAction=True,
        )
    if pose is None:
        pose = choose_pose(controller, target)
    else:
        step(controller, "TeleportFull", **pose)
    return target, pose, object_poses


def create_unit(controller, root, scene, typ, predicate):
    unit = f"extension__{predicate}__{scene}__{typ}"
    folder = root / unit
    folder.mkdir(parents=True, exist_ok=True)
    target, pose, object_poses = prepare(controller, scene, typ, predicate)
    name, target_id = target["name"], target["objectId"]
    variants, traces, raw = {}, {}, {}
    for role in ROLES:
        target, _, _ = prepare(
            controller,
            scene,
            typ,
            predicate,
            name=name,
            pose=pose,
            object_poses=object_poses,
        )
        commands = []
        if role == "causal":
            action, kw = {
                "isSliced": ("SliceObject", {}),
                "isFilledWithLiquid": ("FillObjectWithLiquid", {"fillLiquid": "water"}),
                "isUsedUp": ("UseUpObject", {}),
            }[predicate]
            step(controller, action, objectId=target_id, forceAction=True, **kw)
            commands.append(
                {"action": action, "objectId": target_id, "forceAction": True, **kw}
            )
        if role == "nuisance":
            step(
                controller,
                "RandomizeLighting",
                brightness=[0.70, 0.70],
                randomizeColor=False,
                synchronized=True,
            )
            commands.append(
                {
                    "action": "RandomizeLighting",
                    "brightness": [0.70, 0.70],
                    "randomizeColor": False,
                    "synchronized": True,
                }
            )
        event = controller.last_event
        metadata = copy.deepcopy(event.metadata)
        obj = object_by_name(controller, name)
        ids = [target_id]
        if predicate == "isSliced" and role == "causal":
            ids += [
                o["objectId"]
                for o in metadata["objects"]
                if o["objectId"].startswith(target_id + "|")
            ]
        b = bbox(event, ids)
        metadata_visible = any(
            o["visible"] for o in metadata["objects"] if o["objectId"] in ids
        )
        mask_pixels = sum(
            int(event.instance_masks[i].sum()) for i in ids if i in event.instance_masks
        )
        # Exhaustion disables the full-roll raycast surfaces. Instance pixels
        # directly establish that the remaining core is rendered in this tier.
        visible = metadata_visible or (predicate == "isUsedUp" and mask_pixels >= 32)
        area = (
            (b[2] - b[0])
            * (b[3] - b[1])
            / (event.frame.shape[0] * event.frame.shape[1])
        )
        minimum_area = 0.0015 if predicate == "isUsedUp" else 0.01
        if not visible or area < minimum_area:
            raise ValueError(
                f"{role}: target/descendant union box below {minimum_area:.2%} or invisible"
            )
        # isUsedUp is a constant-false placeholder in this engine. Preserve it;
        # use a separately declared action-trace predicate, never falsify metadata.
        value = role == "causal" if predicate == "isUsedUp" else obj[predicate]
        if value != (role == "causal"):
            raise ValueError(f"{role}: predicate not attained")
        correct = (
            (
                {"action": "done", "object": "none"}
                if value
                else {"action": "slice", "object": typ}
            )
            if predicate == "isSliced"
            else (
                {"action": "empty", "object": typ}
                if value
                else {"action": "done", "object": "none"}
            )
            if predicate == "isFilledWithLiquid"
            else (
                {"action": "done", "object": "none"}
                if value
                else {"action": "pickup", "object": typ}
            )
        )
        Image.fromarray(event.frame).save(folder / f"{role}.png")
        write_json(folder / f"{role}.metadata.json", metadata)
        state = snapshot(metadata)
        raw[role] = state
        variants[role] = {
            "name": role,
            "image": f"{unit}/{role}.png",
            "metadata": f"{unit}/{role}.metadata.json",
            "acceptable_actions": [correct],
            "symbolic_state": state,
            "oracle_state": [
                f"{typ}.{predicate}={str(value).lower()}",
                "inventory: holding Knife"
                if predicate == "isSliced"
                else "inventory: empty hand",
            ],
            "validation": {
                "predicate_value": bool(value),
                "raw_metadata_predicate_value": obj[predicate],
                "predicate_source": "successful_intervention_trace"
                if predicate == "isUsedUp"
                else "simulator_metadata",
                "predicate_bbox": b,
                "predicate_bbox_area_ratio": area,
                "target_or_descendants_visible": visible,
                "minimum_bbox_area_ratio": minimum_area,
                "raw_metadata_visible": metadata_visible,
                "instance_mask_pixels": mask_pixels,
                "visibility_source": "instance_segmentation"
                if predicate == "isUsedUp"
                else "metadata_and_instance_segmentation",
                "last_action_success": True,
            },
        }
        traces[role] = commands
    tolerance = {"position": 1e-5, "rotation": 1e-3}
    differences = certification_diff(raw["anchor"], raw["causal"], tolerance)
    nuisance_differences = certification_diff(raw["anchor"], raw["nuisance"], tolerance)
    if nuisance_differences:
        raise ValueError(
            f"Reconstruction mismatch in nuisance: {nuisance_differences[:10]}"
        )
    expected = f"objects.{name}.{predicate}"
    permitted = {expected}
    tier = "pilot-v3-strict"
    if predicate == "isFilledWithLiquid":
        permitted.add(f"objects.{name}.fillLiquid")
        tier = "extension-liquid-content-v1"
    if predicate == "isSliced":
        tier = "extension-target-topology-v1"
        original_names = set(raw["anchor"]["objects"])
        children = set(raw["causal"]["objects"]) - original_names
        causal_meta = json.loads((folder / "causal.metadata.json").read_text())
        actual_children = {
            o["name"]
            for o in causal_meta["objects"]
            if o["objectId"].startswith(target_id + "|")
        }
        if children != actual_children:
            raise ValueError("Unrelated object creation during slicing")
        permitted.update(f"objects.{n}" for n in children)
        permitted.add(f"objects.{name}.parentReceptacles")
        # Only the original parent's reciprocal membership may gain slices.
        parent_ids = target.get("parentReceptacles") or []
        parent_names = {
            o["name"] for o in causal_meta["objects"] if o["objectId"] in parent_ids
        }
        permitted.update(f"objects.{n}.receptacleObjectIds" for n in parent_names)
        allowed_ids = {target_id} | {
            o["objectId"] for o in causal_meta["objects"] if o["name"] in children
        }
        for parent in parent_names:
            before = set(raw["anchor"]["objects"][parent]["receptacleObjectIds"] or [])
            after = set(raw["causal"]["objects"][parent]["receptacleObjectIds"] or [])
            if not (before ^ after).issubset(allowed_ids):
                raise ValueError("Unrelated containment change during slicing")
    if predicate == "isUsedUp":
        tier = "extension-intervention-trace-v1"
        permitted = set()
    if not set(differences).issubset(permitted):
        raise ValueError(
            f"Uncontrolled causal change: {sorted(set(differences) - permitted)[:12]}"
        )
    if predicate != "isUsedUp" and expected not in differences:
        raise ValueError("Missing causal predicate contrast")
    images = {
        r: np.asarray(Image.open(folder / f"{r}.png"), dtype=float) for r in ROLES
    }
    boxes = [variants[r]["validation"]["predicate_bbox"] for r in ("anchor", "causal")]
    x1, y1, x2, y2 = (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )
    causal_rms = float(
        np.sqrt(
            np.mean(
                (images["anchor"][y1:y2, x1:x2] - images["causal"][y1:y2, x1:x2]) ** 2
            )
        )
    )
    nuisance_rms = float(np.sqrt(np.mean((images["anchor"] - images["nuisance"]) ** 2)))
    if causal_rms < 5 or nuisance_rms < 1:
        raise ValueError(
            f"Insufficient visible contrast: causal={causal_rms}, nuisance={nuisance_rms}"
        )
    instruction = {
        "isSliced": f"Slice the {typ}. Finish when it has been sliced.",
        "isFilledWithLiquid": f"Empty the {typ} of liquid. Finish when it is empty.",
        "isUsedUp": f"Collect the {typ} for use if it has any supply remaining. Leave an exhausted one in place and finish.",
    }[predicate]
    record = {
        "triplet_id": unit,
        "causal_state_unit": unit,
        "predicate": predicate,
        "task_type": {
            "isSliced": "slice_food",
            "isFilledWithLiquid": "empty_vessel",
            "isUsedUp": "collect_usable_supply",
        }[predicate],
        "instruction": instruction,
        "scene": scene,
        "history": ["find Knife", "pickup Knife", f"find {typ}"]
        if predicate == "isSliced"
        else [f"find {typ}"],
        "action_interface": INTERFACE,
        "source_plan": f"{unit}/recipe.json",
        "variants": variants,
        "target": {
            "predicate_object_name": name,
            "predicate_object_type": typ,
            "predicate": predicate,
            "predicate_object_id": target_id,
            "held_object": "Knife" if predicate == "isSliced" else None,
            "decision_object": typ,
        },
        "certification_tier": tier,
        "review_status": "automated_checks_only",
        "validation": {
            "passed": True,
            "causal_symbolic_differences": differences,
            "raw_causal_symbolic_differences": diff(raw["anchor"], raw["causal"]),
            "raw_nuisance_symbolic_differences": diff(raw["anchor"], raw["nuisance"]),
            "numeric_geometry_tolerance": tolerance,
            "allowed_causal_symbolic_differences": sorted(permitted),
            "nuisance_symbolic_differences": [],
            "causal_target_union_bbox_rms": causal_rms,
            "nuisance_image_rms": nuisance_rms,
            "action_sets_disjoint": True,
            "selection_uses_model_outputs": False,
        },
    }
    record["available_object_types"] = sorted(
        {
            o["objectType"]
            for role in ROLES
            for o in json.loads((folder / f"{role}.metadata.json").read_text())[
                "objects"
            ]
        }
    )
    write_json(
        folder / "recipe.json",
        {
            "scene": scene,
            "target_type": typ,
            "target_name": name,
            "predicate": predicate,
            "inspection_pose": pose,
            "physics": "PausePhysicsAutoSim after each fresh scene reset",
            "object_poses": object_poses,
            "commit_id": BUILD,
            "ai2thor_version": "5.0.0",
            "image_size": [512, 512],
            "role_interventions": traces,
            "construction_module": "construction.expand_simulator",
        },
    )
    write_json(folder / "unit.json", record)
    return record


def main():
    import json

    from ai2thor.controller import Controller
    from ai2thor.platform import Linux64

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=[
            "FloorPlan1",
            "FloorPlan2",
            "FloorPlan3",
            "FloorPlan4",
            "FloorPlan5",
            "FloorPlan401",
            "FloorPlan402",
            "FloorPlan403",
        ],
    )
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument(
        "--predicates",
        nargs="+",
        choices=["isSliced", "isFilledWithLiquid", "isUsedUp"],
        default=["isSliced", "isFilledWithLiquid", "isUsedUp"],
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, rejected = [], []
    controller = Controller(
        scene=args.scenes[0],
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
    try:
        for scene in args.scenes:
            families = {
                "isFilledWithLiquid": ["Mug", "Cup", "Bowl"],
                "isSliced": ["Tomato", "Apple", "Potato"],
                "isUsedUp": ["PaperTowelRoll", "ToiletPaper"],
            }
            if int(scene.replace("FloorPlan", "")) >= 400:
                families = {"isUsedUp": ["PaperTowelRoll", "ToiletPaper"]}
            for predicate, types in families.items():
                if predicate not in args.predicates:
                    continue
                for typ in types:
                    if len(rows) >= args.limit:
                        break
                    cached = (
                        args.output
                        / f"extension__{predicate}__{scene}__{typ}"
                        / "unit.json"
                    )
                    try:
                        row = (
                            json.loads(cached.read_text())
                            if cached.exists()
                            else create_unit(
                                controller, args.output, scene, typ, predicate
                            )
                        )
                        rows.append(row)
                        print(
                            "ACCEPT",
                            scene,
                            predicate,
                            typ,
                            row["certification_tier"],
                            flush=True,
                        )
                    except (ValueError, StopIteration) as error:
                        rejected.append(
                            {
                                "scene": scene,
                                "predicate": predicate,
                                "object_type": typ,
                                "reason": str(error),
                            }
                        )
                        print(
                            "REJECT",
                            scene,
                            predicate,
                            typ,
                            str(error)[:300],
                            flush=True,
                        )
                    write_json(
                        args.output / "dataset.json",
                        {"format_version": 2, "triplets": rows},
                    )
                    write_json(args.output / "rejections.json", rejected)
    finally:
        controller.stop()


if __name__ == "__main__":
    main()
