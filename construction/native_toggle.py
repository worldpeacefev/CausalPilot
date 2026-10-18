"""Construct a fresh strict television unit, holding its remote control.

Fresh resets with identical object poses, inventory and reachable inspection
pose; accept only exact Boolean-only causal and empty nuisance state diffs.
The first visible screen pose with target RMS >5 is chosen without model calls.
Review still must be performed separately; this module does not certify humans.
"""

import copy
import math
from pathlib import Path

import numpy as np
from ai2thor.controller import Controller
from ai2thor.platform import Linux64
from PIL import Image

from AsgardBench.causal.generator import _symbolic_snapshot
from construction.common import ROLES, diff, write_json
from construction.expand_simulator import BUILD, INTERFACE, bbox, object_by_name, step


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scenes", nargs="+", default=["FloorPlan216", "FloorPlan205", "FloorPlan211"]
    )
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    c = Controller(
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
            try:
                c.reset(scene=scene)
                step(c, "PausePhysicsAutoSim")
                objects = c.last_event.metadata["objects"]
                target = next(o for o in objects if o["objectType"] == "Television")
                remote = next(o for o in objects if o["objectType"] == "RemoteControl")
                poses = [
                    {
                        "objectName": o["name"],
                        "position": o["position"],
                        "rotation": o["rotation"],
                    }
                    for o in objects
                    if o["pickupable"] or o["moveable"]
                ]
                name = target["name"]
                oid = target["objectId"]
                remoteid = remote["objectId"]

                def prepare(
                    pose=None,
                    scene=scene,
                    poses=poses,
                    remoteid=remoteid,
                    name=name,
                    oid=oid,
                ):
                    c.reset(scene=scene)
                    step(c, "PausePhysicsAutoSim")
                    step(c, "SetObjectPoses", objectPoses=poses)
                    step(c, "PickupObject", objectId=remoteid, forceAction=True)
                    if object_by_name(c, name)["isToggled"]:
                        step(c, "ToggleObjectOff", objectId=oid, forceAction=True)
                    if pose:
                        step(c, "TeleportFull", **pose)

                prepare()
                center = target["axisAlignedBoundingBox"]["center"]
                positions = step(c, "GetReachablePositions").metadata["actionReturn"]
                positions = sorted(
                    positions,
                    key=lambda p: abs(
                        math.hypot(p["x"] - center["x"], p["z"] - center["z"]) - 0.8
                    ),
                )
                chosen = None
                for p in positions[:64]:
                    dist = math.hypot(p["x"] - center["x"], p["z"] - center["z"])
                    if dist < 0.35 or dist > 1.8:
                        continue
                    pose = {
                        "position": p,
                        "rotation": {
                            "x": 0,
                            "y": math.degrees(
                                math.atan2(center["x"] - p["x"], center["z"] - p["z"])
                            )
                            % 360,
                            "z": 0,
                        },
                        "horizon": max(
                            -30,
                            min(
                                60,
                                math.degrees(
                                    math.atan2(p["y"] + 0.675 - center["y"], dist)
                                ),
                            ),
                        ),
                        "standing": True,
                    }
                    e = c.step(action="TeleportFull", **pose)
                    if (
                        not e.metadata["lastActionSuccess"]
                        or not object_by_name(c, name)["visible"]
                    ):
                        continue
                    try:
                        b = bbox(e, [oid])
                    except ValueError:
                        continue
                    if (b[2] - b[0]) * (b[3] - b[1]) / 512**2 < 0.01:
                        continue
                    a = np.asarray(e.frame, dtype=float)
                    e = step(c, "ToggleObjectOn", objectId=oid, forceAction=True)
                    ca = np.asarray(e.frame, dtype=float)
                    rms = float(
                        np.sqrt(
                            np.mean(
                                (
                                    a[b[1] : b[3], b[0] : b[2]]
                                    - ca[b[1] : b[3], b[0] : b[2]]
                                )
                                ** 2
                            )
                        )
                    )
                    step(c, "ToggleObjectOff", objectId=oid, forceAction=True)
                    if rms > 5:
                        chosen = pose
                        print("POSE", scene, rms, flush=True)
                        break
                if chosen is None:
                    raise ValueError("No visible causal contrast")
                unit = f"extension__native_isToggled__{scene}__Television"
                folder = out / unit
                folder.mkdir(exist_ok=True)
                variants = {}
                states = {}
                ims = {}
                traces = {}
                for role in ROLES:
                    prepare(chosen)
                    commands = []
                    if role == "causal":
                        step(c, "ToggleObjectOn", objectId=oid, forceAction=True)
                        commands.append(
                            {
                                "action": "ToggleObjectOn",
                                "objectId": oid,
                                "forceAction": True,
                            }
                        )
                    if role == "nuisance":
                        step(
                            c,
                            "RandomizeLighting",
                            brightness=[0.7, 0.7],
                            randomizeColor=False,
                            synchronized=True,
                        )
                        commands.append(
                            {
                                "action": "RandomizeLighting",
                                "brightness": [0.7, 0.7],
                                "randomizeColor": False,
                                "synchronized": True,
                            }
                        )
                    event = c.last_event
                    m = copy.deepcopy(event.metadata)
                    obj = object_by_name(c, name)
                    b = bbox(event, [oid])
                    area = (b[2] - b[0]) * (b[3] - b[1]) / 512**2
                    assert (
                        obj["visible"]
                        and area >= 0.01
                        and obj["isToggled"] == (role == "causal")
                    )
                    Image.fromarray(event.frame).save(folder / f"{role}.png")
                    write_json(folder / f"{role}.metadata.json", m)
                    states[role] = _symbolic_snapshot(m)
                    ims[role] = np.asarray(event.frame, dtype=float)
                    traces[role] = commands
                    variants[role] = {
                        "name": role,
                        "image": f"{unit}/{role}.png",
                        "metadata": f"{unit}/{role}.metadata.json",
                        "acceptable_actions": [
                            {"action": "done", "object": "none"}
                            if role == "causal"
                            else {"action": "toggle_on", "object": "Television"}
                        ],
                        "oracle_state": [
                            f"Television.isToggled={str(obj['isToggled']).lower()}",
                            "inventory: holding RemoteControl",
                        ],
                        "symbolic_state": states[role],
                        "validation": {
                            "predicate_value": obj["isToggled"],
                            "predicate_bbox": b,
                            "predicate_bbox_area_ratio": area,
                            "visible": {name: obj["visible"]},
                            "last_action_success": True,
                        },
                    }
                expected = f"objects.{name}.isToggled"
                assert diff(states["anchor"], states["causal"]) == [expected], diff(
                    states["anchor"], states["causal"]
                )
                assert diff(states["anchor"], states["nuisance"]) == [], diff(
                    states["anchor"], states["nuisance"]
                )
                row = {
                    "triplet_id": unit,
                    "causal_state_unit": unit,
                    "predicate": "isToggled",
                    "task_type": "turn_on_tv",
                    "instruction": "Turn on the television. Finish when it is on.",
                    "scene": scene,
                    "history": [
                        "find RemoteControl",
                        "pickup RemoteControl",
                        "find Television",
                    ],
                    "action_interface": INTERFACE,
                    "source_plan": f"{unit}/recipe.json",
                    "variants": variants,
                    "target": {
                        "predicate_object_name": name,
                        "predicate_object_type": "Television",
                        "predicate": "isToggled",
                        "predicate_object_id": oid,
                        "held_object": "RemoteControl",
                        "decision_object": "Television",
                    },
                    "certification_tier": "pilot-v3-strict",
                    "review_status": "automated_checks_only",
                    "validation": {
                        "passed": True,
                        "causal_symbolic_differences": [expected],
                        "allowed_causal_symbolic_differences": [expected],
                        "nuisance_symbolic_differences": [],
                        "action_sets_disjoint": True,
                        "selection_uses_model_outputs": False,
                    },
                }
                write_json(
                    folder / "recipe.json",
                    {
                        "scene": scene,
                        "predicate": "isToggled",
                        "target_name": name,
                        "target_id": oid,
                        "remote_id": remoteid,
                        "inspection_pose": chosen,
                        "object_poses": poses,
                        "role_interventions": traces,
                        "commit_id": BUILD,
                        "ai2thor_version": "5.0.0",
                        "image_size": [512, 512],
                        "preparation": [
                            "Reset scene",
                            "PausePhysicsAutoSim",
                            "SetObjectPoses",
                            "PickupObject RemoteControl forceAction",
                            "ToggleObjectOff Television if needed",
                            "TeleportFull inspection_pose",
                        ],
                    },
                )
                write_json(out / "dataset.json", {"triplets": [row]})
                print("ACCEPT", unit, flush=True)
                break
            except (ValueError, StopIteration, AssertionError) as exc:
                print("REJECT", scene, repr(exc), flush=True)
        else:
            raise RuntimeError("No native unit")
    finally:
        c.stop()


if __name__ == "__main__":
    main()
