from __future__ import annotations

import argparse
import copy
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageChops, ImageStat

from AsgardBench import constants as c
from AsgardBench.plan import Plan, PlanType
from AsgardBench.scenario import Scenario
from AsgardBench.specifier import Specifier

from AsgardBench.causal.schema import ActionChoice, TripletRecord, VariantRecord

ACTION_INTERFACE = [
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

SYMBOLIC_KEYS = (
    "isDirty",
    "isOpen",
    "isToggled",
    "isFilledWithLiquid",
    "fillLiquid",
    "isCooked",
    "isSliced",
    "isBroken",
    "isUsedUp",
    "temperature",
    "isPickedUp",
    "parentReceptacles",
    "receptacleObjectIds",
)


class InterventionValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Candidate:
    predicate: str
    plan_path: Path
    target_type: str
    target_name: str | None = None
    decision_type: str | None = None
    checkpoint_pose: dict[str, Any] | None = None
    generation_stage: str = "candidate_discovery"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _load_plan(path: Path) -> Plan:
    with path.open("r", encoding="utf-8") as handle:
        return Plan.from_dict(json.load(handle))


def discover_candidates(dataset_dir: Path) -> dict[str, list[Candidate]]:
    candidates: dict[str, list[Candidate]] = {
        "isOpen": [],
        "isDirty": [],
        "isToggled": [],
        "isCooked": [],
    }
    # The same physical dirty object can occur in several task templates. Keep
    # only the shortest source plan for each scene/object identity so the
    # dataset does not count near-duplicate rendered checkpoints as evidence.
    dirty_candidates: dict[tuple[str, str], tuple[int, Candidate]] = {}
    openable_homes = {"Fridge", "Cabinet", "Drawer"}
    for path in sorted(dataset_dir.glob("*/plan.json")):
        plan = _load_plan(path)

        if plan.name.startswith("putaway__") and plan.goal.put_away_goals:
            dirty_types = {
                action.object_name.split("_", 1)[0]
                for action in plan.setup_actions
                if str(action.action) == c.Action.DIRTY.value
            }
            open_candidate = None
            for goal in plan.goal.put_away_goals:
                homes = c.PREFERED_OBJECT_HOMES.get(goal.object_type, [])
                destination = next(
                    (home for home in homes if home in openable_homes), None
                )
                # Prefer an object that is already clean in the source plan;
                # otherwise washing would be the rational next action.
                if destination and goal.object_type not in dirty_types:
                    open_candidate = Candidate(
                        "isOpen",
                        path,
                        target_type=goal.object_type,
                        decision_type=destination,
                    )
                    break
            if open_candidate is not None:
                candidates["isOpen"].append(open_candidate)

        for action in plan.setup_actions:
            if str(action.action) != c.Action.DIRTY.value:
                continue
            target_type = action.object_name.split("_", 1)[0]
            # Pan and Pot use the same native wash-in-sink semantics. Mirror
            # uses the benchmark's native spray-then-wipe sequence.
            if target_type not in {
                "Bowl",
                "Cup",
                "Mug",
                "Plate",
                "Pan",
                "Pot",
                "Mirror",
            }:
                continue
            has_clean_goal = any(
                goal.object_type == target_type
                and goal.state == "isDirty"
                and goal.value is False
                for goal in plan.goal.state_goals
            )
            if not has_clean_goal:
                continue
            candidate = Candidate(
                "isDirty",
                path,
                target_type=target_type,
                target_name=action.object_name,
                decision_type=target_type,
            )
            key = (plan.scene, action.object_name)
            previous = dirty_candidates.get(key)
            if previous is None or plan.step_count < previous[0]:
                dirty_candidates[key] = (plan.step_count, candidate)

        if plan.name.startswith("turn_on_tv"):
            candidates["isToggled"].append(
                Candidate(
                    "isToggled",
                    path,
                    target_type="Television",
                    decision_type="Television",
                )
            )
            # One predeclared representative scene screens the Box family.
            # Opening a Box moves both its root pose and contained remote in
            # FloorPlan202, a structural violation that is not worth repeating
            # across identical TV templates under a limited compute budget.
            if plan.scene == "FloorPlan202":
                candidates["isOpen"].append(
                    Candidate(
                        "isOpen",
                        path,
                        target_type="RemoteControl",
                        decision_type="Box",
                        generation_stage="remote_container_checkpoint",
                    )
                )

        # Cooking a whole potato uses the Microwave in the native Asgard
        # task.  Immediately before insertion, Microwave.isOpen is the sole
        # visual fact separating OPEN Microwave from PUT Microwave.  This is
        # the same certified checkpoint construction as food/Fridge and
        # dish/Cabinet, but broadens both the task and decision-object support.
        # Sliced potato, bread, and egg recipes use appliances whose doors are
        # not part of their decision boundary and are deliberately excluded.
        microwave_food = next(
            (
                goal.object_type
                for goal in plan.goal.state_goals
                if goal.state == "isCooked"
                and goal.value is True
                and goal.object_type == "Potato"
            ),
            None,
        )
        if plan.name.startswith("cook__") and microwave_food is not None:
            candidates["isOpen"].append(
                Candidate(
                    "isOpen",
                    path,
                    target_type=microwave_food,
                    decision_type="Microwave",
                    generation_stage="predicate_object_task_expansion",
                )
            )
            candidates["isCooked"].append(
                Candidate(
                    "isCooked",
                    path,
                    target_type=microwave_food,
                    decision_type=microwave_food,
                    generation_stage="cooked_state_checkpoint",
                )
            )
        bread_goal = any(
            goal.state == "isCooked"
            and goal.value is True
            and goal.object_type == "Bread"
            for goal in plan.goal.state_goals
        )
        if (
            plan.name.startswith("cook__Bread_")
            and bread_goal
            and not plan.setup_actions
        ):
            candidates["isCooked"].append(
                Candidate(
                    "isCooked",
                    path,
                    target_type="BreadSliced",
                    decision_type="BreadSliced",
                    generation_stage="cooked_bread_state_checkpoint",
                )
            )
    candidates["isDirty"] = [item[1] for item in dirty_candidates.values()]
    return candidates


def _round_robin_candidates(
    candidates: list[Candidate], attribute: str
) -> list[Candidate]:
    """Prefer category coverage before repeated scenes."""
    groups: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        groups.setdefault(str(getattr(candidate, attribute)), []).append(candidate)

    ordered = []
    depth = 0
    while any(depth < len(group) for group in groups.values()):
        for group in groups.values():
            if depth < len(group):
                ordered.append(group[depth])
        depth += 1
    return ordered


def _round_robin_target_types(candidates: list[Candidate]) -> list[Candidate]:
    """Backward-compatible helper used by tests and dirty candidate ordering."""
    return _round_robin_candidates(candidates, "target_type")


def _object_by_name(metadata: dict[str, Any], name: str) -> dict[str, Any]:
    for obj in metadata["objects"]:
        if obj["name"] == name:
            return obj
    raise InterventionValidationError(f"Object {name!r} not present in metadata")


def _objects_by_type(
    metadata: dict[str, Any], object_type: str
) -> list[dict[str, Any]]:
    return [obj for obj in metadata["objects"] if obj["objectType"] == object_type]


def _controller_step(scenario: Scenario, action: str, **kwargs: Any) -> None:
    event = scenario.controller.step(action=action, **kwargs)
    if not event.metadata.get("lastActionSuccess"):
        raise InterventionValidationError(
            f"AI2-THOR action {action} failed: {event.metadata.get('errorMessage', '')}"
        )


def _set_boolean_state(
    scenario: Scenario,
    object_name: str,
    predicate: str,
    value: bool,
) -> None:
    obj = scenario.get_obj_by_name(object_name)
    if bool(obj.get(predicate)) == value:
        return
    if predicate == "isCooked":
        if not value:
            raise InterventionValidationError(
                f"Cannot reverse cooked state for {object_name}"
            )
        _controller_step(
            scenario,
            "CookObject",
            objectId=obj["objectId"],
            forceAction=True,
        )
        return
    actions = {
        ("isOpen", True): "OpenObject",
        ("isOpen", False): "CloseObject",
        ("isToggled", True): "ToggleObjectOn",
        ("isToggled", False): "ToggleObjectOff",
        ("isDirty", True): "DirtyObject",
        ("isDirty", False): "CleanObject",
    }
    action = actions[(predicate, value)]
    kwargs: dict[str, Any] = {"objectId": obj["objectId"], "forceAction": True}
    if action == "OpenObject":
        kwargs["openness"] = 1
    _controller_step(scenario, action, **kwargs)


def _named_specifier(name: str) -> Specifier:
    return Specifier(names=[name], preferred_name=name)


def _do(scenario: Scenario, action: c.Action, object_name: str) -> str:
    result = scenario.do(action, _named_specifier(object_name), "checkpoint setup", [])
    if result is None or scenario.step_error is not None:
        error = scenario.step_error.error_msg if scenario.step_error else "no result"
        raise InterventionValidationError(
            f"Asgard action {action.value} {object_name} failed: {error}"
        )
    return result


def _symbolic_snapshot(metadata: dict[str, Any]) -> dict[str, Any]:
    objects: dict[str, Any] = {}
    for obj in metadata["objects"]:
        state = {
            key: copy.deepcopy(obj.get(key)) for key in SYMBOLIC_KEYS if key in obj
        }
        # Geometry is part of the controlled world state even though it is not a
        # task predicate. Including root pose catches accidental object motion
        # across independently reconstructed nuisance variants.
        state["position"] = copy.deepcopy(obj.get("position"))
        state["rotation"] = copy.deepcopy(obj.get("rotation"))
        for key in ("parentReceptacles", "receptacleObjectIds"):
            if isinstance(state.get(key), list):
                state[key] = sorted(state[key])
        objects[obj["name"]] = state
    inventory = sorted(
        item["objectId"] for item in metadata.get("inventoryObjects", [])
    )
    agent = metadata["agent"]
    agent_pose = {
        "position": copy.deepcopy(agent["position"]),
        "rotation": copy.deepcopy(agent["rotation"]),
        "cameraHorizon": agent["cameraHorizon"],
        "isStanding": agent["isStanding"],
    }
    return {
        "objects": objects,
        "inventoryObjectIds": inventory,
        "agentPose": agent_pose,
    }


def _task_relevant_types(plan: Plan, context_types: Iterable[str]) -> set[str]:
    result = set(context_types)
    result.update(goal.object_type for goal in plan.goal.location_goals)
    result.update(goal.object_type for goal in plan.goal.put_away_goals)
    result.update(goal.object_type for goal in plan.goal.state_goals)
    result.update(goal.container_type for goal in plan.goal.contents_goals)
    return result


def _oracle_state(metadata: dict[str, Any], relevant_types: set[str]) -> list[str]:
    facts = []
    inventory = metadata.get("inventoryObjects", [])
    if inventory:
        held = ", ".join(item["objectType"] for item in inventory)
        facts.append(f"inventory: holding {held}")
    else:
        facts.append("inventory: empty hand")

    for obj in metadata["objects"]:
        if obj["objectType"] not in relevant_types:
            continue
        values = []
        for key in (
            "isDirty",
            "isOpen",
            "isToggled",
            "isFilledWithLiquid",
            "fillLiquid",
            "isCooked",
            "isSliced",
            "isPickedUp",
        ):
            if key in obj and obj[key] is not None:
                values.append(f"{key}={str(obj[key]).lower()}")
        parents = obj.get("parentReceptacles") or []
        if parents:
            parent_types = sorted({parent.split("|", 1)[0] for parent in parents})
            values.append(f"inside_or_on={','.join(parent_types)}")
        if values:
            facts.append(f"{obj['objectType']}: " + ", ".join(values))
    return sorted(set(facts))


def _diff(left: Any, right: Any, prefix: str = "") -> list[str]:
    if type(left) is not type(right):
        return [prefix]
    if isinstance(left, dict):
        differences = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                differences.append(path)
            else:
                differences.extend(_diff(left[key], right[key], path))
        return differences
    if isinstance(left, list):
        return [] if left == right else [prefix]
    return [] if left == right else [prefix]


def _image_rms(left_path: Path, right_path: Path) -> float:
    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")
    stat = ImageStat.Stat(ImageChops.difference(left, right))
    return math.sqrt(sum(value * value for value in stat.rms) / len(stat.rms))


class TripletGenerator:
    def __init__(
        self,
        output_dir: Path,
        *,
        min_bbox_area_ratio: float = 0.01,
        allow_target_articulation: bool = False,
        certification_tier: str = "pilot-v3-strict",
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sim_dir = self.output_dir / "_sim_runs"
        self.sim_dir.mkdir(parents=True, exist_ok=True)
        if not 0 < min_bbox_area_ratio <= 1:
            raise ValueError("min_bbox_area_ratio must be in (0, 1]")
        self.min_bbox_area_ratio = min_bbox_area_ratio
        self.allow_target_articulation = allow_target_articulation
        self.certification_tier = certification_tier

    def _create_scenario(self, plan: Plan, run_name: str) -> Scenario:
        initial_pose = copy.deepcopy(plan.initial_pose)
        if initial_pose:
            initial_pose["standing"] = initial_pose.get("isStanding", True)
        return Scenario(
            task=plan.task_description,
            scene=plan.scene,
            name=run_name,
            plan_type=PlanType.REPLAY,
            data_folder=str(self.sim_dir),
            hand_transparency=60,
            setup_actions=copy.deepcopy(plan.setup_actions),
            object_setup=copy.deepcopy(plan.object_setup),
            randomization=copy.deepcopy(plan.randomization),
            goal=copy.deepcopy(plan.goal),
            initial_pose=initial_pose,
        )

    def _prepare_open(
        self, scenario: Scenario, candidate: Candidate
    ) -> tuple[str, str, list[str], dict[str, Any]]:
        target = scenario.get_obj_by_type(candidate.target_type)
        destination_type = candidate.decision_type
        if destination_type is None:
            raise InterventionValidationError("Open candidate has no destination type")
        destination = scenario.get_obj_by_type(destination_type)
        _set_boolean_state(scenario, destination["name"], "isOpen", False)
        _do(scenario, c.Action.FIND, target["name"])
        _do(scenario, c.Action.PICKUP, target["name"])
        _do(scenario, c.Action.FIND, destination["name"])
        history = [
            f"find {candidate.target_type}",
            f"pickup {candidate.target_type}",
            f"find {destination_type}",
        ]
        return (
            target["name"],
            destination["name"],
            history,
            {
                "held_object": candidate.target_type,
                "decision_object": destination_type,
            },
        )

    def _prepare_dirty(
        self, scenario: Scenario, candidate: Candidate
    ) -> tuple[str, str, list[str], dict[str, Any]]:
        target = scenario.get_obj_by_name(candidate.target_name)
        sink = scenario.get_obj_by_type("SinkBasin")
        sponge = scenario.get_obj_by_type("DishSponge")
        faucet = scenario.get_obj_by_type("Faucet")
        _do(scenario, c.Action.FIND, target["name"])
        _do(scenario, c.Action.PICKUP, target["name"])
        _do(scenario, c.Action.FIND, sink["name"])
        if candidate.target_type in {"Pan", "Pot"}:
            # A failed large-vessel placement reports that the receptacle has
            # no valid placement geometry; changing the agent pose cannot add
            # receptacle capacity. One native placement attempt is therefore
            # sufficient for screening these expansion candidates.
            if not scenario.agent_put(sink["name"]):
                message = scenario.controller.last_event.metadata.get(
                    "errorMessage", "native PutObject failed"
                )
                raise InterventionValidationError(
                    f"AI2-THOR PutObject failed for {candidate.target_type}: {message}"
                )
        else:
            _do(scenario, c.Action.PUT, sink["name"])
        _do(scenario, c.Action.FIND, sponge["name"])
        _do(scenario, c.Action.PICKUP, sponge["name"])
        _set_boolean_state(scenario, faucet["name"], "isToggled", True)
        _do(scenario, c.Action.FIND, target["name"])
        history = [
            f"find {candidate.target_type}",
            f"pickup {candidate.target_type}",
            "find SinkBasin",
            "put SinkBasin",
            "find DishSponge",
            "pickup DishSponge",
            "toggle_on Faucet",
        ]
        return (
            target["name"],
            faucet["name"],
            history,
            {
                "held_object": "DishSponge",
                "decision_object": candidate.target_type,
                "support_object": "Faucet",
            },
        )

    def _prepare_cooked(
        self, scenario: Scenario, candidate: Candidate
    ) -> tuple[str, str, list[str], dict[str, Any]]:
        if candidate.generation_stage == "cooked_bread_state_checkpoint":
            bread = scenario.get_obj_by_type("Bread")
            knife = scenario.get_obj_by_type("Knife")
            counter = scenario.get_obj_by_type("CounterTop")
            _do(scenario, c.Action.FIND, bread["name"])
            _do(scenario, c.Action.PICKUP, bread["name"])
            _do(scenario, c.Action.FIND, counter["name"])
            _do(scenario, c.Action.PUT, counter["name"])
            _do(scenario, c.Action.FIND, knife["name"])
            _do(scenario, c.Action.PICKUP, knife["name"])
            _do(scenario, c.Action.FIND, bread["name"])
            _controller_step(
                scenario,
                "SliceObject",
                objectId=bread["objectId"],
                forceAction=False,
            )
            _do(scenario, c.Action.FIND, counter["name"])
            _do(scenario, c.Action.PUT, counter["name"])
            target = scenario.get_obj_by_type("BreadSliced")
            _do(scenario, c.Action.FIND, target["name"])
            history = [
                "find Bread",
                "pickup Bread",
                "find CounterTop",
                "put CounterTop",
                "find Knife",
                "pickup Knife",
                "find Bread",
                "slice Bread",
                "find CounterTop",
                "put CounterTop",
                "find BreadSliced",
            ]
            return (
                target["name"],
                target["name"],
                history,
                {
                    "held_object": None,
                    "decision_object": "BreadSliced",
                },
            )
        target = scenario.get_obj_by_type(candidate.target_type)
        _do(scenario, c.Action.FIND, target["name"])
        return (
            target["name"],
            target["name"],
            [f"find {candidate.target_type}"],
            {
                "held_object": None,
                "decision_object": candidate.target_type,
            },
        )

    def _prepare_remote_box(
        self, scenario: Scenario, candidate: Candidate
    ) -> tuple[str, str, list[str], dict[str, Any]]:
        """Place the task-relevant remote inside an openable living-room Box."""
        remote = scenario.get_obj_by_type("RemoteControl")
        box = scenario.get_obj_by_type("Box")
        _set_boolean_state(scenario, box["name"], "isOpen", True)
        _do(scenario, c.Action.FIND, remote["name"])
        _do(scenario, c.Action.PICKUP, remote["name"])
        _do(scenario, c.Action.FIND, box["name"])
        _do(scenario, c.Action.PUT, box["name"])
        _do(scenario, c.Action.FIND, box["name"])
        history = [
            "find RemoteControl",
            "pickup RemoteControl",
            "find Box",
            "put Box",
            "find Box",
        ]
        return (
            remote["name"],
            box["name"],
            history,
            {
                "held_object": None,
                "decision_object": "Box",
                "dependent_object": "RemoteControl",
            },
        )

    def _prepare_mirror(
        self, scenario: Scenario, candidate: Candidate
    ) -> tuple[str, str, list[str], dict[str, Any]]:
        mirror = scenario.get_obj_by_name(candidate.target_name)
        spray = scenario.get_obj_by_type("SprayBottle")
        cloth = scenario.get_obj_by_type("Cloth")
        counter = scenario.get_obj_by_type("CounterTop")
        _do(scenario, c.Action.FIND, spray["name"])
        _do(scenario, c.Action.PICKUP, spray["name"])
        _do(scenario, c.Action.FIND, mirror["name"])
        agent = scenario.controller.last_event.metadata["agent"]
        mirror_pose = {
            "position": copy.deepcopy(agent["position"]),
            "rotation": copy.deepcopy(agent["rotation"]),
            "horizon": agent["cameraHorizon"],
            "standing": agent["isStanding"],
        }
        _do(scenario, c.Action.SPRAY, mirror["name"])
        _do(scenario, c.Action.FIND, counter["name"])
        _do(scenario, c.Action.PUT, counter["name"])
        _do(scenario, c.Action.FIND, cloth["name"])
        _do(scenario, c.Action.PICKUP, cloth["name"])
        if not scenario.agent_teleport(mirror_pose, mirror["name"]):
            raise InterventionValidationError(
                "Could not restore the validated mirror inspection pose"
            )
        history = [
            "find SprayBottle",
            "pickup SprayBottle",
            "find Mirror",
            "spray Mirror",
            "find CounterTop",
            "put CounterTop",
            "find Cloth",
            "pickup Cloth",
            "find Mirror",
        ]
        return (
            mirror["name"],
            mirror["name"],
            history,
            {
                "held_object": "Cloth",
                "decision_object": "Mirror",
                "support_object": "SprayBottle",
            },
        )

    def _prepare_faucet_toggle(
        self, scenario: Scenario, candidate: Candidate
    ) -> tuple[str, str, list[str], dict[str, Any]]:
        target = scenario.get_obj_by_name(candidate.target_name)
        sink = scenario.get_obj_by_type("SinkBasin")
        sponge = scenario.get_obj_by_type("DishSponge")
        faucet = scenario.get_obj_by_type("Faucet")
        if candidate.checkpoint_pose is not None:
            _controller_step(
                scenario,
                "PickupObject",
                objectId=target["objectId"],
                forceAction=True,
            )
            _controller_step(
                scenario,
                "PutObject",
                objectId=sink["objectId"],
                forceAction=True,
                placeStationary=True,
            )
            _controller_step(
                scenario,
                "PickupObject",
                objectId=sponge["objectId"],
                forceAction=True,
            )
            pose = candidate.checkpoint_pose
            if not scenario.agent_teleport(
                {
                    "position": copy.deepcopy(pose["position"]),
                    "rotation": copy.deepcopy(pose["rotation"]),
                    "horizon": pose["cameraHorizon"],
                    "standing": pose["isStanding"],
                },
                target["name"],
            ):
                raise InterventionValidationError(
                    "Could not restore accepted wash checkpoint pose"
                )
        else:
            _do(scenario, c.Action.FIND, target["name"])
            _do(scenario, c.Action.PICKUP, target["name"])
            _do(scenario, c.Action.FIND, sink["name"])
            _do(scenario, c.Action.PUT, sink["name"])
            _do(scenario, c.Action.FIND, sponge["name"])
            _do(scenario, c.Action.PICKUP, sponge["name"])
        _set_boolean_state(scenario, faucet["name"], "isToggled", False)
        history = [
            f"find {candidate.target_type}",
            f"pickup {candidate.target_type}",
            "find SinkBasin",
            "put SinkBasin",
            "find DishSponge",
            "pickup DishSponge",
            f"find {candidate.target_type}",
        ]
        return (
            target["name"],
            faucet["name"],
            history,
            {
                "held_object": "DishSponge",
                "decision_object": "Faucet",
                "dependent_object": candidate.target_type,
            },
        )

    def _prepare_toggled(
        self, scenario: Scenario, candidate: Candidate
    ) -> tuple[str, str, list[str], dict[str, Any]]:
        remote = scenario.get_obj_by_type("RemoteControl")
        television = scenario.get_obj_by_type("Television")
        _set_boolean_state(scenario, television["name"], "isToggled", False)
        _do(scenario, c.Action.FIND, remote["name"])
        _do(scenario, c.Action.PICKUP, remote["name"])
        _do(scenario, c.Action.FIND, television["name"])
        # The stock face-object helper can center the TV stand while looking 60
        # degrees down, leaving only a thin strip of the screen visible. Raise
        # the camera so the toggled visual state is actually inspectable.
        scenario.look_vert(0)
        history = ["find RemoteControl", "pickup RemoteControl", "find Television"]
        return (
            television["name"],
            remote["name"],
            history,
            {
                "held_object": "RemoteControl",
                "decision_object": "Television",
            },
        )

    def _build_variant(
        self,
        plan: Plan,
        candidate: Candidate,
        triplet_id: str,
        variant_name: str,
        triplet_dir: Path,
    ) -> tuple[VariantRecord, list[str], dict[str, Any]]:
        run_name = f"{triplet_id}__{variant_name}"
        scenario = self._create_scenario(plan, run_name)
        base_variant = variant_name
        for suffix in ("_viewpoint", "_material"):
            base_variant = base_variant.removesuffix(suffix)
        try:
            if candidate.predicate == "isCooked":
                target_name, support_name, history, context = self._prepare_cooked(
                    scenario, candidate
                )
                predicate_object = target_name
                predicate_value = base_variant == "causal"
                _set_boolean_state(
                    scenario, predicate_object, candidate.predicate, predicate_value
                )
                acceptable = (
                    [ActionChoice("pickup", candidate.target_type)]
                    if predicate_value
                    else [ActionChoice("cook", candidate.target_type)]
                )
                visibility_names = [predicate_object]
                relevant_types = {candidate.target_type, "Microwave", "Plate", "Bowl"}
            elif (
                candidate.predicate == "isOpen"
                and candidate.generation_stage == "remote_container_checkpoint"
            ):
                target_name, support_name, history, context = self._prepare_remote_box(
                    scenario, candidate
                )
                predicate_object = support_name
                predicate_value = base_variant == "causal"
                _set_boolean_state(
                    scenario, predicate_object, candidate.predicate, predicate_value
                )
                acceptable = (
                    [ActionChoice("pickup", "RemoteControl")]
                    if predicate_value
                    else [ActionChoice("open", "Box")]
                )
                visibility_names = [predicate_object]
                relevant_types = {"RemoteControl", "Box", "Television"}
            elif candidate.predicate == "isOpen":
                target_name, support_name, history, context = self._prepare_open(
                    scenario, candidate
                )
                predicate_object = support_name
                predicate_value = base_variant == "causal"
                _set_boolean_state(
                    scenario, predicate_object, candidate.predicate, predicate_value
                )
                destination_type = candidate.decision_type
                acceptable = (
                    [ActionChoice("put", destination_type)]
                    if predicate_value
                    else [ActionChoice("open", destination_type)]
                )
                visibility_names = [predicate_object]
                relevant_types = {candidate.target_type, destination_type}
            elif candidate.predicate == "isDirty" and candidate.target_type == "Mirror":
                target_name, support_name, history, context = self._prepare_mirror(
                    scenario, candidate
                )
                predicate_object = target_name
                predicate_value = base_variant != "causal"
                if base_variant == "nuisance":
                    self._apply_nuisance(scenario)
                _set_boolean_state(
                    scenario, predicate_object, candidate.predicate, predicate_value
                )
                acceptable = (
                    [ActionChoice("clean", "Mirror")]
                    if predicate_value
                    else [ActionChoice("done", None)]
                )
                visibility_names = [predicate_object]
                relevant_types = {"Mirror", "Cloth", "SprayBottle"}
            elif candidate.predicate == "isDirty":
                target_name, support_name, history, context = self._prepare_dirty(
                    scenario, candidate
                )
                predicate_object = target_name
                predicate_value = base_variant != "causal"
                if base_variant == "nuisance":
                    self._apply_nuisance(scenario)
                _set_boolean_state(
                    scenario, predicate_object, candidate.predicate, predicate_value
                )
                scenario.agent_face_object(predicate_object)
                acceptable = (
                    [ActionChoice("clean", candidate.target_type)]
                    if predicate_value
                    else [ActionChoice("toggle_off", "Faucet")]
                )
                visibility_names = [predicate_object, support_name]
                relevant_types = {
                    candidate.target_type,
                    "SinkBasin",
                    "DishSponge",
                    "Faucet",
                }
            elif (
                candidate.predicate == "isToggled"
                and candidate.decision_type == "Faucet"
            ):
                target_name, support_name, history, context = (
                    self._prepare_faucet_toggle(scenario, candidate)
                )
                predicate_object = support_name
                predicate_value = base_variant == "causal"
                # Face the dish first, then apply the intervention. DirtyObject
                # is deliberately last so running water cannot auto-clean it
                # before the validated snapshot is captured.
                scenario.agent_face_object(target_name)
                _set_boolean_state(
                    scenario, predicate_object, candidate.predicate, predicate_value
                )
                _set_boolean_state(scenario, target_name, "isDirty", True)
                acceptable = (
                    [ActionChoice("clean", candidate.target_type)]
                    if predicate_value
                    else [ActionChoice("toggle_on", "Faucet")]
                )
                visibility_names = [predicate_object, target_name]
                relevant_types = {
                    candidate.target_type,
                    "SinkBasin",
                    "DishSponge",
                    "Faucet",
                }
            else:
                target_name, support_name, history, context = self._prepare_toggled(
                    scenario, candidate
                )
                predicate_object = target_name
                predicate_value = base_variant == "causal"
                _set_boolean_state(
                    scenario, predicate_object, candidate.predicate, predicate_value
                )
                acceptable = (
                    [ActionChoice("done", None)]
                    if predicate_value
                    else [ActionChoice("toggle_on", "Television")]
                )
                visibility_names = [predicate_object]
                relevant_types = {"RemoteControl", "Television"}

            if base_variant == "nuisance" and candidate.predicate != "isDirty":
                self._apply_nuisance(scenario)

            if variant_name.endswith("_material"):
                self._apply_material_nuisance(scenario)

            scenario.capture_world_state()
            viewpoint_camera = None
            viewpoint_masks = None
            viewpoint_integrity = None
            viewpoint_source_image = None
            viewpoint_source_metadata = None
            captured_image = scenario.cur_image
            if variant_name.endswith("_viewpoint"):
                pre_camera_metadata = copy.deepcopy(
                    scenario.controller.last_event.metadata
                )
                viewpoint_source_image = scenario.cur_image.copy()
                viewpoint_source_metadata = pre_camera_metadata
                predicate_id_for_camera = _object_by_name(
                    pre_camera_metadata, predicate_object
                )["objectId"]
                captured_image, viewpoint_masks, viewpoint_camera = (
                    self._render_viewpoint_nuisance(scenario, predicate_id_for_camera)
                )
                post_camera_metadata = scenario.controller.last_event.metadata

                def canonical_objects(items: list[dict[str, Any]]) -> dict[str, Any]:
                    result = {}
                    for original in items:
                        item = copy.deepcopy(original)
                        # AI2-THOR may reorder these set-valued arrays between
                        # otherwise state-preserving events.
                        for field in (
                            "parentReceptacles",
                            "receptacleObjectIds",
                            "salientMaterials",
                        ):
                            if isinstance(item.get(field), list):
                                item[field] = sorted(item[field])
                        result[item["name"]] = item
                    return result

                pre_objects = canonical_objects(pre_camera_metadata["objects"])
                post_objects = canonical_objects(post_camera_metadata["objects"])
                object_diffs = _diff(pre_objects, post_objects)
                agent_diffs = _diff(
                    pre_camera_metadata["agent"],
                    post_camera_metadata["agent"],
                )
                inventory_diffs = _diff(
                    pre_camera_metadata.get("inventoryObjects", []),
                    post_camera_metadata.get("inventoryObjects", []),
                )
                if object_diffs or agent_diffs or inventory_diffs:
                    raise InterventionValidationError(
                        "Third-party camera changed simulator state: "
                        f"objects={object_diffs}, agent={agent_diffs}, "
                        f"inventory={inventory_diffs}"
                    )
                viewpoint_integrity = {
                    "object_metadata_differences": object_diffs,
                    "agent_metadata_differences": agent_diffs,
                    "inventory_metadata_differences": inventory_diffs,
                    "state_frozen": True,
                }
            metadata = copy.deepcopy(scenario.controller.last_event.metadata)
            current = _object_by_name(metadata, predicate_object)
            if bool(current[candidate.predicate]) != predicate_value:
                raise InterventionValidationError(
                    f"{predicate_object}.{candidate.predicate} expected "
                    f"{predicate_value}, observed {current[candidate.predicate]}"
                )

            if viewpoint_masks is None:
                visibility = {
                    name: bool(_object_by_name(metadata, name).get("visible"))
                    for name in visibility_names
                }
            else:
                # The manipulated predicate object is the decision-critical
                # target. Dependent/support visibility was already certified
                # in v2 and need not survive a small camera perturbation.
                viewpoint_visibility_names = [predicate_object]
                visibility = {
                    name: bool(
                        viewpoint_masks.get(_object_by_name(metadata, name)["objectId"])
                        is not None
                        and viewpoint_masks[
                            _object_by_name(metadata, name)["objectId"]
                        ].any()
                    )
                    for name in viewpoint_visibility_names
                }
            if not all(visibility.values()):
                raise InterventionValidationError(
                    f"Decision-critical object not visible: {visibility}"
                )

            predicate_id = current["objectId"]
            if viewpoint_masks is None:
                bbox = scenario.controller.last_event.instance_detections2D.get(
                    predicate_id
                )
            else:
                mask = viewpoint_masks.get(predicate_id)
                if mask is None or not mask.any():
                    bbox = None
                else:
                    ys, xs = np.nonzero(mask)
                    bbox = [
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max()) + 1,
                        int(ys.max()) + 1,
                    ]
            if bbox is None:
                raise InterventionValidationError(
                    f"No 2D detection for decision-critical object {predicate_object}"
                )
            x1, y1, x2, y2 = bbox
            image_width, image_height = captured_image.size
            bbox_area_ratio = ((x2 - x1) * (y2 - y1)) / (image_width * image_height)
            if bbox_area_ratio < self.min_bbox_area_ratio:
                raise InterventionValidationError(
                    f"Decision-critical object is too small in frame: "
                    f"area_ratio={bbox_area_ratio:.4f}, bbox={list(bbox)}"
                )

            held_types = sorted(
                item["objectType"] for item in metadata.get("inventoryObjects", [])
            )
            expected_held = context["held_object"]
            if expected_held is None and held_types:
                raise InterventionValidationError(
                    f"Expected empty inventory, inventory={held_types}"
                )
            if expected_held is not None and expected_held not in held_types:
                raise InterventionValidationError(
                    f"Expected to hold {expected_held}, inventory={held_types}"
                )

            triplet_dir.mkdir(parents=True, exist_ok=True)
            image_path = triplet_dir / f"{variant_name}.png"
            captured_image.save(image_path)
            metadata_path = triplet_dir / f"{variant_name}_metadata.json"
            with metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2, default=_json_default)
            viewpoint_source_image_path = None
            viewpoint_source_metadata_path = None
            if viewpoint_source_image is not None:
                source_stem = variant_name.replace("_viewpoint", "_egocentric_replay")
                source_image_path = triplet_dir / f"{source_stem}.png"
                viewpoint_source_image.save(source_image_path)
                source_metadata_path = triplet_dir / f"{source_stem}_metadata.json"
                with source_metadata_path.open("w", encoding="utf-8") as handle:
                    json.dump(
                        viewpoint_source_metadata,
                        handle,
                        indent=2,
                        default=_json_default,
                    )
                viewpoint_source_image_path = str(
                    source_image_path.relative_to(self.output_dir)
                )
                viewpoint_source_metadata_path = str(
                    source_metadata_path.relative_to(self.output_dir)
                )

            task_types = _task_relevant_types(plan, relevant_types)
            record = VariantRecord(
                name=variant_name,
                image=str(image_path.relative_to(self.output_dir)),
                acceptable_actions=acceptable,
                symbolic_state=_symbolic_snapshot(metadata),
                oracle_state=_oracle_state(metadata, task_types),
                metadata=str(metadata_path.relative_to(self.output_dir)),
                validation={
                    "predicate_value": predicate_value,
                    "visible": visibility,
                    "predicate_bbox": list(bbox),
                    "predicate_bbox_area_ratio": bbox_area_ratio,
                    "inventory_types": held_types,
                    "last_action_success": metadata.get("lastActionSuccess"),
                    "viewpoint_camera": viewpoint_camera,
                    "viewpoint_integrity": viewpoint_integrity,
                    "viewpoint_source_image": viewpoint_source_image_path,
                    "viewpoint_source_metadata": viewpoint_source_metadata_path,
                },
            )
            target = {
                "predicate_object_name": predicate_object,
                "predicate_object_type": current["objectType"],
                "predicate": candidate.predicate,
                **context,
            }
            if candidate.generation_stage != "candidate_discovery":
                target["checkpoint_family"] = candidate.generation_stage
            return record, history, target
        finally:
            scenario.controller.stop()

    @staticmethod
    def _apply_nuisance(scenario: Scenario) -> None:
        _controller_step(
            scenario,
            "RandomizeLighting",
            brightness=(0.65, 0.65),
            randomizeColor=False,
            synchronized=True,
        )

    @staticmethod
    def _apply_material_nuisance(scenario: Scenario) -> None:
        """Change appearance without touching task predicates or geometry."""
        _controller_step(
            scenario,
            "RandomizeMaterials",
            useTrainMaterials=False,
            useValMaterials=True,
            useTestMaterials=True,
        )

    @staticmethod
    def _render_viewpoint_nuisance(
        scenario: Scenario,
        predicate_object_id: str,
    ) -> tuple[Image.Image, Any, dict[str, Any]]:
        metadata = scenario.controller.last_event.metadata
        agent = metadata["agent"]
        yaw = float(agent["rotation"]["y"])
        yaw_radians = math.radians(yaw)
        source_position = copy.deepcopy(metadata["cameraPosition"])
        perturbations = (
            (0.05, 0.0, 5.0, 3.0),
            (-0.05, 0.0, -5.0, -3.0),
            (0.03, 0.0, -3.0, 2.0),
            (-0.03, 0.0, 3.0, -2.0),
            (0.01, 0.0, -1.0, 1.0),
            (-0.01, 0.0, 1.0, -1.0),
            (0.005, 0.0, -0.5, 0.5),
            (-0.005, 0.0, 0.5, -0.5),
            (0.0, 0.05, -0.5, 0.5),
            (0.0, 0.03, 0.5, -0.5),
        )
        for attempt, (right_shift, forward_shift, yaw_shift, pitch_shift) in enumerate(
            perturbations
        ):
            position = copy.deepcopy(source_position)
            position["x"] += right_shift * math.cos(yaw_radians)
            position["z"] -= right_shift * math.sin(yaw_radians)
            position["x"] += forward_shift * math.sin(yaw_radians)
            position["z"] += forward_shift * math.cos(yaw_radians)
            rotation = {
                "x": float(agent["cameraHorizon"]) + pitch_shift,
                "y": (yaw + yaw_shift) % 360.0,
                "z": 0.0,
            }
            event = scenario.controller.step(
                action="AddThirdPartyCamera",
                position=position,
                rotation=rotation,
                fieldOfView=float(metadata.get("fov", 60.0)),
            )
            if not event.metadata.get("lastActionSuccess"):
                continue
            masks = event.third_party_instance_masks[-1]
            mask = masks.get(predicate_object_id)
            area_ratio = float(mask.mean()) if mask is not None else 0.0
            if area_ratio < 0.01:
                continue
            camera = {
                "mechanism": "third_party_camera",
                "perturbation_attempt": attempt,
                "right_shift": right_shift,
                "forward_shift": forward_shift,
                "yaw_shift": yaw_shift,
                "pitch_shift": pitch_shift,
                "source_position": source_position,
                "position": position,
                "source_yaw": yaw,
                "yaw": rotation["y"],
                "source_pitch": float(agent["cameraHorizon"]),
                "pitch": rotation["x"],
                "field_of_view": float(metadata.get("fov", 60.0)),
            }
            return Image.fromarray(event.third_party_camera_frames[-1]), masks, camera
        raise InterventionValidationError(
            "No predeclared small viewpoint perturbation retained a 1% target mask"
        )

    def generate(self, candidate: Candidate) -> TripletRecord:
        plan = _load_plan(candidate.plan_path)
        triplet_id = _candidate_triplet_id(candidate)
        triplet_dir = self.output_dir / "triplets" / triplet_id

        variants: dict[str, VariantRecord] = {}
        histories = []
        targets = []
        for variant_name in ("anchor", "causal", "nuisance"):
            variant, history, target = self._build_variant(
                plan, candidate, triplet_id, variant_name, triplet_dir
            )
            variants[variant_name] = variant
            histories.append(history)
            targets.append(target)

        if not all(history == histories[0] for history in histories[1:]):
            raise InterventionValidationError("Variant histories are not identical")
        if not all(target == targets[0] for target in targets[1:]):
            raise InterventionValidationError(
                "Variant target identities are not identical"
            )

        causal_diffs = _diff(
            variants["anchor"].symbolic_state,
            variants["causal"].symbolic_state,
        )
        nuisance_diffs = _diff(
            variants["anchor"].symbolic_state,
            variants["nuisance"].symbolic_state,
        )
        expected_path = (
            f"objects.{targets[0]['predicate_object_name']}.{candidate.predicate}"
        )
        allowed_diffs = {expected_path}
        if self.allow_target_articulation and candidate.predicate == "isOpen":
            target_prefix = f"objects.{targets[0]['predicate_object_name']}."
            allowed_diffs.update(
                path
                for path in causal_diffs
                if path.startswith(f"{target_prefix}position.")
                or path.startswith(f"{target_prefix}rotation.")
            )
        if set(causal_diffs) != allowed_diffs:
            raise InterventionValidationError(
                f"Causal variant changed more than the target predicate: {causal_diffs}"
            )
        if nuisance_diffs:
            raise InterventionValidationError(
                f"Nuisance variant changed symbolic state: {nuisance_diffs}"
            )

        nuisance_rms = _image_rms(
            self.output_dir / variants["anchor"].image,
            self.output_dir / variants["nuisance"].image,
        )
        if nuisance_rms < 1.0:
            raise InterventionValidationError(
                f"Nuisance image change too small (RMS={nuisance_rms:.3f})"
            )

        anchor_actions = {a.canonical() for a in variants["anchor"].acceptable_actions}
        causal_actions = {a.canonical() for a in variants["causal"].acceptable_actions}
        if anchor_actions & causal_actions:
            raise InterventionValidationError(
                "Causal acceptable action set did not change disjointly"
            )

        validation = {
            "passed": True,
            "causal_symbolic_differences": causal_diffs,
            "allowed_causal_symbolic_differences": sorted(allowed_diffs),
            "nuisance_symbolic_differences": nuisance_diffs,
            "nuisance_image_rms": nuisance_rms,
            "action_sets_disjoint": True,
            "certification_contract": {
                "tier": self.certification_tier,
                "minimum_bbox_area_ratio": self.min_bbox_area_ratio,
                "target_local_articulation_allowed": self.allow_target_articulation,
            },
        }
        record = TripletRecord(
            triplet_id=triplet_id,
            predicate=candidate.predicate,
            task_type=plan.name.split("__", 1)[0],
            source_plan=str(candidate.plan_path),
            instruction=plan.task_description,
            scene=plan.scene,
            history=histories[0],
            action_interface=ACTION_INTERFACE,
            target=targets[0],
            variants=variants,
            validation=validation,
            certification_tier=self.certification_tier,
            causal_state_unit=triplet_id,
        )
        with (triplet_dir / "triplet.json").open("w", encoding="utf-8") as handle:
            json.dump(record.to_dict(), handle, indent=2, default=_json_default)
        return record


def _candidate_triplet_id(candidate: Candidate) -> str:
    base = candidate.plan_path.parent.name.split(" [", 1)[0]
    return f"{candidate.predicate.lower()}__{base}"


def generate_dataset(
    dataset_dir: Path,
    output_dir: Path,
    max_per_predicate: int,
    predicate_limits: dict[str, int] | None = None,
    manifest_name: str = "dataset.json",
    seed_rejections_path: Path | None = None,
    allow_target_articulation: bool = False,
) -> dict[str, Any]:
    generator = TripletGenerator(
        output_dir, allow_target_articulation=allow_target_articulation
    )
    candidates = discover_candidates(dataset_dir)
    # Small pilots should cover visual categories before repeating scenes.
    candidates["isDirty"] = _round_robin_target_types(candidates["isDirty"])
    candidates["isOpen"] = _round_robin_candidates(
        candidates["isOpen"], "decision_type"
    )
    triplets = []
    for triplet_path in sorted((output_dir / "triplets").glob("*/triplet.json")):
        with triplet_path.open("r", encoding="utf-8") as handle:
            triplets.append(json.load(handle))

    existing_ids = {triplet["triplet_id"] for triplet in triplets}

    rejected = []
    existing_manifest_path = output_dir / manifest_name
    if existing_manifest_path.exists():
        with existing_manifest_path.open("r", encoding="utf-8") as handle:
            existing_manifest = json.load(handle)
        rejected.extend(existing_manifest.get("rejected", []))
    if seed_rejections_path is not None:
        with seed_rejections_path.open("r", encoding="utf-8") as handle:
            seed = json.load(handle)
        known = {
            (item.get("predicate"), item.get("source_plan"), item.get("error"))
            for item in rejected
        }
        for item in seed.get("rejected", seed.get("rejected_candidates_detail", [])):
            key = (item.get("predicate"), item.get("source_plan"), item.get("error"))
            if key not in known:
                rejected.append(item)
                known.add(key)
    rejected_keys = {(item["predicate"], item["source_plan"]) for item in rejected}

    def persist() -> dict[str, Any]:
        ordered_triplets = sorted(
            triplets, key=lambda item: (item["predicate"], item["triplet_id"])
        )
        manifest = {
            "format_version": 2,
            "generator": "AsgardBench.causal.generator",
            "dataset_dir": str(dataset_dir),
            "max_per_predicate": max_per_predicate,
            "predicate_limits": predicate_limits or {},
            "allow_target_articulation": allow_target_articulation,
            "triplets": ordered_triplets,
            "rejected": rejected,
        }
        with (output_dir / manifest_name).open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, default=_json_default)
        _write_validation_report(manifest, output_dir / "validation_report.json")
        # Submission-level aliases keep accepted and rejected evidence easy to
        # inspect without parsing the full manifest.  They are deterministic
        # projections, not a second source of truth.
        with (output_dir / "accepted_samples.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(ordered_triplets, handle, indent=2, default=_json_default)
        with (output_dir / "rejected_samples.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(rejected, handle, indent=2, default=_json_default)
        return manifest

    for predicate, predicate_candidates in candidates.items():
        if predicate == "isToggled":
            # Every accepted wash checkpoint also admits a second,
            # independently validated causal question: with the dirty dish and
            # sponge fixed, whether the faucet is running determines if
            # TOGGLE_ON or CLEAN is executable.  Derive this *after* the dirty
            # phase so a fresh run need not be invoked twice.
            derived_keys = {
                (str(item.plan_path), item.target_name)
                for item in predicate_candidates
                if item.decision_type == "Faucet"
            }
            for triplet in sorted(triplets, key=lambda item: item["triplet_id"]):
                if triplet["predicate"] != "isDirty":
                    continue
                target_type = triplet["target"]["predicate_object_type"]
                if target_type == "Mirror":
                    continue
                key = (
                    str(triplet["source_plan"]),
                    triplet["target"]["predicate_object_name"],
                )
                if key in derived_keys:
                    continue
                predicate_candidates.append(
                    Candidate(
                        predicate="isToggled",
                        plan_path=Path(triplet["source_plan"]),
                        target_type=target_type,
                        target_name=triplet["target"]["predicate_object_name"],
                        decision_type="Faucet",
                        checkpoint_pose=triplet["variants"]["anchor"]["symbolic_state"][
                            "agentPose"
                        ],
                        generation_stage="derived_faucet_checkpoint",
                    )
                )
                derived_keys.add(key)
            predicate_candidates[:] = _round_robin_candidates(
                predicate_candidates, "decision_type"
            )
        accepted = sum(t["predicate"] == predicate for t in triplets)
        predicate_limit = (predicate_limits or {}).get(predicate, max_per_predicate)
        for candidate in predicate_candidates:
            if accepted >= predicate_limit:
                break
            triplet_id = _candidate_triplet_id(candidate)
            if triplet_id in existing_ids:
                continue
            rejection_key = (predicate, str(candidate.plan_path))
            if rejection_key in rejected_keys:
                continue
            try:
                record = generator.generate(candidate)
                triplets.append(record.to_dict())
                existing_ids.add(record.triplet_id)
                accepted += 1
                print(f"ACCEPT {record.triplet_id}")
            except (
                Exception
            ) as error:  # Keep pilot generation moving; audit every rejection.
                try:
                    rejected_plan = _load_plan(candidate.plan_path)
                    scene = rejected_plan.scene
                    task_type = rejected_plan.name.split("__", 1)[0]
                except Exception:
                    scene = None
                    task_type = None
                message = f"{type(error).__name__}: {error}"
                rejected.append(
                    {
                        "predicate": predicate,
                        "source_plan": str(candidate.plan_path),
                        "object": candidate.decision_type or candidate.target_type,
                        "target_object": candidate.target_type,
                        "scene": scene,
                        "task_type": task_type,
                        "generation_stage": candidate.generation_stage,
                        "rejection_reason": _rejection_category(message),
                        "error": message,
                    }
                )
                rejected_keys.add(rejection_key)
                print(f"REJECT {candidate.plan_path.parent.name}: {error}")
            persist()

    return persist()


def _rejection_category(message: str) -> str:
    checks = (
        ("changed more than", "causal_state_drift"),
        ("Nuisance variant changed", "nuisance_state_drift"),
        ("too small", "target_too_small"),
        ("No 2D detection", "target_not_detected"),
        ("not visible", "target_not_visible"),
        ("expected", "predicate_state_mismatch"),
        ("failed", "simulator_action_failure"),
        ("action set", "action_set_failure"),
    )
    return next((label for text, label in checks if text in message), "other")


def _write_validation_report(manifest: dict[str, Any], path: Path) -> None:
    triplets = manifest["triplets"]
    rejected = manifest["rejected"]
    bbox_areas = [
        variant["validation"]["predicate_bbox_area_ratio"]
        for triplet in triplets
        for variant in triplet["variants"].values()
    ]
    nuisance_rms = [triplet["validation"]["nuisance_image_rms"] for triplet in triplets]
    report = {
        "accepted_triplets": len(triplets),
        "rejected_candidates": len(rejected),
        "predicate_distribution": dict(
            sorted(Counter(t["predicate"] for t in triplets).items())
        ),
        "scene_distribution": dict(
            sorted(Counter(t["scene"] for t in triplets).items())
        ),
        "decision_object_distribution": dict(
            sorted(
                Counter(t["target"]["predicate_object_type"] for t in triplets).items()
            )
        ),
        "checkpoint_object_distribution": dict(
            sorted(
                Counter(
                    str(t["target"].get("held_object", "none")) for t in triplets
                ).items()
            )
        ),
        "task_type_distribution": dict(
            sorted(Counter(t["task_type"] for t in triplets).items())
        ),
        "rejected_reason_distribution": dict(
            sorted(Counter(_rejection_category(r["error"]) for r in rejected).items())
        ),
        "rejected_candidates_detail": rejected,
        "validation_ranges": {
            "predicate_bbox_area_ratio_min": min(bbox_areas, default=None),
            "predicate_bbox_area_ratio_max": max(bbox_areas, default=None),
            "lighting_nuisance_rms_min": min(nuisance_rms, default=None),
            "lighting_nuisance_rms_max": max(nuisance_rms, default=None),
        },
        "validation_contract": {
            "causal_only_target_predicate_diff": True,
            "nuisance_zero_symbolic_diff": True,
            "target_bbox_area_ratio_minimum": 0.01,
            "causal_action_sets_disjoint": True,
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=_json_default)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate validated intervention triplets"
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("Generated/magt_benchmark")
    )
    parser.add_argument("--output", type=Path, default=Path("CausalPilot/data"))
    parser.add_argument("--max-per-predicate", type=int, default=1)
    parser.add_argument("--max-is-open", type=int)
    parser.add_argument("--max-is-dirty", type=int)
    parser.add_argument("--max-is-toggled", type=int)
    parser.add_argument("--max-is-cooked", type=int)
    parser.add_argument("--manifest-name", default="dataset.json")
    parser.add_argument(
        "--allow-target-articulation",
        action="store_true",
        help=(
            "Allow an isOpen intervention to change only the target object's "
            "root position/rotation in addition to isOpen (door articulation)."
        ),
    )
    parser.add_argument(
        "--seed-rejections",
        type=Path,
        help="Reuse already audited rejection records from a prior manifest/report",
    )
    args = parser.parse_args()
    limits = {
        predicate: value
        for predicate, value in (
            ("isOpen", args.max_is_open),
            ("isDirty", args.max_is_dirty),
            ("isToggled", args.max_is_toggled),
            ("isCooked", args.max_is_cooked),
        )
        if value is not None
    }
    manifest = generate_dataset(
        args.dataset,
        args.output,
        args.max_per_predicate,
        predicate_limits=limits,
        manifest_name=args.manifest_name,
        seed_rejections_path=args.seed_rejections,
        allow_target_articulation=args.allow_target_articulation,
    )
    print(
        f"Generated {len(manifest['triplets'])} triplets; "
        f"rejected {len(manifest['rejected'])} candidates"
    )


if __name__ == "__main__":
    main()
