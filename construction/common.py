"""Shared release taxonomy and deterministic serialization."""

import hashlib
import json
from pathlib import Path

ASPECTS = {
    "isDirty": "cleaning",
    "isOpen": "access",
    "isToggled": "device_control",
    "isCooked": "food_preparation",
    "isSliced": "food_preparation",
    "isFilledWithLiquid": "liquid_and_consumables",
    "isUsedUp": "liquid_and_consumables",
}
ROLES = ("anchor", "causal", "nuisance")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def diff(left, right, prefix=""):
    if type(left) is not type(right):
        return [prefix]
    if isinstance(left, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            name = f"{prefix}.{key}" if prefix else key
            result += (
                [name]
                if key not in left or key not in right
                else diff(left[key], right[key], name)
            )
        return result
    return [] if left == right else [prefix]


def certification_diff(left, right, tolerance=None):
    """Compare raw snapshots, optionally accounting for declared float noise.

    Only numeric geometry coordinates can receive tolerance. Predicate values,
    identity, inventory, and containment always require exact equality.
    """
    paths = diff(left, right)
    if tolerance is None:
        return paths
    result = []
    for path in paths:
        bits = path.split(".")
        field = bits[-2] if len(bits) >= 2 else None
        if field not in ("position", "rotation") or bits[-1] not in ("x", "y", "z"):
            result.append(path)
            continue
        a, b = left, right
        for key in bits:
            a, b = a[key], b[key]
        if not isinstance(a, (float, int)) or not isinstance(b, (float, int)):
            result.append(path)
            continue
        delta = abs(a - b) if field == "position" else abs((a - b + 180) % 360 - 180)
        if delta > tolerance[field]:
            result.append(path)
    return result


def snapshot(metadata):
    keys = (
        "isDirty",
        "isOpen",
        "isToggled",
        "isCooked",
        "isSliced",
        "isUsedUp",
        "isFilledWithLiquid",
        "fillLiquid",
        "isBroken",
        "isPickedUp",
        "temperature",
        "position",
        "rotation",
        "parentReceptacles",
        "receptacleObjectIds",
    )
    objects = {}
    for obj in metadata["objects"]:
        value = {k: obj.get(k) for k in keys}
        for key in ("parentReceptacles", "receptacleObjectIds"):
            if isinstance(value[key], list):
                value[key] = sorted(value[key])
        objects[obj["name"]] = value
    return {
        "objects": objects,
        "inventoryObjects": metadata.get("inventoryObjects", []),
        "agent": metadata["agent"],
    }
