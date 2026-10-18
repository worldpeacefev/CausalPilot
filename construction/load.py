"""Stream classified benchmark records and construct observation-only inputs."""

import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=2048)
def _metadata_object_types(path):
    """Immutable metadata paths recur across many nuisance replicas."""
    return frozenset(
        o["objectType"] for o in json.loads(Path(path).read_text())["objects"]
    )


def iter_triplets(
    data_root, *, predicate=None, scene=None, partition=None, metadata_only=True
):
    """Yield records from the catalog; all referenced paths resolve under data_root.

    metadata_only=True excludes the trace-derived isUsedUp tier by default.
    Set it to False explicitly to include that qualified extension.
    """
    root = Path(data_root)
    catalog = json.loads((root / "catalog.json").read_text())
    for shard in catalog["shards"]:
        if predicate and shard["predicate"] != predicate:
            continue
        if scene and shard["scene"] != scene:
            continue
        with (root / shard["manifest"]).open() as stream:
            for line in stream:
                row = json.loads(line)
                if partition and row["release_partition"] != partition:
                    continue
                if (
                    metadata_only
                    and row["certification_tier"] == "extension-intervention-trace-v1"
                ):
                    continue
                yield row


def observation_input(row, role, data_root):
    """Return fixed context plus one image, withholding labels/intervention IDs."""
    root = Path(data_root)
    types = row.get("available_object_types")
    if types is None:
        # Union across roles is fixed for the matched context, including slices.
        types = sorted(
            set().union(
                *(
                    _metadata_object_types(str((root / v["metadata"]).resolve()))
                    for v in row["variants"].values()
                )
            )
        )
    return {
        "instruction": row["instruction"],
        "history": row["history"],
        "action_interface": row["action_interface"],
        "available_object_types": types,
        "image": root / row["variants"][role]["image"],
    }
