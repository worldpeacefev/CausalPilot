"""Bind proxy visual review annotations to the exact reviewed assets and labels.

This module verifies provenance, never assigns visual verdicts. Verdicts must
be supplied by a reviewer who actually inspected the corresponding triplets.
"""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from construction.common import ROLES, sha256


@lru_cache(maxsize=8192)
def _file_hash(path, size, modified_ns):
    return sha256(path)


def file_hash(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _file_hash(str(path), stat.st_size, stat.st_mtime_ns)


def binding(row, data_root):
    root = Path(data_root)
    context = {
        key: row[key]
        for key in (
            "triplet_id",
            "causal_state_unit",
            "predicate",
            "instruction",
            "history",
            "action_interface",
            "target",
            "certification_tier",
        )
    }
    context["acceptable_actions"] = {
        role: row["variants"][role]["acceptable_actions"] for role in ROLES
    }
    context_hash = hashlib.sha256(
        json.dumps(context, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    artifacts = {
        role: {
            field: file_hash(root / row["variants"][role][field])
            for field in ("image", "metadata", "symbolic_state_path")
        }
        for role in ROLES
    }
    return {"context_sha256": context_hash, "artifact_sha256": artifacts}


def load_reviews(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        identity = row["triplet_id"]
        if identity in result:
            raise ValueError(f"Duplicate visual review: {identity}")
        if row["verdict"] not in ("pass", "reject", "uncertain"):
            raise ValueError(f"Invalid visual verdict: {identity}")
        if row["reviewer_kind"] != "agent" or row["actual_human_review"] is not False:
            raise ValueError(
                "This release records agent proxy review, not human review"
            )
        if row["verdict"] == "pass" and not all(
            row["criteria"].get(key) is True
            for key in (
                "causal_state_visually_separable",
                "nuisance_retains_task_cue",
                "context_and_action_labels_consistent",
                "occlusion_and_artifacts_acceptable",
            )
        ):
            raise ValueError(f"Passing visual review has failed criteria: {identity}")
        result[identity] = row
    return result


def verify_binding(row, review, data_root):
    actual = binding(row, data_root)
    if any(review[key] != actual[key] for key in actual):
        raise ValueError(f"Stale visual review assets or labels: {row['triplet_id']}")
