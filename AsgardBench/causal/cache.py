"""Content-addressed persistent inference cache."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

CACHE_SCHEMA = "causalpilot-inference-cache-v1"


def image_sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_key(
    *, provider: str, model: str, prompt: str, image_hash: str | None,
    decoding: dict[str, Any],
) -> str:
    payload = {
        "schema": CACHE_SCHEMA, "provider": provider, "model": model,
        "prompt": prompt, "image_sha256": image_hash, "decoding": decoding,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class InferenceCache:
    """Store raw responses separately from parsed/scored experiment rows."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self.path(key)
        if not path.exists():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema") != CACHE_SCHEMA or value.get("key") != key:
            raise ValueError(f"Invalid inference cache record: {path}")
        return value

    def put(self, key: str, value: dict[str, Any]) -> Path:
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"schema": CACHE_SCHEMA, "key": key, **value}
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
        temporary.replace(path)
        return path
