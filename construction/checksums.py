"""Write the distributed-file integrity manifest (not an evaluation report)."""

import argparse
import subprocess
from pathlib import Path

from construction.common import sha256, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args()
    root = args.root.resolve()
    files = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
    )
    names = sorted({n.decode() for n in files.split(b"\0") if n})
    records = []
    for name in names:
        path = root / name
        if name == "checksums.json" or not path.is_file():
            continue
        records.append(
            {"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        )
    write_json(
        root / "checksums.json",
        {"algorithm": "sha256", "self_excluded": True, "files": records},
    )
    print(f"Integrity manifest: {len(records)} distributed files")


if __name__ == "__main__":
    main()
