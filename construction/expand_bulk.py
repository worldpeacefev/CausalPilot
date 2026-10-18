"""Materialize unreviewed nuisance replicas; these are never paper certificates.

uv run python -m construction.expand_bulk --workers 4
All paths in output records remain relative to data/. Rebuild the two collection
views with construction.partition after generation. Source manifests are kept.
"""

import argparse
import copy
import hashlib
import io
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from construction.common import ROLES, sha256, write_json
from construction.load import iter_triplets

FAMILIES = (
    "exposure",
    "contrast",
    "saturation",
    "gamma",
    "temperature",
    "gaussian_blur",
    "motion_blur",
    "jpeg",
    "quantization",
    "resolution",
    "vignette",
    "illumination_gradient",
    "gaussian_noise",
    "shot_noise",
    "sensor_banding",
    "chromatic_shift",
    "translation",
    "view_scale",
    "occlusion",
    "pixel_dropout",
)


def transform(image, family, strength, seed):
    """Common random fields and parameters for matched A/C images."""
    s = strength
    if family in ("exposure", "contrast", "saturation"):
        cls = {
            "exposure": ImageEnhance.Brightness,
            "contrast": ImageEnhance.Contrast,
            "saturation": ImageEnhance.Color,
        }[family]
        return cls(image).enhance(max(0.03, 1 - 0.9 * s))
    if family == "gamma":
        table = [round(255 * (v / 255) ** (1 + 2.5 * s)) for v in range(256)]
        return image.point(table * 3)
    if family == "gaussian_blur":
        return image.filter(ImageFilter.GaussianBlur(5 * s))
    if family == "motion_blur":
        a = np.asarray(image, dtype=float)
        radius = max(1, round(12 * s))
        p = np.pad(a, ((0, 0), (radius, radius), (0, 0)), mode="edge")
        return Image.fromarray(
            np.mean(
                [p[:, i : i + a.shape[1]] for i in range(2 * radius + 1)], axis=0
            ).astype("uint8")
        )
    if family == "jpeg":
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=max(3, round(98 - 94 * s)))
        buf.seek(0)
        with Image.open(buf) as decoded:
            return decoded.convert("RGB").copy()
    if family == "quantization":
        # A fixed RGB lattice, rather than a separately fitted palette per role.
        levels = max(2, round(32 * (1 - s)))
        a = np.asarray(image, dtype=float)
        return Image.fromarray(
            (np.round(a / 255 * (levels - 1)) / (levels - 1) * 255).astype("uint8")
        )
    if family == "resolution":
        size = max(8, round(image.width * (1 - 0.94 * s)))
        return image.resize((size, size), Image.Resampling.BILINEAR).resize(image.size)
    if family == "translation":
        d = max(1, round(image.width * 0.25 * s))
        result = Image.new("RGB", image.size, (127, 127, 127))
        result.paste(image, (d, -d))
        return result
    if family == "view_scale":
        side = max(8, round(image.width * (1 - 0.7 * s)))
        result = Image.new("RGB", image.size, (127, 127, 127))
        result.paste(image.resize((side, side)), ((image.width - side) // 2,) * 2)
        return result
    a = np.asarray(image, dtype=float).copy()
    height, width = a.shape[:2]
    rng = np.random.default_rng(seed)
    if family == "temperature":
        a *= np.array([1 + 0.7 * s, 1, 1 - 0.7 * s])
    elif family == "vignette":
        y, x = np.mgrid[-1 : 1 : complex(height), -1 : 1 : complex(width)]
        a *= np.maximum(0.03, 1 - 0.85 * s * (x * x + y * y))[..., None]
    elif family == "illumination_gradient":
        a *= np.linspace(1 - 0.9 * s, 1 + 0.9 * s, width)[None, :, None]
    elif family == "gaussian_noise":
        a += rng.normal(size=a.shape) * (60 * s)
    elif family == "shot_noise":
        a += rng.normal(size=a.shape) * np.sqrt(a * 255 / (4096 ** (1 - s)))
    elif family == "sensor_banding":
        phase = rng.uniform(0, 2 * np.pi)
        a *= (1 + 0.7 * s * np.sin(np.arange(height) * 2 * np.pi / 19 + phase))[
            :, None, None
        ]
    elif family == "chromatic_shift":
        d = max(1, round(10 * s))
        a[:, d:, 0] = a[:, :-d, 0]
        a[:, :d, 0] = a[:, d : d + 1, 0]
        a[:, :-d, 2] = a[:, d:, 2]
        a[:, -d:, 2] = a[:, -d - 1 : -d, 2]
    elif family == "occlusion":
        side = max(1, round(width * 0.55 * s))
        # The same normalized position at all strengths, shared across A/C.
        x, y = rng.integers(0, max(1, width - side + 1), size=2)
        a[y : y + side, x : x + side] = 127
    elif family == "pixel_dropout":
        a[rng.uniform(size=(height, width)) < 0.5 * s] = 127
    else:
        raise ValueError(family)
    return Image.fromarray(np.clip(a, 0, 255).astype("uint8"))


def expand_unit(row, root, severities, size):
    data = Path(root) / "data"
    originals = {}
    sources = {}
    for role in ("anchor", "causal"):
        path = data / row["variants"][role]["image"]
        sources[role] = sha256(path)
        with Image.open(path) as im:
            originals[role] = im.convert("RGB").resize(
                (size, size), Image.Resampling.LANCZOS
            )
    folder = (
        Path("extended/scenarios")
        / row["scenario_aspect"]
        / row["predicate"]
        / row["scene"]
        / row["causal_state_unit"]
    )
    rows = []
    for family in FAMILIES:
        for level in range(1, severities + 1):
            profile = f"bulk_{family}_s{level:02}"
            seed = int.from_bytes(
                hashlib.sha256(
                    f"20261019:{row['causal_state_unit']}:{profile}".encode()
                ).digest()[:4],
                "big",
            )
            context, nuisance = 0.025 + 0.018 * level, level / severities
            result = copy.deepcopy(row)
            result["triplet_id"] = row["causal_state_unit"] + "__" + profile
            result["release_partition"] = "bulk_extension"
            result["collection"] = "extended"
            result["replica_profile"] = profile
            result["source_certification_tier"] = row["certification_tier"]
            result["certification_tier"] = "extension-unreviewed-render-v1"
            result["review_status"] = "unreviewed_expansion"
            result["independence_role"] = "correlated_render_replica"
            result["construction"] = {
                "algorithm_version": 1,
                "source_triplet_id": row["triplet_id"],
                "source_image_sha256": sources,
                "family": family,
                "severity": level,
                "context_strength": context,
                "nuisance_strength": nuisance,
                "seed": seed,
                "size": [size, size],
                "image_encoding": {"format": "JPEG", "quality": 82},
                "quality_screened": False,
                "may_remove_decision_evidence": True,
                "common_random_field_for_anchor_and_causal": True,
                "labels_inherited_from_simulator_source": True,
            }
            result.pop("primary_visual_signal", None)
            # These are source certificates only, not certificates of the output.
            result["source_validation"] = result.pop("validation", {})
            for role in ROLES:
                source_role = "anchor" if role == "nuisance" else role
                variant = copy.deepcopy(row["variants"][source_role])
                variant["source_validation"] = variant.pop("validation", {})
                variant["source_image"] = variant["image"]
                path = folder / profile / f"{role}.jpg"
                destination = data / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                rendered = transform(
                    originals[source_role],
                    family,
                    nuisance if role == "nuisance" else context,
                    seed,
                )
                rendered.save(destination, format="JPEG", quality=82)
                variant["image"], variant["name"] = path.as_posix(), role
                # Source bounding boxes are retained as source coordinates only.
                result["variants"][role] = variant
            rows.append(result)
    manifest = data / folder / "triplets.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
            for r in rows
        )
    )
    return {
        "manifest": manifest.relative_to(data).as_posix(),
        "scenario_aspect": row["scenario_aspect"],
        "predicate": row["predicate"],
        "scene": row["scene"],
        "physical_scene_type": row["physical_scene_type"],
        "triplets": len(rows),
        "causal_state_units": 1,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--severities", type=int, default=6)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.severities < 1 or args.size < 32:
        parser.error("positive severity count and size >=32 required")
    units = {}
    for row in iter_triplets(args.root / "data", metadata_only=False):
        if row.get("replica_profile", "simulator_lighting") == "simulator_lighting":
            units.setdefault(row["causal_state_unit"], row)
    work = partial(
        expand_unit, root=str(args.root), severities=args.severities, size=args.size
    )
    shards = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, shard in enumerate(pool.map(work, units.values())):
            shards.append(shard)
            print(f"expanded {index + 1}/{len(units)}", flush=True)
    write_json(
        args.root / "data/extended/bulk_catalog.json",
        {
            "format_version": 2,
            "path_base": "data",
            "review_status": "unreviewed_expansion",
            "families": list(FAMILIES),
            "severities": args.severities,
            "size": [args.size, args.size],
            "shards": shards,
            "counts": {
                "causal_state_units": len(units),
                "triplets": sum(s["triplets"] for s in shards),
            },
        },
    )


if __name__ == "__main__":
    main()
