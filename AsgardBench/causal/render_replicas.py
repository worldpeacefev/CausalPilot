"""Build a 200--300 item evaluation pack from certified causal state units.

Render replicas improve nuisance coverage; they are *not* counted as
independent causal interventions.  Every instance carries ``causal_state_unit``
so uncertainty can be clustered at the intervention level.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageStat

CORE = ("anchor", "causal", "nuisance")


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _rms(left: Image.Image, right: Image.Image) -> float:
    stat = ImageStat.Stat(
        ImageChops.difference(left.convert("RGB"), right.convert("RGB"))
    )
    return math.sqrt(sum(value * value for value in stat.rms) / len(stat.rms))


def _gamma(image: Image.Image, value: float) -> Image.Image:
    table = [round(255 * ((index / 255) ** value)) for index in range(256)]
    return image.convert("RGB").point(table * 3)


def _temperature(image: Image.Image, red: float, blue: float) -> Image.Image:
    red_channel, green_channel, blue_channel = image.convert("RGB").split()
    red_channel = red_channel.point(lambda value: max(0, min(255, round(value * red))))
    blue_channel = blue_channel.point(
        lambda value: max(0, min(255, round(value * blue)))
    )
    return Image.merge("RGB", (red_channel, green_channel, blue_channel))


def _channel_cast(
    image: Image.Image, red: float, green: float, blue: float
) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    array *= np.array([red, green, blue], dtype=np.float32)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB").copy()


def _resolution(image: Image.Image, scale: float) -> Image.Image:
    width, height = image.size
    reduced = image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.LANCZOS,
    )
    return reduced.resize((width, height), Image.Resampling.LANCZOS)


def _vignette(image: Image.Image, edge_scale: float) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = array.shape[:2]
    yy, xx = np.mgrid[-1 : 1 : complex(height), -1 : 1 : complex(width)]
    radius = np.clip(np.sqrt(xx * xx + yy * yy) / math.sqrt(2), 0, 1)
    mask = 1.0 - (1.0 - edge_scale) * radius**1.7
    return Image.fromarray(np.clip(array * mask[..., None], 0, 255).astype(np.uint8))


def _sensor_noise(image: Image.Image, sigma: float, seed: int) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    noise = np.random.default_rng(seed).normal(0.0, sigma, size=array.shape)
    return Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8), "RGB")


def _quantize(image: Image.Image, colors: int) -> Image.Image:
    return image.convert("RGB").quantize(colors=colors).convert("RGB")


def _illumination_gradient(image: Image.Image, amplitude: float) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    gradient = np.linspace(1.0 - amplitude, 1.0 + amplitude, array.shape[1])
    return Image.fromarray(
        np.clip(array * gradient[None, :, None], 0, 255).astype(np.uint8), "RGB"
    )


Transform = Callable[[Image.Image], Image.Image]


CORE_PROFILES: dict[str, tuple[Transform, Transform, dict[str, Any]]] = {
    "exposure": (
        lambda image: ImageEnhance.Brightness(image).enhance(0.92),
        lambda image: ImageEnhance.Brightness(image).enhance(0.68),
        {"context_brightness": 0.92, "nuisance_brightness": 0.68},
    ),
    "color_temperature": (
        lambda image: _temperature(image, 1.03, 0.97),
        lambda image: _temperature(image, 1.14, 0.84),
        {
            "context_red_blue_scale": [1.03, 0.97],
            "nuisance_red_blue_scale": [1.14, 0.84],
        },
    ),
    "contrast": (
        lambda image: ImageEnhance.Contrast(image).enhance(0.95),
        lambda image: ImageEnhance.Contrast(image).enhance(0.70),
        {"context_contrast": 0.95, "nuisance_contrast": 0.70},
    ),
    "gamma": (
        lambda image: _gamma(image, 0.96),
        lambda image: _gamma(image, 1.28),
        {"context_gamma": 0.96, "nuisance_gamma": 1.28},
    ),
    "saturation": (
        lambda image: ImageEnhance.Color(image).enhance(0.96),
        lambda image: ImageEnhance.Color(image).enhance(0.55),
        {"context_saturation": 0.96, "nuisance_saturation": 0.55},
    ),
}


LARGE_EXTRA_PROFILES: dict[str, tuple[Transform, Transform, dict[str, Any]]] = {
    "sharpness": (
        lambda image: ImageEnhance.Sharpness(image).enhance(1.05),
        lambda image: ImageEnhance.Sharpness(image).enhance(2.20),
        {"context_sharpness": 1.05, "nuisance_sharpness": 2.20},
    ),
    "gaussian_blur": (
        lambda image: image.filter(ImageFilter.GaussianBlur(radius=0.15)),
        lambda image: image.filter(ImageFilter.GaussianBlur(radius=1.35)),
        {"context_blur_radius": 0.15, "nuisance_blur_radius": 1.35},
    ),
    "jpeg_compression": (
        lambda image: _jpeg(image, 95),
        lambda image: _jpeg(image, 38),
        {"context_jpeg_quality": 95, "nuisance_jpeg_quality": 38},
    ),
    "resolution": (
        lambda image: _resolution(image, 0.90),
        lambda image: _resolution(image, 0.42),
        {"context_resolution_scale": 0.90, "nuisance_resolution_scale": 0.42},
    ),
    "vignette": (
        lambda image: _vignette(image, 0.96),
        lambda image: _vignette(image, 0.58),
        {"context_edge_scale": 0.96, "nuisance_edge_scale": 0.58},
    ),
    "sensor_noise": (
        lambda image: _sensor_noise(image, 1.0, 1729),
        lambda image: _sensor_noise(image, 10.0, 2718),
        {"context_noise_sigma": 1.0, "nuisance_noise_sigma": 10.0},
    ),
    "color_quantization": (
        lambda image: _quantize(image, 192),
        lambda image: _quantize(image, 24),
        {"context_colors": 192, "nuisance_colors": 24},
    ),
    "illumination_gradient": (
        lambda image: _illumination_gradient(image, 0.025),
        lambda image: _illumination_gradient(image, 0.28),
        {"context_gradient_amplitude": 0.025, "nuisance_gradient_amplitude": 0.28},
    ),
    "channel_cast": (
        lambda image: _channel_cast(image, 1.0, 1.01, 0.99),
        lambda image: _channel_cast(image, 0.88, 1.10, 0.94),
        {
            "context_rgb_scale": [1.0, 1.01, 0.99],
            "nuisance_rgb_scale": [0.88, 1.10, 0.94],
        },
    ),
}

LARGE_PROFILES = {**CORE_PROFILES, **LARGE_EXTRA_PROFILES}


def _save(image: Image.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.stem}.tmp{destination.suffix}")
    image.save(temporary)
    temporary.replace(destination)


def _crop_rms(
    left: Image.Image, right: Image.Image, boxes: tuple[list[int], list[int]]
) -> float:
    x1 = max(0, min(int(box[0]) for box in boxes))
    y1 = max(0, min(int(box[1]) for box in boxes))
    x2 = min(left.width, max(int(box[2]) for box in boxes))
    y2 = min(left.height, max(int(box[3]) for box in boxes))
    return _rms(left.crop((x1, y1, x2, y2)), right.crop((x1, y1, x2, y2)))


def _derived_instance(
    root: Path,
    triplet: dict[str, Any],
    profile: str,
    profiles: dict[str, tuple[Transform, Transform, dict[str, Any]]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    context_transform, nuisance_transform, parameters = profiles[profile]
    unit = triplet.get("causal_state_unit", triplet["triplet_id"])
    output = root / "render_replicas" / unit / profile
    source_anchor = Image.open(root / triplet["variants"]["anchor"]["image"]).convert(
        "RGB"
    )
    source_causal = Image.open(root / triplet["variants"]["causal"]["image"]).convert(
        "RGB"
    )
    images = {
        "anchor": context_transform(source_anchor),
        "causal": context_transform(source_causal),
        "nuisance": nuisance_transform(source_anchor),
    }
    nuisance_rms = _rms(images["anchor"], images["nuisance"])
    boxes = (
        triplet["variants"]["anchor"]["validation"]["predicate_bbox"],
        triplet["variants"]["causal"]["validation"]["predicate_bbox"],
    )
    causal_target_rms = _crop_rms(images["anchor"], images["causal"], boxes)
    certificate = {
        "profile": profile,
        "parameters": parameters,
        "state_source": "certified_simulator_snapshot",
        "symbolic_state_changed": False,
        "acceptable_action_branch_changed": False,
        "same_transform_for_anchor_and_causal": True,
        "nuisance_image_rms": nuisance_rms,
        "causal_target_union_bbox_rms": causal_target_rms,
        "minimum_nuisance_image_rms": 1.0,
        "minimum_causal_target_union_bbox_rms": 1.0,
    }
    if nuisance_rms < 1.0 or causal_target_rms < 1.0:
        return None, certificate

    variants: dict[str, Any] = {}
    for name in CORE:
        source_name = name if name != "nuisance" else "anchor"
        value = copy.deepcopy(triplet["variants"][source_name])
        destination = output / f"{name}.png"
        _save(images[name], destination)
        value["name"] = name
        value["image"] = str(destination.relative_to(root))
        value.setdefault("validation", {})["render_replica"] = certificate
        variants[name] = value

    instance = {
        **{
            key: copy.deepcopy(value)
            for key, value in triplet.items()
            if key != "variants"
        },
        "triplet_id": f"{unit}__render_{profile}",
        "causal_state_unit": unit,
        "replica_profile": profile,
        "independence_role": "correlated_render_replica",
        "variants": variants,
    }
    return instance, certificate


def _simulator_instance(
    triplet: dict[str, Any], suffix: str | None, profile: str
) -> dict[str, Any] | None:
    source_names = {
        name: name if suffix is None else f"{name}_{suffix}" for name in CORE
    }
    if not all(name in triplet["variants"] for name in source_names.values()):
        return None
    unit = triplet.get("causal_state_unit", triplet["triplet_id"])
    variants = {}
    for name, source_name in source_names.items():
        variants[name] = copy.deepcopy(triplet["variants"][source_name])
        variants[name]["name"] = name
    return {
        **{
            key: copy.deepcopy(value)
            for key, value in triplet.items()
            if key != "variants"
        },
        "triplet_id": f"{unit}__render_{profile}",
        "causal_state_unit": unit,
        "replica_profile": profile,
        "independence_role": (
            "primary_simulator_render"
            if suffix is None
            else "correlated_simulator_replica"
        ),
        "variants": variants,
    }


def build_pack(
    dataset_path: Path,
    output_path: Path,
    maximum: int = 300,
    include_optional_simulator_replicas: bool = False,
    profile_set: str = "core",
) -> dict[str, Any]:
    root = dataset_path.parent
    source = json.loads(dataset_path.read_text(encoding="utf-8"))
    profiles = CORE_PROFILES if profile_set == "core" else LARGE_PROFILES
    instances: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for triplet in source["triplets"]:
        base = _simulator_instance(triplet, None, "simulator_lighting")
        assert base is not None
        instances.append(base)
        for profile in profiles:
            instance, certificate = _derived_instance(root, triplet, profile, profiles)
            if instance is None:
                rejected.append(
                    {
                        "triplet_id": triplet["triplet_id"],
                        "predicate": triplet["predicate"],
                        "object": triplet["target"]["predicate_object_type"],
                        "scene": triplet["scene"],
                        "generation_stage": "render_replica",
                        "rejection_reason": "insufficient_visual_signal",
                        "profile": profile,
                        "certificate": certificate,
                    }
                )
            else:
                instances.append(instance)
        if include_optional_simulator_replicas:
            for suffix, profile in (
                ("viewpoint", "simulator_viewpoint"),
                ("material", "simulator_material"),
            ):
                instance = _simulator_instance(triplet, suffix, profile)
                if instance is not None:
                    instances.append(instance)

    # If future expansion exceeds the requested range, retain every primary
    # simulator render, then balance replicas round-robin across profiles and
    # predicates.  Selection never uses model behavior.
    if len(instances) > maximum:
        primary = [
            item
            for item in instances
            if item["replica_profile"] == "simulator_lighting"
        ]
        replicas = [
            item
            for item in instances
            if item["replica_profile"] != "simulator_lighting"
        ]
        replicas.sort(
            key=lambda item: (
                item["replica_profile"],
                item["predicate"],
                item["causal_state_unit"],
            )
        )
        instances = [*primary, *replicas[: maximum - len(primary)]]

    instances.sort(key=lambda item: item["triplet_id"])
    pack = {
        "format_version": 1,
        "generator": "AsgardBench.causal.render_replicas",
        "profile_set": profile_set,
        "source_dataset": str(dataset_path),
        "counting_policy": {
            "evaluation_triplets": "scorable anchor/causal/nuisance image triplets",
            "independent_unit": "causal_state_unit",
            "statistical_rule": "cluster confidence intervals by causal_state_unit",
            "balanced_primary_profiles": not include_optional_simulator_replicas,
            "optional_simulator_replicas": (
                "included"
                if include_optional_simulator_replicas
                else "separate auxiliary suites"
            ),
        },
        "triplets": instances,
        "rejected": rejected,
    }
    _atomic_json(output_path, pack)
    visual_signal_ranges = {}
    for profile in profiles:
        certificates = [
            item["variants"]["anchor"].get("validation", {}).get("render_replica")
            for item in instances
            if item["replica_profile"] == profile
        ]
        certificates = [item for item in certificates if item]
        if certificates:
            nuisance_values = [item["nuisance_image_rms"] for item in certificates]
            causal_values = [
                item["causal_target_union_bbox_rms"] for item in certificates
            ]
            visual_signal_ranges[profile] = {
                "nuisance_image_rms": [min(nuisance_values), max(nuisance_values)],
                "causal_target_union_bbox_rms": [
                    min(causal_values),
                    max(causal_values),
                ],
            }
    report = {
        "evaluation_triplets": len(instances),
        "independent_causal_state_units": len(
            {item["causal_state_unit"] for item in instances}
        ),
        "replica_profile_histogram": dict(
            sorted(Counter(item["replica_profile"] for item in instances).items())
        ),
        "predicate_histogram": dict(
            sorted(Counter(item["predicate"] for item in instances).items())
        ),
        "certification_tier_histogram": dict(
            sorted(
                Counter(
                    item.get("certification_tier", "pilot-v3-strict")
                    for item in instances
                ).items()
            )
        ),
        "rejected_render_replicas": len(rejected),
        "visual_signal_ranges": visual_signal_ranges,
        "within_requested_200_300_range": 200 <= len(instances) <= 300,
        "within_large_500_1000_range": 500 <= len(instances) <= 1000,
        "auxiliary_simulator_nuisance_availability": {
            "viewpoint_triplets": sum(
                all(f"{base}_viewpoint" in item["variants"] for base in CORE)
                for item in source["triplets"]
            ),
            "material_triplets": sum(
                all(f"{base}_material" in item["variants"] for base in CORE)
                for item in source["triplets"]
            ),
        },
    }
    report_path = (
        root / "render_replica_validation_report.json"
        if output_path.name == "benchmark.json"
        else output_path.with_name(f"{output_path.stem}_validation_report.json")
    )
    _atomic_json(report_path, report)
    return pack


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clustered evaluation triplets")
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/causal_states.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("CausalPilot/data_final/benchmark.json")
    )
    parser.add_argument("--maximum", type=int, default=300)
    parser.add_argument("--profile-set", choices=("core", "large"), default="core")
    parser.add_argument("--include-optional-simulator-replicas", action="store_true")
    args = parser.parse_args()
    pack = build_pack(
        args.dataset,
        args.output,
        args.maximum,
        args.include_optional_simulator_replicas,
        args.profile_set,
    )
    units = len({item["causal_state_unit"] for item in pack["triplets"]})
    print(
        f"Built {len(pack['triplets'])} evaluation triplets from {units} causal state units"
    )


if __name__ == "__main__":
    main()
