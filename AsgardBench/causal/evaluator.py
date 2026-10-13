from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from AsgardBench.causal.cache import InferenceCache, cache_key, image_sha256
from AsgardBench.causal.metrics import grouped_metrics
from AsgardBench.causal.providers import ModelSpec, build_backend

METHODS = ("direct", "no_image", "oracle_state")
DEFAULT_VARIANTS = ("anchor", "causal", "nuisance")
PROMPT_VARIANTS = ("v1", "v2", "v3")


def _json_safe(value: Any) -> Any:
    """Convert partial-run NaNs to null without weakening final metrics."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _object_types(metadata_path: Path) -> list[str]:
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return sorted({obj["objectType"] for obj in metadata["objects"]})


def build_prompt(
    triplet: dict[str, Any],
    variant: dict[str, Any],
    method: str,
    object_types: list[str],
    prompt_variant: str = "v1",
) -> str:
    history = "\n".join(
        f"{index + 1}. {step}" for index, step in enumerate(triplet["history"])
    )
    interface_actions = []
    for action in triplet["action_interface"]:
        if action == "put OBJECT":  # Backward-compatible with v0 manifests.
            action = "put DESTINATION_RECEPTACLE"
        interface_actions.append(action)
    interface = "\n".join(f"- {action}" for action in interface_actions)
    state_section = ""
    if method == "oracle_state":
        facts = "\n".join(f"- {fact}" for fact in variant["oracle_state"])
        state_section = (
            "\nGROUND-TRUTH TASK-RELEVANT SYMBOLIC STATE "
            "(use it for planning; it does not state the answer):\n"
            f"{facts}\n"
        )
    elif method == "no_image":
        state_section = "\nNo image or symbolic state is available in this baseline.\n"
    else:
        state_section = (
            "\nInspect the attached current image for the present world state. "
            "Do not replace visual evidence with a typical task script.\n"
        )

    v1 = f"""You are choosing exactly one next high-level action in AI2-THOR.

TASK INSTRUCTION:
{triplet['instruction']}

ACTION HISTORY (already completed successfully):
{history}
{state_section}
AVAILABLE OBJECT TYPES:
{', '.join(object_types)}

ACTION INTERFACE:
{interface}

ACTION SEMANTICS (same in every sample):
- The object field for put is the destination receptacle, not the held item.
- A closed container must be opened before put; an open container can receive put.
- Washing a dirty dish follows: dish in SinkBasin, hold DishSponge, Faucet on,
  clean the dish, then toggle_off Faucet before removing/storing the dish.
- cook OBJECT invokes the benchmark's native high-level cooking routine; do not
  cook an item that is already visibly cooked.
- toggle_on/off targets the device. done none is valid only when the task goal is
  already satisfied in the current state.

Choose the action that is appropriate for the CURRENT state. If the task goal is
already satisfied, choose done none. Return only one JSON object with exactly this
schema: {{"action": "lowercase_action", "object": "ObjectType_or_none"}}
""".strip()
    if prompt_variant == "v1":
        return v1
    if prompt_variant not in PROMPT_VARIANTS:
        raise ValueError(f"Unknown prompt variant: {prompt_variant}")
    common = f"""TASK: {triplet['instruction']}

COMPLETED ACTIONS:
{history}
{state_section}
VALID OBJECT TYPES: {', '.join(object_types)}

VALID ACTION FORMS:
{interface}

FIXED SEMANTICS:
- For put, name the destination receptacle rather than the held object.
- Open a closed container before put; put directly into an open container.
- For a dirty dish in SinkBasin while holding DishSponge with Faucet on, clean it and then toggle_off Faucet before removal/storage.
- cook OBJECT is the benchmark native cooking routine; never cook an already visibly cooked item.
- toggle_on/off names the device; done none is allowed only when the current task goal is already satisfied.
""".strip()
    if prompt_variant == "v2":
        return f"""Select one next AI2-THOR high-level action for the CURRENT state.

{common}

Use supplied current-state evidence, not a stereotyped task sequence. Output only JSON with exactly two fields:
{{"action": "lowercase_action", "object": "ObjectType_or_none"}}""".strip()
    return f"""Return exactly one next action as {{"action": "lowercase_action", "object": "ObjectType_or_none"}} and no other text.

Decide from the CURRENT state; a usual task script is not evidence.

{common}

If the task goal is already satisfied now, use done none.""".strip()


def parse_action(
    response: str, valid_object_types: list[str]
) -> tuple[str | None, str | None]:
    action = None
    obj = None
    candidates = [response]
    candidates.extend(re.findall(r"\{[^{}]+\}", response, flags=re.DOTALL))
    for candidate in reversed(candidates):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "action" in data:
            action = str(data["action"])
            obj = data.get("object")
            obj = None if obj is None else str(obj)
            break

    if action is None:
        match = re.search(
            r"(?i)(?:action\s*[:=]\s*)?"
            r"(toggle_on|toggle_off|clean|close|cook|empty|drink|find|open|pickup|put|slice|spray|done)"
            r"(?:\s+|[\"']?\s*,?\s*[\"']?object[\"']?\s*[:=]\s*[\"']?)"
            r"([A-Za-z_]+)?",
            response,
        )
        if match:
            action, obj = match.group(1), match.group(2)

    if action is None:
        return None, None
    action = action.strip().lower().replace(" ", "_")
    if action in {"done", "done_none"}:
        return "done", None
    if obj is None:
        return action, None
    object_lookup = {name.lower(): name for name in valid_object_types}
    normalized_object = obj.strip().strip("`\"'.:, ").lower()
    return action, object_lookup.get(normalized_object, obj.strip())


def canonical_action(action: str | None, obj: str | None) -> str | None:
    if action is None:
        return None
    return f"{action.lower()} {(obj or 'none').lower()}"


class NextActionEvaluator:
    def __init__(
        self,
        dataset_path: Path,
        output_dir: Path,
        models: list[str],
        methods: list[str],
        temperature: float,
        max_completion_tokens: int,
        bootstrap_iterations: int,
        variants: list[str] | None = None,
        extra_body: dict[str, Any] | None = None,
        model_specs: list[ModelSpec] | None = None,
        cache_dir: Path | None = None,
        retry_attempts: int = 2,
        prompt_variant: str = "v1",
    ):
        with dataset_path.open("r", encoding="utf-8") as handle:
            self.dataset = json.load(handle)
        self.dataset_root = dataset_path.parent
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if model_specs is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is required")
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            model_specs = [
                ModelSpec(
                    name=model,
                    model=model,
                    base_url=base_url,
                    extra_body=extra_body or {},
                )
                for model in models
            ]
        if len({spec.name for spec in model_specs}) != len(model_specs):
            raise ValueError("Model-spec names must be unique")
        self.model_specs = {spec.name: spec for spec in model_specs}
        self.models = list(self.model_specs)
        self.methods = methods
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.bootstrap_iterations = bootstrap_iterations
        self.variants = variants or list(DEFAULT_VARIANTS)
        self.extra_body = extra_body or {}
        self.retry_attempts = retry_attempts
        self.prompt_variant = prompt_variant
        self.cache = InferenceCache(cache_dir or output_dir.parent / "_inference_cache")
        self.inference_audit = {
            "requested_inputs": 0,
            "persistent_cache_hits": 0,
            "provider_attempts": 0,
            "provider_successes": 0,
            "provider_failures": 0,
            "reported_prompt_tokens": 0,
            "reported_completion_tokens": 0,
            "estimated_cost": None,
        }
        self.backends = {
            name: build_backend(spec) for name, spec in self.model_specs.items()
        }
        self.response_cache: dict[
            tuple[str, str], tuple[str, dict[str, Any], float]
        ] = {}
        self.results_path = self.output_dir / "results.jsonl"
        self.rows = self._load_existing()
        self.completed = {
            (row["model"], row["method"], row["triplet_id"], row["variant"])
            for row in self.rows
        }
        self.config = {
            "dataset": str(dataset_path),
            "models": self.models,
            "model_specs": [spec.public_dict() for spec in self.model_specs.values()],
            "methods": methods,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
            "variants": self.variants,
            "extra_body": self.extra_body,
            "cache_dir": str(self.cache.root),
            "retry_attempts": retry_attempts,
            "prompt_variant": prompt_variant,
            "git_commit": _git_commit(),
        }
        config_path = self.output_dir / "config.json"
        run_configs = []
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                previous = json.load(handle)
            run_configs = previous.get("runs", [previous])
        if self.config not in run_configs:
            run_configs.append(self.config)
        temporary_config = config_path.with_name(
            f"{config_path.name}.{os.getpid()}.tmp"
        )
        with temporary_config.open("w", encoding="utf-8") as handle:
            json.dump({"runs": run_configs}, handle, indent=2)
        temporary_config.replace(config_path)

    def _load_existing(self) -> list[dict[str, Any]]:
        if not self.results_path.exists():
            return []
        rows = []
        with self.results_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    @staticmethod
    def _image_content(image_path: Path) -> dict[str, Any]:
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encoded}"},
        }

    def _call(
        self,
        model: str,
        prompt: str,
        image_path: Path | None,
    ) -> tuple[str, dict[str, Any], float]:
        self.inference_audit["requested_inputs"] += 1
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image_path is not None:
            content.append(self._image_content(image_path))
        spec = self.model_specs[model]
        merged_extra_body = {**spec.extra_body, **self.extra_body}
        decoding = {
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens,
            "extra_body": merged_extra_body,
        }
        image_hash = image_sha256(image_path)
        key = cache_key(
            provider=f"{spec.backend}:{spec.base_url}",
            model=spec.model,
            prompt=prompt,
            image_hash=image_hash,
            decoding=decoding,
        )
        cached = self.cache.get(key)
        if cached is not None:
            self.inference_audit["persistent_cache_hits"] += 1
            return cached["raw_output"], cached["response"], 0.0
        errors = []
        for attempt in range(self.retry_attempts + 1):
            self.inference_audit["provider_attempts"] += 1
            try:
                raw, response, latency = self.backends[model].complete(
                    model=spec.model,
                    content=content,
                    temperature=self.temperature,
                    max_completion_tokens=self.max_completion_tokens,
                    extra_body=merged_extra_body,
                )
                self.cache.put(
                    key,
                    {
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "logical_model": model,
                        "served_model": spec.model,
                        "provider": spec.backend,
                        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                        "image_sha256": image_hash,
                        "decoding": decoding,
                        "raw_output": raw,
                        "response": response,
                        "latency_seconds": latency,
                        "attempt": attempt + 1,
                    },
                )
                self.inference_audit["provider_successes"] += 1
                usage = response.get("usage") or {}
                self.inference_audit["reported_prompt_tokens"] += int(
                    usage.get("prompt_tokens") or 0
                )
                self.inference_audit["reported_completion_tokens"] += int(
                    usage.get("completion_tokens") or 0
                )
                return raw, response, latency
            except Exception as error:  # Provider exception types vary.
                self.inference_audit["provider_failures"] += 1
                errors.append(f"{type(error).__name__}: {error}")
        raise RuntimeError(
            f"provider call failed after {self.retry_attempts + 1} attempts: "
            + " | ".join(errors)
        )

    def run(
        self,
        limit_triplets: int | None = None,
        shard_index: int = 0,
        shard_count: int = 1,
    ) -> list[dict[str, Any]]:
        if shard_count < 1 or not 0 <= shard_index < shard_count:
            raise ValueError("Require 0 <= shard_index < shard_count")
        triplets = self.dataset["triplets"]
        if limit_triplets is not None:
            triplets = triplets[:limit_triplets]
        triplets = [
            item
            for index, item in enumerate(triplets)
            if index % shard_count == shard_index
        ]
        total = (
            len(self.models) * len(self.methods) * len(triplets) * len(self.variants)
        )
        call_number = 0
        for model in self.models:
            for method in self.methods:
                for triplet in triplets:
                    for variant_name in self.variants:
                        call_number += 1
                        key = (model, method, triplet["triplet_id"], variant_name)
                        if key in self.completed:
                            continue
                        variant = triplet["variants"][variant_name]
                        image_path = self.dataset_root / variant["image"]
                        metadata_path = self.dataset_root / variant["metadata"]
                        object_types = _object_types(metadata_path)
                        prompt = build_prompt(
                            triplet,
                            variant,
                            method,
                            object_types,
                            self.prompt_variant,
                        )
                        send_image = image_path if method == "direct" else None
                        print(
                            f"[{call_number}/{total}] {model} {method} "
                            f"{triplet['triplet_id']} {variant_name}"
                        )
                        cache_key = (model, prompt)
                        # Non-visual baselines can have byte-identical prompts
                        # across paired observations. Reuse them to remove
                        # provider sampling noise and unnecessary API cost.
                        reused_identical_input = (
                            method != "direct" and cache_key in self.response_cache
                        )
                        try:
                            if reused_identical_input:
                                raw, response_dump, _ = self.response_cache[cache_key]
                                latency = 0.0
                            else:
                                raw, response_dump, latency = self._call(
                                    model, prompt, send_image
                                )
                                if method != "direct":
                                    self.response_cache[cache_key] = (
                                        raw,
                                        response_dump,
                                        latency,
                                    )
                        except Exception as error:
                            error_row = {
                                "model": model,
                                "method": method,
                                "triplet_id": triplet["triplet_id"],
                                "variant": variant_name,
                                "error": f"{type(error).__name__}: {error}",
                            }
                            with (self.output_dir / "errors.jsonl").open(
                                "a", encoding="utf-8"
                            ) as handle:
                                handle.write(json.dumps(error_row) + "\n")
                            print(f"ERROR {error_row}", flush=True)
                            continue
                        action, obj = parse_action(raw, object_types)
                        prediction = canonical_action(action, obj)
                        acceptable = {
                            canonical_action(choice["action"], choice.get("object"))
                            for choice in variant["acceptable_actions"]
                        }
                        row = {
                            "model": model,
                            "method": method,
                            "triplet_id": triplet["triplet_id"],
                            "causal_state_unit": triplet.get(
                                "causal_state_unit", triplet["triplet_id"]
                            ),
                            "replica_profile": triplet.get(
                                "replica_profile", "simulator_lighting"
                            ),
                            "certification_tier": triplet.get(
                                "certification_tier", "pilot-v3-strict"
                            ),
                            "variant": variant_name,
                            "predicate": triplet["predicate"],
                            "task_type": triplet["task_type"],
                            "scene": triplet["scene"],
                            "decision_object": triplet["target"][
                                "predicate_object_type"
                            ],
                            "prediction": prediction,
                            "parsed_action": action,
                            "parsed_object": obj,
                            "acceptable_actions": sorted(acceptable),
                            "correct": prediction in acceptable,
                            "parse_success": prediction is not None,
                            "latency_seconds": latency,
                            "reused_identical_input": reused_identical_input,
                        }
                        call_dir = (
                            self.output_dir
                            / "calls"
                            / model.replace("/", "__")
                            / method
                            / triplet["triplet_id"]
                        )
                        call_dir.mkdir(parents=True, exist_ok=True)
                        audit = {
                            **row,
                            "prompt": prompt,
                            "image": str(image_path) if send_image else None,
                            "raw_output": raw,
                            "response": response_dump,
                            "experiment_config": self.config,
                            "simulator_metadata": str(metadata_path),
                        }
                        with (call_dir / f"{variant_name}.json").open(
                            "w", encoding="utf-8"
                        ) as handle:
                            json.dump(audit, handle, indent=2)
                        with self.results_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(row) + "\n")
                        self.rows.append(row)
                        self.completed.add(key)
        self.summarize()
        return self.rows

    def summarize(self) -> list[dict[str, Any]]:
        summary = grouped_metrics(
            self.rows,
            bootstrap_iterations=self.bootstrap_iterations,
            seed=0,
        )
        summary_path = self.output_dir / "summary.json"
        temporary_summary = summary_path.with_name(
            f"{summary_path.name}.{os.getpid()}.tmp"
        )
        with temporary_summary.open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(summary), handle, indent=2, allow_nan=False)
        temporary_summary.replace(summary_path)
        audit_path = self.output_dir / "inference_audit.json"
        audit_path.write_text(
            json.dumps(self.inference_audit, indent=2), encoding="utf-8"
        )
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run causal next-action evaluation")
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/benchmark.json")
    )
    parser.add_argument("--output", type=Path, default=Path("CausalPilot/results"))
    parser.add_argument("--model", action="append")
    parser.add_argument(
        "--model-config",
        type=Path,
        help=(
            "JSON list (or {models:[...]}) of logical name, served model, "
            "backend, base_url, api_key_env, and optional extra_body"
        ),
    )
    parser.add_argument(
        "--methods",
        default=",".join(METHODS),
        help="Comma-separated direct,no_image,oracle_state",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-completion-tokens", type=int, default=256)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--retry-attempts", type=int, default=2)
    parser.add_argument("--prompt-variant", choices=PROMPT_VARIANTS, default="v1")
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated dataset variants to evaluate",
    )
    parser.add_argument(
        "--extra-body-json",
        default="{}",
        help="Provider-specific request fields, e.g. '{\"enable_thinking\": false}'",
    )
    parser.add_argument("--limit-triplets", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    methods = args.methods.split(",")
    invalid = sorted(set(methods) - set(METHODS))
    if invalid:
        parser.error(f"Unknown methods: {invalid}")
    model_specs = None
    if args.model_config:
        config = json.loads(args.model_config.read_text(encoding="utf-8"))
        values = config.get("models", []) if isinstance(config, dict) else config
        model_specs = [ModelSpec.from_dict(value) for value in values]
    if not args.model and not model_specs:
        parser.error("--model or --model-config is required")
    evaluator = NextActionEvaluator(
        dataset_path=args.dataset,
        output_dir=args.output,
        models=args.model or [],
        methods=methods,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
        bootstrap_iterations=args.bootstrap_iterations,
        variants=[name for name in args.variants.split(",") if name],
        extra_body=json.loads(args.extra_body_json),
        model_specs=model_specs,
        cache_dir=args.cache_dir,
        retry_attempts=args.retry_attempts,
        prompt_variant=args.prompt_variant,
    )
    evaluator.run(args.limit_triplets, args.shard_index, args.shard_count)


if __name__ == "__main__":
    main()
