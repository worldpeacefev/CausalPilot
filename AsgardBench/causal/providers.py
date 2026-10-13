"""Small provider boundary for model-agnostic causal evaluation.

The benchmark consumes a text prompt plus an optional image.  Provider SDK
details live here so scoring, parsing, and sample definitions remain unchanged.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from openai import OpenAI


class ChatBackend(Protocol):
    def complete(
        self,
        model: str,
        content: list[dict[str, Any]],
        temperature: float,
        max_completion_tokens: int,
        extra_body: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any], float]: ...


@dataclass(frozen=True)
class ModelSpec:
    """A logical model and the endpoint used to serve it."""

    name: str
    model: str
    backend: str = "openai_compatible"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    extra_body: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelSpec":
        allowed = {
            "name",
            "model",
            "backend",
            "base_url",
            "api_key_env",
            "extra_body",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"Unknown model-spec fields: {unknown}")
        return cls(**value)

    def public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "backend": self.backend,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "extra_body": self.extra_body,
        }


class OpenAICompatibleBackend:
    """OpenAI, Azure-compatible gateways, vLLM, and compatible closed APIs."""

    def __init__(self, spec: ModelSpec):
        key = os.getenv(spec.api_key_env)
        if not key:
            raise ValueError(f"{spec.api_key_env} is required for model {spec.name}")
        self.client = OpenAI(
            api_key=key,
            base_url=spec.base_url,
            timeout=90.0,
            max_retries=0,
        )

    def complete(
        self,
        model: str,
        content: list[dict[str, Any]],
        temperature: float,
        max_completion_tokens: int,
        extra_body: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any], float]:
        started = time.time()
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            extra_body=extra_body or None,
        )
        latency = time.time() - started
        return (
            response.choices[0].message.content or "",
            response.model_dump(mode="json"),
            latency,
        )


def build_backend(spec: ModelSpec) -> ChatBackend:
    if spec.backend == "openai_compatible":
        return OpenAICompatibleBackend(spec)
    raise ValueError(
        f"Unsupported backend {spec.backend!r}; register a ChatBackend adapter"
    )
