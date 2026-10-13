from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Core names remain fixed, while certified nuisance augmenters may add names
# such as ``anchor_material`` without changing the dataset schema again.
VariantName = str


@dataclass(frozen=True)
class ActionChoice:
    action: str
    object: str | None

    def canonical(self) -> str:
        obj = self.object or "none"
        return f"{self.action.strip().lower()} {obj.strip().lower()}"

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


@dataclass
class VariantRecord:
    name: VariantName
    image: str
    acceptable_actions: list[ActionChoice]
    symbolic_state: dict[str, Any]
    oracle_state: list[str]
    metadata: str
    validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["acceptable_actions"] = [a.to_dict() for a in self.acceptable_actions]
        return result


@dataclass
class TripletRecord:
    triplet_id: str
    predicate: str
    task_type: str
    source_plan: str
    instruction: str
    scene: str
    history: list[str]
    action_interface: list[str]
    target: dict[str, Any]
    variants: dict[VariantName, VariantRecord]
    validation: dict[str, Any]
    # The pilot contract remains the default.  Submission expansion may use a
    # separately reported, explicitly weaker contract, but never silently
    # mixes the two populations.
    certification_tier: str = "pilot-v3-strict"
    causal_state_unit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["variants"] = {
            name: variant.to_dict() for name, variant in self.variants.items()
        }
        return result
