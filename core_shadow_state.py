from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class StabilityProfileShadowState:
    """Transient, non-persistent Core-attached shadow state.

    This state is diagnostic only. It must not participate in propagation,
    candidate selection, thresholding, learning, or persistence.
    """

    kind: str
    motif: str
    stability_class: str | None
    baseline_present: bool | None
    echo_resistant: bool | None
    position_resistant: bool | None
    common_resistant: bool | None
    evidence_conditions: int
    source: str = "core_activity_derived_stability_profile"
    age: int = 0
    ttl: int = 2


def attach_shadow_state(brain: Any, state: StabilityProfileShadowState) -> None:
    """Attach a transient state directly to a SphereBrain instance.

    No adjacency, weight, usage, activation, threshold, learning, or save/load
    behavior is touched. The attribute is deliberately absent from brain.save().
    """
    brain.shadow_state = deepcopy(asdict(state))


def clear_shadow_state(brain: Any) -> None:
    brain.shadow_state = None


def snapshot_shadow_state(brain: Any) -> dict | None:
    value = getattr(brain, "shadow_state", None)
    return deepcopy(value)


def tick_shadow_state(brain: Any) -> dict | None:
    state = getattr(brain, "shadow_state", None)
    if not state:
        return None
    state = deepcopy(state)
    state["age"] = int(state.get("age", 0)) + 1
    ttl = int(state.get("ttl", 0))
    if state["age"] >= ttl:
        brain.shadow_state = None
        return None
    brain.shadow_state = state
    return deepcopy(state)
