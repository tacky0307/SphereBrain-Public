from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SemanticBridgeHomeostasisConfig:
    """Bounded, label-free consolidation for repeatedly re-observed internal bridge edges."""

    success_gain: float = 0.22
    provisional_gain_scale: float = 0.45
    consolidate_after: int = 2
    passive_decay: float = 0.985
    contradiction_loss: float = 0.12
    protection_gain: float = 0.018
    max_stability: float = 1.0


class SemanticBridgeHomeostasis:
    """Tracks repeated structural bridge observations without semantic answer labels.

    The caller supplies only Core-internal candidate edges that were independently
    attributed as context-linked. Repeated observations raise stability
    asymptotically; absent observations decay it. Stable candidates receive only a
    small bounded weight protection, never a direct route or semantic target.
    """

    def __init__(self, config: SemanticBridgeHomeostasisConfig | None = None) -> None:
        self.config = config or SemanticBridgeHomeostasisConfig()
        self.observation_counts: dict[str, int] = {}
        self.stability: dict[str, float] = {}

    @staticmethod
    def edge_key(edge: Iterable[int]) -> str:
        a, b = sorted(int(x) for x in edge)
        return f"{a}>{b}"

    @staticmethod
    def key_edge(key: str) -> tuple[int, int]:
        a, b = key.split(">", 1)
        return int(a), int(b)

    def snapshot(self) -> dict:
        return {
            "observation_counts": dict(self.observation_counts),
            "stability": {k: float(v) for k, v in self.stability.items()},
        }

    def _protect(self, brain, edge: tuple[int, int], stability: float) -> float:
        a, b = edge
        before = float(brain.weights[a, b])
        delta = self.config.protection_gain * float(stability) * (1.0 - before)
        after = min(1.0, before + delta)
        brain.weights[a, b] = after
        brain.weights[b, a] = after
        return after - before

    def observe(self, brain, candidate_edges: Iterable[Iterable[int]]) -> dict:
        observed = {self.edge_key(edge) for edge in candidate_edges}
        protected = []
        promoted = 0
        decayed = 0

        for key in sorted(observed):
            count = int(self.observation_counts.get(key, 0)) + 1
            self.observation_counts[key] = count
            before = float(self.stability.get(key, 0.0))
            scale = 1.0 if count >= self.config.consolidate_after else self.config.provisional_gain_scale
            gain = self.config.success_gain * scale * (1.0 - before)
            after = min(self.config.max_stability, before + gain)
            self.stability[key] = after
            if before < 0.35 <= after:
                promoted += 1
            delta = self._protect(brain, self.key_edge(key), after)
            protected.append({
                "edge": list(self.key_edge(key)),
                "count": count,
                "stability_before": before,
                "stability_after": after,
                "weight_delta": delta,
            })

        for key in list(self.stability):
            if key in observed:
                continue
            before = float(self.stability[key])
            if before <= 0:
                continue
            # Absence acts as contradiction evidence, while still allowing slow forgetting.
            loss = self.config.contradiction_loss * (0.5 + before)
            after = max(0.0, before - loss)
            after *= self.config.passive_decay
            self.stability[key] = after
            decayed += int(after != before)

        return {
            "observed_count": len(observed),
            "promoted": promoted,
            "decayed": decayed,
            "protected": protected,
            "snapshot": self.snapshot(),
        }
