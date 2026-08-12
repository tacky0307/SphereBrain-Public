from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class DelayedSelectiveConfig:
    observation_start: int = 10
    protection_start: int = 20
    min_observations: int = 3
    min_persistence_ratio: float = 0.45
    max_hub_score: float = 0.88
    protection_gain: float = 0.014
    stability_gain: float = 0.18
    passive_decay: float = 0.992


class DelayedSelectiveConsolidation:
    """Label-free semantic bridge consolidation with a competition window.

    Candidate edges are only observed during the early phase. Protection is
    disabled until protection_start, then only repeatedly re-observed, persistent,
    non-hub candidates receive bounded protection.
    """

    def __init__(self, config: DelayedSelectiveConfig | None = None) -> None:
        self.config = config or DelayedSelectiveConfig()
        self.cycle = 0
        self.observation_counts: dict[str, int] = {}
        self.first_seen: dict[str, int] = {}
        self.last_seen: dict[str, int] = {}
        self.stability: dict[str, float] = {}
        self.protected_counts: dict[str, int] = {}

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
            "cycle": self.cycle,
            "observation_counts": dict(self.observation_counts),
            "first_seen": dict(self.first_seen),
            "last_seen": dict(self.last_seen),
            "stability": {k: float(v) for k, v in self.stability.items()},
            "protected_counts": dict(self.protected_counts),
        }

    def observe(self, brain, candidates: list[dict]) -> dict:
        self.cycle += 1
        observed_keys: set[str] = set()
        candidate_rows = []
        protected = []

        for row in candidates:
            edge = tuple(sorted(int(x) for x in row["edge"]))
            key = self.edge_key(edge)
            observed_keys.add(key)
            if self.cycle >= self.config.observation_start:
                self.observation_counts[key] = int(self.observation_counts.get(key, 0)) + 1
                self.first_seen.setdefault(key, self.cycle)
                self.last_seen[key] = self.cycle

            count = int(self.observation_counts.get(key, 0))
            first = int(self.first_seen.get(key, self.cycle))
            window = max(1, self.cycle - first + 1)
            persistence = count / window
            hub_score = float(row.get("hub_score", 1.0))
            context_score = float(row.get("context_score", 0.0))
            recurrence = min(1.0, count / max(1, self.config.min_observations))
            score = recurrence * persistence * context_score * max(0.0, 1.0 - hub_score)

            eligible = (
                self.cycle >= self.config.protection_start
                and count >= self.config.min_observations
                and persistence >= self.config.min_persistence_ratio
                and hub_score <= self.config.max_hub_score
                and context_score > 0.0
            )

            candidate_rows.append({
                "edge": list(edge),
                "count": count,
                "persistence": persistence,
                "hub_score": hub_score,
                "context_score": context_score,
                "score": score,
                "eligible": eligible,
            })

            if eligible:
                before_stability = float(self.stability.get(key, 0.0))
                after_stability = min(1.0, before_stability + self.config.stability_gain * (1.0 - before_stability))
                self.stability[key] = after_stability
                a, b = edge
                before_weight = float(brain.weights[a, b])
                delta = self.config.protection_gain * after_stability * score * (1.0 - before_weight)
                after_weight = min(1.0, before_weight + delta)
                brain.weights[a, b] = after_weight
                brain.weights[b, a] = after_weight
                self.protected_counts[key] = int(self.protected_counts.get(key, 0)) + 1
                protected.append({
                    "edge": list(edge),
                    "score": score,
                    "stability_before": before_stability,
                    "stability_after": after_stability,
                    "weight_delta": after_weight - before_weight,
                })

        for key in list(self.stability):
            if key in observed_keys:
                continue
            self.stability[key] = max(0.0, float(self.stability[key]) * self.config.passive_decay)

        return {
            "cycle": self.cycle,
            "candidate_rows": candidate_rows,
            "protected": protected,
            "snapshot": self.snapshot(),
        }
