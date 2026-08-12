from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EventTriggeredConfig:
    min_observations: int = 4
    min_consecutive: int = 3
    min_age: int = 4
    min_context_score: float = 0.72
    max_hub_score: float = 0.80
    min_maturity_score: float = 0.10
    stability_gain: float = 0.16
    protection_gain: float = 0.014
    passive_decay: float = 0.993


class EventTriggeredConsolidation:
    """Label-free consolidation triggered by structural maturity, not elapsed time.

    An edge becomes protectable only after it repeatedly and consecutively
    reappears, survives for several observations, remains close to the current
    context structure, and is not hub-like. No semantic answer label, PASS flag,
    fixed protection-start episode, or control result is used.
    """

    def __init__(self, config: EventTriggeredConfig | None = None) -> None:
        self.config = config or EventTriggeredConfig()
        self.cycle = 0
        self.observation_counts: dict[str, int] = {}
        self.first_seen: dict[str, int] = {}
        self.last_seen: dict[str, int] = {}
        self.consecutive: dict[str, int] = {}
        self.stability: dict[str, float] = {}
        self.trigger_cycle: dict[str, int] = {}
        self.protected_counts: dict[str, int] = {}

    @staticmethod
    def edge_key(edge) -> str:
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
            "consecutive": dict(self.consecutive),
            "stability": {k: float(v) for k, v in self.stability.items()},
            "trigger_cycle": dict(self.trigger_cycle),
            "protected_counts": dict(self.protected_counts),
        }

    def observe(self, brain, candidates: list[dict]) -> dict:
        self.cycle += 1
        current: dict[str, dict] = {}
        for row in candidates:
            edge = tuple(sorted(int(x) for x in row["edge"]))
            current[self.edge_key(edge)] = {
                "edge": edge,
                "context_score": float(row.get("context_score", 0.0)),
                "hub_score": float(row.get("hub_score", 1.0)),
            }

        candidate_rows = []
        protected = []
        triggered = []

        known = set(self.observation_counts) | set(current)
        for key in known:
            if key not in current:
                self.consecutive[key] = 0
                if key in self.stability:
                    self.stability[key] = max(0.0, float(self.stability[key]) * self.config.passive_decay)
                continue

            row = current[key]
            prev_seen = self.last_seen.get(key)
            self.observation_counts[key] = int(self.observation_counts.get(key, 0)) + 1
            self.first_seen.setdefault(key, self.cycle)
            self.last_seen[key] = self.cycle
            self.consecutive[key] = int(self.consecutive.get(key, 0)) + 1 if prev_seen == self.cycle - 1 else 1

            count = int(self.observation_counts[key])
            streak = int(self.consecutive[key])
            age = int(self.cycle - self.first_seen[key] + 1)
            context_score = row["context_score"]
            hub_score = row["hub_score"]
            recurrence = min(1.0, count / max(1, self.config.min_observations))
            persistence = min(1.0, streak / max(1, self.config.min_consecutive))
            age_score = min(1.0, age / max(1, self.config.min_age))
            maturity_score = recurrence * persistence * age_score * context_score * max(0.0, 1.0 - hub_score)

            mature = (
                count >= self.config.min_observations
                and streak >= self.config.min_consecutive
                and age >= self.config.min_age
                and context_score >= self.config.min_context_score
                and hub_score <= self.config.max_hub_score
                and maturity_score >= self.config.min_maturity_score
            )

            if mature and key not in self.trigger_cycle:
                self.trigger_cycle[key] = self.cycle
                triggered.append({"edge": list(row["edge"]), "cycle": self.cycle, "maturity_score": maturity_score})

            active = key in self.trigger_cycle and mature
            candidate_rows.append({
                "edge": list(row["edge"]),
                "count": count,
                "consecutive": streak,
                "age": age,
                "context_score": context_score,
                "hub_score": hub_score,
                "maturity_score": maturity_score,
                "mature": mature,
                "triggered": key in self.trigger_cycle,
                "active": active,
            })

            if active:
                before_stability = float(self.stability.get(key, 0.0))
                after_stability = min(1.0, before_stability + self.config.stability_gain * (1.0 - before_stability))
                self.stability[key] = after_stability
                a, b = row["edge"]
                before_weight = float(brain.weights[a, b])
                delta = self.config.protection_gain * after_stability * maturity_score * (1.0 - before_weight)
                after_weight = min(1.0, before_weight + delta)
                brain.weights[a, b] = after_weight
                brain.weights[b, a] = after_weight
                self.protected_counts[key] = int(self.protected_counts.get(key, 0)) + 1
                protected.append({
                    "edge": list(row["edge"]),
                    "trigger_cycle": int(self.trigger_cycle[key]),
                    "maturity_score": maturity_score,
                    "stability_before": before_stability,
                    "stability_after": after_stability,
                    "weight_delta": after_weight - before_weight,
                })

        return {
            "cycle": self.cycle,
            "candidate_rows": candidate_rows,
            "triggered": triggered,
            "protected": protected,
            "snapshot": self.snapshot(),
        }
