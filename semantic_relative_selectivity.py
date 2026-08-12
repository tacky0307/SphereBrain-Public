from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RelativeSelectivityConfig:
    min_consecutive: int = 3
    min_context_score: float = 0.72
    max_hub_score: float = 0.80
    drop_trigger_ratio: float = 0.14
    restore_fraction: float = 0.35
    max_restore_per_edge: float = 0.012
    peak_decay: float = 0.997
    cooldown: int = 2


class RelativeSelectivityPreserver:
    """Label-free, reversible preservation of a mature local bridge ensemble.

    The preserver never sees semantic answers, control results, or PASS flags.
    It watches only repeatedly observed, context-near, non-hub candidate edges
    inside one Core. When the aggregate intrinsic selectivity of that mature
    ensemble drops sharply relative to its own recent peak, a bounded fraction
    of weight loss is restored. No new edge is created and protection stops as
    soon as the local selectivity recovers.
    """

    def __init__(self, config: RelativeSelectivityConfig | None = None) -> None:
        self.config = config or RelativeSelectivityConfig()
        self.cycle = 0
        self.consecutive: dict[str, int] = {}
        self.last_seen: dict[str, int] = {}
        self.peak_score: float = 0.0
        self.last_score: float = 0.0
        self.last_intervention_cycle: int | None = None
        self.interventions = 0
        self.restored_total = 0.0

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
            "consecutive": dict(self.consecutive),
            "last_seen": dict(self.last_seen),
            "peak_score": float(self.peak_score),
            "last_score": float(self.last_score),
            "last_intervention_cycle": self.last_intervention_cycle,
            "interventions": int(self.interventions),
            "restored_total": float(self.restored_total),
        }

    def observe(self, brain, candidates: list[dict], previous_weights) -> dict:
        self.cycle += 1
        current: dict[str, dict] = {}
        for row in candidates:
            edge = tuple(sorted(int(x) for x in row["edge"]))
            key = self.edge_key(edge)
            current[key] = {
                "edge": edge,
                "context_score": float(row.get("context_score", 0.0)),
                "hub_score": float(row.get("hub_score", 1.0)),
            }

        known = set(self.consecutive) | set(current)
        for key in known:
            if key not in current:
                self.consecutive[key] = 0
                continue
            prev = self.last_seen.get(key)
            self.consecutive[key] = int(self.consecutive.get(key, 0)) + 1 if prev == self.cycle - 1 else 1
            self.last_seen[key] = self.cycle

        mature = []
        for key, row in current.items():
            if (
                self.consecutive.get(key, 0) >= self.config.min_consecutive
                and row["context_score"] >= self.config.min_context_score
                and row["hub_score"] <= self.config.max_hub_score
            ):
                a, b = row["edge"]
                weight = float(brain.weights[a, b])
                intrinsic = weight * row["context_score"] * max(0.0, 1.0 - row["hub_score"])
                mature.append({**row, "key": key, "weight": weight, "intrinsic": intrinsic})

        score = sum(x["intrinsic"] for x in mature)
        if self.peak_score > 0:
            self.peak_score *= self.config.peak_decay
        self.peak_score = max(self.peak_score, score)
        drop_ratio = (self.peak_score - score) / self.peak_score if self.peak_score > 1e-12 else 0.0

        cooldown_ok = (
            self.last_intervention_cycle is None
            or self.cycle - self.last_intervention_cycle >= self.config.cooldown
        )
        trigger = bool(mature) and drop_ratio >= self.config.drop_trigger_ratio and cooldown_ok
        restored = []

        if trigger:
            for item in mature:
                a, b = item["edge"]
                before_cycle = float(previous_weights[a, b])
                current_weight = float(brain.weights[a, b])
                lost = max(0.0, before_cycle - current_weight)
                # If raw weight did not fall, allow only a very small bounded
                # ensemble support proportional to aggregate selectivity loss.
                support_base = lost if lost > 0 else self.config.max_restore_per_edge * drop_ratio
                delta = min(
                    self.config.max_restore_per_edge,
                    self.config.restore_fraction * support_base,
                )
                if delta <= 0:
                    continue
                after = min(1.0, current_weight + delta)
                brain.weights[a, b] = after
                brain.weights[b, a] = after
                actual = after - current_weight
                if actual > 0:
                    restored.append({
                        "edge": [a, b],
                        "before_cycle_weight": before_cycle,
                        "current_weight": current_weight,
                        "restored": actual,
                        "context_score": item["context_score"],
                        "hub_score": item["hub_score"],
                    })
                    self.restored_total += actual
            if restored:
                self.interventions += 1
                self.last_intervention_cycle = self.cycle

        self.last_score = score
        return {
            "cycle": self.cycle,
            "mature_edge_count": len(mature),
            "local_selectivity_score": score,
            "peak_selectivity_score": self.peak_score,
            "drop_ratio": drop_ratio,
            "triggered": bool(restored),
            "restored": restored,
            "snapshot": self.snapshot(),
        }
