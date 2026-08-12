from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class NativeLearningConfig:
    replay_repeats: int = 2
    loop_gate: int = 2
    min_failure_evidence: float = 0.55
    return_weight_decay: float = 0.965
    return_usage_decay: float = 0.72
    entry_weight_decay: float = 0.997
    entry_usage_decay: float = 0.97
    success_gain: float = 0.26
    provisional_gain_scale: float = 0.45
    passive_decay: float = 0.992
    contradiction_loss: float = 0.18
    consolidate_after: int = 2
    extra_replay_gate: float = 0.55
    max_extra_replay: int = 1
    protection_scale: float = 0.55


class CoreNativeLearningState:
    """Native Core learning state validated by the v73-v79 experiment line.

    This object owns only learning-time state. World semantics remain outside Core:
    callers provide opaque specific/relative source sets and opaque condition tokens.
    """

    SCHEMA_VERSION = 1

    def __init__(self, config: NativeLearningConfig | None = None) -> None:
        self.config = config or NativeLearningConfig()
        self.episode_counter = 0
        self.motif_counts: dict[str, int] = {}
        self.success_counts: dict[str, int] = {}
        self.stability: dict[str, float] = {}
        self.last_events: list[dict[str, Any]] = []

    def clear(self) -> None:
        self.episode_counter = 0
        self.motif_counts = {}
        self.success_counts = {}
        self.stability = {}
        self.last_events = []

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "episode_counter": int(self.episode_counter),
            "motif_counts": {str(k): int(v) for k, v in self.motif_counts.items()},
            "success_counts": {str(k): int(v) for k, v in self.success_counts.items()},
            "stability": {str(k): float(v) for k, v in self.stability.items()},
        }

    def replace_from_mapping(self, value: dict[str, Any] | None) -> None:
        self.clear()
        if not isinstance(value, dict):
            return
        self.episode_counter = int(value.get("episode_counter", 0))
        self.motif_counts = {str(k): int(v) for k, v in dict(value.get("motif_counts", {})).items()}
        self.success_counts = {str(k): int(v) for k, v in dict(value.get("success_counts", {})).items()}
        self.stability = {
            str(k): max(0.0, min(1.0, float(v)))
            for k, v in dict(value.get("stability", {})).items()
        }

    @staticmethod
    def _pair_key(source: int, target: int) -> str:
        return f"{int(source)}>{int(target)}"

    @staticmethod
    def _unique_transitions(transitions: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[tuple[int, int]] = set()
        out: list[dict[str, Any]] = []
        for row in transitions:
            pair = (int(row["source"]), int(row["target"]))
            if pair in seen:
                continue
            seen.add(pair)
            out.append(row)
        return out

    @staticmethod
    def _route(brain: Any, sources: Iterable[int], *, learn: bool) -> Any:
        return brain.propagate(list(sources), steps=8, threshold=0.18, noise=0.0, learn=learn)

    @staticmethod
    def _decay_route(brain: Any, sources: Iterable[int], weight_factor: float, usage_factor: float) -> dict[str, Any]:
        result = CoreNativeLearningState._route(brain, sources, learn=False)
        changed = []
        for a, b in [tuple(x) for x in result.traversed_edges]:
            before_w = float(brain.weights[a, b])
            before_u = float(brain.usage[a, b])
            after_w = max(0.0, min(1.0, before_w * float(weight_factor)))
            after_u = max(0.0, before_u * float(usage_factor))
            brain.weights[a, b] = after_w
            brain.weights[b, a] = after_w
            # usage has historically been integer, but adaptive forgetting makes
            # effective usage continuous. Keep the ndarray dtype behavior of the
            # current Core by assigning the computed value directly.
            brain.usage[a, b] = after_u
            brain.usage[b, a] = after_u
            changed.append({
                "edge": [int(a), int(b)],
                "weight_before": before_w,
                "weight_after": float(brain.weights[a, b]),
                "usage_before": before_u,
                "usage_after": float(brain.usage[a, b]),
            })
        return {"changed_edges": changed}

    @staticmethod
    def _immediate_return_motifs(transitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not transitions:
            return []
        path = [int(transitions[0]["source"])] + [int(x["target"]) for x in transitions]
        motifs = []
        for i in range(len(path) - 2):
            a, b, c = path[i], path[i + 1], path[i + 2]
            if a != b and a == c:
                motifs.append({
                    "nodes": [a, b, c],
                    "entry": [a, b],
                    "return": [b, a],
                    "key": f"{a}>{b}>{a}",
                })
        return motifs

    @staticmethod
    def _evidence_probability(brain: Any, token: str) -> float:
        row = brain.experience_state.condition_evidence.get(token, {})
        t = float(row.get("true_weight", 0.0))
        f = float(row.get("false_weight", 0.0))
        total = t + f
        return 0.5 if total <= 1e-12 else t / total

    def _passive_decay(self, used_keys: set[str]) -> int:
        changed = 0
        for key in list(self.stability):
            if key in used_keys:
                continue
            before = float(self.stability[key])
            after = max(0.0, before * self.config.passive_decay)
            self.stability[key] = after
            changed += int(after != before)
        return changed

    def _success_update(self, brain: Any, transitions: list[dict[str, Any]]) -> dict[str, int]:
        promoted = 0
        extra_replayed = 0
        used_keys: set[str] = set()
        for row in self._unique_transitions(transitions):
            source, target = int(row["source"]), int(row["target"])
            key = self._pair_key(source, target)
            used_keys.add(key)
            self.success_counts[key] = int(self.success_counts.get(key, 0)) + 1
            before = float(self.stability.get(key, 0.0))
            scale = 1.0 if self.success_counts[key] >= self.config.consolidate_after else self.config.provisional_gain_scale
            gain = self.config.success_gain * scale * (1.0 - before)
            after = min(1.0, before + gain)
            self.stability[key] = after
            if before < 0.35 <= after:
                promoted += 1
            if after >= self.config.extra_replay_gate:
                for _ in range(self.config.max_extra_replay):
                    self._route(brain, row["specific_sources"], learn=True)
                extra_replayed += 1
        passive = self._passive_decay(used_keys)
        return {"promoted": promoted, "extra_replayed": extra_replayed, "passive_decayed": passive}

    def _failure_stability_decay(self, transitions: list[dict[str, Any]]) -> int:
        changed = 0
        used_keys: set[str] = set()
        for row in self._unique_transitions(transitions):
            key = self._pair_key(int(row["source"]), int(row["target"]))
            used_keys.add(key)
            before = float(self.stability.get(key, 0.0))
            if before <= 0:
                continue
            loss = self.config.contradiction_loss * (0.5 + before)
            after = max(0.0, before - loss)
            self.stability[key] = after
            changed += int(after != before)
        changed += self._passive_decay(used_keys)
        return changed

    def _find_transition(self, transitions: list[dict[str, Any]], source: int, target: int) -> dict[str, Any] | None:
        for row in transitions:
            if int(row["source"]) == int(source) and int(row["target"]) == int(target):
                return row
        return None

    def _temporal_attribution(self, brain: Any, transitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for motif in self._immediate_return_motifs(transitions):
            key = str(motif["key"])
            self.motif_counts[key] = int(self.motif_counts.get(key, 0)) + 1
            count = self.motif_counts[key]
            source, target = [int(x) for x in motif["return"]]
            row = self._find_transition(transitions, source, target)
            entry_row = self._find_transition(transitions, int(motif["entry"][0]), int(motif["entry"][1]))
            if row is None or entry_row is None:
                continue
            stability_key = self._pair_key(source, target)
            stable = float(self.stability.get(stability_key, 0.0))
            failure_evidence = 1.0 - self._evidence_probability(brain, str(row["specific_token"]))
            event: dict[str, Any] = {
                "episode": int(self.episode_counter),
                "motif": list(motif["nodes"]),
                "motif_count": int(count),
                "entry": list(motif["entry"]),
                "return": list(motif["return"]),
                "failure_evidence": float(failure_evidence),
                "stability_before": stable,
                "protection_strength": stable,
                "credited": False,
            }
            if count >= self.config.loop_gate and failure_evidence >= self.config.min_failure_evidence:
                wf = self.config.return_weight_decay + (1.0 - self.config.return_weight_decay) * self.config.protection_scale * stable
                uf = self.config.return_usage_decay + (1.0 - self.config.return_usage_decay) * self.config.protection_scale * stable
                event["return_decay"] = self._decay_route(
                    brain, row["specific_sources"], weight_factor=wf, usage_factor=uf
                )
                event["entry_decay"] = self._decay_route(
                    brain,
                    entry_row["specific_sources"],
                    weight_factor=self.config.entry_weight_decay,
                    usage_factor=self.config.entry_usage_decay,
                )
                if stable > 0:
                    loss = self.config.contradiction_loss * (0.5 + stable)
                    self.stability[stability_key] = max(0.0, stable - loss)
                event["stability_after"] = float(self.stability.get(stability_key, 0.0))
                event["credited"] = True
            events.append(event)
        return events

    def observe_episode(
        self,
        brain: Any,
        transitions: Iterable[dict[str, Any]],
        *,
        success: bool,
        expected_conditions: Iterable[str],
        motif: str = "native_learning",
    ) -> dict[str, Any]:
        rows = [dict(x) for x in transitions]
        self.episode_counter += 1

        # Native Experience State receives the same unique condition observations
        # used by the validated runner implementation.
        seen_tokens: set[str] = set()
        expected = list(expected_conditions)
        for row in rows:
            for token_key in ("specific_token", "relative_token"):
                token = str(row[token_key])
                if token in seen_tokens:
                    continue
                seen_tokens.add(token)
                brain.experience_state.observe(
                    condition=token,
                    present=bool(success),
                    motif=motif,
                    expected_conditions=expected,
                )

        events: list[dict[str, Any]] = []
        consolidation = {"promoted": 0, "extra_replayed": 0, "passive_decayed": 0, "eroded": 0}

        if success:
            # Validated success reinforcement: unique specific transitions and
            # unique relative transitions are replayed independently.
            seen_specific: set[tuple[int, int]] = set()
            seen_relative: set[str] = set()
            for row in rows:
                pair = (int(row["source"]), int(row["target"]))
                if pair not in seen_specific:
                    seen_specific.add(pair)
                    for _ in range(self.config.replay_repeats):
                        self._route(brain, row["specific_sources"], learn=True)
                relative_token = str(row["relative_token"])
                if relative_token not in seen_relative:
                    seen_relative.add(relative_token)
                    for _ in range(self.config.replay_repeats):
                        self._route(brain, row["relative_sources"], learn=True)
            consolidation.update(self._success_update(brain, rows))
        else:
            consolidation["eroded"] = self._failure_stability_decay(rows)
            events = self._temporal_attribution(brain, rows)

        self.last_events = events
        return {
            "episode": int(self.episode_counter),
            "events": events,
            "consolidation": consolidation,
            "snapshot": self.snapshot(),
        }
