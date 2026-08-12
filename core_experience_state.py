from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable
import json


@dataclass
class CoreExperienceState:
    """Native, serializable experience-derived state owned by SphereBrain.

    The state can now update itself directly from sequential Core experience
    evidence. It remains deliberately behavior-neutral: propagation, thresholds,
    weights, topology, usage, learning and Decoder output do not read this state.
    """

    schema_version: int = 2
    kind: str = "motif_stability_profile"
    motif: str | None = None
    stability_class: str | None = None
    baseline_present: bool | None = None
    echo_resistant: bool | None = None
    position_resistant: bool | None = None
    common_resistant: bool | None = None
    evidence_conditions: int = 0
    evidence_experiences: int = 0
    confidence: float = 0.0
    surprise_ewma: float = 0.0
    drift_suspected: bool = False
    adaptive_forgetting: bool = False
    forgetting_decay: float | None = None
    condition_evidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # v60-validated defaults. They live with the native state rather than in
    # propagation so that experience updating stays behavior-neutral.
    BASE_DECAY = 0.92
    FAST_DECAY = 0.62
    SURPRISE_CONFIDENT_MARGIN = 0.45
    RECENT_WINDOW = 12
    MIN_SURPRISE_HITS = 4
    MIN_DISTINCT_CONFLICTS = 4
    MIN_MARGIN = 0.60
    MAX_FRESH_GAP = 9
    EWMA_GATE = 0.22
    EXIT_STABLE_STEPS = 10
    EXIT_EWMA = 0.10
    EPS = 1e-12

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(asdict(self))

    def replace_from_mapping(self, value: dict[str, Any] | None) -> None:
        if not value:
            self.clear()
            return
        # Backward-compatible load: older schema-1 mappings simply lack the
        # online detector metadata and are accepted as-is.
        allowed = set(asdict(self))
        for key in allowed:
            if key in value:
                setattr(self, key, deepcopy(value[key]))
        self.schema_version = max(2, int(getattr(self, "schema_version", 1)))
        self._ensure_runtime_metadata()

    def update_profile(
        self,
        *,
        motif: str | None,
        stability_class: str | None,
        baseline_present: bool | None,
        echo_resistant: bool | None,
        position_resistant: bool | None,
        common_resistant: bool | None,
        evidence_conditions: int,
        evidence_experiences: int,
        confidence: float,
        surprise_ewma: float = 0.0,
        drift_suspected: bool = False,
        adaptive_forgetting: bool = False,
        forgetting_decay: float | None = None,
        condition_evidence: dict[str, dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.kind = "motif_stability_profile"
        self.motif = motif
        self.stability_class = stability_class
        self.baseline_present = baseline_present
        self.echo_resistant = echo_resistant
        self.position_resistant = position_resistant
        self.common_resistant = common_resistant
        self.evidence_conditions = int(evidence_conditions)
        self.evidence_experiences = int(evidence_experiences)
        self.confidence = float(confidence)
        self.surprise_ewma = float(surprise_ewma)
        self.drift_suspected = bool(drift_suspected)
        self.adaptive_forgetting = bool(adaptive_forgetting)
        self.forgetting_decay = None if forgetting_decay is None else float(forgetting_decay)
        if condition_evidence is not None:
            self.condition_evidence = deepcopy(condition_evidence)
        if metadata is not None:
            self.metadata = deepcopy(metadata)
        self._ensure_runtime_metadata()

    def configure(
        self,
        *,
        motif: str | None = None,
        expected_conditions: Iterable[str] | None = None,
    ) -> None:
        """Configure the native evidence universe without adding observations."""
        if motif is not None:
            self.motif = str(motif)
        if expected_conditions is not None:
            names = [str(x) for x in expected_conditions]
            self.metadata["expected_conditions"] = list(dict.fromkeys(names))
            for name in self.metadata["expected_conditions"]:
                self._ensure_condition(name)
        self._ensure_runtime_metadata()
        self._refresh_profile()

    def observe(
        self,
        *,
        condition: str,
        present: bool,
        motif: str | None = None,
        expected_conditions: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Update native experience state from one sequential experience.

        Inputs are deliberately minimal and label-free with respect to the tested
        context identity: a condition token and whether the selected structural
        motif was present in the Core-derived experience. No target Node, answer,
        position label, route choice or reward enters this update.
        """
        condition = str(condition)
        if motif is not None:
            self.motif = str(motif)
        if expected_conditions is not None:
            self.configure(motif=self.motif, expected_conditions=expected_conditions)
        else:
            self._ensure_runtime_metadata()

        self._ensure_condition(condition)
        step = int(self.evidence_experiences) + 1
        entry = self.condition_evidence[condition]

        probability_before = self._condition_probability(condition)
        margin_before = self._condition_margin(condition)
        confident_prediction = bool(
            probability_before is not None
            and margin_before is not None
            and margin_before >= self.SURPRISE_CONFIDENT_MARGIN
        )
        mismatch = bool(
            confident_prediction
            and ((float(probability_before) >= 0.5) != bool(present))
        )
        surprise = 1 if mismatch else 0

        previous_seen = entry.get("last_seen_step")
        last_seen_gap = None if previous_seen is None else step - int(previous_seen)

        recent = list(self.metadata.get("recent_conflicts", []))
        recent.append({
            "step": step,
            "condition": condition,
            "surprise": surprise,
            "margin": None if margin_before is None else float(margin_before),
            "last_seen_gap": last_seen_gap,
            "fresh": last_seen_gap is None or last_seen_gap <= self.MAX_FRESH_GAP,
            "high_margin": margin_before is not None and float(margin_before) >= self.MIN_MARGIN,
        })
        recent = recent[-self.RECENT_WINDOW:]
        self.metadata["recent_conflicts"] = recent

        self.surprise_ewma = 0.72 * float(self.surprise_ewma) + 0.28 * surprise
        surprise_conflicts = [x for x in recent if int(x.get("surprise", 0)) == 1]
        recent_hits = len(surprise_conflicts)
        distinct_conditions = len({str(x["condition"]) for x in surprise_conflicts})
        high_margin_hits = sum(1 for x in surprise_conflicts if bool(x.get("high_margin")))
        fresh_high_margin_conditions = {
            str(x["condition"])
            for x in surprise_conflicts
            if bool(x.get("high_margin")) and bool(x.get("fresh"))
        }

        factors = {
            "ewma": self.surprise_ewma >= self.EWMA_GATE,
            "persistent_hits": recent_hits >= self.MIN_SURPRISE_HITS,
            "multi_condition": distinct_conditions >= self.MIN_DISTINCT_CONFLICTS,
            "high_margin": high_margin_hits >= self.MIN_SURPRISE_HITS,
            "fresh_evidence": len(fresh_high_margin_conditions) >= 2,
        }
        gate_open = all(factors.values())

        stable_steps = int(self.metadata.get("stable_steps", 0))
        trigger_count = int(self.metadata.get("trigger_count", 0))
        if not self.adaptive_forgetting and gate_open:
            self.adaptive_forgetting = True
            self.drift_suspected = True
            stable_steps = 0
            trigger_count += 1
        elif self.adaptive_forgetting:
            if surprise == 0:
                stable_steps += 1
            else:
                stable_steps = 0
            if stable_steps >= self.EXIT_STABLE_STEPS and self.surprise_ewma < self.EXIT_EWMA:
                self.adaptive_forgetting = False
                self.drift_suspected = False

        self.metadata["stable_steps"] = stable_steps
        self.metadata["trigger_count"] = trigger_count
        self.metadata["last_gate_factors"] = deepcopy(factors)
        self.metadata["last_gate_open"] = bool(gate_open)

        decay = self.FAST_DECAY if self.adaptive_forgetting else self.BASE_DECAY
        self.forgetting_decay = float(decay)
        for value in self.condition_evidence.values():
            value["true_weight"] = float(value.get("true_weight", 0.0)) * decay
            value["false_weight"] = float(value.get("false_weight", 0.0)) * decay

        entry = self.condition_evidence[condition]
        if present:
            entry["true_weight"] = float(entry.get("true_weight", 0.0)) + 1.0
        else:
            entry["false_weight"] = float(entry.get("false_weight", 0.0)) + 1.0
        entry["observations"] = int(entry.get("observations", 0)) + 1
        entry["last_seen_step"] = step
        self.evidence_experiences = step

        self._refresh_profile()

        return {
            "condition": condition,
            "present": bool(present),
            "probability_before": probability_before,
            "margin_before": margin_before,
            "confident_prediction": confident_prediction,
            "mismatch": mismatch,
            "surprise": surprise,
            "surprise_ewma": float(self.surprise_ewma),
            "recent_surprise_hits": recent_hits,
            "distinct_surprise_conditions": distinct_conditions,
            "high_margin_surprise_hits": high_margin_hits,
            "fresh_high_margin_condition_count": len(fresh_high_margin_conditions),
            "last_seen_gap": last_seen_gap,
            "gate_factors": factors,
            "multi_factor_gate_open": bool(gate_open),
            "drift_suspected": bool(self.drift_suspected),
            "adaptive_forgetting": bool(self.adaptive_forgetting),
            "active_decay": float(self.forgetting_decay),
            "trigger_count": trigger_count,
            "state": self.snapshot(),
        }

    def _ensure_runtime_metadata(self) -> None:
        self.metadata.setdefault("expected_conditions", [])
        self.metadata.setdefault("recent_conflicts", [])
        self.metadata.setdefault("stable_steps", 0)
        self.metadata.setdefault("trigger_count", 0)
        self.metadata.setdefault("last_gate_factors", {})
        self.metadata.setdefault("last_gate_open", False)

    def _ensure_condition(self, condition: str) -> None:
        if condition not in self.condition_evidence:
            self.condition_evidence[condition] = {
                "true_weight": 0.0,
                "false_weight": 0.0,
                "observations": 0,
                "last_seen_step": None,
            }
        expected = list(self.metadata.get("expected_conditions", []))
        if not expected:
            # With no explicit universe, grow it naturally from observed conditions.
            self.metadata["expected_conditions"] = list(dict.fromkeys(
                list(self.condition_evidence.keys())
            ))

    def _condition_probability(self, condition: str) -> float | None:
        value = self.condition_evidence.get(condition)
        if value is None:
            return None
        t = float(value.get("true_weight", 0.0))
        f = float(value.get("false_weight", 0.0))
        total = t + f
        if total <= self.EPS:
            return None
        return t / total

    def _condition_margin(self, condition: str) -> float | None:
        p = self._condition_probability(condition)
        if p is None:
            return None
        return abs(float(p) - 0.5) * 2.0

    def _condition_presence(self, condition: str) -> bool | None:
        p = self._condition_probability(condition)
        if p is None:
            return None
        return bool(p >= 0.5)

    def _expected_conditions(self) -> list[str]:
        expected = [str(x) for x in self.metadata.get("expected_conditions", [])]
        if expected:
            return expected
        return list(self.condition_evidence.keys())

    def _refresh_profile(self) -> None:
        names = self._expected_conditions()
        if not names:
            self.stability_class = None
            self.evidence_conditions = 0
            self.confidence = 0.0
            return

        presence = {name: self._condition_presence(name) for name in names}
        known = [name for name in names if presence[name] is not None]
        present_count = sum(1 for name in known if bool(presence[name]))
        self.evidence_conditions = len(known)

        if not known:
            self.stability_class = "unknown"
        else:
            fraction = present_count / len(known)
            all_expected_known = len(known) == len(names)
            if all_expected_known and fraction >= 1.0 - self.EPS:
                self.stability_class = "stable"
            elif fraction >= 5.0 / 7.0:
                self.stability_class = "mostly"
            elif fraction >= 2.0 / 7.0:
                self.stability_class = "unstable"
            else:
                self.stability_class = "absent"

        self.baseline_present = presence.get("baseline")

        def resistant(prefix: str) -> bool | None:
            baseline = self.baseline_present
            if baseline is None or not baseline:
                return None
            group = [name for name in names if name.startswith(prefix + "_")]
            if len(group) < 2:
                return None
            values = [presence[name] for name in group]
            if any(value is None for value in values):
                return None
            return all(bool(value) == bool(baseline) for value in values)

        self.echo_resistant = resistant("echo")
        self.position_resistant = resistant("position")
        self.common_resistant = resistant("common")

        margins = [self._condition_margin(name) for name in known]
        masses = [
            float(self.condition_evidence[name].get("true_weight", 0.0))
            + float(self.condition_evidence[name].get("false_weight", 0.0))
            for name in known
        ]
        coverage = len(known) / len(names)
        certainty = 0.0 if not margins else sum(float(x) for x in margins if x is not None) / len(margins)
        mass_confidence = 0.0 if not masses else min(1.0, sum(masses) / (len(names) * 2.0))
        self.confidence = float(0.50 * coverage + 0.35 * certainty + 0.15 * mass_confidence)

        self.metadata["condition_presence"] = presence
        self.metadata["present_count"] = present_count
        self.metadata["coverage"] = coverage
        self.metadata["certainty"] = certainty
        self.metadata["effective_mass_confidence"] = mass_confidence

    def clear(self) -> None:
        fresh = CoreExperienceState()
        for key, value in asdict(fresh).items():
            setattr(self, key, deepcopy(value))

    def contains_forbidden_position_label(self) -> bool:
        text = json.dumps(self.snapshot(), ensure_ascii=False)
        return any(label in text for label in ["左", "中央", "右", "left", "center", "right"])
