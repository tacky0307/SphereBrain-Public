from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from structural_core_assist import StructuralAssistConfig, StructuralCoreAssist


@dataclass(frozen=True)
class BehavioralShadowAssistConfig:
    enabled: bool = False
    minimum_confidence: float = 0.90
    tie_margin: float = 0.0025
    near_zero_margin: float = 1e-8
    relative_cap_ratio: float = 0.20
    absolute_cap: float = 2.5e-5
    structural_gain: float = 0.02


class BoundedBehavioralShadowAssist:
    """Shadow-gated, tie-only candidate reordering.

    The Shadow State never names or scores a target Node. It only permits a tiny
    tie-break when its experience-derived stability is sufficiently confident
    and no drift is suspected. Candidate-local preference is computed from the
    Core's current local structural context. Candidate values, thresholds,
    weights, topology, usage, and learning are never modified here.
    """

    def __init__(self, config: BehavioralShadowAssistConfig) -> None:
        self.config = config
        self._structural = StructuralCoreAssist(
            StructuralAssistConfig(
                enabled=True,
                gain=config.structural_gain,
                tie_margin=config.tie_margin,
                near_zero_margin=config.near_zero_margin,
                relative_cap_ratio=config.relative_cap_ratio,
                absolute_cap=config.absolute_cap,
            )
        )

    def reorder(
        self,
        brain,
        ranked: Sequence[tuple[int, tuple[float, int]]],
        history: Sequence[Sequence[int]],
        edges_by_step: Sequence[Sequence[tuple[int, int]]],
    ) -> tuple[list[tuple[int, tuple[float, int]]], dict]:
        ranked_list = list(ranked)
        state = getattr(brain, "shadow_state", None) or {}
        confidence = float(state.get("evidence_confidence", 0.0) or 0.0)
        drift = bool(state.get("drift_suspected", False))
        stable_kind = str(state.get("kind", "")).startswith("motif_stability_profile")

        trace = {
            "enabled": self.config.enabled,
            "shadow_present": bool(state),
            "shadow_confidence": confidence,
            "drift_suspected": drift,
            "stable_shadow_kind": stable_kind,
            "confidence_gate": confidence >= self.config.minimum_confidence,
            "shadow_gate_open": False,
            "tie_gate_active": False,
            "near_zero_tie": False,
            "baseline_margin": None,
            "top_candidate_changed": False,
            "absolute_modulation": 0.0,
            "meaningful_relative_ratio": 0.0,
            "candidate_values_changed": False,
            "threshold_crossing_possible": False,
            "winner_forced_by_shadow": False,
        }
        if not self.config.enabled or len(ranked_list) < 2:
            return ranked_list, trace
        if not state or not stable_kind or drift or confidence < self.config.minimum_confidence:
            return ranked_list, trace

        baseline = np.asarray([payload[0] for _, payload in ranked_list], dtype=float)
        order = np.argsort(-baseline)
        margin = float(baseline[order[0]] - baseline[order[1]])
        near_zero = margin <= self.config.near_zero_margin
        trace["baseline_margin"] = margin
        trace["near_zero_tie"] = near_zero
        if margin > self.config.tie_margin:
            return ranked_list, trace

        trace["shadow_gate_open"] = True
        # Reuse the Core-local structural affinity, but apply stricter caps than
        # StructuralCoreAssist. The Shadow contributes no target-specific score.
        episode = self._structural.context_builder.instant(
            __import__("structural_observer").StructuralEpisode.from_lists(
                [list(step) for step in history],
                [list(step) for step in edges_by_step] + [[]],
            )
        )
        terms = np.asarray(
            [
                self._structural._affinity(brain, source, target, episode)
                for target, (_, source) in ranked_list
            ],
            dtype=float,
        )
        if terms.size:
            terms -= terms.mean()
        raw = self.config.structural_gain * terms
        if near_zero:
            cap = self.config.absolute_cap
        else:
            cap = min(self.config.absolute_cap, self.config.relative_cap_ratio * margin)
        modulation = np.clip(raw, -cap, cap)
        assisted = baseline + modulation
        assisted_order = np.argsort(-assisted)
        reordered = [ranked_list[int(i)] for i in assisted_order]

        max_abs = float(np.max(np.abs(modulation))) if modulation.size else 0.0
        trace.update({
            "tie_gate_active": True,
            "top_candidate_changed": int(assisted_order[0]) != int(order[0]),
            "absolute_modulation": max_abs,
            "meaningful_relative_ratio": 0.0 if near_zero else max_abs / max(margin, 1e-30),
            "cap_used": float(cap),
        })
        return reordered, trace
