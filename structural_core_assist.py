from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from structural_context_v2 import StructuralContextV2
from structural_observer import StructuralEpisode


@dataclass(frozen=True)
class StructuralAssistConfig:
    enabled: bool = False
    gain: float = 0.02
    tie_margin: float = 0.0025
    near_zero_margin: float = 1e-8
    relative_cap_ratio: float = 0.35
    absolute_cap: float = 5e-5
    usage_scale: float = 20.0


class StructuralCoreAssist:
    """Bounded tie-only structural assistance for focused Core propagation.

    The assist never changes candidate values, Edge weights, usage, learning, or
    the candidate set. It may only reorder candidates when the top baseline
    candidates are tied or nearly tied. Strong baseline decisions are untouched.
    """

    def __init__(self, config: StructuralAssistConfig) -> None:
        self.config = config
        self.context_builder = StructuralContextV2()

    def reorder(
        self,
        brain,
        ranked: Sequence[tuple[int, tuple[float, int]]],
        history: Sequence[Sequence[int]],
        edges_by_step: Sequence[Sequence[tuple[int, int]]],
    ) -> tuple[list[tuple[int, tuple[float, int]]], dict]:
        ranked_list = list(ranked)
        trace = {
            "enabled": self.config.enabled,
            "tie_gate_active": False,
            "near_zero_tie": False,
            "baseline_margin": None,
            "top_candidate_changed": False,
            "absolute_modulation": 0.0,
            "meaningful_relative_ratio": 0.0,
        }
        if not self.config.enabled or len(ranked_list) < 2:
            return ranked_list, trace

        baseline = np.asarray([payload[0] for _, payload in ranked_list], dtype=float)
        order = np.argsort(-baseline)
        margin = float(baseline[order[0]] - baseline[order[1]])
        near_zero = margin <= self.config.near_zero_margin
        trace["baseline_margin"] = margin
        trace["near_zero_tie"] = near_zero

        if margin > self.config.tie_margin:
            return ranked_list, trace

        episode = StructuralEpisode.from_lists(
            [list(step) for step in history],
            [list(step) for step in edges_by_step] + [[]],
        )
        context = self.context_builder.instant(episode)
        terms = np.asarray(
            [self._affinity(brain, source, target, context) for target, (_, source) in ranked_list],
            dtype=float,
        )
        terms -= terms.mean() if terms.size else 0.0
        raw = self.config.gain * terms

        if near_zero:
            cap = self.config.absolute_cap
        else:
            cap = min(
                self.config.absolute_cap,
                self.config.relative_cap_ratio * margin,
            )
        modulation = np.clip(raw, -cap, cap)
        assisted = baseline + modulation
        assisted_order = np.argsort(-assisted)
        reordered = [ranked_list[int(index)] for index in assisted_order]

        max_abs = float(np.max(np.abs(modulation))) if modulation.size else 0.0
        trace.update({
            "tie_gate_active": True,
            "top_candidate_changed": int(assisted_order[0]) != int(order[0]),
            "absolute_modulation": max_abs,
            "meaningful_relative_ratio": 0.0 if near_zero else max_abs / max(margin, 1e-30),
        })
        return reordered, trace

    def _affinity(self, brain, source: int, target: int, context: np.ndarray) -> float:
        weight = float(brain.weights[source, target])
        usage = min(1.0, float(brain.usage[source, target]) / self.config.usage_scale)
        target_degree = float(np.count_nonzero(brain.adjacency[target])) / max(
            1.0, brain.neighbors_per_node * 2.0
        )
        edge_length = float(np.linalg.norm(brain.positions[target] - brain.positions[source]))
        length_affinity = 1.0 / (1.0 + edge_length)
        radial_delta = float(
            np.linalg.norm(brain.positions[target]) - np.linalg.norm(brain.positions[source])
        )
        direction = float(np.tanh(3.0 * radial_delta))
        merge, split, depth, parallel, reuse, repetition, cycle = (
            context[2], context[3], context[5], context[6],
            context[8], context[9], context[10],
        )
        return float(
            merge * target_degree
            + split * abs(direction)
            + depth * weight
            + parallel * length_affinity
            + reuse * usage
            + repetition * usage * length_affinity
            + cycle * direction * weight
        )
