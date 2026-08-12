from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import hashlib

import numpy as np

from brain import SphereBrain
from structural_context_v2 import StructuralContextV2
from structural_observer import StructuralEpisode


@dataclass(frozen=True)
class ShadowV3Config:
    gain: float = 0.02
    usage_scale: float = 20.0
    tie_margin: float = 0.0025
    near_zero_margin: float = 1e-8
    relative_cap_ratio: float = 0.35
    absolute_cap: float = 5e-5


class CoreIntegrationShadowV3:
    """Read-only virtual integration with explicit near-zero tie handling.

    Strong baseline decisions are never modulated. Near-zero ties are classified
    separately as structure-resolved ties, and modulation is bounded by both an
    absolute cap and a relative cap where the baseline margin is meaningful.
    """

    def __init__(self, config: ShadowV3Config | None = None) -> None:
        self.config = config or ShadowV3Config()
        self.context_builder = StructuralContextV2()

    def run(self, brain: SphereBrain, source_nodes: Iterable[int], steps: int = 18, threshold: float = 0.18) -> dict:
        before = self._array_digest(brain)
        baseline = self._run_route(brain, source_nodes, 0.0, steps, threshold)
        shadow = self._run_route(brain, source_nodes, self.config.gain, steps, threshold)
        after = self._array_digest(brain)

        baseline_edges = {tuple(edge) for edge in baseline["route_edges"]}
        shadow_edges = {tuple(edge) for edge in shadow["route_edges"]}
        baseline_nodes = set(baseline["activated_nodes"])
        shadow_nodes = set(shadow["activated_nodes"])
        return {
            "gain": self.config.gain,
            "read_only": True,
            "core_unchanged": before == after,
            "baseline": baseline,
            "shadow": shadow,
            "route_edge_symmetric_difference": len(baseline_edges ^ shadow_edges),
            "route_edge_jaccard": self._jaccard(baseline_edges, shadow_edges),
            "activated_node_symmetric_difference": len(baseline_nodes ^ shadow_nodes),
            "activated_node_jaccard": self._jaccard(baseline_nodes, shadow_nodes),
            "step_count_difference": abs(baseline["step_count"] - shadow["step_count"]),
            "strong_decision_override_count": shadow["strong_decision_override_count"],
            "tie_resolved_by_structure_count": shadow["tie_resolved_by_structure_count"],
            "selected_set_difference_count": shadow["selected_set_difference_count"],
            "max_absolute_modulation": shadow["max_absolute_modulation"],
            "max_meaningful_relative_ratio": shadow["max_meaningful_relative_ratio"],
        }

    def _run_route(self, brain: SphereBrain, source_nodes: Iterable[int], gain: float, steps: int, threshold: float) -> dict:
        sources, activation = brain._initial_activation(source_nodes, None)
        activated_nodes = set(np.flatnonzero(activation > 0).tolist())
        history = [sorted(activated_nodes)]
        edges_by_step: list[list[tuple[int, int]]] = []
        route_edges: list[list[int]] = []
        records: list[dict] = []
        strong_overrides = 0
        ties_resolved = 0
        selected_set_differences = 0
        max_abs_mod = 0.0
        max_relative = 0.0

        for step_index in range(steps):
            active_sources = np.flatnonzero(activation > 0)
            if active_sources.size == 0:
                break
            candidates: dict[int, tuple[float, int]] = {}
            for source in active_sources:
                neighbors = np.flatnonzero(brain.adjacency[source])
                if neighbors.size == 0:
                    continue
                scores = activation[source] * brain.weights[source, neighbors]
                branch_count = min(brain.max_branches, neighbors.size)
                best_indices = np.argpartition(scores, -branch_count)[-branch_count:]
                for local_index in best_indices:
                    target = int(neighbors[local_index])
                    value = float(scores[local_index]) * brain.signal_decay
                    if value < threshold:
                        continue
                    previous = candidates.get(target)
                    if previous is None or value > previous[0]:
                        candidates[target] = (value, int(source))
            if not candidates:
                break

            ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)
            baseline_logits = np.asarray([payload[0] for _, payload in ranked], dtype=float)
            baseline_order = np.argsort(-baseline_logits)
            margin = float(baseline_logits[baseline_order[0]] - baseline_logits[baseline_order[1]]) if len(ranked) > 1 else float("inf")
            near_zero = bool(len(ranked) > 1 and margin <= self.config.near_zero_margin)
            tie_gate = bool(len(ranked) > 1 and margin <= self.config.tie_margin and gain > 0)

            episode = StructuralEpisode.from_lists(history, edges_by_step + [[]])
            context = self.context_builder.instant(episode)
            terms = np.asarray([self._affinity(brain, source, target, context) for target, (_, source) in ranked], dtype=float)
            terms -= terms.mean() if terms.size else 0.0

            if tie_gate:
                raw = gain * terms
                if near_zero:
                    cap = self.config.absolute_cap
                else:
                    cap = min(self.config.absolute_cap, self.config.relative_cap_ratio * margin)
                modulation = np.clip(raw, -cap, cap)
            else:
                modulation = np.zeros_like(terms)

            shadow_logits = baseline_logits + modulation
            shadow_order = np.argsort(-shadow_logits)
            top_changed = bool(len(ranked) > 1 and int(shadow_order[0]) != int(baseline_order[0]))
            if top_changed and near_zero:
                ties_resolved += 1
            elif top_changed and not near_zero:
                strong_overrides += 1

            remaining_capacity = max(0, brain.max_total_active_nodes - len(activated_nodes))
            step_limit = min(brain.max_active_per_step, len(ranked))
            baseline_selected = self._select(ranked, baseline_order, activated_nodes, remaining_capacity, step_limit)
            shadow_selected = self._select(ranked, shadow_order, activated_nodes, remaining_capacity, step_limit)
            baseline_set = {(payload[1], target) for target, payload in baseline_selected}
            shadow_set = {(payload[1], target) for target, payload in shadow_selected}
            selected_set_changed = baseline_set != shadow_set
            selected_set_differences += int(selected_set_changed)

            if not shadow_selected:
                break
            next_activation = np.zeros(brain.node_count, dtype=float)
            accepted_edges: list[tuple[int, int]] = []
            for target, (value, source) in shadow_selected:
                value = float(np.clip(value, 0.0, 1.0))
                if value < threshold:
                    continue
                next_activation[target] = max(next_activation[target], value)
                accepted_edges.append((source, target))
                route_edges.append([int(source), int(target)])
            active_now = np.flatnonzero(next_activation > 0).tolist()
            if not active_now:
                break

            abs_mod = float(np.max(np.abs(modulation))) if modulation.size else 0.0
            relative = abs_mod / margin if np.isfinite(margin) and margin > self.config.near_zero_margin else 0.0
            max_abs_mod = max(max_abs_mod, abs_mod)
            max_relative = max(max_relative, relative)
            records.append({
                "step": step_index,
                "candidate_count": len(ranked),
                "baseline_margin": None if not np.isfinite(margin) else margin,
                "near_zero_tie": near_zero,
                "tie_gate_active": tie_gate,
                "closed_cycle": float(context[-1]),
                "local_repetition": float(context[-2]),
                "top_candidate_changed": top_changed,
                "tie_resolved_by_structure": bool(top_changed and near_zero),
                "strong_decision_overridden": bool(top_changed and not near_zero),
                "selected_set_changed": selected_set_changed,
                "absolute_modulation": abs_mod,
                "meaningful_relative_ratio": relative,
            })
            edges_by_step.append(accepted_edges)
            history.append(active_now)
            activated_nodes.update(active_now)
            activation = next_activation
            if len(activated_nodes) >= brain.max_total_active_nodes:
                break

        return {
            "gain": gain,
            "source_nodes": [int(node) for node in sources],
            "step_count": len(records),
            "activated_nodes": sorted(int(node) for node in activated_nodes),
            "route_edges": route_edges,
            "records": records,
            "strong_decision_override_count": strong_overrides,
            "tie_resolved_by_structure_count": ties_resolved,
            "selected_set_difference_count": selected_set_differences,
            "max_absolute_modulation": max_abs_mod,
            "max_meaningful_relative_ratio": max_relative,
        }

    @staticmethod
    def _select(ranked, order, activated_nodes, remaining_capacity, step_limit):
        selected = []
        new_nodes_selected = 0
        for index in order:
            target, payload = ranked[int(index)]
            is_new = target not in activated_nodes
            if is_new and new_nodes_selected >= remaining_capacity:
                continue
            selected.append((target, payload))
            new_nodes_selected += int(is_new)
            if len(selected) >= step_limit:
                break
        return selected

    def _affinity(self, brain: SphereBrain, source: int, target: int, context: np.ndarray) -> float:
        weight = float(brain.weights[source, target])
        usage = min(1.0, float(brain.usage[source, target]) / self.config.usage_scale)
        target_degree = float(np.count_nonzero(brain.adjacency[target])) / max(1.0, brain.neighbors_per_node * 2.0)
        edge_length = float(np.linalg.norm(brain.positions[target] - brain.positions[source]))
        length_affinity = 1.0 / (1.0 + edge_length)
        radial_delta = float(np.linalg.norm(brain.positions[target]) - np.linalg.norm(brain.positions[source]))
        direction = float(np.tanh(3.0 * radial_delta))
        merge, split, depth, parallel, reuse, repetition, cycle = context[2], context[3], context[5], context[6], context[8], context[9], context[10]
        return float(merge * target_degree + split * abs(direction) + depth * weight + parallel * length_affinity + reuse * usage + repetition * usage * length_affinity + cycle * direction * weight)

    @staticmethod
    def _jaccard(left: set, right: set) -> float:
        union = left | right
        return len(left & right) / len(union) if union else 1.0

    @staticmethod
    def _array_digest(brain: SphereBrain) -> str:
        digest = hashlib.sha256()
        for array in (brain.positions, brain.adjacency, brain.weights, brain.usage, brain.node_usage):
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()
