from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import hashlib

import numpy as np

from brain import SphereBrain
from structural_observer import StructuralEpisode, StructuralObserver


@dataclass(frozen=True)
class ShadowConfig:
    structural_gain: float = 0.05
    usage_scale: float = 20.0
    enabled: bool = True


class CoreIntegrationShadow:
    """Replay focused Core propagation and compute a non-intervening shadow route.

    The baseline route is reproduced from the current Core arrays. Structural
    modulation is calculated beside it and never fed back into activation,
    weights, usage, node usage, route selection, or learning.
    """

    CONTEXT_NAMES = (
        "source_ratio", "sink_ratio", "merge_ratio", "split_ratio",
        "component_ratio", "depth_ratio", "parallel_ratio",
        "temporal_overlap", "edge_reuse", "cycle_hint",
    )

    def __init__(self, config: ShadowConfig | None = None) -> None:
        self.config = config or ShadowConfig()
        self.observer = StructuralObserver()

    def run(
        self,
        brain: SphereBrain,
        source_nodes: Iterable[int],
        steps: int = 18,
        threshold: float = 0.18,
    ) -> dict:
        if brain.propagation_mode != "focused":
            raise ValueError("Core Integration Shadow v1 supports focused mode only")

        before = self._array_digest(brain)
        sources, activation = brain._initial_activation(source_nodes, None)
        activated_nodes = set(np.flatnonzero(activation > 0).tolist())
        baseline_edges_by_step: list[list[tuple[int, int]]] = []
        history: list[list[int]] = [sorted(activated_nodes)]
        step_records: list[dict] = []

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
            remaining_capacity = max(0, brain.max_total_active_nodes - len(activated_nodes))
            step_limit = min(brain.max_active_per_step, len(ranked))
            selected: list[tuple[int, tuple[float, int]]] = []
            new_nodes_selected = 0
            for target, payload in ranked:
                is_new = target not in activated_nodes
                if is_new and new_nodes_selected >= remaining_capacity:
                    continue
                selected.append((target, payload))
                new_nodes_selected += int(is_new)
                if len(selected) >= step_limit:
                    break
            if not selected:
                break

            baseline_edges = [(source, target) for target, (_, source) in selected]
            partial_episode = StructuralEpisode.from_lists(
                history,
                baseline_edges_by_step + [[]],
            )
            context = self._context(self.observer.observe(partial_episode))

            baseline_logits = np.asarray([payload[0] for _, payload in selected], dtype=float)
            structural_terms = np.asarray(
                [self._affinity(brain, source, target, context) for target, (_, source) in selected],
                dtype=float,
            )
            structural_terms -= structural_terms.mean() if structural_terms.size else 0.0
            modulation = (
                self.config.structural_gain * structural_terms
                if self.config.enabled else np.zeros_like(structural_terms)
            )
            shadow_logits = baseline_logits + modulation
            shadow_order = np.argsort(-shadow_logits).tolist()
            baseline_order = list(range(len(selected)))

            baseline_prob = self._softmax(baseline_logits)
            shadow_prob = self._softmax(shadow_logits)
            top_k = min(brain.max_branches, len(selected))
            baseline_top = set(baseline_order[:top_k])
            shadow_top = set(shadow_order[:top_k])

            next_activation = np.zeros(brain.node_count, dtype=float)
            accepted_edges: list[tuple[int, int]] = []
            for target, (value, source) in selected:
                value = float(np.clip(value, 0.0, 1.0))
                if value < threshold:
                    continue
                next_activation[target] = max(next_activation[target], value)
                accepted_edges.append((source, target))

            active_now = np.flatnonzero(next_activation > 0).tolist()
            if not active_now:
                break

            step_records.append({
                "step": step_index,
                "active_sources": active_sources.astype(int).tolist(),
                "candidate_count": len(selected),
                "candidate_edges": [[int(source), int(target)] for target, (_, source) in selected],
                "baseline_logits": baseline_logits.tolist(),
                "structural_context": context.tolist(),
                "shadow_modulation": modulation.tolist(),
                "shadow_logits": shadow_logits.tolist(),
                "baseline_probabilities": baseline_prob.tolist(),
                "shadow_probabilities": shadow_prob.tolist(),
                "probability_distance": float(np.linalg.norm(baseline_prob - shadow_prob)),
                "top_k_overlap": len(baseline_top & shadow_top) / max(1, top_k),
                "top_candidate_changed": bool(shadow_order and shadow_order[0] != 0),
            })

            baseline_edges_by_step.append(accepted_edges)
            activated_nodes.update(active_now)
            history.append(active_now)
            activation = next_activation
            if len(activated_nodes) >= brain.max_total_active_nodes:
                break

        after = self._array_digest(brain)
        probability_distances = [item["probability_distance"] for item in step_records]
        overlaps = [item["top_k_overlap"] for item in step_records]
        changed = [item["top_candidate_changed"] for item in step_records]

        return {
            "shadow_version": 1,
            "read_only": True,
            "intervenes_in_core": False,
            "learning": False,
            "noise": 0.0,
            "structural_gain": self.config.structural_gain,
            "source_nodes": [int(node) for node in sources],
            "step_count": len(step_records),
            "activated_node_count": len(activated_nodes),
            "baseline_route_edges": [
                [int(a), int(b)] for edges in baseline_edges_by_step for a, b in edges
            ],
            "steps": step_records,
            "summary": {
                "mean_probability_distance": float(np.mean(probability_distances)) if probability_distances else 0.0,
                "max_probability_distance": float(np.max(probability_distances)) if probability_distances else 0.0,
                "mean_top_k_overlap": float(np.mean(overlaps)) if overlaps else 1.0,
                "top_candidate_change_count": int(sum(changed)),
                "top_candidate_change_ratio": float(np.mean(changed)) if changed else 0.0,
            },
            "core_arrays_unchanged": before == after,
            "core_digest_before": before,
            "core_digest_after": after,
        }

    def _context(self, observation: dict) -> np.ndarray:
        shape = observation["shape"]
        temporal = observation["temporal"]
        n = max(1.0, float(shape["node_count"]))
        e = max(1.0, float(shape["edge_count"]))
        cycle_hint = 1.0 if shape["sources"] == 0 and shape["sinks"] == 0 and e > 0 else 0.0
        return np.asarray([
            shape["sources"] / n,
            shape["sinks"] / n,
            shape["merges"] / n,
            shape["splits"] / n,
            shape["components"] / n,
            shape["max_depth"] / max(1.0, n - 1.0),
            shape["parallel_width"] / n,
            temporal["temporal_overlap"],
            temporal["edge_reuse"],
            cycle_hint,
        ], dtype=float)

    def _affinity(
        self,
        brain: SphereBrain,
        source: int,
        target: int,
        context: np.ndarray,
    ) -> float:
        weight = float(brain.weights[source, target])
        usage = min(1.0, float(brain.usage[source, target]) / self.config.usage_scale)
        target_degree = float(np.count_nonzero(brain.adjacency[target])) / max(1.0, brain.neighbors_per_node * 2.0)
        edge_length = float(np.linalg.norm(brain.positions[target] - brain.positions[source]))
        length_affinity = 1.0 / (1.0 + edge_length)
        radial_delta = float(np.linalg.norm(brain.positions[target]) - np.linalg.norm(brain.positions[source]))
        direction = float(np.tanh(3.0 * radial_delta))

        merge, split, depth, parallel, reuse, cycle = (
            context[2], context[3], context[5], context[6], context[8], context[9]
        )
        return float(
            merge * target_degree
            + split * abs(direction)
            + depth * weight
            + parallel * length_affinity
            + reuse * usage
            + cycle * direction * weight
        )

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values.copy()
        shifted = values - values.max()
        exp = np.exp(shifted)
        return exp / exp.sum()

    @staticmethod
    def _array_digest(brain: SphereBrain) -> str:
        digest = hashlib.sha256()
        for array in (brain.positions, brain.adjacency, brain.weights, brain.usage, brain.node_usage):
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()
