from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import numpy as np

from brain import SignalResult, SphereBrain


@dataclass(frozen=True)
class ObserverConfig:
    """Read-only configuration for the Core State Observer."""

    temperature: float = 1.0
    include_inactive_neighbors: bool = False
    inactive_neighbor_limit: int = 0


class CoreStateObserver:
    """Observe a completed Core propagation without changing the Core.

    This module deliberately has:
    - no learning parameters,
    - no call to SphereBrain.propagate,
    - no writes to Node/Edge state,
    - no feedback path into the Core.

    It borrows only the Transformer's parallel relation calculation:
    all observed Node feature vectors are compared at once with a
    deterministic self-attention calculation.
    """

    FEATURE_NAMES = (
        "pos_x",
        "pos_y",
        "pos_z",
        "node_usage",
        "weighted_degree",
        "route_degree",
        "first_step",
        "last_step",
        "source_flag",
        "final_activation",
    )

    def __init__(self, config: ObserverConfig | None = None) -> None:
        self.config = config or ObserverConfig()

    def observe(self, brain: SphereBrain, result: SignalResult) -> dict[str, Any]:
        node_ids = self._observed_nodes(brain, result)
        if not node_ids:
            return self._empty_observation()

        raw_features = self._build_features(brain, result, node_ids)
        normalized = self._normalize_features(raw_features)
        relation_matrix, contextualized = self._parallel_relations(normalized)

        route_edges = {tuple(sorted(edge)) for edge in result.traversed_edges}
        relation_summary = self._relation_summary(node_ids, relation_matrix, route_edges)
        state_vector = self._pool_state(raw_features, normalized, contextualized, relation_matrix)
        temporal = self._temporal_summary(result, node_ids)

        return {
            "observer_version": 1,
            "read_only": True,
            "method": "deterministic identity self-attention over Core node features",
            "feature_names": list(self.FEATURE_NAMES),
            "observed_node_ids": node_ids,
            "observed_node_count": len(node_ids),
            "route_edge_count": len(route_edges),
            "state_vector": state_vector.tolist(),
            "state_vector_dim": int(state_vector.size),
            "relation_summary": relation_summary,
            "temporal_summary": temporal,
        }

    def _observed_nodes(self, brain: SphereBrain, result: SignalResult) -> list[int]:
        active = set(int(node) for node in result.activated_nodes)
        if self.config.include_inactive_neighbors and self.config.inactive_neighbor_limit > 0:
            candidates: list[tuple[float, int]] = []
            for node in active:
                for other in np.flatnonzero(brain.adjacency[node]):
                    other_id = int(other)
                    if other_id in active:
                        continue
                    candidates.append((float(brain.weights[node, other_id]), other_id))
            candidates.sort(reverse=True)
            for _, node in candidates:
                active.add(node)
                if len(active) >= len(result.activated_nodes) + self.config.inactive_neighbor_limit:
                    break
        return sorted(active)

    def _build_features(
        self,
        brain: SphereBrain,
        result: SignalResult,
        node_ids: list[int],
    ) -> np.ndarray:
        first_step: dict[int, int] = {}
        last_step: dict[int, int] = {}
        for step_index, nodes in enumerate(result.activation_history):
            for node in nodes:
                node_id = int(node)
                first_step.setdefault(node_id, step_index)
                last_step[node_id] = step_index

        max_step = max(1, len(result.activation_history) - 1)
        max_usage = max(1.0, float(np.max(brain.node_usage)))
        weighted_degree_all = np.sum(brain.weights * brain.adjacency, axis=1)
        max_weighted_degree = max(1e-12, float(np.max(weighted_degree_all)))

        route_degree: dict[int, int] = {node: 0 for node in node_ids}
        for left, right in result.traversed_edges:
            a, b = int(left), int(right)
            if a in route_degree:
                route_degree[a] += 1
            if b in route_degree:
                route_degree[b] += 1
        max_route_degree = max(1, max(route_degree.values(), default=0))

        source_set = {int(node) for node in result.source_nodes}
        final_activation = np.asarray(result.final_activation, dtype=float)

        rows: list[list[float]] = []
        for node in node_ids:
            x, y, z = brain.positions[node]
            rows.append(
                [
                    float(x),
                    float(y),
                    float(z),
                    float(brain.node_usage[node]) / max_usage,
                    float(weighted_degree_all[node]) / max_weighted_degree,
                    float(route_degree.get(node, 0)) / max_route_degree,
                    float(first_step.get(node, max_step)) / max_step,
                    float(last_step.get(node, max_step)) / max_step,
                    1.0 if node in source_set else 0.0,
                    float(final_activation[node]) if node < final_activation.size else 0.0,
                ]
            )
        return np.asarray(rows, dtype=float)

    @staticmethod
    def _normalize_features(features: np.ndarray) -> np.ndarray:
        means = features.mean(axis=0, keepdims=True)
        stds = features.std(axis=0, keepdims=True)
        safe_stds = np.where(stds < 1e-9, 1.0, stds)
        normalized = (features - means) / safe_stds
        return np.nan_to_num(normalized, copy=False)

    def _parallel_relations(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dimension = max(1, features.shape[1])
        scale = math.sqrt(dimension) * max(1e-6, self.config.temperature)
        logits = (features @ features.T) / scale
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        denominators = exp_logits.sum(axis=1, keepdims=True)
        attention = exp_logits / np.where(denominators == 0.0, 1.0, denominators)
        contextualized = attention @ features
        return attention, contextualized

    @staticmethod
    def _pool_state(
        raw_features: np.ndarray,
        normalized: np.ndarray,
        contextualized: np.ndarray,
        relation_matrix: np.ndarray,
    ) -> np.ndarray:
        raw_mean = raw_features.mean(axis=0)
        raw_std = raw_features.std(axis=0)
        normalized_std = normalized.std(axis=0)
        mean_context = contextualized.mean(axis=0)
        max_context = contextualized.max(axis=0)
        diagonal = float(np.trace(relation_matrix) / max(1, relation_matrix.shape[0]))
        entropy = -np.sum(
            relation_matrix * np.log(np.clip(relation_matrix, 1e-12, 1.0)),
            axis=1,
        )
        relation_stats = np.asarray(
            [
                diagonal,
                float(entropy.mean()),
                float(entropy.std()),
                float(relation_matrix.max(axis=1).mean()),
            ],
            dtype=float,
        )
        return np.concatenate([
            raw_mean,
            raw_std,
            normalized_std,
            mean_context,
            max_context,
            relation_stats,
        ])

    @staticmethod
    def _relation_summary(
        node_ids: list[int],
        relation_matrix: np.ndarray,
        route_edges: set[tuple[int, int]],
    ) -> dict[str, Any]:
        pairs: list[dict[str, Any]] = []
        route_scores: list[float] = []
        non_route_scores: list[float] = []

        for i, source in enumerate(node_ids):
            for j in range(i + 1, len(node_ids)):
                target = node_ids[j]
                score = float((relation_matrix[i, j] + relation_matrix[j, i]) / 2.0)
                is_route_edge = tuple(sorted((source, target))) in route_edges
                (route_scores if is_route_edge else non_route_scores).append(score)
                pairs.append(
                    {
                        "node_a": source,
                        "node_b": target,
                        "relation": round(score, 8),
                        "is_route_edge": is_route_edge,
                    }
                )

        pairs.sort(key=lambda item: item["relation"], reverse=True)
        return {
            "top_pairs": pairs[:20],
            "route_pair_mean": float(np.mean(route_scores)) if route_scores else 0.0,
            "non_route_pair_mean": float(np.mean(non_route_scores)) if non_route_scores else 0.0,
            "route_minus_non_route": (
                float(np.mean(route_scores) - np.mean(non_route_scores))
                if route_scores and non_route_scores
                else 0.0
            ),
        }

    @staticmethod
    def _temporal_summary(result: SignalResult, node_ids: list[int]) -> dict[str, Any]:
        active_set = set(node_ids)
        steps = []
        previous: set[int] = set()
        for index, step_nodes in enumerate(result.activation_history):
            current = {int(node) for node in step_nodes if int(node) in active_set}
            union = previous | current
            retention = len(previous & current) / len(union) if union else 0.0
            steps.append(
                {
                    "step": index,
                    "active_count": len(current),
                    "new_count": len(current - previous),
                    "retention_from_previous": retention,
                }
            )
            previous = current
        return {
            "step_count": len(steps),
            "steps": steps,
        }

    @staticmethod
    def cosine_similarity(left: list[float], right: list[float]) -> float:
        a = np.asarray(left, dtype=float)
        b = np.asarray(right, dtype=float)
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denominator) if denominator > 0.0 else 0.0

    @staticmethod
    def _empty_observation() -> dict[str, Any]:
        return {
            "observer_version": 1,
            "read_only": True,
            "method": "deterministic identity self-attention over Core node features",
            "feature_names": list(CoreStateObserver.FEATURE_NAMES),
            "observed_node_ids": [],
            "observed_node_count": 0,
            "route_edge_count": 0,
            "state_vector": [],
            "state_vector_dim": 0,
            "relation_summary": {
                "top_pairs": [],
                "route_pair_mean": 0.0,
                "non_route_pair_mean": 0.0,
                "route_minus_non_route": 0.0,
            },
            "temporal_summary": {"step_count": 0, "steps": []},
        }
