from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json
import hashlib
import numpy as np

from core_experience_state import CoreExperienceState
from core_native_learning import CoreNativeLearningState
from structural_core_assist import StructuralAssistConfig, StructuralCoreAssist


@dataclass
class SignalResult:
    source_nodes: list[int]
    activated_nodes: list[int]
    traversed_edges: list[tuple[int, int]]
    activation_history: list[list[int]]
    final_activation: np.ndarray


class SphereBrain:
    def __init__(
        self,
        node_count: int = 240,
        neighbors_per_node: int = 7,
        seed: int = 42,
        learning_rate: float = 0.07,
        decay_rate: float = 0.0008,
        propagation_mode: str = "focused",
        signal_decay: float = 0.78,
        max_branches: int = 2,
        max_active_per_step: int = 72,
        max_total_active_nodes: int = 100,
        structural_assist_enabled: bool = False,
        structural_gain: float = 0.02,
        structural_tie_margin: float = 0.0025,
        structural_near_zero_margin: float = 1e-8,
        structural_relative_cap_ratio: float = 0.35,
        structural_absolute_cap: float = 5e-5,
    ) -> None:
        self.node_count = node_count
        self.neighbors_per_node = neighbors_per_node
        self.seed = seed
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.propagation_mode = propagation_mode
        self.signal_decay = signal_decay
        self.max_branches = max_branches
        self.max_active_per_step = max_active_per_step
        self.max_total_active_nodes = max_total_active_nodes
        self.structural_assist_enabled = structural_assist_enabled
        self.structural_gain = structural_gain
        self.structural_tie_margin = structural_tie_margin
        self.structural_near_zero_margin = structural_near_zero_margin
        self.structural_relative_cap_ratio = structural_relative_cap_ratio
        self.structural_absolute_cap = structural_absolute_cap
        self.rng = np.random.default_rng(seed)

        self.structural_assist = StructuralCoreAssist(
            StructuralAssistConfig(
                enabled=structural_assist_enabled,
                gain=structural_gain,
                tie_margin=structural_tie_margin,
                near_zero_margin=structural_near_zero_margin,
                relative_cap_ratio=structural_relative_cap_ratio,
                absolute_cap=structural_absolute_cap,
            )
        )
        self.last_structural_assist_trace: list[dict] = []

        # Native experience-derived Core state.
        self.experience_state = CoreExperienceState()

        # Primary Native Learning state validated by v73-v80B:
        # Temporal Credit + Homeostatic Consolidation.
        self.learning_state = CoreNativeLearningState()

        self.positions = self._generate_points_in_sphere(node_count)
        self.adjacency = np.zeros((node_count, node_count), dtype=bool)
        self.weights = np.zeros((node_count, node_count), dtype=float)
        self.usage = np.zeros((node_count, node_count), dtype=int)
        self.node_usage = np.zeros(node_count, dtype=int)

        self._connect_nearest_nodes()

    def snapshot_experience_state(self) -> dict:
        return self.experience_state.snapshot()

    def update_experience_state(self, **kwargs) -> None:
        """Update the native experience state without affecting propagation."""
        self.experience_state.update_profile(**kwargs)

    def clear_experience_state(self) -> None:
        self.experience_state.clear()

    def snapshot_learning_state(self) -> dict:
        return self.learning_state.snapshot()

    def clear_learning_state(self) -> None:
        self.learning_state.clear()

    def observe_learning_episode(
        self,
        transitions,
        *,
        success: bool,
        expected_conditions,
        motif: str = "native_learning",
    ) -> dict:
        """Apply Primary Core Native Learning to one externally observed episode."""
        return self.learning_state.observe_episode(
            self,
            transitions,
            success=success,
            expected_conditions=expected_conditions,
            motif=motif,
        )

    def set_structural_assist(self, enabled: bool) -> None:
        """Enable or disable bounded structural assistance at runtime."""
        self.structural_assist_enabled = bool(enabled)
        self.structural_assist = StructuralCoreAssist(
            StructuralAssistConfig(
                enabled=self.structural_assist_enabled,
                gain=self.structural_gain,
                tie_margin=self.structural_tie_margin,
                near_zero_margin=self.structural_near_zero_margin,
                relative_cap_ratio=self.structural_relative_cap_ratio,
                absolute_cap=self.structural_absolute_cap,
            )
        )

    def _generate_points_in_sphere(self, count: int) -> np.ndarray:
        directions = self.rng.normal(size=(count, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        radii = self.rng.random(count) ** (1.0 / 3.0)
        return directions * radii[:, None]

    def _connect_nearest_nodes(self) -> None:
        differences = self.positions[:, None, :] - self.positions[None, :, :]
        distances = np.linalg.norm(differences, axis=2)
        np.fill_diagonal(distances, np.inf)

        for node in range(self.node_count):
            nearest = np.argsort(distances[node])[: self.neighbors_per_node]
            for other in nearest:
                a, b = sorted((node, int(other)))
                self.adjacency[a, b] = True
                self.adjacency[b, a] = True
                if self.weights[a, b] == 0:
                    base = 0.22 + 0.42 * np.exp(-2.0 * distances[a, b])
                    weight = min(0.92, base + float(self.rng.uniform(0.0, 0.07)))
                    self.weights[a, b] = weight
                    self.weights[b, a] = weight

    def text_to_sources(self, text: str, count: int = 3) -> list[int]:
        clean = text.strip()
        if not clean:
            raise ValueError("入力が空です。")

        digest = hashlib.sha256(clean.encode("utf-8")).digest()
        sources: list[int] = []
        offset = 0
        while len(sources) < count:
            value = int.from_bytes(digest[offset:offset+4], "big")
            node = value % self.node_count
            if node not in sources:
                sources.append(node)
            offset += 4
            if offset + 4 > len(digest):
                digest = hashlib.sha256(digest).digest()
                offset = 0
        return sources

    def propagate(
        self,
        source_nodes: Iterable[int],
        steps: int = 18,
        threshold: float = 0.15,
        noise: float = 0.018,
        learn: bool = True,
        context_nodes: Iterable[int] | None = None,
    ) -> SignalResult:
        self.last_structural_assist_trace = []
        if self.propagation_mode == "legacy":
            return self._propagate_legacy(
                source_nodes=source_nodes,
                steps=steps,
                threshold=threshold,
                noise=noise,
                learn=learn,
                context_nodes=context_nodes,
            )

        return self._propagate_focused(
            source_nodes=source_nodes,
            steps=steps,
            threshold=max(threshold, 0.18),
            noise=min(noise, 0.006),
            learn=learn,
            context_nodes=context_nodes,
        )

    def _initial_activation(
        self,
        source_nodes: Iterable[int],
        context_nodes: Iterable[int] | None,
    ) -> tuple[list[int], np.ndarray]:
        sources = list(source_nodes)
        activation = np.zeros(self.node_count, dtype=float)

        for index, node in enumerate(sources):
            activation[node] = max(activation[node], 1.0 - index * 0.08)

        if context_nodes:
            for node in context_nodes:
                activation[node] = max(activation[node], 0.34)

        return sources, activation

    def _propagate_focused(
        self,
        source_nodes: Iterable[int],
        steps: int,
        threshold: float,
        noise: float,
        learn: bool,
        context_nodes: Iterable[int] | None,
    ) -> SignalResult:
        sources, activation = self._initial_activation(source_nodes, context_nodes)
        activated_nodes = set(np.flatnonzero(activation > 0).tolist())
        traversed_edges: set[tuple[int, int]] = set()
        history = [sorted(activated_nodes)]
        edges_by_step: list[list[tuple[int, int]]] = []

        for step_index in range(steps):
            active_sources = np.flatnonzero(activation > 0)
            if active_sources.size == 0:
                break

            candidates: dict[int, tuple[float, int]] = {}

            for source in active_sources:
                neighbors = np.flatnonzero(self.adjacency[source])
                if neighbors.size == 0:
                    continue

                scores = activation[source] * self.weights[source, neighbors]
                branch_count = min(self.max_branches, neighbors.size)
                best_indices = np.argpartition(scores, -branch_count)[-branch_count:]

                for local_index in best_indices:
                    target = int(neighbors[local_index])
                    value = float(scores[local_index]) * self.signal_decay
                    if value < threshold:
                        continue
                    previous = candidates.get(target)
                    if previous is None or value > previous[0]:
                        candidates[target] = (value, int(source))

            if not candidates:
                break

            ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)
            ranked, assist_trace = self.structural_assist.reorder(
                self, ranked, history, edges_by_step
            )
            assist_trace["step"] = step_index
            self.last_structural_assist_trace.append(assist_trace)

            remaining_capacity = max(0, self.max_total_active_nodes - len(activated_nodes))
            step_limit = min(self.max_active_per_step, len(ranked))

            selected: list[tuple[int, tuple[float, int]]] = []
            new_nodes_selected = 0
            for target, payload in ranked:
                is_new = target not in activated_nodes
                if is_new and new_nodes_selected >= remaining_capacity:
                    continue
                selected.append((target, payload))
                if is_new:
                    new_nodes_selected += 1
                if len(selected) >= step_limit:
                    break

            if not selected:
                break

            next_activation = np.zeros(self.node_count, dtype=float)
            accepted_edges: list[tuple[int, int]] = []
            for target, (value, source) in selected:
                if noise:
                    value += float(self.rng.normal(0.0, noise))
                value = float(np.clip(value, 0.0, 1.0))
                if value < threshold:
                    continue
                next_activation[target] = max(next_activation[target], value)
                accepted_edges.append((source, target))
                traversed_edges.add(tuple(sorted((source, target))))

            active_now = np.flatnonzero(next_activation > 0).tolist()
            if not active_now:
                break

            activated_nodes.update(active_now)
            history.append(active_now)
            edges_by_step.append(accepted_edges)
            activation = next_activation

            if len(activated_nodes) >= self.max_total_active_nodes:
                break

        if learn and traversed_edges:
            self._reinforce(traversed_edges, activated_nodes)

        return SignalResult(
            source_nodes=sources,
            activated_nodes=sorted(activated_nodes),
            traversed_edges=sorted(traversed_edges),
            activation_history=history,
            final_activation=activation.copy(),
        )

    def _propagate_legacy(
        self,
        source_nodes: Iterable[int],
        steps: int,
        threshold: float,
        noise: float,
        learn: bool,
        context_nodes: Iterable[int] | None,
    ) -> SignalResult:
        sources, activation = self._initial_activation(source_nodes, context_nodes)
        activated_nodes = set(np.flatnonzero(activation > 0).tolist())
        traversed_edges: set[tuple[int, int]] = set()
        history = [sorted(activated_nodes)]

        for _ in range(steps):
            transmitted = activation[:, None] * self.weights
            next_activation = transmitted.max(axis=0) * 0.82

            if noise:
                next_activation += self.rng.normal(0.0, noise, self.node_count)

            next_activation = np.clip(next_activation, 0.0, 1.0)
            next_activation[next_activation < threshold] = 0.0

            active_now = np.flatnonzero(next_activation > 0).tolist()
            history.append(active_now)
            activated_nodes.update(active_now)

            for target in active_now:
                incoming = transmitted[:, target]
                source = int(np.argmax(incoming))
                if incoming[source] >= threshold and self.adjacency[source, target]:
                    traversed_edges.add(tuple(sorted((source, target))))

            activation = next_activation
            if not active_now:
                break

        if learn and traversed_edges:
            self._reinforce(traversed_edges, activated_nodes)

        return SignalResult(
            source_nodes=sources,
            activated_nodes=sorted(activated_nodes),
            traversed_edges=sorted(traversed_edges),
            activation_history=history,
            final_activation=activation.copy(),
        )

    def _reinforce(self, edges: Iterable[tuple[int, int]], nodes: Iterable[int]) -> None:
        self.weights[self.adjacency] *= 1.0 - self.decay_rate

        for a, b in edges:
            current = self.weights[a, b]
            new = current + self.learning_rate * (1.0 - current)
            self.weights[a, b] = new
            self.weights[b, a] = new
            self.usage[a, b] += 1
            self.usage[b, a] += 1

        for node in nodes:
            self.node_usage[node] += 1

        self.weights = np.clip(self.weights, 0.0, 1.0)

    def idle_cycle(self, remembered_nodes: Iterable[int]) -> SignalResult | None:
        nodes = list(remembered_nodes)
        if not nodes:
            return None

        source_count = min(2, len(nodes))
        sources = self.rng.choice(nodes, size=source_count, replace=False).tolist()
        return self.propagate(
            sources,
            steps=10,
            threshold=0.19,
            noise=0.025,
            learn=True,
        )

    def strongest_edges(self, limit: int = 40) -> list[dict]:
        upper = np.triu_indices(self.node_count, k=1)
        mask = self.adjacency[upper]
        pairs = list(zip(upper[0][mask], upper[1][mask]))
        pairs.sort(key=lambda e: self.weights[e[0], e[1]], reverse=True)
        return [
            {
                "a": int(a),
                "b": int(b),
                "weight": float(self.weights[a, b]),
                "usage": int(self.usage[a, b]),
            }
            for a, b in pairs[:limit]
        ]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        data = {
            "node_count": self.node_count,
            "neighbors_per_node": self.neighbors_per_node,
            "seed": self.seed,
            "learning_rate": self.learning_rate,
            "decay_rate": self.decay_rate,
            "propagation_mode": self.propagation_mode,
            "signal_decay": self.signal_decay,
            "max_branches": self.max_branches,
            "max_active_per_step": self.max_active_per_step,
            "max_total_active_nodes": self.max_total_active_nodes,
            "structural_assist_enabled": self.structural_assist_enabled,
            "structural_gain": self.structural_gain,
            "structural_tie_margin": self.structural_tie_margin,
            "structural_near_zero_margin": self.structural_near_zero_margin,
            "structural_relative_cap_ratio": self.structural_relative_cap_ratio,
            "structural_absolute_cap": self.structural_absolute_cap,
            "experience_state": self.experience_state.snapshot(),
            "learning_state": self.learning_state.snapshot(),
            "positions": self.positions.tolist(),
            "adjacency": self.adjacency.astype(int).tolist(),
            "weights": self.weights.tolist(),
            "usage": self.usage.tolist(),
            "node_usage": self.node_usage.tolist(),
        }
        path.write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SphereBrain":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        brain = cls(
            node_count=data["node_count"],
            neighbors_per_node=data["neighbors_per_node"],
            seed=data["seed"],
            learning_rate=data["learning_rate"],
            decay_rate=data["decay_rate"],
            propagation_mode=data.get("propagation_mode", "focused"),
            signal_decay=data.get("signal_decay", 0.78),
            max_branches=data.get("max_branches", 2),
            max_active_per_step=data.get("max_active_per_step", 72),
            max_total_active_nodes=data.get("max_total_active_nodes", 100),
            structural_assist_enabled=data.get("structural_assist_enabled", False),
            structural_gain=data.get("structural_gain", 0.02),
            structural_tie_margin=data.get("structural_tie_margin", 0.0025),
            structural_near_zero_margin=data.get("structural_near_zero_margin", 1e-8),
            structural_relative_cap_ratio=data.get("structural_relative_cap_ratio", 0.35),
            structural_absolute_cap=data.get("structural_absolute_cap", 5e-5),
        )
        brain.positions = np.asarray(data["positions"], dtype=float)
        brain.adjacency = np.asarray(data["adjacency"], dtype=bool)
        brain.weights = np.asarray(data["weights"], dtype=float)
        brain.usage = np.asarray(data["usage"], dtype=int)
        brain.node_usage = np.asarray(data.get("node_usage", [0] * brain.node_count), dtype=int)
        brain.experience_state.replace_from_mapping(data.get("experience_state"))
        brain.learning_state.replace_from_mapping(data.get("learning_state"))
        return brain