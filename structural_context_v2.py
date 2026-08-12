from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from structural_observer import StructuralEpisode, StructuralObserver


@dataclass(frozen=True)
class StructuralContextConfig:
    decay: float = 0.78


class StructuralContextV2:
    """Language-free structural context with separated reuse, repetition and cycle.

    This component observes activity only. It has no answer labels, trainable
    parameters, route selection, or long-term memory.
    """

    FEATURE_NAMES = (
        "source_ratio",
        "sink_ratio",
        "merge_ratio",
        "split_ratio",
        "component_ratio",
        "depth_ratio",
        "parallel_ratio",
        "temporal_overlap",
        "edge_reuse",
        "local_repetition",
        "closed_cycle",
    )

    def __init__(self, config: StructuralContextConfig | None = None) -> None:
        self.config = config or StructuralContextConfig()
        self.observer = StructuralObserver()

    def sequence(self, episode: StructuralEpisode) -> list[np.ndarray]:
        state = np.zeros(len(self.FEATURE_NAMES), dtype=float)
        states: list[np.ndarray] = []
        for step_index in range(len(episode.steps)):
            partial = StructuralEpisode(
                steps=episode.steps[: step_index + 1],
                edges_by_step=episode.edges_by_step[: step_index + 1],
            )
            instant = self.instant(partial)
            state = self.config.decay * state + (1.0 - self.config.decay) * instant
            states.append(state.copy())
        return states

    def instant(self, episode: StructuralEpisode) -> np.ndarray:
        observation = self.observer.observe(episode)
        shape = observation["shape"]
        temporal = observation["temporal"]
        n = max(1.0, float(shape["node_count"]))
        edge_count = float(shape["edge_count"])
        closed_cycle = self._closed_cycle(episode)
        local_repetition = self._local_repetition(episode.edges_by_step)
        return np.asarray(
            [
                shape["sources"] / n,
                shape["sinks"] / n,
                shape["merges"] / n,
                shape["splits"] / n,
                shape["components"] / n,
                shape["max_depth"] / max(1.0, n - 1.0),
                shape["parallel_width"] / n,
                temporal["temporal_overlap"],
                temporal["edge_reuse"] if edge_count > 0 else 0.0,
                local_repetition,
                closed_cycle,
            ],
            dtype=float,
        )

    @staticmethod
    def _closed_cycle(episode: StructuralEpisode) -> float:
        edges = {edge for step in episode.edges_by_step for edge in step}
        if not edges:
            return 0.0
        adjacency: dict[int, set[int]] = {}
        for source, target in edges:
            adjacency.setdefault(source, set()).add(target)
        visiting: set[int] = set()
        visited: set[int] = set()

        def visit(node: int) -> bool:
            if node in visiting:
                return True
            if node in visited:
                return False
            visiting.add(node)
            for other in adjacency.get(node, set()):
                if visit(other):
                    return True
            visiting.remove(node)
            visited.add(node)
            return False

        nodes = {node for edge in edges for node in edge}
        return 1.0 if any(visit(node) for node in nodes if node not in visited) else 0.0

    @staticmethod
    def _local_repetition(edges_by_step: Sequence[Sequence[tuple[int, int]]]) -> float:
        signatures: list[tuple[tuple[int, int], ...]] = []
        for edges in edges_by_step:
            if not edges:
                continue
            # ID-invariant local shape: in/out role counts within the step.
            indegree: dict[int, int] = {}
            outdegree: dict[int, int] = {}
            for source, target in edges:
                outdegree[source] = outdegree.get(source, 0) + 1
                indegree[target] = indegree.get(target, 0) + 1
                indegree.setdefault(source, 0)
                outdegree.setdefault(target, 0)
            signature = tuple(sorted((indegree[node], outdegree[node]) for node in set(indegree) | set(outdegree)))
            signatures.append(signature)
        if len(signatures) < 2:
            return 0.0
        repeated = 0
        comparisons = 0
        for left in range(len(signatures)):
            for right in range(left + 1, len(signatures)):
                comparisons += 1
                repeated += int(signatures[left] == signatures[right])
        return repeated / comparisons if comparisons else 0.0
