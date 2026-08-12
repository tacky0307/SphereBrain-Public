from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


DirectedEdge = tuple[int, int]


@dataclass(frozen=True)
class StructuralEpisode:
    """One non-linguistic activity episode.

    `steps` records simultaneously active nodes at each time step.
    `edges_by_step` records directed activity flow observed during that step.
    The observer never modifies the Core or chooses a route.
    """

    steps: tuple[tuple[int, ...], ...]
    edges_by_step: tuple[tuple[DirectedEdge, ...], ...]

    @classmethod
    def from_lists(
        cls,
        steps: Sequence[Iterable[int]],
        edges_by_step: Sequence[Iterable[DirectedEdge]],
    ) -> "StructuralEpisode":
        if len(steps) != len(edges_by_step):
            raise ValueError("steps and edges_by_step must have the same length")
        return cls(
            steps=tuple(tuple(sorted({int(node) for node in step})) for step in steps),
            edges_by_step=tuple(
                tuple((int(source), int(target)) for source, target in edges)
                for edges in edges_by_step
            ),
        )


class StructuralObserver:
    """Read-only observer for activity topology and timing.

    It does not use node names, language labels, correct answers, or trainable
    parameters.  It describes shape using ID-invariant graph statistics.
    """

    FEATURE_NAMES = (
        "node_count",
        "edge_count",
        "source_count",
        "sink_count",
        "merge_count",
        "split_count",
        "isolated_count",
        "component_count",
        "max_depth",
        "mean_depth",
        "parallel_width",
        "temporal_overlap",
        "edge_reuse",
        "degree_entropy",
    )

    def observe(self, episode: StructuralEpisode) -> dict:
        nodes = self._all_nodes(episode)
        edges = self._all_edges(episode)
        indegree = {node: 0 for node in nodes}
        outdegree = {node: 0 for node in nodes}
        undirected = {node: set() for node in nodes}

        for source, target in edges:
            nodes.add(source)
            nodes.add(target)
            indegree.setdefault(source, 0)
            indegree.setdefault(target, 0)
            outdegree.setdefault(source, 0)
            outdegree.setdefault(target, 0)
            undirected.setdefault(source, set()).add(target)
            undirected.setdefault(target, set()).add(source)
            outdegree[source] += 1
            indegree[target] += 1

        sources = [node for node in nodes if outdegree[node] > 0 and indegree[node] == 0]
        sinks = [node for node in nodes if indegree[node] > 0 and outdegree[node] == 0]
        merges = [node for node in nodes if indegree[node] >= 2]
        splits = [node for node in nodes if outdegree[node] >= 2]
        isolated = [node for node in nodes if indegree[node] == 0 and outdegree[node] == 0]
        depths = self._depths(nodes, edges, sources)
        components = self._component_count(nodes, undirected)

        vector = np.asarray(
            [
                len(nodes),
                len(edges),
                len(sources),
                len(sinks),
                len(merges),
                len(splits),
                len(isolated),
                components,
                max(depths.values(), default=0),
                float(np.mean(list(depths.values()))) if depths else 0.0,
                max((len(step) for step in episode.steps), default=0),
                self._temporal_overlap(episode.steps),
                self._edge_reuse(episode.edges_by_step),
                self._degree_entropy(indegree, outdegree),
            ],
            dtype=float,
        )

        return {
            "observer_version": 1,
            "read_only": True,
            "language_free": True,
            "feature_names": list(self.FEATURE_NAMES),
            "feature_vector": vector.tolist(),
            "shape": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "sources": len(sources),
                "sinks": len(sinks),
                "merges": len(merges),
                "splits": len(splits),
                "isolated": len(isolated),
                "components": components,
                "max_depth": max(depths.values(), default=0),
                "parallel_width": max((len(step) for step in episode.steps), default=0),
            },
            "temporal": {
                "step_count": len(episode.steps),
                "temporal_overlap": self._temporal_overlap(episode.steps),
                "edge_reuse": self._edge_reuse(episode.edges_by_step),
            },
            "canonical_signature": self.canonical_signature(episode),
        }

    def observe_sequence(self, episodes: Sequence[StructuralEpisode]) -> dict:
        observations = [self.observe(episode) for episode in episodes]
        signatures = [item["canonical_signature"] for item in observations]
        pair_scores = []
        for left in range(len(observations)):
            for right in range(left + 1, len(observations)):
                pair_scores.append(
                    self.cosine_similarity(
                        observations[left]["feature_vector"],
                        observations[right]["feature_vector"],
                    )
                )
        repeated_pairs = 0
        total_pairs = 0
        for left in range(len(signatures)):
            for right in range(left + 1, len(signatures)):
                total_pairs += 1
                repeated_pairs += int(signatures[left] == signatures[right])
        return {
            "episode_count": len(episodes),
            "episodes": observations,
            "exact_repeat_ratio": repeated_pairs / total_pairs if total_pairs else 0.0,
            "mean_structural_similarity": float(np.mean(pair_scores)) if pair_scores else 1.0,
        }

    def canonical_signature(self, episode: StructuralEpisode) -> str:
        """Create an ID-invariant temporal topology signature."""
        nodes = sorted(self._all_nodes(episode))
        first_step = {}
        for step_index, step_nodes in enumerate(episode.steps):
            for node in step_nodes:
                first_step.setdefault(node, step_index)

        edges = self._all_edges(episode)
        indegree = {node: 0 for node in nodes}
        outdegree = {node: 0 for node in nodes}
        for source, target in edges:
            indegree[source] = indegree.get(source, 0)
            indegree[target] = indegree.get(target, 0) + 1
            outdegree[source] = outdegree.get(source, 0) + 1
            outdegree[target] = outdegree.get(target, 0)

        node_roles = sorted(
            (first_step.get(node, -1), indegree.get(node, 0), outdegree.get(node, 0))
            for node in nodes
        )
        step_shapes = []
        for step_index, (step_nodes, step_edges) in enumerate(
            zip(episode.steps, episode.edges_by_step)
        ):
            role_counts = sorted(
                (indegree.get(node, 0), outdegree.get(node, 0)) for node in step_nodes
            )
            edge_roles = sorted(
                (
                    indegree.get(source, 0),
                    outdegree.get(source, 0),
                    indegree.get(target, 0),
                    outdegree.get(target, 0),
                )
                for source, target in step_edges
            )
            step_shapes.append((step_index, tuple(role_counts), tuple(edge_roles)))
        return repr((tuple(node_roles), tuple(step_shapes)))

    @staticmethod
    def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
        a = np.asarray(left, dtype=float)
        b = np.asarray(right, dtype=float)
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denominator) if denominator else 0.0

    @staticmethod
    def _all_nodes(episode: StructuralEpisode) -> set[int]:
        nodes = {node for step in episode.steps for node in step}
        for edges in episode.edges_by_step:
            for source, target in edges:
                nodes.add(source)
                nodes.add(target)
        return nodes

    @staticmethod
    def _all_edges(episode: StructuralEpisode) -> set[DirectedEdge]:
        return {edge for edges in episode.edges_by_step for edge in edges}

    @staticmethod
    def _depths(
        nodes: set[int], edges: set[DirectedEdge], sources: Sequence[int]
    ) -> dict[int, int]:
        incoming = {node: [] for node in nodes}
        for source, target in edges:
            incoming.setdefault(target, []).append(source)
        depths = {node: 0 for node in sources}
        for _ in range(max(1, len(nodes))):
            changed = False
            for node in nodes:
                parents = incoming.get(node, [])
                known = [depths[parent] for parent in parents if parent in depths]
                if known:
                    candidate = max(known) + 1
                    if candidate > depths.get(node, -1):
                        depths[node] = candidate
                        changed = True
            if not changed:
                break
        for node in nodes:
            depths.setdefault(node, 0)
        return depths

    @staticmethod
    def _component_count(nodes: set[int], adjacency: dict[int, set[int]]) -> int:
        unseen = set(nodes)
        count = 0
        while unseen:
            count += 1
            stack = [unseen.pop()]
            while stack:
                node = stack.pop()
                for other in adjacency.get(node, set()):
                    if other in unseen:
                        unseen.remove(other)
                        stack.append(other)
        return count

    @staticmethod
    def _temporal_overlap(steps: Sequence[Sequence[int]]) -> float:
        scores = []
        previous: set[int] | None = None
        for step in steps:
            current = set(step)
            if previous is not None:
                union = previous | current
                scores.append(len(previous & current) / len(union) if union else 0.0)
            previous = current
        return float(np.mean(scores)) if scores else 0.0

    @staticmethod
    def _edge_reuse(edges_by_step: Sequence[Sequence[DirectedEdge]]) -> float:
        seen: set[DirectedEdge] = set()
        repeated = 0
        total = 0
        for edges in edges_by_step:
            for edge in edges:
                total += 1
                repeated += int(edge in seen)
                seen.add(edge)
        return repeated / total if total else 0.0

    @staticmethod
    def _degree_entropy(
        indegree: dict[int, int], outdegree: dict[int, int]
    ) -> float:
        values = np.asarray(
            [indegree[node] + outdegree[node] for node in indegree], dtype=float
        )
        total = float(values.sum())
        if total <= 0:
            return 0.0
        probabilities = values[values > 0] / total
        return float(-np.sum(probabilities * np.log2(probabilities)))
