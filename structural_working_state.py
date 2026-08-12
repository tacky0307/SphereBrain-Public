from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from structural_observer import StructuralEpisode, StructuralObserver


@dataclass
class WorkingStateConfig:
    decay: float = 0.72
    structural_gain: float = 0.35
    enabled: bool = True


class StructuralWorkingState:
    """Ephemeral, language-free structural context.

    The state is rebuilt during one episode and discarded afterwards.
    It has no trainable parameters, no answer labels, and no long-term memory.
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
        "cycle_hint",
    )

    def __init__(self, config: WorkingStateConfig | None = None) -> None:
        self.config = config or WorkingStateConfig()
        self.observer = StructuralObserver()

    def run(self, episode: StructuralEpisode) -> dict:
        state = np.zeros(len(self.FEATURE_NAMES), dtype=float)
        timeline: list[dict] = []

        for step_index in range(len(episode.steps)):
            partial = StructuralEpisode(
                steps=episode.steps[: step_index + 1],
                edges_by_step=episode.edges_by_step[: step_index + 1],
            )
            observation = self.observer.observe(partial)
            instant = self._to_context(observation)
            if self.config.enabled:
                state = self.config.decay * state + (1.0 - self.config.decay) * instant
            else:
                state[:] = 0.0

            active = list(partial.steps[-1]) if partial.steps else []
            node_states = {
                str(node): self._node_state(node, partial, state) for node in active
            }
            timeline.append(
                {
                    "step": step_index,
                    "active_nodes": active,
                    "instant_structure": instant.tolist(),
                    "working_state": state.tolist(),
                    "node_states": node_states,
                }
            )

        terminal_nodes = list(episode.steps[-1]) if episode.steps else []
        terminal = {
            str(node): self._node_state(node, episode, state) for node in terminal_nodes
        }
        return {
            "working_state_version": 1,
            "enabled": self.config.enabled,
            "read_only_core": True,
            "language_free": True,
            "ephemeral": True,
            "feature_names": list(self.FEATURE_NAMES),
            "final_working_state": state.tolist(),
            "terminal_node_states": terminal,
            "timeline": timeline,
        }

    def _to_context(self, observation: dict) -> np.ndarray:
        shape = observation["shape"]
        temporal = observation["temporal"]
        n = max(1.0, float(shape["node_count"]))
        e = max(1.0, float(shape["edge_count"]))
        cycle_hint = 1.0 if shape["sources"] == 0 and shape["sinks"] == 0 and e > 0 else 0.0
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
                temporal["edge_reuse"],
                cycle_hint,
            ],
            dtype=float,
        )

    def _node_state(
        self,
        node: int,
        episode: StructuralEpisode,
        structural_state: np.ndarray,
    ) -> list[float]:
        indegree = 0
        outdegree = 0
        first_step = len(episode.steps)
        last_step = -1
        for step_index, (nodes, edges) in enumerate(zip(episode.steps, episode.edges_by_step)):
            if node in nodes:
                first_step = min(first_step, step_index)
                last_step = max(last_step, step_index)
            for source, target in edges:
                if target == node:
                    indegree += 1
                if source == node:
                    outdegree += 1

        local = np.asarray(
            [
                min(1.0, indegree / 3.0),
                min(1.0, outdegree / 3.0),
                first_step / max(1.0, len(episode.steps) - 1.0),
                last_step / max(1.0, len(episode.steps) - 1.0),
            ],
            dtype=float,
        )
        context = structural_state * self.config.structural_gain if self.config.enabled else np.zeros_like(structural_state)
        return np.concatenate([local, context]).tolist()

    @staticmethod
    def cosine(left: Sequence[float], right: Sequence[float]) -> float:
        a = np.asarray(left, dtype=float)
        b = np.asarray(right, dtype=float)
        d = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / d) if d else 0.0

    @staticmethod
    def distance(left: Sequence[float], right: Sequence[float]) -> float:
        return float(np.linalg.norm(np.asarray(left, dtype=float) - np.asarray(right, dtype=float)))
