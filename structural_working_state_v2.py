from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from structural_observer import StructuralEpisode, StructuralObserver


@dataclass(frozen=True)
class ControlledCase:
    name: str
    left: StructuralEpisode
    right: StructuralEpisode
    terminal_node_left: int
    terminal_node_right: int
    common_suffix_start: int


@dataclass
class WorkingStateV2Config:
    decay: float = 0.78
    structural_gain: float = 0.5
    enabled: bool = True


class StructuralWorkingStateV2:
    """Controlled, ephemeral structural context experiment.

    The terminal local state is calculated only from a declared common suffix.
    Therefore differences before that suffix can reach the terminal only through
    the ephemeral structural working state. No route choice, labels, language,
    learning, or Core mutation is used.
    """

    CONTEXT_NAMES = (
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

    LOCAL_NAMES = (
        "suffix_in_degree",
        "suffix_out_degree",
        "suffix_first_step",
        "suffix_last_step",
    )

    def __init__(self, config: WorkingStateV2Config | None = None) -> None:
        self.config = config or WorkingStateV2Config()
        self.observer = StructuralObserver()

    def run_case(self, case: ControlledCase) -> dict:
        left = self.run_episode(
            case.left,
            terminal_node=case.terminal_node_left,
            common_suffix_start=case.common_suffix_start,
        )
        right = self.run_episode(
            case.right,
            terminal_node=case.terminal_node_right,
            common_suffix_start=case.common_suffix_start,
        )
        left_state = left["terminal_state"]
        right_state = right["terminal_state"]
        return {
            "case": case.name,
            "common_suffix_start": case.common_suffix_start,
            "left": left,
            "right": right,
            "comparison": {
                "distance": self.distance(left_state, right_state),
                "similarity": self.cosine(left_state, right_state),
                "local_distance": self.distance(left["terminal_local_state"], right["terminal_local_state"]),
                "structural_distance": self.distance(left["terminal_structural_state"], right["terminal_structural_state"]),
            },
        }

    def run_episode(
        self,
        episode: StructuralEpisode,
        terminal_node: int | None = None,
        common_suffix_start: int = 0,
    ) -> dict:
        """Run one episode and return its ephemeral structural state.

        `terminal_node` defaults to the first node active in the final step.
        This public wrapper is used by propagation experiments; `_run` remains
        the implementation detail.
        """
        if not episode.steps or not episode.steps[-1]:
            raise ValueError("episode must contain at least one terminal node")
        resolved_terminal = (
            int(terminal_node)
            if terminal_node is not None
            else int(episode.steps[-1][0])
        )
        if common_suffix_start < 0 or common_suffix_start >= len(episode.steps):
            raise ValueError("common_suffix_start is outside the episode")
        return self._run(episode, resolved_terminal, common_suffix_start)

    def _run(self, episode: StructuralEpisode, terminal_node: int, suffix_start: int) -> dict:
        state = np.zeros(len(self.CONTEXT_NAMES), dtype=float)
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
            timeline.append({
                "step": step_index,
                "active_nodes": list(episode.steps[step_index]),
                "instant_structure": instant.tolist(),
                "working_state": state.tolist(),
            })

        local = self._suffix_local_state(episode, terminal_node, suffix_start)
        structural = state * self.config.structural_gain if self.config.enabled else np.zeros_like(state)
        terminal = np.concatenate([local, structural])
        return {
            "enabled": self.config.enabled,
            "ephemeral": True,
            "language_free": True,
            "long_term_learning": False,
            "terminal_node": terminal_node,
            "terminal_local_state": local.tolist(),
            "terminal_structural_state": structural.tolist(),
            "terminal_state": terminal.tolist(),
            "final_working_state": state.tolist(),
            "timeline": timeline,
        }

    def _suffix_local_state(self, episode: StructuralEpisode, node: int, suffix_start: int) -> np.ndarray:
        suffix_steps = episode.steps[suffix_start:]
        suffix_edges = episode.edges_by_step[suffix_start:]
        indegree = 0
        outdegree = 0
        first = len(suffix_steps)
        last = -1
        for index, (nodes, edges) in enumerate(zip(suffix_steps, suffix_edges)):
            if node in nodes:
                first = min(first, index)
                last = max(last, index)
            for source, target in edges:
                indegree += int(target == node)
                outdegree += int(source == node)
        denominator = max(1.0, len(suffix_steps) - 1.0)
        return np.asarray([
            min(1.0, indegree / 3.0),
            min(1.0, outdegree / 3.0),
            first / denominator,
            last / denominator,
        ], dtype=float)

    @staticmethod
    def _to_context(observation: dict) -> np.ndarray:
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

    @staticmethod
    def cosine(left: Sequence[float], right: Sequence[float]) -> float:
        a = np.asarray(left, dtype=float)
        b = np.asarray(right, dtype=float)
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denominator) if denominator else 1.0

    @staticmethod
    def distance(left: Sequence[float], right: Sequence[float]) -> float:
        return float(np.linalg.norm(np.asarray(left, dtype=float) - np.asarray(right, dtype=float)))
