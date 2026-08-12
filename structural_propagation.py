from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from structural_observer import StructuralEpisode
from structural_working_state_v2 import StructuralWorkingStateV2, WorkingStateV2Config


@dataclass(frozen=True)
class PropagationConfig:
    structural_gain: float = 0.18
    temperature: float = 1.0
    enabled: bool = True
    candidate_seed: int = 20260806


class StructuralPropagation:
    """Apply ephemeral structural context weakly to an otherwise neutral branch.

    There are no answer labels and no preferred branch. Candidate context vectors
    are deterministic, zero-mean numerical directions independent of node IDs and
    episode names. Structural state changes branch activation only through a small
    dot-product modulation. No long-term learning is performed.
    """

    def __init__(self, config: PropagationConfig | None = None) -> None:
        self.config = config or PropagationConfig()

    def propagate(
        self,
        episode: StructuralEpisode,
        terminal_node: int,
        common_suffix_start: int,
        candidate_count: int = 2,
        candidate_order: Sequence[int] | None = None,
    ) -> dict:
        worker = StructuralWorkingStateV2(
            WorkingStateV2Config(enabled=self.config.enabled)
        )
        working = worker._run(episode, terminal_node, common_suffix_start)
        context = np.asarray(working["terminal_structural_state"], dtype=float)
        directions = self._candidate_directions(candidate_count, context.size)

        order = list(candidate_order) if candidate_order is not None else list(range(candidate_count))
        if sorted(order) != list(range(candidate_count)):
            raise ValueError("candidate_order must be a permutation of candidate indexes")
        ordered = directions[order]

        baseline = np.zeros(candidate_count, dtype=float)
        if self.config.enabled and context.size:
            modulation = self.config.structural_gain * (ordered @ context)
            modulation -= modulation.mean()
        else:
            modulation = np.zeros(candidate_count, dtype=float)

        final_logits = baseline + modulation
        logits = final_logits / max(1e-9, self.config.temperature)
        logits -= logits.max()
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()

        return {
            "propagation_version": 1,
            "enabled": self.config.enabled,
            "language_free": True,
            "answer_labels": False,
            "long_term_learning": False,
            "structural_gain": self.config.structural_gain,
            "candidate_order": order,
            "terminal_structural_state": context.tolist(),
            "baseline_logits": baseline.tolist(),
            "structural_modulation": modulation.tolist(),
            "final_logits": final_logits.tolist(),
            "branch_probabilities": probabilities.tolist(),
            "probability_spread": float(probabilities.max() - probabilities.min()),
            "working_state": working,
        }

    def _candidate_directions(self, count: int, dimension: int) -> np.ndarray:
        rng = np.random.default_rng(self.config.candidate_seed)
        matrix = rng.normal(0.0, 1.0, size=(count, dimension))
        matrix -= matrix.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix /= np.where(norms < 1e-12, 1.0, norms)
        return matrix

    @staticmethod
    def distance(left: Sequence[float], right: Sequence[float]) -> float:
        return float(np.linalg.norm(np.asarray(left, dtype=float) - np.asarray(right, dtype=float)))
