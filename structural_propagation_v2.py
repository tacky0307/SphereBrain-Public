from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from structural_observer import StructuralEpisode
from structural_working_state_v2 import StructuralWorkingStateV2, WorkingStateV2Config


@dataclass(frozen=True)
class CandidateEdgeState:
    edge_id: int
    weight: float
    usage: float
    recency: float
    target_degree: float
    direction: float

    def vector(self) -> np.ndarray:
        return np.asarray(
            [self.weight, self.usage, self.recency, self.target_degree, self.direction],
            dtype=float,
        )


@dataclass
class PropagationV2Config:
    structural_gain: float = 0.20
    local_gain: float = 1.0
    temperature: float = 1.0
    enabled: bool = True


class StructuralPropagationV2:
    """Weakly modulate candidate Edge transmission using real candidate state.

    Candidate edges are described only by local numerical properties: weight,
    usage, recency, target degree, and direction. There are no semantic labels,
    correct answers, random candidate vectors, trainable parameters, or
    long-term learning. Structural context is ephemeral and only changes the
    current propagation distribution.
    """

    CONTEXT_INDEX = {
        "source_ratio": 0,
        "sink_ratio": 1,
        "merge_ratio": 2,
        "split_ratio": 3,
        "component_ratio": 4,
        "depth_ratio": 5,
        "parallel_ratio": 6,
        "temporal_overlap": 7,
        "edge_reuse": 8,
        "cycle_hint": 9,
    }

    def __init__(self, config: PropagationV2Config | None = None) -> None:
        self.config = config or PropagationV2Config()

    def propagate(
        self,
        episode: StructuralEpisode,
        common_suffix_start: int,
        candidates: Sequence[CandidateEdgeState],
    ) -> dict:
        if len(candidates) < 2:
            raise ValueError("at least two candidate edges are required")

        worker = StructuralWorkingStateV2(
            WorkingStateV2Config(enabled=self.config.enabled)
        )
        working = worker.run_episode(
            episode,
            common_suffix_start=common_suffix_start,
        )
        context = np.asarray(working["terminal_structural_state"], dtype=float)

        local_logits = np.asarray(
            [self._local_logit(candidate) for candidate in candidates], dtype=float
        )
        if self.config.enabled:
            structural_terms = np.asarray(
                [self._structural_affinity(candidate, context) for candidate in candidates],
                dtype=float,
            )
            structural_terms -= structural_terms.mean()
            structural_modulation = self.config.structural_gain * structural_terms
        else:
            structural_modulation = np.zeros(len(candidates), dtype=float)

        final_logits = self.config.local_gain * local_logits + structural_modulation
        probabilities = self._softmax(final_logits / max(1e-9, self.config.temperature))

        return {
            "propagation_version": 2,
            "enabled": self.config.enabled,
            "language_free": True,
            "answer_labels": False,
            "random_candidate_vectors": False,
            "long_term_learning": False,
            "candidate_states": [
                {
                    "edge_id": c.edge_id,
                    "weight": c.weight,
                    "usage": c.usage,
                    "recency": c.recency,
                    "target_degree": c.target_degree,
                    "direction": c.direction,
                }
                for c in candidates
            ],
            "terminal_structural_state": context.tolist(),
            "local_logits": local_logits.tolist(),
            "structural_modulation": structural_modulation.tolist(),
            "final_logits": final_logits.tolist(),
            "branch_probabilities": probabilities.tolist(),
            "probability_spread": float(probabilities.max() - probabilities.min()),
            "working_state": working,
        }

    @staticmethod
    def _local_logit(candidate: CandidateEdgeState) -> float:
        # Local Edge transmission before structural context.
        return float(
            0.55 * candidate.weight
            + 0.20 * candidate.usage
            + 0.15 * candidate.recency
            + 0.10 * candidate.target_degree
        )

    def _structural_affinity(
        self, candidate: CandidateEdgeState, context: np.ndarray
    ) -> float:
        idx = self.CONTEXT_INDEX
        merge = context[idx["merge_ratio"]]
        split = context[idx["split_ratio"]]
        depth = context[idx["depth_ratio"]]
        parallel = context[idx["parallel_ratio"]]
        reuse = context[idx["edge_reuse"]]
        cycle = context[idx["cycle_hint"]]

        # No candidate has a predefined meaning. The interaction uses only its
        # measurable local state and the current structural context.
        return float(
            merge * candidate.target_degree
            + split * abs(candidate.direction)
            + depth * candidate.recency
            + parallel * candidate.weight
            + reuse * candidate.usage
            + cycle * candidate.direction * candidate.recency
        )

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - values.max()
        exp = np.exp(shifted)
        return exp / exp.sum()

    @staticmethod
    def distance(left: Sequence[float], right: Sequence[float]) -> float:
        return float(
            np.linalg.norm(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))
        )
