from __future__ import annotations

from brain import SphereBrain


class NativeLearningSphereBrain(SphereBrain):
    """Backward-compatible name for the v80 Native Learning candidate.

    Native Learning was promoted into the primary SphereBrain in v81.
    This subclass intentionally adds no learning behavior of its own.
    """

    @classmethod
    def from_sphere_brain(cls, source: SphereBrain) -> "NativeLearningSphereBrain":
        brain = cls(
            node_count=source.node_count,
            neighbors_per_node=source.neighbors_per_node,
            seed=source.seed,
            learning_rate=source.learning_rate,
            decay_rate=source.decay_rate,
            propagation_mode=source.propagation_mode,
            signal_decay=source.signal_decay,
            max_branches=source.max_branches,
            max_active_per_step=source.max_active_per_step,
            max_total_active_nodes=source.max_total_active_nodes,
            structural_assist_enabled=source.structural_assist_enabled,
            structural_gain=source.structural_gain,
            structural_tie_margin=source.structural_tie_margin,
            structural_near_zero_margin=source.structural_near_zero_margin,
            structural_relative_cap_ratio=source.structural_relative_cap_ratio,
            structural_absolute_cap=source.structural_absolute_cap,
        )
        brain.positions = source.positions.copy()
        brain.adjacency = source.adjacency.copy()
        brain.weights = source.weights.copy()
        brain.usage = source.usage.copy()
        brain.node_usage = source.node_usage.copy()
        brain.experience_state.replace_from_mapping(source.experience_state.snapshot())
        brain.learning_state.replace_from_mapping(source.learning_state.snapshot())
        return brain
