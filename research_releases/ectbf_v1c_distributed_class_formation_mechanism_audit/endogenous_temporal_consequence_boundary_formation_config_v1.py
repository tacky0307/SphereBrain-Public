from __future__ import annotations

"""Configuration for Endogenous Temporal Consequence Boundary Formation v1."""

STUDY = (
    "Endogenous Temporal Consequence Boundary Formation v1 / "
    "Can the Core Learn Which Moments in a Continuous Experience Stream "
    "Should Be Compared?"
)
TARGET_BRANCH = "experiment/endogenous-temporal-consequence-boundary-formation-v1"
ANCESTRY_BRANCH = "experiment/endogenous-relation-primitive-formation-v1"
ANCESTRY_COMPLETION_PATH = (
    "results/endogenous_relation_primitive_formation_v1_completion_receipt.json"
)
ANCESTRY_FORMAL_SUMMARY_PATH = (
    "results/endogenous_relation_primitive_formation_v1_formal_summary.json"
)
PROTOCOL_PATH = (
    "research/protocols/"
    "endogenous_temporal_consequence_boundary_formation_v1.md"
)

QUERY_COUNT = 5
DEV_FAMILY_SEEDS = (81119, 82129, 83137)
HOLDOUT_FAMILY_SEEDS = (91121, 92143, 93151, 94153, 95177, 96181)

SUITES = (
    {
        "name": "clean_continuous_stream",
        "stream_noise": 0.10,
        "sensor_dropout": 0.00,
        "distractor_count": 4,
        "distractor_gain": 1.25,
        "timing_jitter": 2,
    },
    {
        "name": "noisy_partial_continuous_stream",
        "stream_noise": 0.18,
        "sensor_dropout": 0.10,
        "distractor_count": 7,
        "distractor_gain": 1.55,
        "timing_jitter": 3,
    },
)

STREAM_LENGTH = 22
RAW_DIM = 18
CONSEQUENCE_SUBSPACE_DIM = 3
TRUE_WINDOW_RADIUS = 1
MIN_PAIR_LAG = 4
MAX_PAIR_LAG = 18
FORMATION_EPISODES_PER_OBSERVED_QUERY = 72
FORMATION_CREDIT_DELAY = 3
DIRECT_TEST_EPISODES = 160
PLAN_EPISODES = 48
PLAN_DEPTH = 3

TEMPORAL_RIDGE_LAMBDA = 4.0
TEMPORAL_SMOOTHNESS_LAMBDA = 3.5
WEIGHT_PRUNE_FRACTION = 0.55

CONSEQUENCE_GAIN = 2.55
IDENTITY_GAIN = 1.05
CONTEXT_GAIN = 0.72
ACTION_FOOTPRINT_GAIN = 1.95
DRIFT_GAIN = 0.62

RELATION_STATE_DIM = 48
RELATION_CLEANUP_STEPS = 4
RELATION_DELAY_STEPS = 20
RELATION_DELAY_RETENTION = 0.91
RELATION_DELAY_DRIFT = 0.11
NO_MAINTENANCE_DRIFT = 0.44
RUNTIME_PARTIAL_FRACTION = 0.14
RUNTIME_NOISE_STD = 0.15
RUNTIME_COMBINED_NOISE_STD = 0.12
TEMPORAL_SHIFT_CONTROL = 5

EXPLICIT_UPPER = "explicit_temporal_boundary_upper"
PRIMARY = "learned_temporal_boundary_clean"
DELAY = "learned_temporal_boundary_delay"
PARTIAL = "learned_temporal_boundary_partial_cue"
NOISE = "learned_temporal_boundary_noise"
COMBINED = "learned_temporal_boundary_combined"
SHIFTED = "learned_temporal_boundary_shifted"
REVERSED = "learned_temporal_boundary_reversed"
BOUNDARY_ABLATED = "learned_temporal_boundary_ablated"
NO_MAINTENANCE = "learned_temporal_boundary_no_maintenance"
FIRST_LAST = "fixed_first_last_control"
UNIFORM = "uniform_temporal_pair_control"
SHUFFLED = "shuffled_delayed_credit_control"
RANDOM = "random_norm_matched_boundary_control"

ROBUST_METHODS = (PRIMARY, DELAY, PARTIAL, NOISE, COMBINED)
CONTROL_METHODS = (
    SHIFTED,
    REVERSED,
    BOUNDARY_ABLATED,
    NO_MAINTENANCE,
    FIRST_LAST,
    UNIFORM,
    SHUFFLED,
    RANDOM,
)
METHODS = (EXPLICIT_UPPER, *ROBUST_METHODS, *CONTROL_METHODS)

FORBIDDEN_SNAPSHOT_TOKENS = (
    "true_pre",
    "true_post",
    "pre_center",
    "post_center",
    "boundary_mask",
    "causal_window",
    "correct_route",
    "actual_relation",
    "same",
    "flip",
    "teacher",
    "oracle",
    "held_answer",
    "future_outcome",
    "carrier_indices",
)

# Patched exactly once by the freeze utility after the declared development bank.
FROZEN_THRESHOLDS = {
    "boundary_ablation_causal_gap": 0.75,
    "clean_sequence_identity_rate": 0.93,
    "combined_sequence_identity_rate": 0.86,
    "delay_sequence_identity_rate": 0.93,
    "first_last_sequence_causal_gap": 0.75,
    "maintenance_causal_gap": 0.42,
    "maximum_positive_pair_fraction": 0.55,
    "maximum_temporal_center_mae": 2.75,
    "minimum_gain_over_first_last": 0.32,
    "minimum_gain_over_random_boundary": 0.15,
    "minimum_gain_over_shifted": 0.3,
    "minimum_gain_over_shuffled_credit": 0.3,
    "minimum_gain_over_uniform": 0.3,
    "minimum_robust_relation_accuracy": 0.86,
    "minimum_support_enrichment": 1.6,
    "minimum_top_pair_support_enrichment": 1.2,
    "minimum_unseen_relation_accuracy": 0.9,
    "noise_sequence_identity_rate": 0.9,
    "partial_sequence_identity_rate": 0.88,
    "shifted_sequence_causal_gap": 0.72
}

STATUS_POSITIVE = (
    "ENDOGENOUS_TEMPORAL_CONSEQUENCE_BOUNDARY_FORMATION_V1__"
    "TEMPORAL_BOUNDARY_AND_PLAN_SUPPORTED"
)
STATUS_PLAN_ONLY = (
    "ENDOGENOUS_TEMPORAL_CONSEQUENCE_BOUNDARY_FORMATION_V1__"
    "PLAN_CONTINUATION_WITHOUT_TEMPORAL_BOUNDARY_RECOVERY"
)
STATUS_BOUNDARY_ONLY = (
    "ENDOGENOUS_TEMPORAL_CONSEQUENCE_BOUNDARY_FORMATION_V1__"
    "TEMPORAL_BOUNDARY_RECOVERY_WITHOUT_PLAN_EQUIVALENCE"
)
STATUS_FRAGILE = (
    "ENDOGENOUS_TEMPORAL_CONSEQUENCE_BOUNDARY_FORMATION_V1__CLEAN_ONLY"
)
STATUS_NEGATIVE = (
    "ENDOGENOUS_TEMPORAL_CONSEQUENCE_BOUNDARY_FORMATION_V1__"
    "NO_RELIABLE_TEMPORAL_CONSEQUENCE_BOUNDARY"
)
STATUS_INVALID = (
    "ENDOGENOUS_TEMPORAL_CONSEQUENCE_BOUNDARY_FORMATION_V1__INVALID_AUDIT"
)
