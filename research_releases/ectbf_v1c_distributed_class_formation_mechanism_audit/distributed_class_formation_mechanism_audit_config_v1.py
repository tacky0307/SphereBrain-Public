from __future__ import annotations

"""Configuration for ECTBF v1C — Distributed Class Formation Mechanism Audit."""

from pathlib import Path
import json

STUDY = (
    "ECTBF v1C — Distributed Class Formation Mechanism Audit / "
    "Does the Consequence Field Depend on Cooperative Covariance Rather Than "
    "Interchangeable Members?"
)
TARGET_BRANCH = "experiment/distributed-class-formation-mechanism-audit-v1"
ANCESTRY_BRANCH = "experiment/distributed-temporal-boundary-equivalence-v1"
ANCESTRY_COMPLETION_PATH = (
    "results/distributed_temporal_boundary_equivalence_v1_completion_receipt.json"
)
ANCESTRY_FORMAL_SUMMARY_PATH = (
    "results/distributed_temporal_boundary_equivalence_v1_formal_summary.json"
)
PROTOCOL_PATH = "research/protocols/distributed_class_formation_mechanism_audit_v1.md"
THRESHOLD_DECLARATION_PATH = (
    "research/frozen/DISTRIBUTED_CLASS_FORMATION_MECHANISM_AUDIT_V1_THRESHOLDS.json"
)

QUERY_COUNT = 5
DEV_FAMILY_SEEDS = (121119, 122129, 123137)
HOLDOUT_FAMILY_SEEDS = (131121, 132143, 133151, 134153, 135177, 136181)

FORMATION_EPISODES_PER_OBSERVED_QUERY = 72
DIRECT_TEST_EPISODES = 160
PLAN_EPISODES = 48
PLAN_DEPTH = 3
COOPERATIVE_SUBSET_SIZE = 24
BOOTSTRAP_REPLICATES = 5000

FULL = "full_covariance_field"
COOPERATIVE = "cooperative_covariance_subset"
MARGINAL = "marginal_score_subset"
CONDITIONAL_SHUFFLED = "conditional_marginal_preserving_covariance_destroyed"
DIAGONAL = "diagonal_gram_marginal_only"
COOPERATIVE_DESTROYED = "cooperative_members_covariance_destroyed"
QUERY_FACTORIZED = "query_factorized_field"
SMOOTH_NULL = "smoothness_matched_shuffled_credit"
FIELD_ABLATED = "field_ablated"
COOPERATIVE_DELAY = "cooperative_subset_delay"
COOPERATIVE_PARTIAL = "cooperative_subset_partial_cue"
COOPERATIVE_NOISE = "cooperative_subset_noise"
COOPERATIVE_COMBINED = "cooperative_subset_combined"
COOPERATIVE_NO_MAINTENANCE = "cooperative_subset_no_maintenance"

ROBUST_COOPERATIVE_METHODS = (
    COOPERATIVE,
    COOPERATIVE_DELAY,
    COOPERATIVE_PARTIAL,
    COOPERATIVE_NOISE,
    COOPERATIVE_COMBINED,
)
METHODS = (
    FULL,
    COOPERATIVE,
    MARGINAL,
    CONDITIONAL_SHUFFLED,
    DIAGONAL,
    COOPERATIVE_DESTROYED,
    QUERY_FACTORIZED,
    SMOOTH_NULL,
    FIELD_ABLATED,
    COOPERATIVE_DELAY,
    COOPERATIVE_PARTIAL,
    COOPERATIVE_NOISE,
    COOPERATIVE_COMBINED,
    COOPERATIVE_NO_MAINTENANCE,
)

FORBIDDEN_SNAPSHOT_TOKENS = (
    "true_pre",
    "true_post",
    "pre_center",
    "post_center",
    "boundary_mask",
    "correct_route",
    "actual_relation",
    "same",
    "flip",
    "teacher",
    "oracle",
    "held_answer",
    "future_outcome",
    "support_frequency",
    "truth_window",
    "training_stream",
    "reward_history",
)

EXPECTED_THRESHOLD_KEYS = (
    "minimum_full_relation_accuracy",
    "minimum_cooperative_relation_accuracy",
    "minimum_robust_cooperative_relation_accuracy",
    "minimum_cooperative_complete_sequence",
    "minimum_combined_complete_sequence",
    "minimum_mean_gain_full_over_conditional_shuffle",
    "minimum_bootstrap_low_full_over_conditional_shuffle",
    "minimum_mean_gain_full_over_diagonal",
    "minimum_bootstrap_low_full_over_diagonal",
    "minimum_mean_gain_cooperative_over_marginal",
    "minimum_bootstrap_low_cooperative_over_marginal",
    "minimum_mean_gain_cooperative_over_destroyed",
    "minimum_bootstrap_low_cooperative_over_destroyed",
    "minimum_positive_fraction_cooperative_over_marginal",
    "minimum_positive_fraction_cooperative_over_destroyed",
    "minimum_covariance_destruction_ratio",
    "maximum_conditional_marginal_preservation_error",
    "minimum_cooperative_formation_r2",
    "minimum_gain_full_over_smooth_null",
    "minimum_field_ablation_sequence_gap",
    "minimum_no_maintenance_sequence_gap",
)

STATUS_POSITIVE = (
    "DISTRIBUTED_CLASS_FORMATION_MECHANISM_AUDIT_V1__"
    "COOPERATIVE_COVARIANCE_SUPPORTED"
)
STATUS_MARGINAL_SUFFICIENT = (
    "DISTRIBUTED_CLASS_FORMATION_MECHANISM_AUDIT_V1__"
    "MARGINAL_STRUCTURE_SUFFICIENT"
)
STATUS_COOPERATIVE_NO_PLAN = (
    "DISTRIBUTED_CLASS_FORMATION_MECHANISM_AUDIT_V1__"
    "COOPERATIVE_COVARIANCE_WITHOUT_PERSISTENT_PLAN_VALUE"
)
STATUS_INCONCLUSIVE = (
    "DISTRIBUTED_CLASS_FORMATION_MECHANISM_AUDIT_V1__"
    "INCONCLUSIVE_COOPERATIVE_STRUCTURE"
)
STATUS_INVALID = "DISTRIBUTED_CLASS_FORMATION_MECHANISM_AUDIT_V1__INVALID_AUDIT"


def load_frozen_thresholds() -> dict[str, float] | None:
    path = Path(THRESHOLD_DECLARATION_PATH)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, dict):
        raise RuntimeError("threshold declaration lacks thresholds")
    if tuple(sorted(thresholds)) != tuple(sorted(EXPECTED_THRESHOLD_KEYS)):
        missing = sorted(set(EXPECTED_THRESHOLD_KEYS) - set(thresholds))
        extra = sorted(set(thresholds) - set(EXPECTED_THRESHOLD_KEYS))
        raise RuntimeError(f"threshold key mismatch missing={missing} extra={extra}")
    return {str(key): float(value) for key, value in thresholds.items()}


FROZEN_THRESHOLDS = load_frozen_thresholds()
