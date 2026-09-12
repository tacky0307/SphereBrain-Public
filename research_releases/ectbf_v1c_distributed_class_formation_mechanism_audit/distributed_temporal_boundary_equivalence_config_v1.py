from __future__ import annotations

"""Configuration for ECTBF v1B — Distributed Temporal Boundary Equivalence."""

from pathlib import Path
import json

STUDY = (
    "ECTBF v1B — Distributed Temporal Boundary Equivalence / "
    "Does the Core Represent a Consequence as a Distributed Equivalence Class "
    "of Moment Pairs Rather Than a Single Boundary?"
)
TARGET_BRANCH = "experiment/distributed-temporal-boundary-equivalence-v1"
ANCESTRY_BRANCH = "experiment/endogenous-temporal-consequence-boundary-formation-v1"
ANCESTRY_COMPLETION_PATH = (
    "results/endogenous_temporal_consequence_boundary_formation_v1_completion_receipt.json"
)
ANCESTRY_FORMAL_SUMMARY_PATH = (
    "results/endogenous_temporal_consequence_boundary_formation_v1_formal_summary.json"
)
PROTOCOL_PATH = (
    "research/protocols/distributed_temporal_boundary_equivalence_v1.md"
)
THRESHOLD_DECLARATION_PATH = (
    "research/frozen/"
    "DISTRIBUTED_TEMPORAL_BOUNDARY_EQUIVALENCE_V1_THRESHOLDS.json"
)

QUERY_COUNT = 5
DEV_FAMILY_SEEDS = (101119, 102129, 103137)
HOLDOUT_FAMILY_SEEDS = (111121, 112143, 113151, 114153, 115177, 116181)

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

FORMATION_EPISODES_PER_OBSERVED_QUERY = 72
DIRECT_TEST_EPISODES = 160
PLAN_EPISODES = 48
PLAN_DEPTH = 3

EQUIVALENCE_SCORE_MASS_FRACTION = 0.82
MIN_EQUIVALENCE_PAIR_COUNT = 24
MAX_EQUIVALENCE_PAIR_FRACTION = 0.45
SECTOR_COUNT = 3
PERMUTATION_REPLICATES = 8
DROPOUT_REPLICATES = 8
DROPOUT_FRACTION = 0.45
CLASS_SHIFT = 4

# These are operational definitions fixed before development. They determine
# whether a disjoint sector counts as an approximately equivalent realization.
SECTOR_EQUIVALENCE_MIN_ACCURACY = 0.72
SECTOR_EQUIVALENCE_MAX_GAP_TO_CLASS = 0.15
SECTOR_EQUIVALENCE_MIN_AGREEMENT = 0.78

FULL = "full_learned_field"
CLASS_ONLY = "endogenous_equivalence_class_only"
EQUALIZED_CLASS = "equalized_equivalence_class"
MEMBER_MAJORITY = "equivalence_member_majority_vote"
SECTOR_0 = "interleaved_equivalence_sector_0"
SECTOR_1 = "interleaved_equivalence_sector_1"
SECTOR_2 = "interleaved_equivalence_sector_2"
BAND_0 = "contiguous_equivalence_band_0"
BAND_1 = "contiguous_equivalence_band_1"
BAND_2 = "contiguous_equivalence_band_2"
BEST_SINGLE = "best_formation_single_pair"
DOMINANT_SINGLE = "dominant_weight_single_pair"
LOCAL_PATCH = "dominant_local_patch"
CLASS_ABLATED = "equivalence_class_ablated"
MATCHED_OUTSIDE = "lag_geometry_matched_outside_relocation"
SHIFTED_CLASS = "shifted_equivalence_class"
SHUFFLED_CREDIT = "shuffled_delayed_credit_field"
CLASS_DELAY = "equivalence_class_delay"
CLASS_PARTIAL = "equivalence_class_partial_cue"
CLASS_NOISE = "equivalence_class_noise"
CLASS_COMBINED = "equivalence_class_combined"
CLASS_NO_MAINTENANCE = "equivalence_class_no_maintenance"

INTERLEAVED_SECTORS = (SECTOR_0, SECTOR_1, SECTOR_2)
CONTIGUOUS_BANDS = (BAND_0, BAND_1, BAND_2)
ROBUST_CLASS_METHODS = (
    CLASS_ONLY,
    CLASS_DELAY,
    CLASS_PARTIAL,
    CLASS_NOISE,
    CLASS_COMBINED,
)
CONTROL_METHODS = (
    BEST_SINGLE,
    DOMINANT_SINGLE,
    LOCAL_PATCH,
    CLASS_ABLATED,
    MATCHED_OUTSIDE,
    SHIFTED_CLASS,
    SHUFFLED_CREDIT,
    CLASS_NO_MAINTENANCE,
)
METHODS = (
    FULL,
    CLASS_ONLY,
    EQUALIZED_CLASS,
    MEMBER_MAJORITY,
    *INTERLEAVED_SECTORS,
    *CONTIGUOUS_BANDS,
    *CONTROL_METHODS,
    CLASS_DELAY,
    CLASS_PARTIAL,
    CLASS_NOISE,
    CLASS_COMBINED,
)

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
    "support_frequency",
    "truth_window",
)

EXPECTED_THRESHOLD_KEYS = (
    "minimum_class_only_relation_accuracy",
    "minimum_equalized_class_relation_accuracy",
    "minimum_member_majority_relation_accuracy",
    "minimum_class_only_complete_sequence",
    "minimum_robust_class_relation_accuracy",
    "minimum_combined_complete_sequence",
    "minimum_effective_pair_number",
    "minimum_normalized_class_entropy",
    "minimum_class_support_enrichment",
    "minimum_class_mass_fraction",
    "minimum_fraction_cases_with_two_equivalent_sectors",
    "minimum_mean_equivalent_sector_count",
    "minimum_interleaved_sector_relation_accuracy",
    "minimum_interleaved_sector_complete_sequence",
    "minimum_permutation_relation_accuracy",
    "minimum_dropout_relation_accuracy",
    "minimum_gain_over_best_single",
    "minimum_gain_over_local_patch",
    "minimum_gain_over_matched_outside",
    "minimum_gain_over_shifted_class",
    "minimum_class_ablation_relation_gap",
    "minimum_class_ablation_sequence_gap",
    "minimum_outside_relocation_sequence_gap",
    "minimum_no_maintenance_sequence_gap",
)

STATUS_POSITIVE = (
    "DISTRIBUTED_TEMPORAL_BOUNDARY_EQUIVALENCE_V1__"
    "DISTRIBUTED_EQUIVALENCE_AND_PLAN_SUPPORTED"
)
STATUS_FUNCTIONAL_NO_EQUIVALENCE = (
    "DISTRIBUTED_TEMPORAL_BOUNDARY_EQUIVALENCE_V1__"
    "CONSEQUENCE_FIELD_WITHOUT_DISTRIBUTED_EQUIVALENCE"
)
STATUS_EQUIVALENCE_NO_PLAN = (
    "DISTRIBUTED_TEMPORAL_BOUNDARY_EQUIVALENCE_V1__"
    "DISTRIBUTED_EQUIVALENCE_WITHOUT_PLAN_CONTINUATION"
)
STATUS_LOCALIZED = (
    "DISTRIBUTED_TEMPORAL_BOUNDARY_EQUIVALENCE_V1__"
    "LOCALIZED_OR_SINGLE_BOUNDARY_SUFFICIENT"
)
STATUS_NEGATIVE = (
    "DISTRIBUTED_TEMPORAL_BOUNDARY_EQUIVALENCE_V1__"
    "NO_RELIABLE_CONSEQUENCE_EQUIVALENCE_STRUCTURE"
)
STATUS_INVALID = (
    "DISTRIBUTED_TEMPORAL_BOUNDARY_EQUIVALENCE_V1__INVALID_AUDIT"
)


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
