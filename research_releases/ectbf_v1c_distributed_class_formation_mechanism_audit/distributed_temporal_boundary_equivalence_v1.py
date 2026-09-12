from __future__ import annotations

"""ECTBF v1B — Distributed Temporal Boundary Equivalence.

The study reuses the frozen ECTBF v1 continuous-stream world and zero-start
temporal-pair learner, but changes the scientific object under inspection.
Instead of asking whether one dominant pair recovers one hidden boundary, it
asks whether the learned field contains a consequence-bearing class of many
moment pairs whose disjoint realizations are approximately interchangeable on
unseen streams.

Class construction uses only formation streams, committed actions, and delayed
scalar continuation feedback. Hidden temporal windows are consulted only after
all predictions for a stream have been committed, for scientific audit.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
import csv
import hashlib
import json
import math

import numpy as np

import distributed_temporal_boundary_equivalence_config_v1 as cfg
import endogenous_temporal_consequence_boundary_formation_v1 as base

EPS = 1e-12
PAIR_COUNT = base.PAIR_COUNT
PAIR_INDICES = base.PAIR_INDICES
PAIR_I = base.PAIR_I
PAIR_J = base.PAIR_J
PAIR_COORDS = np.asarray(
    [
        (
            left / float(base.STREAM_LENGTH - 1),
            right / float(base.STREAM_LENGTH - 1),
            (right - left) / float(base.STREAM_LENGTH - 1),
        )
        for left, right in PAIR_INDICES
    ],
    dtype=float,
)


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    )


def payload_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def normalized_field(weights: np.ndarray, reference_norm: float | None = None) -> np.ndarray:
    values = np.asarray(weights, dtype=float).copy()
    norm = float(np.linalg.norm(values))
    if norm <= EPS:
        return values
    target = norm if reference_norm is None else float(reference_norm)
    return values * (target / norm)


def effective_number(weights: np.ndarray) -> float:
    positive = np.maximum(np.asarray(weights, dtype=float), 0.0)
    total = float(np.sum(positive))
    square = float(np.sum(positive * positive))
    if total <= EPS or square <= EPS:
        return 0.0
    return total * total / square


def normalized_entropy(weights: np.ndarray) -> float:
    positive = np.maximum(np.asarray(weights, dtype=float), 0.0)
    positive = positive[positive > 0.0]
    if positive.size <= 1:
        return 0.0
    probability = positive / float(np.sum(positive))
    entropy = -float(np.sum(probability * np.log(probability + EPS)))
    return entropy / math.log(float(positive.size))


def field_component_count(indices: Sequence[int]) -> int:
    remaining = set(int(index) for index in indices)
    components = 0
    while remaining:
        components += 1
        seed = remaining.pop()
        stack = [seed]
        while stack:
            current = stack.pop()
            left, right = PAIR_INDICES[current]
            neighbors = (
                (left - 1, right),
                (left + 1, right),
                (left, right - 1),
                (left, right + 1),
            )
            for pair in neighbors:
                neighbor = base.PAIR_INDEX.get(pair)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return components


def _weighted_axis(indices: np.ndarray, scores: np.ndarray) -> np.ndarray:
    coordinates = PAIR_COORDS[indices]
    local_scores = np.maximum(scores[indices], 0.0)
    if float(np.sum(local_scores)) <= EPS:
        local_scores = np.ones(indices.size, dtype=float)
    probabilities = local_scores / float(np.sum(local_scores))
    center = probabilities @ coordinates
    centered = coordinates - center
    covariance = (centered * probabilities[:, None]).T @ centered
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    # Resolve eigenvector sign deterministically.
    first_nonzero = next((value for value in axis if abs(float(value)) > EPS), 1.0)
    if first_nonzero < 0.0:
        axis = -axis
    return axis


def _interleaved_partition(
    indices: np.ndarray,
    scores: np.ndarray,
    sector_count: int,
) -> np.ndarray:
    axis = _weighted_axis(indices, scores)
    projection = PAIR_COORDS[indices] @ axis
    ordered = indices[np.argsort(projection, kind="stable")]
    labels = np.full(PAIR_COUNT, -1, dtype=int)
    masses = np.zeros(sector_count, dtype=float)
    for start in range(0, len(ordered), sector_count):
        block = list(int(v) for v in ordered[start : start + sector_count])
        block.sort(key=lambda index: (-float(scores[index]), index))
        available = list(range(sector_count))
        available.sort(key=lambda sector: (masses[sector], sector))
        for index, sector in zip(block, available):
            labels[index] = sector
            masses[sector] += max(float(scores[index]), EPS)
    return labels


def _contiguous_partition(
    indices: np.ndarray,
    scores: np.ndarray,
    sector_count: int,
) -> np.ndarray:
    axis = _weighted_axis(indices, scores)
    projection = PAIR_COORDS[indices] @ axis
    ordered = indices[np.argsort(projection, kind="stable")]
    labels = np.full(PAIR_COUNT, -1, dtype=int)
    for sector, chunk in enumerate(np.array_split(ordered, sector_count)):
        for index in chunk:
            labels[int(index)] = int(sector)
    return labels


def build_equivalence_model(
    features: np.ndarray,
    targets: np.ndarray,
    query_ids: np.ndarray,
    learned_weights: np.ndarray,
    *,
    observed_queries: Sequence[int],
) -> dict[str, Any]:
    """Construct a candidate class without hidden-window information."""

    matrix = np.asarray(features, dtype=float)
    response = np.asarray(targets, dtype=float)
    weights = np.maximum(np.asarray(learned_weights, dtype=float), 0.0)
    if matrix.ndim != 2 or matrix.shape[1] != PAIR_COUNT:
        raise ValueError("formation feature matrix has unexpected shape")
    if response.shape != (matrix.shape[0],):
        raise ValueError("formation target vector has unexpected shape")

    query_alignments: list[np.ndarray] = []
    for query in observed_queries:
        mask = query_ids == int(query)
        local = matrix[mask]
        local_targets = response[mask]
        scale = np.std(local, axis=0) + 1e-6
        query_alignments.append(
            np.mean(local_targets[:, None] * local, axis=0) / scale
        )
    alignment_matrix = np.stack(query_alignments, axis=0)
    positive_consistency = np.mean(alignment_matrix > 0.0, axis=0)
    positive_strength = np.mean(np.maximum(alignment_matrix, 0.0), axis=0)

    full_evidence = matrix @ weights
    pair_agreement = np.mean(
        (matrix >= 0.0) == (full_evidence[:, None] >= 0.0),
        axis=0,
    )
    score = (
        weights
        * positive_strength
        * (0.25 + 0.75 * positive_consistency)
        * (0.50 + 0.50 * pair_agreement)
    )
    active = np.flatnonzero(weights > 0.0)
    if active.size == 0:
        raise AssertionError("learned temporal field is empty")
    if float(np.sum(score[active])) <= EPS:
        score = weights * (0.25 + 0.75 * pair_agreement)

    ranked = active[np.argsort(score[active], kind="stable")[::-1]]
    minimum_count = min(cfg.MIN_EQUIVALENCE_PAIR_COUNT, int(active.size))
    maximum_count = min(
        int(active.size),
        max(
            minimum_count,
            int(math.ceil(cfg.MAX_EQUIVALENCE_PAIR_FRACTION * PAIR_COUNT)),
        ),
    )
    target_mass = cfg.EQUIVALENCE_SCORE_MASS_FRACTION * float(np.sum(score[active]))
    selected: list[int] = []
    accumulated = 0.0
    for index in ranked:
        if len(selected) >= maximum_count:
            break
        selected.append(int(index))
        accumulated += float(score[index])
        if len(selected) >= minimum_count and accumulated >= target_mass:
            break
    if not selected:
        selected = [int(ranked[0])]
    member_indices = np.asarray(sorted(selected), dtype=int)
    member_mask = np.zeros(PAIR_COUNT, dtype=bool)
    member_mask[member_indices] = True

    interleaved_labels = _interleaved_partition(
        member_indices, score, cfg.SECTOR_COUNT
    )
    contiguous_labels = _contiguous_partition(
        member_indices, score, cfg.SECTOR_COUNT
    )

    class_weights = np.where(member_mask, weights, 0.0)
    total_weight = float(np.sum(weights))
    class_mass_fraction = float(np.sum(class_weights)) / max(total_weight, EPS)
    best_single_index = int(member_indices[np.argmax(score[member_indices])])
    dominant_index = int(np.argmax(weights))

    return {
        "member_indices": member_indices,
        "member_mask": member_mask,
        "interleaved_labels": interleaved_labels,
        "contiguous_labels": contiguous_labels,
        "score": score,
        "positive_consistency": positive_consistency,
        "positive_strength": positive_strength,
        "pair_agreement": pair_agreement,
        "class_weights": class_weights,
        "best_single_index": best_single_index,
        "dominant_index": dominant_index,
        "class_mass_fraction": class_mass_fraction,
        "effective_pair_number": effective_number(class_weights),
        "normalized_class_entropy": normalized_entropy(class_weights),
        "component_count": field_component_count(member_indices),
        "left_span": float(
            max(PAIR_INDICES[index][0] for index in member_indices)
            - min(PAIR_INDICES[index][0] for index in member_indices)
        ),
        "right_span": float(
            max(PAIR_INDICES[index][1] for index in member_indices)
            - min(PAIR_INDICES[index][1] for index in member_indices)
        ),
        "lag_span": float(
            max(PAIR_INDICES[index][1] - PAIR_INDICES[index][0] for index in member_indices)
            - min(PAIR_INDICES[index][1] - PAIR_INDICES[index][0] for index in member_indices)
        ),
        "class_construction_hidden_window_reads": 0,
        "class_construction_delayed_scalar_only": True,
    }


@dataclass(frozen=True)
class EquivalenceSnapshot:
    version: str
    stream_length: int
    raw_dim: int
    pair_count: int
    pair_vocabulary_hash: str
    temporal_pair_weights: tuple[float, ...]
    equivalence_member_indices: tuple[int, ...]
    interleaved_sector_labels: tuple[int, ...]
    contiguous_band_labels: tuple[int, ...]
    best_single_index: int
    dominant_index: int
    relation_codes: tuple[tuple[float, ...], ...]
    maintenance_basin: tuple[tuple[float, ...], ...]
    formation_hash: str
    class_score_hash: str
    substrate_hash: str

    def payload_without_hash(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "stream_length": self.stream_length,
            "raw_dim": self.raw_dim,
            "pair_count": self.pair_count,
            "pair_vocabulary_hash": self.pair_vocabulary_hash,
            "temporal_pair_weights": list(self.temporal_pair_weights),
            "equivalence_member_indices": list(self.equivalence_member_indices),
            "interleaved_sector_labels": list(self.interleaved_sector_labels),
            "contiguous_band_labels": list(self.contiguous_band_labels),
            "best_single_index": self.best_single_index,
            "dominant_index": self.dominant_index,
            "relation_codes": [list(row) for row in self.relation_codes],
            "maintenance_basin": [list(row) for row in self.maintenance_basin],
            "formation_hash": self.formation_hash,
            "class_score_hash": self.class_score_hash,
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self.payload_without_hash(), "substrate_hash": self.substrate_hash}

    def to_json(self) -> str:
        return canonical_json(self.to_payload())

    @classmethod
    def from_json(cls, text: str) -> "EquivalenceSnapshot":
        payload = json.loads(text)
        snapshot = cls(
            version=str(payload["version"]),
            stream_length=int(payload["stream_length"]),
            raw_dim=int(payload["raw_dim"]),
            pair_count=int(payload["pair_count"]),
            pair_vocabulary_hash=str(payload["pair_vocabulary_hash"]),
            temporal_pair_weights=tuple(float(v) for v in payload["temporal_pair_weights"]),
            equivalence_member_indices=tuple(
                int(v) for v in payload["equivalence_member_indices"]
            ),
            interleaved_sector_labels=tuple(
                int(v) for v in payload["interleaved_sector_labels"]
            ),
            contiguous_band_labels=tuple(
                int(v) for v in payload["contiguous_band_labels"]
            ),
            best_single_index=int(payload["best_single_index"]),
            dominant_index=int(payload["dominant_index"]),
            relation_codes=tuple(
                tuple(float(v) for v in row) for row in payload["relation_codes"]
            ),
            maintenance_basin=tuple(
                tuple(float(v) for v in row)
                for row in payload["maintenance_basin"]
            ),
            formation_hash=str(payload["formation_hash"]),
            class_score_hash=str(payload["class_score_hash"]),
            substrate_hash=str(payload["substrate_hash"]),
        )
        if snapshot.stream_length != base.STREAM_LENGTH:
            raise AssertionError("snapshot stream length mismatch")
        if snapshot.raw_dim != base.RAW_DIM:
            raise AssertionError("snapshot raw dimension mismatch")
        if snapshot.pair_count != PAIR_COUNT:
            raise AssertionError("snapshot pair count mismatch")
        if snapshot.pair_vocabulary_hash != payload_hash(PAIR_INDICES):
            raise AssertionError("snapshot pair vocabulary mismatch")
        member_set = set(snapshot.equivalence_member_indices)
        if not member_set:
            raise AssertionError("snapshot class is empty")
        if min(member_set) < 0 or max(member_set) >= PAIR_COUNT:
            raise AssertionError("snapshot class index out of range")
        if snapshot.best_single_index not in member_set:
            raise AssertionError("best single is not a class member")
        if not 0 <= snapshot.dominant_index < PAIR_COUNT:
            raise AssertionError("dominant index out of range")
        for labels in (
            snapshot.interleaved_sector_labels,
            snapshot.contiguous_band_labels,
        ):
            if len(labels) != PAIR_COUNT:
                raise AssertionError("snapshot label length mismatch")
            for index in range(PAIR_COUNT):
                if index in member_set:
                    if labels[index] not in range(cfg.SECTOR_COUNT):
                        raise AssertionError("member lacks sector label")
                elif labels[index] != -1:
                    raise AssertionError("nonmember received sector label")
        if payload_hash(snapshot.payload_without_hash()) != snapshot.substrate_hash:
            raise AssertionError("snapshot hash mismatch")
        return snapshot


def snapshot_forbidden_token_count(snapshot: EquivalenceSnapshot) -> int:
    text = snapshot.to_json().lower()
    return sum(text.count(token.lower()) for token in cfg.FORBIDDEN_SNAPSHOT_TOKENS)


def form_equivalence_class(
    family_seed: int,
    held_query: int,
    suite: dict[str, Any],
    *,
    formation_episodes_per_query: int | None = None,
) -> tuple[
    EquivalenceSnapshot,
    base.ContinuousConsequenceWorld,
    dict[str, Any],
    np.ndarray,
    dict[str, Any],
]:
    episodes_per_query = (
        cfg.FORMATION_EPISODES_PER_OBSERVED_QUERY
        if formation_episodes_per_query is None
        else int(formation_episodes_per_query)
    )
    world = base.ContinuousConsequenceWorld(family_seed)
    learner = base.TemporalBoundaryLearner()
    vault = base.DelayedContinuationVault()
    pending: dict[int, tuple[np.ndarray, int, str, int]] = {}
    formation_metadata: list[base.StreamMetadata] = []
    formation_identities: list[str] = []
    released_features: list[np.ndarray] = []
    released_targets: list[float] = []
    released_queries: list[int] = []
    step = 0

    observed_queries = [query for query in range(cfg.QUERY_COUNT) if query != held_query]
    for query in observed_queries:
        for episode in range(episodes_per_query):
            identity = (
                f"dtbe-formation:f{family_seed}:h{held_query}:q{query}:"
                f"s{suite['name']}:e{episode}"
            )
            stream, metadata = world.generate(
                query,
                identity,
                suite,
                episode_index=episode,
            )
            features = base.temporal_pair_features(stream)
            action = int(
                base.stable_u32(
                    "dtbe-v1-formation-action",
                    family_seed,
                    held_query,
                    query,
                    episode,
                    suite["name"],
                )
                % 2
            )
            token = vault.commit(
                step=step,
                action_channel=action,
                target_channel=metadata.relation_channel,
            )
            pending[token] = (features, action, identity, query)
            # Metadata remains environment-private until class construction ends.
            formation_metadata.append(metadata)
            formation_identities.append(identity)
            for released_token, scalar_result in vault.release_ready(step):
                vector, released_action, released_identity, released_query = pending.pop(
                    released_token
                )
                learner.receive(
                    base.FormationRecord(
                        temporal_features=tuple(float(v) for v in vector),
                        committed_action=released_action,
                        continuation_result=scalar_result,
                        identity=released_identity,
                    )
                )
                released_features.append(vector.copy())
                released_targets.append(
                    base.relation_signed_target(released_action, scalar_result)
                )
                released_queries.append(released_query)
            step += 1

    flush_step = step + base.FORMATION_CREDIT_DELAY
    while pending:
        released = vault.release_ready(flush_step)
        if not released:
            flush_step += 1
            continue
        for released_token, scalar_result in released:
            vector, released_action, released_identity, released_query = pending.pop(
                released_token
            )
            learner.receive(
                base.FormationRecord(
                    temporal_features=tuple(float(v) for v in vector),
                    committed_action=released_action,
                    continuation_result=scalar_result,
                    identity=released_identity,
                )
            )
            released_features.append(vector.copy())
            released_targets.append(
                base.relation_signed_target(released_action, scalar_result)
            )
            released_queries.append(released_query)

    learned_weights, shuffled_weights = learner.solve()
    feature_matrix = np.asarray(released_features, dtype=float)
    target_vector = np.asarray(released_targets, dtype=float)
    query_vector = np.asarray(released_queries, dtype=int)
    model = build_equivalence_model(
        feature_matrix,
        target_vector,
        query_vector,
        learned_weights,
        observed_queries=observed_queries,
    )

    code_zero = base.unit_code(
        "dtbe-v1-relation-state",
        family_seed,
        held_query,
        suite["name"],
        0,
        dim=base.RELATION_STATE_DIM,
    )
    code_one_seed = base.unit_code(
        "dtbe-v1-relation-state",
        family_seed,
        held_query,
        suite["name"],
        1,
        dim=base.RELATION_STATE_DIM,
    )
    code_one = base.normalize(
        code_one_seed - float(code_zero @ code_one_seed) * code_zero
    )
    codes = np.stack([code_zero, code_one], axis=0)
    basin = codes.T @ codes

    formation_hash = payload_hash(
        {
            "family_seed": family_seed,
            "held_query": held_query,
            "suite": suite["name"],
            "formation_identity_hash": payload_hash(sorted(formation_identities)),
            "feedback": vault.audit(),
            "learner": learner.audit(),
            "world": world.audit(),
            "class_member_count": int(model["member_indices"].size),
            "class_score_hash": payload_hash(
                np.round(model["score"], 14).tolist()
            ),
        }
    )
    provisional = EquivalenceSnapshot(
        version="ectbf-v1b-dtbe-v1",
        stream_length=base.STREAM_LENGTH,
        raw_dim=base.RAW_DIM,
        pair_count=PAIR_COUNT,
        pair_vocabulary_hash=payload_hash(PAIR_INDICES),
        temporal_pair_weights=tuple(float(v) for v in learned_weights),
        equivalence_member_indices=tuple(
            int(v) for v in model["member_indices"]
        ),
        interleaved_sector_labels=tuple(
            int(v) for v in model["interleaved_labels"]
        ),
        contiguous_band_labels=tuple(
            int(v) for v in model["contiguous_labels"]
        ),
        best_single_index=int(model["best_single_index"]),
        dominant_index=int(model["dominant_index"]),
        relation_codes=tuple(tuple(float(v) for v in row) for row in codes),
        maintenance_basin=tuple(tuple(float(v) for v in row) for row in basin),
        formation_hash=formation_hash,
        class_score_hash=payload_hash(np.round(model["score"], 14).tolist()),
        substrate_hash="",
    )
    snapshot = EquivalenceSnapshot(
        **{
            **provisional.__dict__,
            "substrate_hash": payload_hash(provisional.payload_without_hash()),
        }
    )
    serialized = snapshot.to_json()
    restored = EquivalenceSnapshot.from_json(serialized)

    support_sum = np.zeros(PAIR_COUNT, dtype=float)
    pre_centers: list[float] = []
    post_centers: list[float] = []
    for metadata in formation_metadata:
        support_sum += metadata.support_vector()
        pre_centers.append(metadata.pre_center)
        post_centers.append(metadata.post_center)
    support_frequency = support_sum / float(len(formation_identities))
    baseline = float(np.mean(support_frequency))
    member_indices = model["member_indices"]
    class_weights = model["class_weights"]
    class_mass = np.maximum(class_weights, 0.0)
    if float(np.sum(class_mass)) > EPS:
        class_mass = class_mass / float(np.sum(class_mass))
    formation_truth_audit = {
        "support_baseline_frequency": baseline,
        "class_unweighted_support_frequency": float(
            np.mean(support_frequency[member_indices])
        ),
        "class_unweighted_support_enrichment": float(
            np.mean(support_frequency[member_indices])
        )
        / max(baseline, EPS),
        "class_weighted_support_frequency": float(class_mass @ support_frequency),
        "class_weighted_support_enrichment": float(class_mass @ support_frequency)
        / max(baseline, EPS),
        "mean_true_pre_center": float(np.mean(pre_centers)),
        "mean_true_post_center": float(np.mean(post_centers)),
        "hidden_windows_used_after_class_construction_only": True,
    }

    report = {
        "observed_queries": observed_queries,
        "held_query": held_query,
        "formation_episode_count": len(formation_identities),
        "formation_identity_count": len(set(formation_identities)),
        "formation_identity_hash": payload_hash(sorted(set(formation_identities))),
        "feedback_audit": vault.audit(),
        "learner_audit": learner.audit(),
        "world_audit": world.audit(),
        "initial_temporal_weight_norm": learner.initial_weight_norm,
        "learned_temporal_weight_norm": float(np.linalg.norm(learned_weights)),
        "learned_nonzero_pair_count": int(np.count_nonzero(learned_weights)),
        "equivalence_member_count": int(member_indices.size),
        "equivalence_class_mass_fraction": model["class_mass_fraction"],
        "effective_pair_number": model["effective_pair_number"],
        "normalized_class_entropy": model["normalized_class_entropy"],
        "class_component_count": model["component_count"],
        "class_left_span": model["left_span"],
        "class_right_span": model["right_span"],
        "class_lag_span": model["lag_span"],
        "best_single_pair": list(PAIR_INDICES[model["best_single_index"]]),
        "dominant_pair": list(PAIR_INDICES[model["dominant_index"]]),
        "class_construction_hidden_window_reads": model[
            "class_construction_hidden_window_reads"
        ],
        "class_construction_delayed_scalar_only": model[
            "class_construction_delayed_scalar_only"
        ],
        "snapshot_roundtrip_exact": restored.to_json() == serialized,
        "snapshot_forbidden_token_count": snapshot_forbidden_token_count(snapshot),
        "held_query_feedback_updates": 0,
        "formation_and_evaluation_namespaces_disjoint": True,
        "formation_truth_audit": formation_truth_audit,
    }
    return snapshot, world, report, shuffled_weights, model


def class_mask_from_snapshot(snapshot: EquivalenceSnapshot) -> np.ndarray:
    mask = np.zeros(PAIR_COUNT, dtype=bool)
    mask[list(snapshot.equivalence_member_indices)] = True
    return mask


def sector_field(
    base_weights: np.ndarray,
    labels: Sequence[int],
    sector: int,
) -> np.ndarray:
    label_array = np.asarray(labels, dtype=int)
    return np.where(label_array == int(sector), base_weights, 0.0)


def matched_outside_relocation(
    class_weights: np.ndarray,
    member_indices: Sequence[int],
) -> tuple[np.ndarray, tuple[int, ...]]:
    members = [int(index) for index in member_indices]
    outside = set(range(PAIR_COUNT)) - set(members)
    relocated = np.zeros(PAIR_COUNT, dtype=float)
    assigned: list[int] = []
    ordered = sorted(members, key=lambda index: (-float(class_weights[index]), index))
    for source in ordered:
        left, right = PAIR_INDICES[source]
        lag = right - left
        target = min(
            outside,
            key=lambda index: (
                4.0
                * abs((PAIR_INDICES[index][1] - PAIR_INDICES[index][0]) - lag)
                + abs(PAIR_INDICES[index][0] - left)
                + abs(PAIR_INDICES[index][1] - right),
                index,
            ),
        )
        outside.remove(target)
        relocated[target] = float(class_weights[source])
        assigned.append(target)
    return relocated, tuple(sorted(assigned))


def local_patch_field(
    class_weights: np.ndarray,
    member_indices: Sequence[int],
    center_index: int,
) -> tuple[np.ndarray, tuple[int, ...]]:
    count = max(1, int(math.ceil(len(tuple(member_indices)) / cfg.SECTOR_COUNT)))
    center = PAIR_COORDS[int(center_index)]
    candidates = sorted(
        range(PAIR_COUNT),
        key=lambda index: (
            float(np.sum((PAIR_COORDS[index] - center) ** 2)),
            index,
        ),
    )[:count]
    source_values = sorted(
        (float(class_weights[index]) for index in member_indices),
        reverse=True,
    )[:count]
    patch = np.zeros(PAIR_COUNT, dtype=float)
    for target, value in zip(candidates, source_values):
        patch[int(target)] = value
    return patch, tuple(sorted(int(v) for v in candidates))


def build_method_fields(
    snapshot: EquivalenceSnapshot,
    shuffled_weights: np.ndarray,
    *,
    family_seed: int,
    held_query: int,
    suite_name: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    learned = np.asarray(snapshot.temporal_pair_weights, dtype=float)
    member_mask = class_mask_from_snapshot(snapshot)
    class_weights = np.where(member_mask, learned, 0.0)
    class_norm = float(np.linalg.norm(class_weights))

    equalized = np.where(member_mask, 1.0, 0.0)
    equalized = normalized_field(equalized, class_norm)
    best_index = int(snapshot.best_single_index)
    best_single = np.zeros(PAIR_COUNT, dtype=float)
    best_single[int(best_index)] = max(class_norm, 1.0)
    dominant_index = int(snapshot.dominant_index)
    dominant_single = np.zeros(PAIR_COUNT, dtype=float)
    dominant_single[dominant_index] = max(class_norm, 1.0)

    local_patch, local_indices = local_patch_field(
        class_weights,
        snapshot.equivalence_member_indices,
        int(best_index),
    )
    outside, outside_indices = matched_outside_relocation(
        class_weights,
        snapshot.equivalence_member_indices,
    )
    shifted = base.temporal_shift_weights(class_weights, cfg.CLASS_SHIFT)

    fields: dict[str, np.ndarray] = {
        cfg.FULL: learned,
        cfg.CLASS_ONLY: class_weights,
        cfg.EQUALIZED_CLASS: equalized,
        cfg.SECTOR_0: sector_field(
            class_weights, snapshot.interleaved_sector_labels, 0
        ),
        cfg.SECTOR_1: sector_field(
            class_weights, snapshot.interleaved_sector_labels, 1
        ),
        cfg.SECTOR_2: sector_field(
            class_weights, snapshot.interleaved_sector_labels, 2
        ),
        cfg.BAND_0: sector_field(
            class_weights, snapshot.contiguous_band_labels, 0
        ),
        cfg.BAND_1: sector_field(
            class_weights, snapshot.contiguous_band_labels, 1
        ),
        cfg.BAND_2: sector_field(
            class_weights, snapshot.contiguous_band_labels, 2
        ),
        cfg.BEST_SINGLE: best_single,
        cfg.DOMINANT_SINGLE: dominant_single,
        cfg.LOCAL_PATCH: local_patch,
        cfg.CLASS_ABLATED: np.where(member_mask, 0.0, learned),
        cfg.MATCHED_OUTSIDE: outside,
        cfg.SHIFTED_CLASS: shifted,
        cfg.SHUFFLED_CREDIT: np.asarray(shuffled_weights, dtype=float),
        cfg.CLASS_DELAY: class_weights,
        cfg.CLASS_PARTIAL: class_weights,
        cfg.CLASS_NOISE: class_weights,
        cfg.CLASS_COMBINED: class_weights,
        cfg.CLASS_NO_MAINTENANCE: class_weights,
    }

    rng = np.random.default_rng(
        base.stable_u32(
            "dtbe-v1-field-replicates",
            family_seed,
            held_query,
            suite_name,
        )
    )
    members = np.asarray(snapshot.equivalence_member_indices, dtype=int)
    member_values = class_weights[members]
    permutation_fields: list[np.ndarray] = []
    for _ in range(cfg.PERMUTATION_REPLICATES):
        field = np.zeros(PAIR_COUNT, dtype=float)
        field[members] = rng.permutation(member_values)
        permutation_fields.append(field)

    dropout_fields: list[np.ndarray] = []
    keep_count = max(2, int(round((1.0 - cfg.DROPOUT_FRACTION) * len(members))))
    for _ in range(cfg.DROPOUT_REPLICATES):
        kept = rng.choice(members, size=keep_count, replace=False)
        field = np.zeros(PAIR_COUNT, dtype=float)
        field[kept] = class_weights[kept]
        field = normalized_field(field, class_norm)
        dropout_fields.append(field)

    metadata = {
        "best_single_index": int(best_index),
        "dominant_index": dominant_index,
        "local_patch_indices": local_indices,
        "outside_relocation_indices": outside_indices,
        "outside_disjoint_from_class": not bool(
            set(outside_indices) & set(snapshot.equivalence_member_indices)
        ),
        "local_patch_count": len(local_indices),
        "class_count": len(snapshot.equivalence_member_indices),
        "permutation_fields": permutation_fields,
        "dropout_fields": dropout_fields,
    }
    return fields, metadata


def _variant_for_method(method: str) -> str:
    if method == cfg.CLASS_PARTIAL:
        return "partial"
    if method == cfg.CLASS_NOISE:
        return "noise"
    if method in (cfg.CLASS_COMBINED, cfg.CLASS_NO_MAINTENANCE):
        return "combined"
    return "clean"


def _delay_for_method(method: str) -> tuple[bool, bool, float]:
    if method == cfg.CLASS_DELAY:
        return True, True, base.RELATION_DELAY_DRIFT
    if method == cfg.CLASS_NO_MAINTENANCE:
        return True, False, base.NO_MAINTENANCE_DRIFT
    return False, True, base.RELATION_DELAY_DRIFT


def predict_from_features(
    features: np.ndarray,
    weights: np.ndarray,
    snapshot: EquivalenceSnapshot,
    *,
    delay: bool,
    maintain: bool,
    state_noise: float,
    seed_parts: Sequence[Any],
) -> tuple[int, float]:
    score = float(np.asarray(weights, dtype=float) @ np.asarray(features, dtype=float))
    initial = int(score >= 0.0)
    if not delay:
        return initial, abs(score)
    codes = np.asarray(snapshot.relation_codes, dtype=float)
    basin = np.asarray(snapshot.maintenance_basin, dtype=float)
    rng = np.random.default_rng(base.stable_u32("dtbe-v1-delay", *seed_parts))
    state = codes[initial].copy()
    for _ in range(base.RELATION_DELAY_STEPS):
        state = (
            base.RELATION_DELAY_RETENTION * state
            + state_noise
            * rng.normal(size=base.RELATION_STATE_DIM)
            / math.sqrt(float(base.RELATION_STATE_DIM))
        )
        if maintain:
            state = base.normalize(0.22 * state + 0.78 * (basin @ state))
    scores = codes @ state
    return int(np.argmax(scores)), float(np.max(scores) - np.min(scores))


def member_majority_prediction(
    features: np.ndarray,
    member_indices: Sequence[int],
) -> tuple[int, float]:
    values = np.asarray(features, dtype=float)[list(member_indices)]
    votes = np.where(values >= 0.0, 1, -1)
    total = int(np.sum(votes))
    if total == 0:
        total = 1 if float(np.mean(values)) >= 0.0 else -1
    return int(total > 0), abs(float(total)) / float(len(votes))


def evaluate_direct(
    *,
    family_seed: int,
    held_query: int,
    suite: dict[str, Any],
    world: base.ContinuousConsequenceWorld,
    snapshot: EquivalenceSnapshot,
    shuffled_weights: np.ndarray,
    direct_episodes: int | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    episode_count = (
        cfg.DIRECT_TEST_EPISODES if direct_episodes is None else int(direct_episodes)
    )
    fields, field_metadata = build_method_fields(
        snapshot,
        shuffled_weights,
        family_seed=family_seed,
        held_query=held_query,
        suite_name=suite["name"],
    )
    counts = {method: 0 for method in cfg.METHODS}
    margins = {method: [] for method in cfg.METHODS}
    agreements = {method: 0 for method in cfg.METHODS}
    evaluation_identities: list[str] = []
    decisions_before_truth = 0
    support_sum = np.zeros(PAIR_COUNT, dtype=float)

    member_indices = np.asarray(snapshot.equivalence_member_indices, dtype=int)
    member_correct = np.zeros(member_indices.size, dtype=int)
    member_full_agreement = np.zeros(member_indices.size, dtype=int)

    permutation_correct = np.zeros(cfg.PERMUTATION_REPLICATES, dtype=int)
    permutation_agreement = np.zeros(cfg.PERMUTATION_REPLICATES, dtype=int)
    dropout_correct = np.zeros(cfg.DROPOUT_REPLICATES, dtype=int)
    dropout_agreement = np.zeros(cfg.DROPOUT_REPLICATES, dtype=int)

    for episode in range(episode_count):
        identity = (
            f"dtbe-evaluation-direct:f{family_seed}:h{held_query}:"
            f"s{suite['name']}:e{episode}"
        )
        stream, metadata = world.generate(
            held_query,
            identity,
            suite,
            episode_index=episode,
        )
        evaluation_identities.append(identity)

        variants: dict[str, np.ndarray] = {"clean": stream}
        for variant in ("partial", "noise", "combined"):
            variants[variant] = base.perturb_stream(
                stream,
                variant,
                seed_parts=(
                    "dtbe-v1",
                    family_seed,
                    held_query,
                    suite["name"],
                    episode,
                    variant,
                ),
            )
        feature_cache = {
            variant: base.temporal_pair_features(values)
            for variant, values in variants.items()
        }

        predictions: dict[str, tuple[int, float]] = {}
        for method in cfg.METHODS:
            if method == cfg.MEMBER_MAJORITY:
                predictions[method] = member_majority_prediction(
                    feature_cache["clean"],
                    snapshot.equivalence_member_indices,
                )
                continue
            variant = _variant_for_method(method)
            delay, maintain, state_noise = _delay_for_method(method)
            predictions[method] = predict_from_features(
                feature_cache[variant],
                fields[method],
                snapshot,
                delay=delay,
                maintain=maintain,
                state_noise=state_noise,
                seed_parts=(
                    family_seed,
                    held_query,
                    suite["name"],
                    episode,
                    method,
                ),
            )

        full_prediction = predictions[cfg.FULL][0]
        permutation_predictions = [
            int(float(field @ feature_cache["clean"]) >= 0.0)
            for field in field_metadata["permutation_fields"]
        ]
        dropout_predictions = [
            int(float(field @ feature_cache["clean"]) >= 0.0)
            for field in field_metadata["dropout_fields"]
        ]
        individual_predictions = (
            feature_cache["clean"][member_indices] >= 0.0
        ).astype(int)

        decisions_before_truth += 1
        truth = int(metadata.relation_channel)
        support_sum += metadata.support_vector()
        for method, (prediction, margin) in predictions.items():
            counts[method] += int(prediction == truth)
            margins[method].append(float(margin))
            agreements[method] += int(prediction == full_prediction)
        member_correct += (individual_predictions == truth).astype(int)
        member_full_agreement += (
            individual_predictions == full_prediction
        ).astype(int)
        permutation_correct += np.asarray(
            [int(prediction == truth) for prediction in permutation_predictions],
            dtype=int,
        )
        permutation_agreement += np.asarray(
            [
                int(prediction == full_prediction)
                for prediction in permutation_predictions
            ],
            dtype=int,
        )
        dropout_correct += np.asarray(
            [int(prediction == truth) for prediction in dropout_predictions],
            dtype=int,
        )
        dropout_agreement += np.asarray(
            [
                int(prediction == full_prediction)
                for prediction in dropout_predictions
            ],
            dtype=int,
        )

    methods: dict[str, dict[str, float]] = {}
    for method in cfg.METHODS:
        methods[method] = {
            "relation_accuracy": counts[method] / float(episode_count),
            "agreement_with_full": agreements[method] / float(episode_count),
            "mean_margin": float(np.mean(margins[method])),
            "minimum_margin": float(np.min(margins[method])),
        }

    member_accuracy = member_correct / float(episode_count)
    member_agreement = member_full_agreement / float(episode_count)
    permutation_accuracy = permutation_correct / float(episode_count)
    permutation_full_agreement = permutation_agreement / float(episode_count)
    dropout_accuracy = dropout_correct / float(episode_count)
    dropout_full_agreement = dropout_agreement / float(episode_count)

    class_accuracy = methods[cfg.CLASS_ONLY]["relation_accuracy"]
    equivalent_sectors: list[bool] = []
    for method in cfg.INTERLEAVED_SECTORS:
        equivalent_sectors.append(
            methods[method]["relation_accuracy"]
            >= max(
                cfg.SECTOR_EQUIVALENCE_MIN_ACCURACY,
                class_accuracy - cfg.SECTOR_EQUIVALENCE_MAX_GAP_TO_CLASS,
            )
            and methods[method]["agreement_with_full"]
            >= cfg.SECTOR_EQUIVALENCE_MIN_AGREEMENT
        )

    support_frequency = support_sum / float(episode_count)
    support_baseline = float(np.mean(support_frequency))
    member_mask = class_mask_from_snapshot(snapshot)
    class_weights = np.where(
        member_mask,
        np.asarray(snapshot.temporal_pair_weights, dtype=float),
        0.0,
    )
    mass = np.maximum(class_weights, 0.0)
    mass /= max(float(np.sum(mass)), EPS)
    interleaved_support = []
    for sector in range(cfg.SECTOR_COUNT):
        indices = np.flatnonzero(
            np.asarray(snapshot.interleaved_sector_labels, dtype=int) == sector
        )
        interleaved_support.append(
            float(np.mean(support_frequency[indices])) / max(support_baseline, EPS)
        )

    audit = {
        "evaluation_episode_count": episode_count,
        "evaluation_identity_count": len(set(evaluation_identities)),
        "evaluation_identity_hash": payload_hash(sorted(set(evaluation_identities))),
        "route_decisions_before_truth_reads": decisions_before_truth,
        "truth_reads_after_route_decision": episode_count,
        "explicit_boundary_arguments": 0,
        "explicit_relation_arguments": 0,
        "runtime_planning_calls": 0,
        "class_hidden_window_reads_before_decision": 0,
        "outside_relocation_disjoint": field_metadata[
            "outside_disjoint_from_class"
        ],
        "equivalent_interleaved_sector_flags": equivalent_sectors,
        "equivalent_interleaved_sector_count": int(sum(equivalent_sectors)),
        "member_accuracy_mean": float(np.mean(member_accuracy)),
        "member_accuracy_median": float(np.median(member_accuracy)),
        "member_accuracy_quartile": float(np.quantile(member_accuracy, 0.25)),
        "member_full_agreement_mean": float(np.mean(member_agreement)),
        "fraction_members_accuracy_at_least_0_60": float(
            np.mean(member_accuracy >= 0.60)
        ),
        "fraction_members_within_0_15_of_class": float(
            np.mean(member_accuracy >= class_accuracy - 0.15)
        ),
        "permutation_relation_accuracy_mean": float(
            np.mean(permutation_accuracy)
        ),
        "permutation_relation_accuracy_minimum": float(
            np.min(permutation_accuracy)
        ),
        "permutation_full_agreement_mean": float(
            np.mean(permutation_full_agreement)
        ),
        "permutation_full_agreement_minimum": float(
            np.min(permutation_full_agreement)
        ),
        "dropout_relation_accuracy_mean": float(np.mean(dropout_accuracy)),
        "dropout_relation_accuracy_minimum": float(np.min(dropout_accuracy)),
        "dropout_full_agreement_mean": float(np.mean(dropout_full_agreement)),
        "dropout_full_agreement_minimum": float(np.min(dropout_full_agreement)),
        "unseen_support_baseline_frequency": support_baseline,
        "unseen_class_unweighted_support_frequency": float(
            np.mean(support_frequency[member_mask])
        ),
        "unseen_class_support_enrichment": float(
            np.mean(support_frequency[member_mask])
        )
        / max(support_baseline, EPS),
        "unseen_class_weighted_support_enrichment": float(
            mass @ support_frequency
        )
        / max(support_baseline, EPS),
        "unseen_class_support_coverage": float(
            np.sum(support_frequency[member_mask])
        )
        / max(float(np.sum(support_frequency)), EPS),
        "interleaved_sector_support_enrichments": interleaved_support,
        "best_single_pair_support_frequency": float(
            support_frequency[field_metadata["best_single_index"]]
        ),
        "local_patch_count": field_metadata["local_patch_count"],
        "class_count": field_metadata["class_count"],
    }
    return methods, audit


def evaluate_plan(
    *,
    family_seed: int,
    held_query: int,
    suite: dict[str, Any],
    world: base.ContinuousConsequenceWorld,
    snapshot: EquivalenceSnapshot,
    shuffled_weights: np.ndarray,
    plan_episodes: int | None = None,
    plan_depth: int | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    episode_count = cfg.PLAN_EPISODES if plan_episodes is None else int(plan_episodes)
    depth_count = cfg.PLAN_DEPTH if plan_depth is None else int(plan_depth)
    fields, _metadata = build_method_fields(
        snapshot,
        shuffled_weights,
        family_seed=family_seed,
        held_query=held_query,
        suite_name=suite["name"],
    )
    complete = {method: 0 for method in cfg.METHODS}
    step_correct = {method: 0 for method in cfg.METHODS}
    identities: list[str] = []
    decisions_before_truth = 0

    for episode in range(episode_count):
        streams: list[np.ndarray] = []
        metadata_rows: list[base.StreamMetadata] = []
        for depth in range(depth_count):
            identity = (
                f"dtbe-evaluation-plan:f{family_seed}:h{held_query}:"
                f"s{suite['name']}:e{episode}:d{depth}"
            )
            stream, metadata = world.generate(
                held_query,
                identity,
                suite,
                episode_index=episode * depth_count + depth,
            )
            streams.append(stream)
            metadata_rows.append(metadata)
            identities.append(identity)

        predictions = {method: [] for method in cfg.METHODS}
        for depth, stream in enumerate(streams):
            variants: dict[str, np.ndarray] = {"clean": stream}
            for variant in ("partial", "noise", "combined"):
                variants[variant] = base.perturb_stream(
                    stream,
                    variant,
                    seed_parts=(
                        "dtbe-v1-plan",
                        family_seed,
                        held_query,
                        suite["name"],
                        episode,
                        depth,
                        variant,
                    ),
                )
            feature_cache = {
                variant: base.temporal_pair_features(values)
                for variant, values in variants.items()
            }
            for method in cfg.METHODS:
                if method == cfg.MEMBER_MAJORITY:
                    prediction, _margin = member_majority_prediction(
                        feature_cache["clean"],
                        snapshot.equivalence_member_indices,
                    )
                else:
                    variant = _variant_for_method(method)
                    delay, maintain, state_noise = _delay_for_method(method)
                    prediction, _margin = predict_from_features(
                        feature_cache[variant],
                        fields[method],
                        snapshot,
                        delay=delay,
                        maintain=maintain,
                        state_noise=state_noise,
                        seed_parts=(
                            family_seed,
                            held_query,
                            suite["name"],
                            episode,
                            depth,
                            method,
                        ),
                    )
                predictions[method].append(int(prediction))
            decisions_before_truth += 1

        truths = [int(row.relation_channel) for row in metadata_rows]
        for method, sequence in predictions.items():
            step_correct[method] += sum(
                int(prediction == truth)
                for prediction, truth in zip(sequence, truths)
            )
            complete[method] += int(sequence == truths)

    total_steps = episode_count * depth_count
    methods = {
        method: {
            "step_accuracy": step_correct[method] / float(total_steps),
            "complete_sequence_identity_rate": complete[method]
            / float(episode_count),
        }
        for method in cfg.METHODS
    }
    audit = {
        "plan_episode_count": episode_count,
        "plan_depth": depth_count,
        "plan_identity_count": len(set(identities)),
        "plan_identity_hash": payload_hash(sorted(set(identities))),
        "route_decisions_before_truth_reads": decisions_before_truth,
        "truth_reads_after_route_decision": total_steps,
        "explicit_boundary_arguments": 0,
        "explicit_relation_arguments": 0,
        "runtime_planning_calls": 0,
    }
    return methods, audit


def run_case(
    family_seed: int,
    held_query: int,
    suite: dict[str, Any],
    *,
    formation_episodes_per_query: int | None = None,
    direct_episodes: int | None = None,
    plan_episodes: int | None = None,
    plan_depth: int | None = None,
) -> dict[str, Any]:
    snapshot, world, formation, shuffled_weights, _model = form_equivalence_class(
        family_seed,
        held_query,
        suite,
        formation_episodes_per_query=formation_episodes_per_query,
    )
    restored = EquivalenceSnapshot.from_json(snapshot.to_json())
    direct_methods, direct_audit = evaluate_direct(
        family_seed=family_seed,
        held_query=held_query,
        suite=suite,
        world=world,
        snapshot=restored,
        shuffled_weights=shuffled_weights,
        direct_episodes=direct_episodes,
    )
    plan_methods, plan_audit = evaluate_plan(
        family_seed=family_seed,
        held_query=held_query,
        suite=suite,
        world=world,
        snapshot=restored,
        shuffled_weights=shuffled_weights,
        plan_episodes=plan_episodes,
        plan_depth=plan_depth,
    )
    methods = {
        method: {
            **direct_methods[method],
            **plan_methods[method],
        }
        for method in cfg.METHODS
    }

    class_accuracy = methods[cfg.CLASS_ONLY]["relation_accuracy"]
    class_sequence = methods[cfg.CLASS_ONLY]["complete_sequence_identity_rate"]
    control_gaps = {
        "gain_over_best_single": (
            class_accuracy - methods[cfg.BEST_SINGLE]["relation_accuracy"]
        ),
        "gain_over_dominant_single": (
            class_accuracy - methods[cfg.DOMINANT_SINGLE]["relation_accuracy"]
        ),
        "gain_over_local_patch": (
            class_accuracy - methods[cfg.LOCAL_PATCH]["relation_accuracy"]
        ),
        "gain_over_matched_outside": (
            class_accuracy - methods[cfg.MATCHED_OUTSIDE]["relation_accuracy"]
        ),
        "gain_over_shifted_class": (
            class_accuracy - methods[cfg.SHIFTED_CLASS]["relation_accuracy"]
        ),
        "gain_over_shuffled_credit": (
            class_accuracy - methods[cfg.SHUFFLED_CREDIT]["relation_accuracy"]
        ),
        "class_ablation_relation_gap": (
            class_accuracy - methods[cfg.CLASS_ABLATED]["relation_accuracy"]
        ),
        "class_ablation_sequence_gap": (
            class_sequence
            - methods[cfg.CLASS_ABLATED]["complete_sequence_identity_rate"]
        ),
        "outside_relocation_sequence_gap": (
            class_sequence
            - methods[cfg.MATCHED_OUTSIDE]["complete_sequence_identity_rate"]
        ),
        "shifted_class_sequence_gap": (
            class_sequence
            - methods[cfg.SHIFTED_CLASS]["complete_sequence_identity_rate"]
        ),
        "no_maintenance_sequence_gap": (
            methods[cfg.CLASS_COMBINED]["complete_sequence_identity_rate"]
            - methods[cfg.CLASS_NO_MAINTENANCE][
                "complete_sequence_identity_rate"
            ]
        ),
    }
    information_boundary = {
        "continuous_stream_only_at_runtime": True,
        "endpoint_segmentation_absent": True,
        "true_boundary_reads_during_class_construction": formation[
            "class_construction_hidden_window_reads"
        ],
        "true_relation_reads_before_decision": 0,
        "held_query_formation_updates": formation[
            "held_query_feedback_updates"
        ],
        "explicit_boundary_arguments": (
            direct_audit["explicit_boundary_arguments"]
            + plan_audit["explicit_boundary_arguments"]
        ),
        "explicit_relation_arguments": (
            direct_audit["explicit_relation_arguments"]
            + plan_audit["explicit_relation_arguments"]
        ),
        "runtime_planning_calls": plan_audit["runtime_planning_calls"],
        "generic_ordered_pair_vocabulary_supplied": True,
        "class_construction_delayed_scalar_only": formation[
            "class_construction_delayed_scalar_only"
        ],
        "hidden_windows_postdecision_audit_only": True,
    }
    control_identity = {
        "formation_and_direct_namespaces_disjoint": (
            formation["formation_identity_hash"]
            != direct_audit["evaluation_identity_hash"]
        ),
        "formation_and_plan_namespaces_disjoint": (
            formation["formation_identity_hash"] != plan_audit["plan_identity_hash"]
        ),
        "direct_and_plan_namespaces_disjoint": (
            direct_audit["evaluation_identity_hash"] != plan_audit["plan_identity_hash"]
        ),
        "snapshot_roundtrip_exact": formation["snapshot_roundtrip_exact"],
        "snapshot_forbidden_tokens_zero": (
            formation["snapshot_forbidden_token_count"] == 0
        ),
        "outside_relocation_disjoint": direct_audit[
            "outside_relocation_disjoint"
        ],
        "pair_vocabulary_hash_exact": (
            restored.pair_vocabulary_hash == payload_hash(PAIR_INDICES)
        ),
        "sector_partition_exact": (
            sum(
                1
                for label in restored.interleaved_sector_labels
                if label in range(cfg.SECTOR_COUNT)
            )
            == len(restored.equivalence_member_indices)
        ),
    }
    case = {
        "family_seed": int(family_seed),
        "held_query": int(held_query),
        "suite": suite["name"],
        "methods": methods,
        "formation": formation,
        "direct_audit": direct_audit,
        "plan_audit": plan_audit,
        "control_gaps": control_gaps,
        "information_boundary": information_boundary,
        "control_identity": control_identity,
        "snapshot_sha256": hashlib.sha256(
            restored.to_json().encode("utf-8")
        ).hexdigest(),
    }
    case["case_sha256"] = payload_hash(case)
    return case


def _mean(rows: Iterable[float]) -> float:
    values = [float(value) for value in rows]
    return float(np.mean(values)) if values else 0.0


def _minimum(rows: Iterable[float]) -> float:
    values = [float(value) for value in rows]
    return float(np.min(values)) if values else 0.0


def _maximum(rows: Iterable[float]) -> float:
    values = [float(value) for value in rows]
    return float(np.max(values)) if values else 0.0


def aggregate_cases(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    methods: dict[str, dict[str, float]] = {}
    for method in cfg.METHODS:
        methods[method] = {
            "relation_accuracy": _mean(
                case["methods"][method]["relation_accuracy"] for case in cases
            ),
            "minimum_relation_accuracy": _minimum(
                case["methods"][method]["relation_accuracy"] for case in cases
            ),
            "agreement_with_full": _mean(
                case["methods"][method]["agreement_with_full"] for case in cases
            ),
            "step_accuracy": _mean(
                case["methods"][method]["step_accuracy"] for case in cases
            ),
            "complete_sequence_identity_rate": _mean(
                case["methods"][method]["complete_sequence_identity_rate"]
                for case in cases
            ),
            "minimum_complete_sequence_identity_rate": _minimum(
                case["methods"][method]["complete_sequence_identity_rate"]
                for case in cases
            ),
        }

    equivalence = {
        "case_count": len(cases),
        "mean_equivalence_member_count": _mean(
            case["formation"]["equivalence_member_count"] for case in cases
        ),
        "minimum_equivalence_member_count": int(
            min(case["formation"]["equivalence_member_count"] for case in cases)
        ),
        "maximum_equivalence_member_count": int(
            max(case["formation"]["equivalence_member_count"] for case in cases)
        ),
        "minimum_class_mass_fraction": _minimum(
            case["formation"]["equivalence_class_mass_fraction"] for case in cases
        ),
        "minimum_effective_pair_number": _minimum(
            case["formation"]["effective_pair_number"] for case in cases
        ),
        "minimum_normalized_class_entropy": _minimum(
            case["formation"]["normalized_class_entropy"] for case in cases
        ),
        "minimum_class_support_enrichment": _minimum(
            case["direct_audit"]["unseen_class_support_enrichment"]
            for case in cases
        ),
        "mean_class_support_enrichment": _mean(
            case["direct_audit"]["unseen_class_support_enrichment"]
            for case in cases
        ),
        "minimum_class_weighted_support_enrichment": _minimum(
            case["direct_audit"]["unseen_class_weighted_support_enrichment"]
            for case in cases
        ),
        "minimum_class_only_relation_accuracy": methods[cfg.CLASS_ONLY][
            "minimum_relation_accuracy"
        ],
        "minimum_equalized_class_relation_accuracy": methods[cfg.EQUALIZED_CLASS][
            "minimum_relation_accuracy"
        ],
        "minimum_member_majority_relation_accuracy": methods[cfg.MEMBER_MAJORITY][
            "minimum_relation_accuracy"
        ],
        "minimum_interleaved_sector_relation_accuracy": min(
            methods[method]["minimum_relation_accuracy"]
            for method in cfg.INTERLEAVED_SECTORS
        ),
        "mean_interleaved_sector_relation_accuracy": _mean(
            methods[method]["relation_accuracy"]
            for method in cfg.INTERLEAVED_SECTORS
        ),
        "minimum_contiguous_band_relation_accuracy": min(
            methods[method]["minimum_relation_accuracy"]
            for method in cfg.CONTIGUOUS_BANDS
        ),
        "mean_equivalent_sector_count": _mean(
            case["direct_audit"]["equivalent_interleaved_sector_count"]
            for case in cases
        ),
        "minimum_equivalent_sector_count": int(
            min(
                case["direct_audit"]["equivalent_interleaved_sector_count"]
                for case in cases
            )
        ),
        "fraction_cases_with_two_equivalent_sectors": _mean(
            case["direct_audit"]["equivalent_interleaved_sector_count"] >= 2
            for case in cases
        ),
        "fraction_cases_with_three_equivalent_sectors": _mean(
            case["direct_audit"]["equivalent_interleaved_sector_count"] == 3
            for case in cases
        ),
        "minimum_permutation_relation_accuracy": _minimum(
            case["direct_audit"]["permutation_relation_accuracy_minimum"]
            for case in cases
        ),
        "mean_permutation_relation_accuracy": _mean(
            case["direct_audit"]["permutation_relation_accuracy_mean"]
            for case in cases
        ),
        "minimum_dropout_relation_accuracy": _minimum(
            case["direct_audit"]["dropout_relation_accuracy_minimum"]
            for case in cases
        ),
        "mean_dropout_relation_accuracy": _mean(
            case["direct_audit"]["dropout_relation_accuracy_mean"]
            for case in cases
        ),
        "minimum_member_accuracy_mean": _minimum(
            case["direct_audit"]["member_accuracy_mean"] for case in cases
        ),
        "minimum_fraction_members_accuracy_at_least_0_60": _minimum(
            case["direct_audit"]["fraction_members_accuracy_at_least_0_60"]
            for case in cases
        ),
        "minimum_gain_over_best_single": _minimum(
            case["control_gaps"]["gain_over_best_single"] for case in cases
        ),
        "minimum_gain_over_local_patch": _minimum(
            case["control_gaps"]["gain_over_local_patch"] for case in cases
        ),
        "minimum_gain_over_matched_outside": _minimum(
            case["control_gaps"]["gain_over_matched_outside"] for case in cases
        ),
        "minimum_gain_over_shifted_class": _minimum(
            case["control_gaps"]["gain_over_shifted_class"] for case in cases
        ),
        "minimum_gain_over_shuffled_credit": _minimum(
            case["control_gaps"]["gain_over_shuffled_credit"] for case in cases
        ),
        "minimum_class_ablation_relation_gap": _minimum(
            case["control_gaps"]["class_ablation_relation_gap"] for case in cases
        ),
        "all_initial_fields_zero": all(
            case["formation"]["initial_temporal_weight_norm"] == 0.0
            for case in cases
        ),
        "all_learned_fields_nonzero": all(
            case["formation"]["learned_temporal_weight_norm"] > 0.0
            for case in cases
        ),
        "all_classes_nonempty": all(
            case["formation"]["equivalence_member_count"] > 0 for case in cases
        ),
        "all_feedback_delayed": all(
            case["formation"]["feedback_audit"]["all_results_after_required_delay"]
            for case in cases
        ),
        "all_held_query_updates_zero": all(
            case["formation"]["held_query_feedback_updates"] == 0 for case in cases
        ),
        "all_snapshot_roundtrips_exact": all(
            case["formation"]["snapshot_roundtrip_exact"] for case in cases
        ),
        "maximum_snapshot_forbidden_token_count": int(
            max(
                case["formation"]["snapshot_forbidden_token_count"]
                for case in cases
            )
        ),
        "all_hidden_window_reads_zero_during_class_construction": all(
            case["formation"]["class_construction_hidden_window_reads"] == 0
            for case in cases
        ),
    }
    persistence = {
        "case_count": len(cases),
        "class_only_complete_sequence": methods[cfg.CLASS_ONLY][
            "complete_sequence_identity_rate"
        ],
        "minimum_class_only_complete_sequence": methods[cfg.CLASS_ONLY][
            "minimum_complete_sequence_identity_rate"
        ],
        "equalized_class_complete_sequence": methods[cfg.EQUALIZED_CLASS][
            "complete_sequence_identity_rate"
        ],
        "member_majority_complete_sequence": methods[cfg.MEMBER_MAJORITY][
            "complete_sequence_identity_rate"
        ],
        "minimum_interleaved_sector_complete_sequence": min(
            methods[method]["minimum_complete_sequence_identity_rate"]
            for method in cfg.INTERLEAVED_SECTORS
        ),
        "mean_interleaved_sector_complete_sequence": _mean(
            methods[method]["complete_sequence_identity_rate"]
            for method in cfg.INTERLEAVED_SECTORS
        ),
        "class_delay_complete_sequence": methods[cfg.CLASS_DELAY][
            "complete_sequence_identity_rate"
        ],
        "class_partial_complete_sequence": methods[cfg.CLASS_PARTIAL][
            "complete_sequence_identity_rate"
        ],
        "class_noise_complete_sequence": methods[cfg.CLASS_NOISE][
            "complete_sequence_identity_rate"
        ],
        "class_combined_complete_sequence": methods[cfg.CLASS_COMBINED][
            "complete_sequence_identity_rate"
        ],
        "minimum_robust_class_relation_accuracy": min(
            methods[method]["minimum_relation_accuracy"]
            for method in cfg.ROBUST_CLASS_METHODS
        ),
        "class_ablated_complete_sequence": methods[cfg.CLASS_ABLATED][
            "complete_sequence_identity_rate"
        ],
        "matched_outside_complete_sequence": methods[cfg.MATCHED_OUTSIDE][
            "complete_sequence_identity_rate"
        ],
        "shifted_class_complete_sequence": methods[cfg.SHIFTED_CLASS][
            "complete_sequence_identity_rate"
        ],
        "class_no_maintenance_complete_sequence": methods[
            cfg.CLASS_NO_MAINTENANCE
        ]["complete_sequence_identity_rate"],
        "class_ablation_sequence_gap": (
            methods[cfg.CLASS_ONLY]["complete_sequence_identity_rate"]
            - methods[cfg.CLASS_ABLATED]["complete_sequence_identity_rate"]
        ),
        "outside_relocation_sequence_gap": (
            methods[cfg.CLASS_ONLY]["complete_sequence_identity_rate"]
            - methods[cfg.MATCHED_OUTSIDE]["complete_sequence_identity_rate"]
        ),
        "shifted_class_sequence_gap": (
            methods[cfg.CLASS_ONLY]["complete_sequence_identity_rate"]
            - methods[cfg.SHIFTED_CLASS]["complete_sequence_identity_rate"]
        ),
        "no_maintenance_sequence_gap": (
            methods[cfg.CLASS_COMBINED]["complete_sequence_identity_rate"]
            - methods[cfg.CLASS_NO_MAINTENANCE][
                "complete_sequence_identity_rate"
            ]
        ),
    }

    structural: dict[str, Any] = {}
    for section in ("information_boundary", "control_identity"):
        for key in sorted(cases[0][section]):
            values = [case[section][key] for case in cases]
            name = f"{section}__{key}"
            if isinstance(values[0], bool):
                structural[name] = all(bool(value) for value in values)
            elif isinstance(values[0], (int, float)):
                structural[name] = max(float(value) for value in values)
            else:
                structural[name] = all(value == values[0] for value in values)

    return {
        "methods": methods,
        "distributed_equivalence": equivalence,
        "persistence": persistence,
        "structural_audits": structural,
    }


def decide(aggregate: dict[str, Any]) -> dict[str, Any]:
    if cfg.FROZEN_THRESHOLDS is None:
        raise RuntimeError("DTBE v1 formal thresholds are not frozen")
    threshold = cfg.FROZEN_THRESHOLDS
    equivalence = aggregate["distributed_equivalence"]
    persistence = aggregate["persistence"]
    structural = aggregate["structural_audits"]

    audit_checks = {
        "all_structural_boolean_audits": all(
            value for value in structural.values() if isinstance(value, bool)
        ),
        "initial_fields_zero": equivalence["all_initial_fields_zero"],
        "learned_fields_nonzero": equivalence["all_learned_fields_nonzero"],
        "classes_nonempty": equivalence["all_classes_nonempty"],
        "feedback_delayed": equivalence["all_feedback_delayed"],
        "held_query_updates_zero": equivalence["all_held_query_updates_zero"],
        "snapshot_roundtrip": equivalence["all_snapshot_roundtrips_exact"],
        "snapshot_forbidden_zero": (
            equivalence["maximum_snapshot_forbidden_token_count"] == 0
        ),
        "hidden_window_reads_zero_during_class_construction": equivalence[
            "all_hidden_window_reads_zero_during_class_construction"
        ],
    }
    functional_checks = {
        "minimum_class_only_relation_accuracy": (
            equivalence["minimum_class_only_relation_accuracy"]
            >= threshold["minimum_class_only_relation_accuracy"]
        ),
        "minimum_class_only_complete_sequence": (
            persistence["minimum_class_only_complete_sequence"]
            >= threshold["minimum_class_only_complete_sequence"]
        ),
        "minimum_robust_class_relation_accuracy": (
            persistence["minimum_robust_class_relation_accuracy"]
            >= threshold["minimum_robust_class_relation_accuracy"]
        ),
        "minimum_combined_complete_sequence": (
            persistence["class_combined_complete_sequence"]
            >= threshold["minimum_combined_complete_sequence"]
        ),
        "minimum_class_support_enrichment": (
            equivalence["minimum_class_support_enrichment"]
            >= threshold["minimum_class_support_enrichment"]
        ),
    }
    equivalence_checks = {
        "minimum_equalized_class_relation_accuracy": (
            equivalence["minimum_equalized_class_relation_accuracy"]
            >= threshold["minimum_equalized_class_relation_accuracy"]
        ),
        "minimum_member_majority_relation_accuracy": (
            equivalence["minimum_member_majority_relation_accuracy"]
            >= threshold["minimum_member_majority_relation_accuracy"]
        ),
        "minimum_effective_pair_number": (
            equivalence["minimum_effective_pair_number"]
            >= threshold["minimum_effective_pair_number"]
        ),
        "minimum_normalized_class_entropy": (
            equivalence["minimum_normalized_class_entropy"]
            >= threshold["minimum_normalized_class_entropy"]
        ),
        "minimum_class_mass_fraction": (
            equivalence["minimum_class_mass_fraction"]
            >= threshold["minimum_class_mass_fraction"]
        ),
        "minimum_fraction_cases_with_two_equivalent_sectors": (
            equivalence["fraction_cases_with_two_equivalent_sectors"]
            >= threshold["minimum_fraction_cases_with_two_equivalent_sectors"]
        ),
        "minimum_mean_equivalent_sector_count": (
            equivalence["mean_equivalent_sector_count"]
            >= threshold["minimum_mean_equivalent_sector_count"]
        ),
        "minimum_interleaved_sector_relation_accuracy": (
            equivalence["minimum_interleaved_sector_relation_accuracy"]
            >= threshold["minimum_interleaved_sector_relation_accuracy"]
        ),
        "minimum_interleaved_sector_complete_sequence": (
            persistence["minimum_interleaved_sector_complete_sequence"]
            >= threshold["minimum_interleaved_sector_complete_sequence"]
        ),
        "minimum_permutation_relation_accuracy": (
            equivalence["minimum_permutation_relation_accuracy"]
            >= threshold["minimum_permutation_relation_accuracy"]
        ),
        "minimum_dropout_relation_accuracy": (
            equivalence["minimum_dropout_relation_accuracy"]
            >= threshold["minimum_dropout_relation_accuracy"]
        ),
        "minimum_gain_over_best_single": (
            equivalence["minimum_gain_over_best_single"]
            >= threshold["minimum_gain_over_best_single"]
        ),
        "minimum_gain_over_local_patch": (
            equivalence["minimum_gain_over_local_patch"]
            >= threshold["minimum_gain_over_local_patch"]
        ),
    }
    causal_checks = {
        "minimum_gain_over_matched_outside": (
            equivalence["minimum_gain_over_matched_outside"]
            >= threshold["minimum_gain_over_matched_outside"]
        ),
        "minimum_gain_over_shifted_class": (
            equivalence["minimum_gain_over_shifted_class"]
            >= threshold["minimum_gain_over_shifted_class"]
        ),
        "minimum_class_ablation_relation_gap": (
            equivalence["minimum_class_ablation_relation_gap"]
            >= threshold["minimum_class_ablation_relation_gap"]
        ),
        "minimum_class_ablation_sequence_gap": (
            persistence["class_ablation_sequence_gap"]
            >= threshold["minimum_class_ablation_sequence_gap"]
        ),
        "minimum_outside_relocation_sequence_gap": (
            persistence["outside_relocation_sequence_gap"]
            >= threshold["minimum_outside_relocation_sequence_gap"]
        ),
        "minimum_no_maintenance_sequence_gap": (
            persistence["no_maintenance_sequence_gap"]
            >= threshold["minimum_no_maintenance_sequence_gap"]
        ),
    }

    audits_pass = all(audit_checks.values())
    functional_pass = all(functional_checks.values())
    equivalence_pass = all(equivalence_checks.values())
    causal_pass = all(causal_checks.values())
    localized_signal = (
        functional_checks["minimum_class_only_relation_accuracy"]
        and equivalence["minimum_gain_over_best_single"] < 0.03
    )
    if not audits_pass:
        status = cfg.STATUS_INVALID
    elif functional_pass and equivalence_pass and causal_pass:
        status = cfg.STATUS_POSITIVE
    elif equivalence_pass and causal_pass and not functional_pass:
        status = cfg.STATUS_EQUIVALENCE_NO_PLAN
    elif functional_pass and localized_signal:
        status = cfg.STATUS_LOCALIZED
    elif functional_pass:
        status = cfg.STATUS_FUNCTIONAL_NO_EQUIVALENCE
    else:
        status = cfg.STATUS_NEGATIVE

    return {
        "status": status,
        "candidate_promoted": status == cfg.STATUS_POSITIVE,
        "all_audits_pass": audits_pass,
        "functional_pass": functional_pass,
        "equivalence_pass": equivalence_pass,
        "causal_pass": causal_pass,
        "audit_checks": audit_checks,
        "functional_checks": functional_checks,
        "equivalence_checks": equivalence_checks,
        "causal_checks": causal_checks,
    }


def ancestry_record() -> dict[str, Any]:
    record: dict[str, Any] = {"branch": cfg.ANCESTRY_BRANCH}
    for label, path_text in (
        ("completion_receipt", cfg.ANCESTRY_COMPLETION_PATH),
        ("formal_summary", cfg.ANCESTRY_FORMAL_SUMMARY_PATH),
    ):
        path = Path(path_text)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            record[label] = {
                "path": path_text,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "formal_status": payload.get("formal_status"),
                "candidate_promoted": payload.get("candidate_promoted"),
                "formal_result_sha256": payload.get("formal_result_sha256"),
                "pipeline_complete": payload.get("pipeline_complete"),
            }
        else:
            record[label] = {
                "path": path_text,
                "missing_in_local_probe": True,
            }
    return record


def source_manifest() -> tuple[str, dict[str, str]]:
    paths = [
        "distributed_temporal_boundary_equivalence_config_v1.py",
        "distributed_temporal_boundary_equivalence_v1.py",
        "run_distributed_temporal_boundary_equivalence_v1.py",
        "audit_distributed_temporal_boundary_equivalence_v1.py",
        "freeze_distributed_temporal_boundary_equivalence_v1.py",
        "build_distributed_temporal_boundary_equivalence_v1_formal_summary.py",
        "postprocess_distributed_temporal_boundary_equivalence_v1.py",
        "test_distributed_temporal_boundary_equivalence_v1.py",
        "research/protocols/distributed_temporal_boundary_equivalence_v1.md",
        "endogenous_temporal_consequence_boundary_formation_config_v1.py",
        "endogenous_temporal_consequence_boundary_formation_v1.py",
        cfg.ANCESTRY_COMPLETION_PATH,
        cfg.ANCESTRY_FORMAL_SUMMARY_PATH,
    ]
    threshold_path = Path(cfg.THRESHOLD_DECLARATION_PATH)
    if threshold_path.exists():
        paths.append(cfg.THRESHOLD_DECLARATION_PATH)
    members: dict[str, str] = {}
    for path_text in paths:
        path = Path(path_text)
        if path.exists():
            members[path_text] = hashlib.sha256(path.read_bytes()).hexdigest()
    return payload_hash(members), members


def write_markdown(result: dict[str, Any], path: Path) -> None:
    aggregate = result["aggregate"]
    equivalence = aggregate["distributed_equivalence"]
    persistence = aggregate["persistence"]
    lines = [
        f"# {cfg.STUDY}",
        "",
        f"Phase: **{result['phase']}**",
        "",
        "## Distributed equivalence",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Cases | {result['case_count']} |",
        f"| Minimum class-only relation accuracy | {equivalence['minimum_class_only_relation_accuracy']:.4f} |",
        f"| Minimum equalized-class accuracy | {equivalence['minimum_equalized_class_relation_accuracy']:.4f} |",
        f"| Minimum member-majority accuracy | {equivalence['minimum_member_majority_relation_accuracy']:.4f} |",
        f"| Minimum effective pair number | {equivalence['minimum_effective_pair_number']:.4f} |",
        f"| Minimum normalized class entropy | {equivalence['minimum_normalized_class_entropy']:.4f} |",
        f"| Minimum class support enrichment | {equivalence['minimum_class_support_enrichment']:.4f} |",
        f"| Cases with >=2 equivalent sectors | {equivalence['fraction_cases_with_two_equivalent_sectors']:.4f} |",
        f"| Minimum permutation accuracy | {equivalence['minimum_permutation_relation_accuracy']:.4f} |",
        f"| Minimum dropout accuracy | {equivalence['minimum_dropout_relation_accuracy']:.4f} |",
        f"| Minimum gain over best single | {equivalence['minimum_gain_over_best_single']:.4f} |",
        f"| Minimum gain over matched outside | {equivalence['minimum_gain_over_matched_outside']:.4f} |",
        "",
        "## Plan continuation",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Class-only complete sequence | {persistence['class_only_complete_sequence']:.4f} |",
        f"| Combined complete sequence | {persistence['class_combined_complete_sequence']:.4f} |",
        f"| Interleaved-sector mean complete sequence | {persistence['mean_interleaved_sector_complete_sequence']:.4f} |",
        f"| Class-ablation sequence gap | {persistence['class_ablation_sequence_gap']:.4f} |",
        f"| Outside-relocation sequence gap | {persistence['outside_relocation_sequence_gap']:.4f} |",
        f"| No-maintenance sequence gap | {persistence['no_maintenance_sequence_gap']:.4f} |",
        "",
        "## Methods",
        "",
        "| Method | Relation accuracy | Full agreement | Complete sequence |",
        "|---|---:|---:|---:|",
    ]
    for method in cfg.METHODS:
        row = aggregate["methods"][method]
        lines.append(
            f"| {method} | {row['relation_accuracy']:.4f} | "
            f"{row['agreement_with_full']:.4f} | "
            f"{row['complete_sequence_identity_rate']:.4f} |"
        )
    if result.get("decision") is not None:
        lines.extend(["", f"Formal status: `{result['decision']['status']}`"])
    lines.extend(
        ["", f"Source manifest: `{result['source_manifest_sha256']}`", ""]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_case_csv(cases: Sequence[dict[str, Any]], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for case in cases:
        rows.append(
            {
                "family_seed": case["family_seed"],
                "held_query": case["held_query"],
                "suite": case["suite"],
                "class_member_count": case["formation"][
                    "equivalence_member_count"
                ],
                "class_mass_fraction": case["formation"][
                    "equivalence_class_mass_fraction"
                ],
                "effective_pair_number": case["formation"][
                    "effective_pair_number"
                ],
                "class_support_enrichment": case["direct_audit"][
                    "unseen_class_support_enrichment"
                ],
                "class_accuracy": case["methods"][cfg.CLASS_ONLY][
                    "relation_accuracy"
                ],
                "member_majority_accuracy": case["methods"][
                    cfg.MEMBER_MAJORITY
                ]["relation_accuracy"],
                "equivalent_sector_count": case["direct_audit"][
                    "equivalent_interleaved_sector_count"
                ],
                "permutation_minimum": case["direct_audit"][
                    "permutation_relation_accuracy_minimum"
                ],
                "dropout_minimum": case["direct_audit"][
                    "dropout_relation_accuracy_minimum"
                ],
                "gain_over_best_single": case["control_gaps"][
                    "gain_over_best_single"
                ],
                "gain_over_matched_outside": case["control_gaps"][
                    "gain_over_matched_outside"
                ],
                "class_ablation_sequence_gap": case["control_gaps"][
                    "class_ablation_sequence_gap"
                ],
                "case_sha256": case["case_sha256"],
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_bank(
    *,
    phase: str,
    family_seeds: Sequence[int],
    output_path: Path,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for family_seed in family_seeds:
        for held_query in range(cfg.QUERY_COUNT):
            for suite in cfg.SUITES:
                cases.append(
                    run_case(family_seed, held_query, dict(suite))
                )
    aggregate = aggregate_cases(cases)
    source_sha, source_members = source_manifest()
    result: dict[str, Any] = {
        "study": cfg.STUDY,
        "target_branch": cfg.TARGET_BRANCH,
        "phase": phase,
        "family_seeds": [int(seed) for seed in family_seeds],
        "query_count": cfg.QUERY_COUNT,
        "suite_names": [suite["name"] for suite in cfg.SUITES],
        "case_count": len(cases),
        "pair_count": PAIR_COUNT,
        "formation_episodes_per_case": (
            (cfg.QUERY_COUNT - 1)
            * cfg.FORMATION_EPISODES_PER_OBSERVED_QUERY
        ),
        "source_manifest_sha256": source_sha,
        "source_manifest_members": source_members,
        "ancestry": ancestry_record(),
        "aggregate": aggregate,
        "cases": cases,
        "decision": None,
    }
    if phase == "holdout":
        result["decision"] = decide(aggregate)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    output_path.with_suffix(".sha256").write_text(
        digest + "  " + output_path.name + "\n",
        encoding="utf-8",
    )
    write_markdown(result, output_path.with_suffix(".md"))
    write_case_csv(
        cases,
        output_path.with_name(output_path.stem + "_cases.csv"),
    )
    return result
