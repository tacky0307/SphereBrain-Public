from __future__ import annotations

"""ECTBF v1C — Distributed Class Formation Mechanism Audit.

The study keeps each temporal member's class-conditional marginal evidence
fixed while selectively removing cross-member covariance. It asks whether the
consequence field is supported by a cooperative joint geometry rather than by
independent interchangeable members.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
import csv
import hashlib
import json
import math

import numpy as np

import endogenous_temporal_consequence_boundary_formation_v1 as base
import distributed_temporal_boundary_equivalence_config_v1 as v1b_cfg
import distributed_temporal_boundary_equivalence_v1 as v1b
import distributed_class_formation_mechanism_audit_config_v1 as cfg

EPS = 1e-12
PAIR_COUNT = base.PAIR_COUNT
PAIR_INDICES = base.PAIR_INDICES


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


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= EPS:
        return np.zeros_like(vector)
    return vector / norm


def projected_subset_fit(
    features: np.ndarray,
    targets: np.ndarray,
    indices: Sequence[int],
) -> np.ndarray:
    selected = np.asarray(sorted(set(int(v) for v in indices)), dtype=int)
    local = np.asarray(features, dtype=float)[:, selected]
    gram = local.T @ local + base.TEMPORAL_RIDGE_LAMBDA * np.eye(selected.size)
    rhs = local.T @ np.asarray(targets, dtype=float)
    row_bound = max(float(np.max(np.sum(np.abs(gram), axis=1))), EPS)
    step = 0.92 / row_bound
    weights = np.zeros(selected.size, dtype=float)
    momentum = weights.copy()
    acceleration = 1.0
    for _ in range(260):
        gradient = gram @ momentum - rhs
        updated = np.maximum(momentum - step * gradient, 0.0)
        next_acceleration = 0.5 * (
            1.0 + math.sqrt(1.0 + 4.0 * acceleration**2)
        )
        momentum = updated + ((acceleration - 1.0) / next_acceleration) * (
            updated - weights
        )
        weights = updated
        acceleration = next_acceleration
    full = np.zeros(features.shape[1], dtype=float)
    full[selected] = weights
    scores = features @ full
    denominator = float(scores @ scores)
    if denominator > EPS:
        full *= max(float(scores @ targets) / denominator, EPS)
    return full


def diagonal_marginal_fit(features: np.ndarray, targets: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    response = np.asarray(targets, dtype=float)
    diagonal = (
        np.sum(matrix * matrix, axis=0)
        + base.TEMPORAL_RIDGE_LAMBDA
        + base.TEMPORAL_SMOOTHNESS_LAMBDA * np.diag(base.PAIR_LAPLACIAN)
    )
    rhs = matrix.T @ response
    weights = np.maximum(rhs / np.maximum(diagonal, EPS), 0.0)
    if float(np.linalg.norm(weights)) <= EPS:
        weights[int(np.argmax(np.abs(rhs)))] = 1.0
    threshold = float(np.quantile(weights, base.WEIGHT_PRUNE_FRACTION))
    weights = np.where(weights > threshold, weights, 0.0)
    scores = matrix @ weights
    denominator = float(scores @ scores)
    if denominator > EPS:
        weights *= max(float(scores @ response) / denominator, EPS)
    return weights


def conditional_independent_shuffle(
    features: np.ndarray,
    targets: np.ndarray,
    query_ids: np.ndarray,
    *,
    seed_parts: Sequence[Any],
) -> np.ndarray:
    """Preserve every member's query/target marginal, destroy joint covariance."""
    matrix = np.asarray(features, dtype=float)
    response = np.asarray(targets, dtype=float)
    queries = np.asarray(query_ids, dtype=int)
    shuffled = matrix.copy()
    rng = np.random.default_rng(base.stable_u32("dcfma-v1-conditional-shuffle", *seed_parts))
    for query in sorted(set(int(v) for v in queries)):
        for target in (-1.0, 1.0):
            rows = np.flatnonzero((queries == query) & (response == target))
            if rows.size < 2:
                continue
            for column in range(matrix.shape[1]):
                shuffled[rows, column] = matrix[rng.permutation(rows), column]
    return shuffled


def conditional_marginal_error(
    original: np.ndarray,
    shuffled: np.ndarray,
    targets: np.ndarray,
    query_ids: np.ndarray,
) -> float:
    maximum = 0.0
    for query in sorted(set(int(v) for v in query_ids)):
        for target in (-1.0, 1.0):
            rows = (query_ids == query) & (targets == target)
            if not np.any(rows):
                continue
            maximum = max(
                maximum,
                float(
                    np.max(
                        np.abs(
                            np.mean(original[rows], axis=0)
                            - np.mean(shuffled[rows], axis=0)
                        )
                    )
                ),
            )
    return maximum


def offdiagonal_covariance_change(original: np.ndarray, shuffled: np.ndarray) -> float:
    first = np.cov(np.asarray(original, dtype=float), rowvar=False)
    second = np.cov(np.asarray(shuffled, dtype=float), rowvar=False)
    first = first - np.diag(np.diag(first))
    second = second - np.diag(np.diag(second))
    return float(np.linalg.norm(first - second) / max(np.linalg.norm(first), EPS))


def marginal_subset_indices(
    features: np.ndarray,
    targets: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(features, dtype=float)
    response = np.asarray(targets, dtype=float)
    scale = np.std(matrix, axis=0) + 1e-6
    score = np.abs(np.mean(response[:, None] * matrix, axis=0) / scale)
    indices = np.argsort(score, kind="stable")[-int(count):]
    return np.asarray(sorted(int(v) for v in indices), dtype=int), score


def cooperative_subset_indices(
    features: np.ndarray,
    full_weights: np.ndarray,
    count: int,
) -> tuple[np.ndarray, float]:
    """Greedily preserve full-field activity using residual covariance."""
    matrix = np.asarray(features, dtype=float)
    target = matrix @ np.asarray(full_weights, dtype=float)
    residual = target.copy()
    active = np.flatnonzero(np.asarray(full_weights, dtype=float) > 0.0)
    if active.size < count:
        active = np.arange(matrix.shape[1], dtype=int)
    columns = matrix[:, active]
    norms = np.linalg.norm(columns, axis=0) + EPS
    selected: list[int] = []
    for _ in range(min(int(count), int(active.size))):
        correlation = np.abs(columns.T @ residual) / norms
        if selected:
            correlation[np.isin(active, np.asarray(selected, dtype=int))] = -1.0
        chosen = int(active[int(np.argmax(correlation))])
        selected.append(chosen)
        local = matrix[:, selected]
        coefficients = np.linalg.lstsq(local, target, rcond=None)[0]
        residual = target - local @ coefficients
    r2 = 1.0 - float(residual @ residual) / max(float(target @ target), EPS)
    return np.asarray(sorted(selected), dtype=int), float(r2)


def query_factorized_field(
    features: np.ndarray,
    targets: np.ndarray,
    query_ids: np.ndarray,
    observed_queries: Sequence[int],
    reference_norm: float,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    fields: dict[int, np.ndarray] = {}
    normalized: list[np.ndarray] = []
    for query in observed_queries:
        rows = query_ids == int(query)
        field = base.TemporalBoundaryLearner._projected_ridge(
            features[rows], targets[rows]
        )
        fields[int(query)] = field.copy()
        normalized.append(normalize(field))
    combined = np.mean(np.stack(normalized, axis=0), axis=0)
    combined = normalize(combined) * max(float(reference_norm), 1.0)
    return combined, fields


@dataclass(frozen=True)
class MechanismSnapshot:
    version: str
    pair_count: int
    pair_vocabulary_hash: str
    fields: dict[str, tuple[float, ...]]
    cooperative_indices: tuple[int, ...]
    marginal_indices: tuple[int, ...]
    relation_codes: tuple[tuple[float, ...], ...]
    maintenance_basin: tuple[tuple[float, ...], ...]
    formation_hash: str
    substrate_hash: str

    def payload_without_hash(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "pair_count": self.pair_count,
            "pair_vocabulary_hash": self.pair_vocabulary_hash,
            "fields": {key: list(value) for key, value in sorted(self.fields.items())},
            "cooperative_indices": list(self.cooperative_indices),
            "marginal_indices": list(self.marginal_indices),
            "relation_codes": [list(row) for row in self.relation_codes],
            "maintenance_basin": [list(row) for row in self.maintenance_basin],
            "formation_hash": self.formation_hash,
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self.payload_without_hash(), "substrate_hash": self.substrate_hash}

    def to_json(self) -> str:
        return canonical_json(self.to_payload())

    @classmethod
    def from_json(cls, text: str) -> "MechanismSnapshot":
        payload = json.loads(text)
        snapshot = cls(
            version=str(payload["version"]),
            pair_count=int(payload["pair_count"]),
            pair_vocabulary_hash=str(payload["pair_vocabulary_hash"]),
            fields={
                str(key): tuple(float(v) for v in value)
                for key, value in payload["fields"].items()
            },
            cooperative_indices=tuple(int(v) for v in payload["cooperative_indices"]),
            marginal_indices=tuple(int(v) for v in payload["marginal_indices"]),
            relation_codes=tuple(
                tuple(float(v) for v in row) for row in payload["relation_codes"]
            ),
            maintenance_basin=tuple(
                tuple(float(v) for v in row)
                for row in payload["maintenance_basin"]
            ),
            formation_hash=str(payload["formation_hash"]),
            substrate_hash=str(payload["substrate_hash"]),
        )
        if snapshot.pair_count != PAIR_COUNT:
            raise AssertionError("snapshot pair count mismatch")
        if snapshot.pair_vocabulary_hash != payload_hash(PAIR_INDICES):
            raise AssertionError("snapshot pair vocabulary mismatch")
        if len(snapshot.cooperative_indices) != cfg.COOPERATIVE_SUBSET_SIZE:
            raise AssertionError("cooperative subset size mismatch")
        if len(snapshot.marginal_indices) != cfg.COOPERATIVE_SUBSET_SIZE:
            raise AssertionError("marginal subset size mismatch")
        if payload_hash(snapshot.payload_without_hash()) != snapshot.substrate_hash:
            raise AssertionError("snapshot hash mismatch")
        return snapshot


def snapshot_forbidden_token_count(snapshot: MechanismSnapshot) -> int:
    text = snapshot.to_json().lower()
    return sum(text.count(token.lower()) for token in cfg.FORBIDDEN_SNAPSHOT_TOKENS)


def collect_formation(
    family_seed: int,
    held_query: int,
    suite: dict[str, Any],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    base.ContinuousConsequenceWorld,
    dict[str, Any],
]:
    world = base.ContinuousConsequenceWorld(family_seed)
    vault = base.DelayedContinuationVault()
    pending: dict[int, tuple[np.ndarray, int, int, str]] = {}
    features: list[np.ndarray] = []
    targets: list[float] = []
    query_ids: list[int] = []
    identities: list[str] = []
    step = 0
    observed_queries = [query for query in range(cfg.QUERY_COUNT) if query != held_query]
    for query in observed_queries:
        for episode in range(cfg.FORMATION_EPISODES_PER_OBSERVED_QUERY):
            identity = (
                f"dcfma-formation:f{family_seed}:h{held_query}:q{query}:"
                f"s{suite['name']}:e{episode}"
            )
            stream, metadata = world.generate(
                query, identity, suite, episode_index=episode
            )
            vector = base.temporal_pair_features(stream)
            action = int(
                base.stable_u32(
                    "dcfma-v1-action",
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
            pending[token] = (vector, action, query, identity)
            identities.append(identity)
            for released_token, scalar_result in vault.release_ready(step):
                released_vector, released_action, released_query, _identity = pending.pop(
                    released_token
                )
                features.append(released_vector.copy())
                targets.append(
                    base.relation_signed_target(released_action, scalar_result)
                )
                query_ids.append(released_query)
            step += 1
    flush_step = step + base.FORMATION_CREDIT_DELAY
    while pending:
        released = vault.release_ready(flush_step)
        if not released:
            flush_step += 1
            continue
        for released_token, scalar_result in released:
            released_vector, released_action, released_query, _identity = pending.pop(
                released_token
            )
            features.append(released_vector.copy())
            targets.append(base.relation_signed_target(released_action, scalar_result))
            query_ids.append(released_query)
    report = {
        "observed_queries": observed_queries,
        "formation_episode_count": len(features),
        "formation_identity_count": len(set(identities)),
        "formation_identity_hash": payload_hash(sorted(set(identities))),
        "feedback_audit": vault.audit(),
        "held_query_updates": 0,
        "correct_routes_supplied": 0,
        "hidden_windows_read_during_formation": 0,
    }
    return (
        np.asarray(features, dtype=float),
        np.asarray(targets, dtype=float),
        np.asarray(query_ids, dtype=int),
        world,
        report,
    )


def form_mechanism_snapshot(
    family_seed: int,
    held_query: int,
    suite: dict[str, Any],
) -> tuple[
    MechanismSnapshot,
    base.ContinuousConsequenceWorld,
    dict[str, Any],
    dict[int, np.ndarray],
]:
    features, targets, query_ids, world, report = collect_formation(
        family_seed, held_query, suite
    )
    full = base.TemporalBoundaryLearner._projected_ridge(features, targets)
    shuffled_features = conditional_independent_shuffle(
        features,
        targets,
        query_ids,
        seed_parts=(family_seed, held_query, suite["name"]),
    )
    conditional_shuffled = base.TemporalBoundaryLearner._projected_ridge(
        shuffled_features, targets
    )
    diagonal = diagonal_marginal_fit(features, targets)
    smooth_null = base.TemporalBoundaryLearner._projected_ridge(
        features,
        np.roll(targets, max(1, len(targets) // 3)),
    )
    marginal_indices, marginal_scores = marginal_subset_indices(
        features, targets, cfg.COOPERATIVE_SUBSET_SIZE
    )
    cooperative_indices, cooperative_r2 = cooperative_subset_indices(
        features, full, cfg.COOPERATIVE_SUBSET_SIZE
    )
    marginal = projected_subset_fit(features, targets, marginal_indices)
    cooperative = projected_subset_fit(features, targets, cooperative_indices)
    cooperative_destroyed = projected_subset_fit(
        shuffled_features, targets, cooperative_indices
    )
    query_factorized, query_fields = query_factorized_field(
        features,
        targets,
        query_ids,
        report["observed_queries"],
        float(np.linalg.norm(full)),
    )
    ablated = np.zeros(PAIR_COUNT, dtype=float)

    fields = {
        cfg.FULL: full,
        cfg.COOPERATIVE: cooperative,
        cfg.MARGINAL: marginal,
        cfg.CONDITIONAL_SHUFFLED: conditional_shuffled,
        cfg.DIAGONAL: diagonal,
        cfg.COOPERATIVE_DESTROYED: cooperative_destroyed,
        cfg.QUERY_FACTORIZED: query_factorized,
        cfg.SMOOTH_NULL: smooth_null,
        cfg.FIELD_ABLATED: ablated,
        cfg.COOPERATIVE_DELAY: cooperative,
        cfg.COOPERATIVE_PARTIAL: cooperative,
        cfg.COOPERATIVE_NOISE: cooperative,
        cfg.COOPERATIVE_COMBINED: cooperative,
        cfg.COOPERATIVE_NO_MAINTENANCE: cooperative,
    }

    code_zero = base.unit_code(
        "dcfma-v1-relation-state",
        family_seed,
        held_query,
        suite["name"],
        0,
        dim=base.RELATION_STATE_DIM,
    )
    code_one_seed = base.unit_code(
        "dcfma-v1-relation-state",
        family_seed,
        held_query,
        suite["name"],
        1,
        dim=base.RELATION_STATE_DIM,
    )
    code_one = normalize(code_one_seed - float(code_zero @ code_one_seed) * code_zero)
    codes = np.stack([code_zero, code_one], axis=0)
    basin = codes.T @ codes

    marginal_error = conditional_marginal_error(
        features, shuffled_features, targets, query_ids
    )
    covariance_change = offdiagonal_covariance_change(features, shuffled_features)
    formation_hash = payload_hash(
        {
            "family_seed": family_seed,
            "held_query": held_query,
            "suite": suite["name"],
            "formation_identity_hash": report["formation_identity_hash"],
            "feedback": report["feedback_audit"],
            "conditional_marginal_error": marginal_error,
            "covariance_change": covariance_change,
            "cooperative_indices": cooperative_indices.tolist(),
            "marginal_indices": marginal_indices.tolist(),
        }
    )
    provisional = MechanismSnapshot(
        version="ectbf-v1c-dcfma-v1",
        pair_count=PAIR_COUNT,
        pair_vocabulary_hash=payload_hash(PAIR_INDICES),
        fields={
            key: tuple(float(v) for v in value)
            for key, value in fields.items()
        },
        cooperative_indices=tuple(int(v) for v in cooperative_indices),
        marginal_indices=tuple(int(v) for v in marginal_indices),
        relation_codes=tuple(tuple(float(v) for v in row) for row in codes),
        maintenance_basin=tuple(tuple(float(v) for v in row) for row in basin),
        formation_hash=formation_hash,
        substrate_hash="",
    )
    snapshot = MechanismSnapshot(
        **{
            **provisional.__dict__,
            "substrate_hash": payload_hash(provisional.payload_without_hash()),
        }
    )
    restored = MechanismSnapshot.from_json(snapshot.to_json())

    full_score = features @ full
    marginal_score = features @ marginal
    cooperative_score = features @ cooperative
    report.update(
        {
            "initial_field_norm": 0.0,
            "full_field_norm": float(np.linalg.norm(full)),
            "cooperative_field_norm": float(np.linalg.norm(cooperative)),
            "marginal_field_norm": float(np.linalg.norm(marginal)),
            "cooperative_subset_size": len(cooperative_indices),
            "marginal_subset_size": len(marginal_indices),
            "subset_sizes_equal": len(cooperative_indices) == len(marginal_indices),
            "cooperative_subset_r2": cooperative_r2,
            "marginal_full_score_r2": 1.0
            - float(np.sum((full_score - marginal_score) ** 2))
            / max(float(np.sum(full_score**2)), EPS),
            "cooperative_full_score_r2": 1.0
            - float(np.sum((full_score - cooperative_score) ** 2))
            / max(float(np.sum(full_score**2)), EPS),
            "conditional_marginal_preservation_error": marginal_error,
            "offdiagonal_covariance_change_ratio": covariance_change,
            "conditional_target_alignment_error": float(
                np.max(
                    np.abs(features.T @ targets - shuffled_features.T @ targets)
                )
            ),
            "snapshot_roundtrip_exact": restored.to_json() == snapshot.to_json(),
            "snapshot_forbidden_token_count": snapshot_forbidden_token_count(snapshot),
            "all_fields_nonzero_except_ablation": all(
                np.linalg.norm(value) > 0.0
                for key, value in fields.items()
                if key != cfg.FIELD_ABLATED
            ),
            "query_factorized_field_count": len(query_fields),
            "marginal_score_hash": payload_hash(np.round(marginal_scores, 14).tolist()),
        }
    )
    return snapshot, world, report, query_fields


def method_variant(method: str) -> str:
    if method == cfg.COOPERATIVE_PARTIAL:
        return "partial"
    if method == cfg.COOPERATIVE_NOISE:
        return "noise"
    if method in (cfg.COOPERATIVE_COMBINED, cfg.COOPERATIVE_NO_MAINTENANCE):
        return "combined"
    return "clean"


def method_delay(method: str) -> tuple[bool, bool, float]:
    if method == cfg.COOPERATIVE_DELAY:
        return True, True, base.RELATION_DELAY_DRIFT
    if method == cfg.COOPERATIVE_NO_MAINTENANCE:
        return True, False, base.NO_MAINTENANCE_DRIFT
    return False, True, base.RELATION_DELAY_DRIFT


def predict(
    features: np.ndarray,
    weights: np.ndarray,
    snapshot: MechanismSnapshot,
    *,
    delay: bool,
    maintain: bool,
    state_noise: float,
    seed_parts: Sequence[Any],
) -> tuple[int, float]:
    score = float(np.asarray(weights) @ np.asarray(features))
    initial = int(score >= 0.0)
    if not delay:
        return initial, abs(score)
    codes = np.asarray(snapshot.relation_codes, dtype=float)
    basin = np.asarray(snapshot.maintenance_basin, dtype=float)
    rng = np.random.default_rng(base.stable_u32("dcfma-v1-delay", *seed_parts))
    state = codes[initial].copy()
    for _ in range(base.RELATION_DELAY_STEPS):
        state = (
            base.RELATION_DELAY_RETENTION * state
            + state_noise
            * rng.normal(size=base.RELATION_STATE_DIM)
            / math.sqrt(float(base.RELATION_STATE_DIM))
        )
        if maintain:
            state = normalize(0.22 * state + 0.78 * (basin @ state))
    scores = codes @ state
    return int(np.argmax(scores)), float(np.max(scores) - np.min(scores))


def evaluate_direct(
    family_seed: int,
    held_query: int,
    suite: dict[str, Any],
    world: base.ContinuousConsequenceWorld,
    snapshot: MechanismSnapshot,
    query_fields: dict[int, np.ndarray],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    fields = {key: np.asarray(value, dtype=float) for key, value in snapshot.fields.items()}
    correct = {method: 0 for method in cfg.METHODS}
    margins = {method: [] for method in cfg.METHODS}
    query_correct = {query: 0 for query in query_fields}
    identities: list[str] = []
    decisions_before_truth = 0
    for episode in range(cfg.DIRECT_TEST_EPISODES):
        identity = (
            f"dcfma-evaluation-direct:f{family_seed}:h{held_query}:"
            f"s{suite['name']}:e{episode}"
        )
        stream, metadata = world.generate(
            held_query, identity, suite, episode_index=episode
        )
        identities.append(identity)
        variants = {"clean": stream}
        for variant in ("partial", "noise", "combined"):
            variants[variant] = base.perturb_stream(
                stream,
                variant,
                seed_parts=(
                    "dcfma-v1",
                    family_seed,
                    held_query,
                    suite["name"],
                    episode,
                    variant,
                ),
            )
        feature_cache = {
            variant: base.temporal_pair_features(value)
            for variant, value in variants.items()
        }
        predictions: dict[str, tuple[int, float]] = {}
        for method in cfg.METHODS:
            variant = method_variant(method)
            delay, maintain, noise = method_delay(method)
            predictions[method] = predict(
                feature_cache[variant],
                fields[method],
                snapshot,
                delay=delay,
                maintain=maintain,
                state_noise=noise,
                seed_parts=(family_seed, held_query, suite["name"], episode, method),
            )
        for query, field in query_fields.items():
            query_correct[int(query)] += int(
                int(float(field @ feature_cache["clean"]) >= 0.0)
                == metadata.relation_channel
            )
        decisions_before_truth += 1
        truth = metadata.relation_channel
        for method, (prediction, margin) in predictions.items():
            correct[method] += int(prediction == truth)
            margins[method].append(float(margin))
    methods = {
        method: {
            "relation_accuracy": correct[method] / float(cfg.DIRECT_TEST_EPISODES),
            "mean_margin": float(np.mean(margins[method])),
            "minimum_margin": float(np.min(margins[method])),
        }
        for method in cfg.METHODS
    }
    audit = {
        "evaluation_episode_count": cfg.DIRECT_TEST_EPISODES,
        "evaluation_identity_count": len(set(identities)),
        "evaluation_identity_hash": payload_hash(sorted(set(identities))),
        "decisions_before_truth_reads": decisions_before_truth,
        "query_transfer_accuracy": {
            str(query): query_correct[query] / float(cfg.DIRECT_TEST_EPISODES)
            for query in sorted(query_correct)
        },
        "held_query_updates": 0,
    }
    return methods, audit


def evaluate_plan(
    family_seed: int,
    held_query: int,
    suite: dict[str, Any],
    world: base.ContinuousConsequenceWorld,
    snapshot: MechanismSnapshot,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    fields = {key: np.asarray(value, dtype=float) for key, value in snapshot.fields.items()}
    complete = {method: 0 for method in cfg.METHODS}
    step_correct = {method: 0 for method in cfg.METHODS}
    identities: list[str] = []
    for episode in range(cfg.PLAN_EPISODES):
        streams: list[np.ndarray] = []
        truths: list[int] = []
        for depth in range(cfg.PLAN_DEPTH):
            identity = (
                f"dcfma-evaluation-plan:f{family_seed}:h{held_query}:"
                f"s{suite['name']}:e{episode}:d{depth}"
            )
            stream, metadata = world.generate(
                held_query,
                identity,
                suite,
                episode_index=episode * cfg.PLAN_DEPTH + depth,
            )
            identities.append(identity)
            streams.append(stream)
            truths.append(metadata.relation_channel)
        for method in cfg.METHODS:
            sequence: list[int] = []
            for depth, stream in enumerate(streams):
                variant = method_variant(method)
                perturbed = base.perturb_stream(
                    stream,
                    variant,
                    seed_parts=(
                        "dcfma-v1-plan",
                        family_seed,
                        held_query,
                        suite["name"],
                        episode,
                        depth,
                        method,
                    ),
                )
                features = base.temporal_pair_features(perturbed)
                delay, maintain, noise = method_delay(method)
                prediction, _ = predict(
                    features,
                    fields[method],
                    snapshot,
                    delay=delay,
                    maintain=maintain,
                    state_noise=noise,
                    seed_parts=(
                        family_seed,
                        held_query,
                        suite["name"],
                        episode,
                        depth,
                        method,
                    ),
                )
                sequence.append(prediction)
            step_correct[method] += sum(
                int(prediction == truth)
                for prediction, truth in zip(sequence, truths)
            )
            complete[method] += int(sequence == truths)
    total_steps = cfg.PLAN_EPISODES * cfg.PLAN_DEPTH
    methods = {
        method: {
            "step_accuracy": step_correct[method] / float(total_steps),
            "complete_sequence_identity_rate": complete[method]
            / float(cfg.PLAN_EPISODES),
        }
        for method in cfg.METHODS
    }
    audit = {
        "plan_episode_count": cfg.PLAN_EPISODES,
        "plan_depth": cfg.PLAN_DEPTH,
        "plan_identity_count": len(set(identities)),
        "plan_identity_hash": payload_hash(sorted(set(identities))),
        "runtime_planning_calls": 0,
    }
    return methods, audit


def run_case(family_seed: int, held_query: int, suite: dict[str, Any]) -> dict[str, Any]:
    snapshot, world, formation, query_fields = form_mechanism_snapshot(
        family_seed, held_query, suite
    )
    restored = MechanismSnapshot.from_json(snapshot.to_json())
    direct, direct_audit = evaluate_direct(
        family_seed, held_query, suite, world, restored, query_fields
    )
    plan, plan_audit = evaluate_plan(
        family_seed, held_query, suite, world, restored
    )
    methods = {
        method: {**direct[method], **plan[method]}
        for method in cfg.METHODS
    }
    gains = {
        "full_over_conditional_shuffle": (
            methods[cfg.FULL]["relation_accuracy"]
            - methods[cfg.CONDITIONAL_SHUFFLED]["relation_accuracy"]
        ),
        "full_over_diagonal": (
            methods[cfg.FULL]["relation_accuracy"]
            - methods[cfg.DIAGONAL]["relation_accuracy"]
        ),
        "cooperative_over_marginal": (
            methods[cfg.COOPERATIVE]["relation_accuracy"]
            - methods[cfg.MARGINAL]["relation_accuracy"]
        ),
        "cooperative_over_destroyed": (
            methods[cfg.COOPERATIVE]["relation_accuracy"]
            - methods[cfg.COOPERATIVE_DESTROYED]["relation_accuracy"]
        ),
        "full_over_smooth_null": (
            methods[cfg.FULL]["relation_accuracy"]
            - methods[cfg.SMOOTH_NULL]["relation_accuracy"]
        ),
        "field_ablation_sequence_gap": (
            methods[cfg.FULL]["complete_sequence_identity_rate"]
            - methods[cfg.FIELD_ABLATED]["complete_sequence_identity_rate"]
        ),
        "no_maintenance_sequence_gap": (
            methods[cfg.COOPERATIVE_COMBINED]["complete_sequence_identity_rate"]
            - methods[cfg.COOPERATIVE_NO_MAINTENANCE][
                "complete_sequence_identity_rate"
            ]
        ),
    }
    information_boundary = {
        "continuous_stream_only": True,
        "true_boundary_reads": 0,
        "true_relation_reads_before_decision": 0,
        "held_query_formation_updates": formation["held_query_updates"],
        "correct_routes_supplied": formation["correct_routes_supplied"],
        "hidden_windows_read_during_formation": formation[
            "hidden_windows_read_during_formation"
        ],
        "runtime_planning_calls": plan_audit["runtime_planning_calls"],
    }
    control_identity = {
        "subset_sizes_equal": formation["subset_sizes_equal"],
        "conditional_target_alignment_preserved": (
            formation["conditional_target_alignment_error"] <= 1e-10
        ),
        "snapshot_roundtrip_exact": formation["snapshot_roundtrip_exact"],
        "snapshot_forbidden_tokens_zero": (
            formation["snapshot_forbidden_token_count"] == 0
        ),
        "formation_direct_identities_disjoint": (
            formation["formation_identity_hash"]
            != direct_audit["evaluation_identity_hash"]
        ),
        "formation_plan_identities_disjoint": (
            formation["formation_identity_hash"] != plan_audit["plan_identity_hash"]
        ),
        "direct_plan_identities_disjoint": (
            direct_audit["evaluation_identity_hash"] != plan_audit["plan_identity_hash"]
        ),
    }
    case = {
        "family_seed": int(family_seed),
        "held_query": int(held_query),
        "suite": suite["name"],
        "methods": methods,
        "gains": gains,
        "formation": formation,
        "direct_audit": direct_audit,
        "plan_audit": plan_audit,
        "information_boundary": information_boundary,
        "control_identity": control_identity,
        "snapshot_sha256": hashlib.sha256(snapshot.to_json().encode("utf-8")).hexdigest(),
    }
    case["case_sha256"] = payload_hash(case)
    return case


def _mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return float(np.mean(rows)) if rows else 0.0


def _minimum(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return float(np.min(rows)) if rows else 0.0


def _maximum(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return float(np.max(rows)) if rows else 0.0


def bootstrap_interval(values: Sequence[float], namespace: str) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(base.stable_u32("dcfma-v1-bootstrap", namespace))
    estimates = np.empty(cfg.BOOTSTRAP_REPLICATES, dtype=float)
    for index in range(cfg.BOOTSTRAP_REPLICATES):
        sample = rng.integers(0, len(array), size=len(array))
        estimates[index] = float(np.mean(array[sample]))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


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
    paired: dict[str, dict[str, float]] = {}
    for key in (
        "full_over_conditional_shuffle",
        "full_over_diagonal",
        "cooperative_over_marginal",
        "cooperative_over_destroyed",
    ):
        values = [float(case["gains"][key]) for case in cases]
        low, high = bootstrap_interval(values, key)
        paired[key] = {
            "mean": float(np.mean(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "positive_fraction": float(np.mean(np.asarray(values) > 0.0)),
            "nonnegative_fraction": float(np.mean(np.asarray(values) >= 0.0)),
            "bootstrap_low": low,
            "bootstrap_high": high,
        }
    transfer_values = [
        value
        for case in cases
        for value in case["direct_audit"]["query_transfer_accuracy"].values()
    ]
    mechanism = {
        "case_count": len(cases),
        "maximum_conditional_marginal_preservation_error": _maximum(
            case["formation"]["conditional_marginal_preservation_error"]
            for case in cases
        ),
        "maximum_conditional_target_alignment_error": _maximum(
            case["formation"]["conditional_target_alignment_error"]
            for case in cases
        ),
        "minimum_covariance_destruction_ratio": _minimum(
            case["formation"]["offdiagonal_covariance_change_ratio"]
            for case in cases
        ),
        "mean_covariance_destruction_ratio": _mean(
            case["formation"]["offdiagonal_covariance_change_ratio"]
            for case in cases
        ),
        "minimum_cooperative_formation_r2": _minimum(
            case["formation"]["cooperative_subset_r2"] for case in cases
        ),
        "minimum_cooperative_full_score_r2": _minimum(
            case["formation"]["cooperative_full_score_r2"] for case in cases
        ),
        "maximum_marginal_full_score_r2": _maximum(
            case["formation"]["marginal_full_score_r2"] for case in cases
        ),
        "all_subset_sizes_equal": all(
            case["formation"]["subset_sizes_equal"] for case in cases
        ),
        "all_initial_fields_zero": all(
            case["formation"]["initial_field_norm"] == 0.0 for case in cases
        ),
        "all_fields_nonzero_except_ablation": all(
            case["formation"]["all_fields_nonzero_except_ablation"]
            for case in cases
        ),
        "all_feedback_delayed": all(
            case["formation"]["feedback_audit"][
                "all_results_after_required_delay"
            ]
            for case in cases
        ),
        "all_held_query_updates_zero": all(
            case["formation"]["held_query_updates"] == 0 for case in cases
        ),
        "all_snapshot_roundtrips_exact": all(
            case["formation"]["snapshot_roundtrip_exact"] for case in cases
        ),
        "maximum_snapshot_forbidden_token_count": int(
            max(case["formation"]["snapshot_forbidden_token_count"] for case in cases)
        ),
        "mean_query_transfer_accuracy": float(np.mean(transfer_values)),
        "minimum_query_transfer_accuracy": float(np.min(transfer_values)),
    }
    causal = {
        "minimum_gain_full_over_smooth_null": _minimum(
            case["gains"]["full_over_smooth_null"] for case in cases
        ),
        "mean_gain_full_over_smooth_null": _mean(
            case["gains"]["full_over_smooth_null"] for case in cases
        ),
        "field_ablation_sequence_gap": (
            methods[cfg.FULL]["complete_sequence_identity_rate"]
            - methods[cfg.FIELD_ABLATED]["complete_sequence_identity_rate"]
        ),
        "no_maintenance_sequence_gap": (
            methods[cfg.COOPERATIVE_COMBINED]["complete_sequence_identity_rate"]
            - methods[cfg.COOPERATIVE_NO_MAINTENANCE][
                "complete_sequence_identity_rate"
            ]
        ),
    }
    audits = {
        "all_information_boundaries_pass": all(
            all(
                (
                    value is True
                    if isinstance(value, bool)
                    else value == 0
                )
                for value in case["information_boundary"].values()
            )
            for case in cases
        ),
        "all_control_identity_pass": all(
            all(bool(value) for value in case["control_identity"].values())
            for case in cases
        ),
    }
    return {
        "methods": methods,
        "paired": paired,
        "mechanism": mechanism,
        "causal": causal,
        "audits": audits,
    }


def decide(aggregate: dict[str, Any]) -> dict[str, Any]:
    thresholds = cfg.FROZEN_THRESHOLDS
    if thresholds is None:
        raise RuntimeError("formal decision requires frozen thresholds")
    methods = aggregate["methods"]
    paired = aggregate["paired"]
    mechanism = aggregate["mechanism"]
    causal = aggregate["causal"]
    audit_checks = {
        "information_boundary": aggregate["audits"]["all_information_boundaries_pass"],
        "control_identity": aggregate["audits"]["all_control_identity_pass"],
        "subset_sizes_equal": mechanism["all_subset_sizes_equal"],
        "initial_fields_zero": mechanism["all_initial_fields_zero"],
        "fields_nonzero": mechanism["all_fields_nonzero_except_ablation"],
        "feedback_delayed": mechanism["all_feedback_delayed"],
        "held_query_updates_zero": mechanism["all_held_query_updates_zero"],
        "snapshot_roundtrip": mechanism["all_snapshot_roundtrips_exact"],
        "snapshot_forbidden_zero": mechanism["maximum_snapshot_forbidden_token_count"] == 0,
        "conditional_marginals_preserved": mechanism[
            "maximum_conditional_marginal_preservation_error"
        ] <= thresholds["maximum_conditional_marginal_preservation_error"],
    }
    functional_checks = {
        "minimum_full_relation_accuracy": methods[cfg.FULL][
            "minimum_relation_accuracy"
        ] >= thresholds["minimum_full_relation_accuracy"],
        "minimum_cooperative_relation_accuracy": methods[cfg.COOPERATIVE][
            "minimum_relation_accuracy"
        ] >= thresholds["minimum_cooperative_relation_accuracy"],
        "minimum_robust_cooperative_relation_accuracy": min(
            methods[method]["minimum_relation_accuracy"]
            for method in cfg.ROBUST_COOPERATIVE_METHODS
        ) >= thresholds["minimum_robust_cooperative_relation_accuracy"],
        "minimum_cooperative_complete_sequence": methods[cfg.COOPERATIVE][
            "complete_sequence_identity_rate"
        ] >= thresholds["minimum_cooperative_complete_sequence"],
        "minimum_combined_complete_sequence": methods[cfg.COOPERATIVE_COMBINED][
            "complete_sequence_identity_rate"
        ] >= thresholds["minimum_combined_complete_sequence"],
    }
    cooperative_checks = {
        "mean_full_over_conditional_shuffle": paired[
            "full_over_conditional_shuffle"
        ]["mean"] >= thresholds["minimum_mean_gain_full_over_conditional_shuffle"],
        "bootstrap_full_over_conditional_shuffle": paired[
            "full_over_conditional_shuffle"
        ]["bootstrap_low"] >= thresholds[
            "minimum_bootstrap_low_full_over_conditional_shuffle"
        ],
        "mean_full_over_diagonal": paired["full_over_diagonal"]["mean"]
        >= thresholds["minimum_mean_gain_full_over_diagonal"],
        "bootstrap_full_over_diagonal": paired["full_over_diagonal"][
            "bootstrap_low"
        ] >= thresholds["minimum_bootstrap_low_full_over_diagonal"],
        "mean_cooperative_over_marginal": paired[
            "cooperative_over_marginal"
        ]["mean"] >= thresholds["minimum_mean_gain_cooperative_over_marginal"],
        "bootstrap_cooperative_over_marginal": paired[
            "cooperative_over_marginal"
        ]["bootstrap_low"] >= thresholds[
            "minimum_bootstrap_low_cooperative_over_marginal"
        ],
        "mean_cooperative_over_destroyed": paired[
            "cooperative_over_destroyed"
        ]["mean"] >= thresholds["minimum_mean_gain_cooperative_over_destroyed"],
        "bootstrap_cooperative_over_destroyed": paired[
            "cooperative_over_destroyed"
        ]["bootstrap_low"] >= thresholds[
            "minimum_bootstrap_low_cooperative_over_destroyed"
        ],
        "positive_fraction_cooperative_over_marginal": paired[
            "cooperative_over_marginal"
        ]["positive_fraction"] >= thresholds[
            "minimum_positive_fraction_cooperative_over_marginal"
        ],
        "positive_fraction_cooperative_over_destroyed": paired[
            "cooperative_over_destroyed"
        ]["positive_fraction"] >= thresholds[
            "minimum_positive_fraction_cooperative_over_destroyed"
        ],
        "covariance_destroyed": mechanism["minimum_covariance_destruction_ratio"]
        >= thresholds["minimum_covariance_destruction_ratio"],
        "cooperative_formation_r2": mechanism["minimum_cooperative_formation_r2"]
        >= thresholds["minimum_cooperative_formation_r2"],
    }
    causal_checks = {
        "full_over_smooth_null": causal["minimum_gain_full_over_smooth_null"]
        >= thresholds["minimum_gain_full_over_smooth_null"],
        "field_ablation_sequence_gap": causal["field_ablation_sequence_gap"]
        >= thresholds["minimum_field_ablation_sequence_gap"],
        "no_maintenance_sequence_gap": causal["no_maintenance_sequence_gap"]
        >= thresholds["minimum_no_maintenance_sequence_gap"],
    }
    audit_pass = all(audit_checks.values())
    functional_pass = all(functional_checks.values())
    cooperative_pass = all(cooperative_checks.values())
    causal_pass = all(causal_checks.values())
    if not audit_pass:
        status = cfg.STATUS_INVALID
    elif functional_pass and cooperative_pass and causal_pass:
        status = cfg.STATUS_POSITIVE
    elif functional_pass and not cooperative_pass:
        status = cfg.STATUS_MARGINAL_SUFFICIENT
    elif cooperative_pass and not functional_pass:
        status = cfg.STATUS_COOPERATIVE_NO_PLAN
    else:
        status = cfg.STATUS_INCONCLUSIVE
    return {
        "status": status,
        "candidate_promoted": status == cfg.STATUS_POSITIVE,
        "all_audits_pass": audit_pass,
        "functional_pass": functional_pass,
        "cooperative_pass": cooperative_pass,
        "causal_pass": causal_pass,
        "audit_checks": audit_checks,
        "functional_checks": functional_checks,
        "cooperative_checks": cooperative_checks,
        "causal_checks": causal_checks,
    }


def ancestry_record() -> dict[str, Any]:
    completion_path = Path(cfg.ANCESTRY_COMPLETION_PATH)
    summary_path = Path(cfg.ANCESTRY_FORMAL_SUMMARY_PATH)
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not completion.get("pipeline_complete"):
        raise RuntimeError("v1B ancestry pipeline is incomplete")
    if completion.get("formal_result_sha256") != summary.get("formal_result_sha256"):
        raise RuntimeError("v1B ancestry result hash mismatch")
    return {
        "branch": cfg.ANCESTRY_BRANCH,
        "completion_receipt": {
            "path": cfg.ANCESTRY_COMPLETION_PATH,
            "sha256": sha256_path(completion_path),
            "formal_status": completion.get("formal_status"),
            "candidate_promoted": completion.get("candidate_promoted"),
            "formal_result_sha256": completion.get("formal_result_sha256"),
            "pipeline_complete": completion.get("pipeline_complete"),
        },
        "formal_summary": {
            "path": cfg.ANCESTRY_FORMAL_SUMMARY_PATH,
            "sha256": sha256_path(summary_path),
            "formal_status": summary.get("formal_status"),
            "candidate_promoted": summary.get("candidate_promoted"),
            "formal_result_sha256": summary.get("formal_result_sha256"),
        },
    }


def source_manifest() -> tuple[str, dict[str, str]]:
    members = [
        "distributed_class_formation_mechanism_audit_config_v1.py",
        "distributed_class_formation_mechanism_audit_v1.py",
        "run_distributed_class_formation_mechanism_audit_v1.py",
        cfg.PROTOCOL_PATH,
        cfg.ANCESTRY_COMPLETION_PATH,
        cfg.ANCESTRY_FORMAL_SUMMARY_PATH,
        "endogenous_temporal_consequence_boundary_formation_config_v1.py",
        "endogenous_temporal_consequence_boundary_formation_v1.py",
        "distributed_temporal_boundary_equivalence_config_v1.py",
        "distributed_temporal_boundary_equivalence_v1.py",
    ]
    threshold_path = Path(cfg.THRESHOLD_DECLARATION_PATH)
    if threshold_path.exists():
        members.append(cfg.THRESHOLD_DECLARATION_PATH)
    hashes = {path: sha256_path(Path(path)) for path in members}
    return payload_hash(hashes), hashes


def write_markdown(result: dict[str, Any], path: Path) -> None:
    aggregate = result["aggregate"]
    lines = [
        f"# {cfg.STUDY}",
        "",
        f"Phase: `{result['phase']}`",
        f"Cases: `{result['case_count']}`",
        f"Source manifest: `{result['source_manifest_sha256']}`",
        "",
        "## Decision",
        "",
        f"`{result['decision']['status'] if result['decision'] else 'DEVELOPMENT_NO_FORMAL_DECISION'}`",
        "",
        "## Primary methods",
        "",
        "| Method | Relation | Minimum | Complete sequence |",
        "|---|---:|---:|---:|",
    ]
    for method in (
        cfg.FULL,
        cfg.COOPERATIVE,
        cfg.MARGINAL,
        cfg.CONDITIONAL_SHUFFLED,
        cfg.DIAGONAL,
        cfg.COOPERATIVE_DESTROYED,
        cfg.QUERY_FACTORIZED,
        cfg.SMOOTH_NULL,
    ):
        row = aggregate["methods"][method]
        lines.append(
            f"| {method} | {row['relation_accuracy']:.6f} | "
            f"{row['minimum_relation_accuracy']:.6f} | "
            f"{row['complete_sequence_identity_rate']:.6f} |"
        )
    lines.extend(["", "## Paired mechanism contrasts", ""])
    for name, row in aggregate["paired"].items():
        lines.append(
            f"- `{name}`: mean `{row['mean']:.6f}`, 95% bootstrap "
            f"`[{row['bootstrap_low']:.6f}, {row['bootstrap_high']:.6f}]`, "
            f"positive fraction `{row['positive_fraction']:.4f}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_case_csv(cases: Sequence[dict[str, Any]], path: Path) -> None:
    rows = []
    for case in cases:
        rows.append(
            {
                "family_seed": case["family_seed"],
                "held_query": case["held_query"],
                "suite": case["suite"],
                "full_accuracy": case["methods"][cfg.FULL]["relation_accuracy"],
                "cooperative_accuracy": case["methods"][cfg.COOPERATIVE]["relation_accuracy"],
                "marginal_accuracy": case["methods"][cfg.MARGINAL]["relation_accuracy"],
                "conditional_shuffled_accuracy": case["methods"][cfg.CONDITIONAL_SHUFFLED]["relation_accuracy"],
                "diagonal_accuracy": case["methods"][cfg.DIAGONAL]["relation_accuracy"],
                "cooperative_destroyed_accuracy": case["methods"][cfg.COOPERATIVE_DESTROYED]["relation_accuracy"],
                "gain_cooperative_over_marginal": case["gains"]["cooperative_over_marginal"],
                "gain_cooperative_over_destroyed": case["gains"]["cooperative_over_destroyed"],
                "covariance_change": case["formation"]["offdiagonal_covariance_change_ratio"],
                "conditional_marginal_error": case["formation"]["conditional_marginal_preservation_error"],
                "case_sha256": case["case_sha256"],
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_bank(
    phase: str,
    family_seeds: Sequence[int],
    output_path: Path,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for family_seed in family_seeds:
        for held_query in range(cfg.QUERY_COUNT):
            for suite in v1b_cfg.SUITES:
                cases.append(run_case(int(family_seed), held_query, dict(suite)))
    aggregate = aggregate_cases(cases)
    decision = decide(aggregate) if phase == "holdout" else None
    manifest_sha, manifest_members = source_manifest()
    result = {
        "study": cfg.STUDY,
        "phase": phase,
        "target_branch": cfg.TARGET_BRANCH,
        "family_seeds": [int(value) for value in family_seeds],
        "suite_names": [suite["name"] for suite in v1b_cfg.SUITES],
        "query_count": cfg.QUERY_COUNT,
        "case_count": len(cases),
        "formation_episodes_per_observed_query": cfg.FORMATION_EPISODES_PER_OBSERVED_QUERY,
        "cooperative_subset_size": cfg.COOPERATIVE_SUBSET_SIZE,
        "source_manifest_sha256": manifest_sha,
        "source_manifest_members": manifest_members,
        "ancestry": ancestry_record(),
        "thresholds": cfg.FROZEN_THRESHOLDS,
        "aggregate": aggregate,
        "decision": decision,
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    digest = sha256_path(output_path)
    output_path.with_suffix(".sha256").write_text(
        f"{digest}  {output_path.as_posix()}\n", encoding="utf-8"
    )
    write_markdown(result, output_path.with_suffix(".md"))
    write_case_csv(cases, output_path.with_name(output_path.stem + "_cases.csv"))
    return result
