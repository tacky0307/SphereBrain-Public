from __future__ import annotations

"""Endogenous Temporal Consequence Boundary Formation v1.

The Core receives an uninterrupted raw stream rather than supplied before/after
endpoints.  A non-negative temporal pair field starts at zero and is formed
from route commitments followed by delayed scalar continuation results.  The
field selects which ordered moments should be compared by a generic symmetric
interaction.  Ground-truth temporal windows remain inside the environment and
are used only after decisions for scientific audit.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence
import copy
import csv
import hashlib
import json
import math

import numpy as np

from endogenous_temporal_consequence_boundary_formation_config_v1 import *

EPS = 1e-12
PAIR_INDICES: tuple[tuple[int, int], ...] = tuple(
    (left, right)
    for left in range(STREAM_LENGTH)
    for right in range(left + 1, STREAM_LENGTH)
    if MIN_PAIR_LAG <= right - left <= MAX_PAIR_LAG
)
PAIR_I = np.asarray([pair[0] for pair in PAIR_INDICES], dtype=int)
PAIR_J = np.asarray([pair[1] for pair in PAIR_INDICES], dtype=int)
PAIR_COUNT = len(PAIR_INDICES)
PAIR_INDEX = {pair: index for index, pair in enumerate(PAIR_INDICES)}


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


def stable_u32(namespace: str, *parts: Any) -> int:
    material = namespace + "\0" + "\0".join(canonical_json(part) for part in parts)
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:4], "big")


def normalize(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= EPS:
        return np.zeros_like(vector)
    return vector / norm


def orthogonal_component(value: np.ndarray, basis: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if basis.size:
        vector = vector - basis @ (basis.T @ vector)
    return normalize(vector)


def unit_code(namespace: str, *parts: Any, dim: int) -> np.ndarray:
    rng = np.random.default_rng(stable_u32(namespace, *parts))
    return normalize(rng.normal(size=dim))


def pair_graph_laplacian() -> np.ndarray:
    matrix = np.zeros((PAIR_COUNT, PAIR_COUNT), dtype=float)
    for index, (left, right) in enumerate(PAIR_INDICES):
        neighbors: set[int] = set()
        for candidate in (
            (left - 1, right),
            (left + 1, right),
            (left, right - 1),
            (left, right + 1),
        ):
            other = PAIR_INDEX.get(candidate)
            if other is not None:
                neighbors.add(other)
        matrix[index, index] = float(len(neighbors))
        for other in neighbors:
            matrix[index, other] = -1.0
    return matrix


PAIR_LAPLACIAN = pair_graph_laplacian()


def temporal_pair_features(stream: np.ndarray) -> np.ndarray:
    """Generic ordered-moment interaction; no true boundaries are supplied."""
    values = np.asarray(stream, dtype=float)
    if values.shape != (STREAM_LENGTH, RAW_DIM):
        raise ValueError(f"expected stream {(STREAM_LENGTH, RAW_DIM)}, got {values.shape}")
    centered = values - np.mean(values, axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean(centered * centered)))
    centered = centered / max(scale, EPS)
    raw = -np.einsum("pd,pd->p", centered[PAIR_I], centered[PAIR_J]) / float(RAW_DIM)
    raw = raw - float(np.mean(raw))
    deviation = float(np.std(raw))
    return raw / max(deviation, EPS)


def relation_signed_target(action_channel: int, continuation_result: int) -> float:
    inferred_channel = int(action_channel) if continuation_result else 1 - int(action_channel)
    return 1.0 if inferred_channel == 1 else -1.0


def temporal_shift_weights(weights: np.ndarray, shift: int) -> np.ndarray:
    shifted = np.zeros(PAIR_COUNT, dtype=float)
    for source, (left, right) in enumerate(PAIR_INDICES):
        target = PAIR_INDEX.get((left + shift, right + shift))
        if target is not None:
            shifted[target] += float(weights[source])
    source_norm = float(np.linalg.norm(weights))
    target_norm = float(np.linalg.norm(shifted))
    if source_norm > EPS and target_norm > EPS:
        shifted *= source_norm / target_norm
    return shifted


def temporal_reverse_weights(weights: np.ndarray) -> np.ndarray:
    reversed_weights = np.zeros(PAIR_COUNT, dtype=float)
    for source, (left, right) in enumerate(PAIR_INDICES):
        target_pair = (STREAM_LENGTH - 1 - right, STREAM_LENGTH - 1 - left)
        target = PAIR_INDEX.get(target_pair)
        if target is not None:
            reversed_weights[target] += float(weights[source])
    source_norm = float(np.linalg.norm(weights))
    target_norm = float(np.linalg.norm(reversed_weights))
    if source_norm > EPS and target_norm > EPS:
        reversed_weights *= source_norm / target_norm
    return reversed_weights


def fixed_first_last_weights(reference_norm: float) -> np.ndarray:
    weights = np.zeros(PAIR_COUNT, dtype=float)
    target = max(
        range(PAIR_COUNT),
        key=lambda index: PAIR_INDICES[index][1] - PAIR_INDICES[index][0],
    )
    weights[target] = max(reference_norm, 1.0)
    return weights


def uniform_pair_weights(reference_norm: float) -> np.ndarray:
    weights = np.ones(PAIR_COUNT, dtype=float)
    return normalize(weights) * max(reference_norm, 1.0)


def random_pair_weights(reference_norm: float, *seed_parts: Any) -> np.ndarray:
    rng = np.random.default_rng(stable_u32("ectbf-v1-random-pair-field", *seed_parts))
    weights = np.maximum(rng.normal(size=PAIR_COUNT), 0.0)
    if float(np.linalg.norm(weights)) <= EPS:
        weights[0] = 1.0
    return normalize(weights) * max(reference_norm, 1.0)


@dataclass(frozen=True)
class StreamMetadata:
    relation_channel: int
    initial_value: int
    final_value: int
    pre_window: tuple[int, ...]
    post_window: tuple[int, ...]
    pre_center: float
    post_center: float
    intervention_time: int
    identity: str

    def support_vector(self) -> np.ndarray:
        pre = set(self.pre_window)
        post = set(self.post_window)
        return np.asarray(
            [1.0 if left in pre and right in post else 0.0 for left, right in PAIR_INDICES],
            dtype=float,
        )


@dataclass(frozen=True)
class ContinuousWorldSpec:
    consequence_basis: tuple[tuple[float, ...], ...]
    context_codes: tuple[tuple[float, ...], ...]
    action_direction: tuple[float, ...]
    drift_directions: tuple[tuple[float, ...], ...]
    world_hash: str


class ContinuousConsequenceWorld:
    """Synthetic continuous stream with hidden temporal consequence windows."""

    def __init__(self, family_seed: int) -> None:
        rng = np.random.default_rng(stable_u32("ectbf-v1-world", family_seed))
        raw = rng.normal(size=(RAW_DIM, CONSEQUENCE_SUBSPACE_DIM))
        consequence_basis, _ = np.linalg.qr(raw, mode="reduced")

        nuisance_raw = rng.normal(size=(RAW_DIM, QUERY_COUNT + 3))
        nuisance_raw = nuisance_raw - consequence_basis @ (consequence_basis.T @ nuisance_raw)
        nuisance_basis, _ = np.linalg.qr(nuisance_raw, mode="reduced")
        context_codes = nuisance_basis[:, :QUERY_COUNT].T
        action_direction = nuisance_basis[:, QUERY_COUNT]
        drift_directions = nuisance_basis[:, QUERY_COUNT + 1 : QUERY_COUNT + 3].T

        material = {
            "consequence_basis": np.round(consequence_basis, 14).tolist(),
            "context_codes": np.round(context_codes, 14).tolist(),
            "action_direction": np.round(action_direction, 14).tolist(),
            "drift_directions": np.round(drift_directions, 14).tolist(),
        }
        self.spec = ContinuousWorldSpec(
            consequence_basis=tuple(tuple(float(v) for v in row) for row in consequence_basis),
            context_codes=tuple(tuple(float(v) for v in row) for row in context_codes),
            action_direction=tuple(float(v) for v in action_direction),
            drift_directions=tuple(tuple(float(v) for v in row) for row in drift_directions),
            world_hash=payload_hash(material),
        )
        self.consequence_basis = np.asarray(self.spec.consequence_basis, dtype=float)
        self.context_codes = np.asarray(self.spec.context_codes, dtype=float)
        self.action_direction = np.asarray(self.spec.action_direction, dtype=float)
        self.drift_directions = np.asarray(self.spec.drift_directions, dtype=float)
        self.stream_reads = 0

    def _query_consequence_direction(self, query: int) -> np.ndarray:
        coefficients = unit_code(
            "ectbf-v1-query-consequence-code",
            self.spec.world_hash,
            query,
            dim=CONSEQUENCE_SUBSPACE_DIM,
        )
        return normalize(self.consequence_basis @ coefficients)

    def _identity_direction(self, identity: str) -> np.ndarray:
        vector = unit_code("ectbf-v1-identity", self.spec.world_hash, identity, dim=RAW_DIM)
        return orthogonal_component(vector, self.consequence_basis)

    def _distractor_direction(self, identity: str, index: int) -> np.ndarray:
        vector = unit_code(
            "ectbf-v1-distractor",
            self.spec.world_hash,
            identity,
            index,
            dim=RAW_DIM,
        )
        return orthogonal_component(vector, self.consequence_basis)

    def generate(
        self,
        query: int,
        identity: str,
        suite: dict[str, Any],
        *,
        episode_index: int,
    ) -> tuple[np.ndarray, StreamMetadata]:
        if not 0 <= query < QUERY_COUNT:
            raise ValueError("query outside declared bank")
        rng = np.random.default_rng(
            stable_u32(
                "ectbf-v1-stream",
                self.spec.world_hash,
                query,
                identity,
                suite["name"],
                episode_index,
            )
        )
        initial_value = 1 if stable_u32("ectbf-v1-initial", identity) % 2 else -1
        relation_channel = int(stable_u32("ectbf-v1-relation", identity, query) % 2)
        final_value = initial_value if relation_channel == 0 else -initial_value

        jitter = int(suite["timing_jitter"])
        pre_shift = int(rng.integers(-jitter, jitter + 1))
        pre_center = int(np.clip(5 + pre_shift, 2, 8))
        delay_offset = (-1, 0, 1, -1, 1)[query]
        variable_delay = int(rng.integers(7, 10)) + delay_offset
        post_center = int(np.clip(pre_center + variable_delay, pre_center + 5, STREAM_LENGTH - 3))
        intervention_time = int(np.clip(pre_center + 3, pre_center + 2, post_center - 2))

        pre_window = tuple(
            time
            for time in range(pre_center - TRUE_WINDOW_RADIUS, pre_center + TRUE_WINDOW_RADIUS + 1)
            if 0 <= time < STREAM_LENGTH
        )
        post_window = tuple(
            time
            for time in range(post_center - TRUE_WINDOW_RADIUS, post_center + TRUE_WINDOW_RADIUS + 1)
            if 0 <= time < STREAM_LENGTH
        )

        stream = np.zeros((STREAM_LENGTH, RAW_DIM), dtype=float)
        identity_direction = self._identity_direction(identity)
        context_direction = self.context_codes[query]
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        time_axis = np.arange(STREAM_LENGTH, dtype=float)
        drift_one = np.sin(2.0 * math.pi * time_axis / STREAM_LENGTH + phase)
        drift_two = np.cos(4.0 * math.pi * time_axis / STREAM_LENGTH + 0.5 * phase)
        for time in range(STREAM_LENGTH):
            stream[time] += IDENTITY_GAIN * identity_direction
            stream[time] += CONTEXT_GAIN * context_direction
            stream[time] += DRIFT_GAIN * (
                drift_one[time] * self.drift_directions[0]
                + 0.55 * drift_two[time] * self.drift_directions[1]
            )

        carrier = self._query_consequence_direction(query)
        pulse_shape = {
            -TRUE_WINDOW_RADIUS: 0.72,
            0: 1.0,
            TRUE_WINDOW_RADIUS: 0.72,
        }
        for offset, gain in pulse_shape.items():
            first_time = pre_center + offset
            second_time = post_center + offset
            if 0 <= first_time < STREAM_LENGTH:
                stream[first_time] += CONSEQUENCE_GAIN * gain * initial_value * carrier
            if 0 <= second_time < STREAM_LENGTH:
                stream[second_time] += CONSEQUENCE_GAIN * gain * final_value * carrier

        action_shape = ((-1, 0.45), (0, 1.0), (1, 0.45))
        for offset, gain in action_shape:
            time = intervention_time + offset
            if 0 <= time < STREAM_LENGTH:
                stream[time] += ACTION_FOOTPRINT_GAIN * gain * self.action_direction

        forbidden_times = set(pre_window) | set(post_window)
        distractor_count = int(suite["distractor_count"])
        allowed_times = [time for time in range(1, STREAM_LENGTH - 1) if time not in forbidden_times]
        rng.shuffle(allowed_times)
        for index, time in enumerate(allowed_times[:distractor_count]):
            direction = self._distractor_direction(identity, index)
            sign = 1.0 if stable_u32("ectbf-v1-distractor-sign", identity, index) % 2 else -1.0
            gain = float(suite["distractor_gain"]) * float(rng.uniform(0.75, 1.25))
            stream[time] += sign * gain * direction
            if time + 1 < STREAM_LENGTH and time + 1 not in forbidden_times:
                stream[time + 1] += 0.35 * sign * gain * direction

        stream += float(suite["stream_noise"]) * rng.normal(size=stream.shape)
        dropout = float(suite["sensor_dropout"])
        if dropout > 0.0:
            mask = rng.random(size=stream.shape) < dropout
            stream = stream.copy()
            stream[mask] = 0.0

        self.stream_reads += 1
        return stream, StreamMetadata(
            relation_channel=relation_channel,
            initial_value=initial_value,
            final_value=final_value,
            pre_window=pre_window,
            post_window=post_window,
            pre_center=float(pre_center),
            post_center=float(post_center),
            intervention_time=intervention_time,
            identity=identity,
        )

    def audit(self) -> dict[str, Any]:
        return {
            "stream_reads": int(self.stream_reads),
            "world_hash": self.spec.world_hash,
            "stream_length": STREAM_LENGTH,
            "raw_dim": RAW_DIM,
            "consequence_subspace_dim": CONSEQUENCE_SUBSPACE_DIM,
            "consequence_basis_hidden_from_core": True,
            "true_temporal_boundaries_exposed_to_core": False,
            "endpoint_segmentation_absent": True,
            "generic_ordered_pair_vocabulary_supplied": True,
        }


@dataclass
class FeedbackAudit:
    route_commits: int = 0
    delayed_scalar_results: int = 0
    precommit_reads: int = 0
    early_reads: int = 0
    relation_tokens_returned: int = 0
    minimum_observed_delay: int = 10**9
    maximum_observed_delay: int = 0

    def payload(self) -> dict[str, Any]:
        minimum = 0 if self.delayed_scalar_results == 0 else self.minimum_observed_delay
        return {
            "route_commits": self.route_commits,
            "delayed_scalar_results": self.delayed_scalar_results,
            "precommit_reads": self.precommit_reads,
            "early_reads": self.early_reads,
            "relation_tokens_returned": self.relation_tokens_returned,
            "minimum_observed_delay": minimum,
            "maximum_observed_delay": self.maximum_observed_delay,
            "all_results_after_required_delay": (
                self.precommit_reads == 0
                and self.early_reads == 0
                and minimum >= FORMATION_CREDIT_DELAY
                and self.delayed_scalar_results == self.route_commits
            ),
        }


class DelayedContinuationVault:
    """Environment-only vault: route first, scalar result after a fixed delay."""

    def __init__(self) -> None:
        self._next_token = 0
        self._pending: dict[int, tuple[int, int, int]] = {}
        self.audit_state = FeedbackAudit()

    def commit(self, *, step: int, action_channel: int, target_channel: int) -> int:
        if action_channel not in (0, 1) or target_channel not in (0, 1):
            raise ValueError("binary route channels required")
        token = self._next_token
        self._next_token += 1
        result = int(action_channel == target_channel)
        ready_step = int(step) + FORMATION_CREDIT_DELAY
        self._pending[token] = (ready_step, int(step), result)
        self.audit_state.route_commits += 1
        return token

    def release_ready(self, step: int) -> list[tuple[int, int]]:
        released: list[tuple[int, int]] = []
        for token, (ready_step, commit_step, result) in list(self._pending.items()):
            if step < ready_step:
                continue
            delay = int(step) - commit_step
            if delay < FORMATION_CREDIT_DELAY:
                self.audit_state.early_reads += 1
                raise AssertionError("continuation result read before delay")
            released.append((token, result))
            self.audit_state.delayed_scalar_results += 1
            self.audit_state.minimum_observed_delay = min(
                self.audit_state.minimum_observed_delay, delay
            )
            self.audit_state.maximum_observed_delay = max(
                self.audit_state.maximum_observed_delay, delay
            )
            del self._pending[token]
        return released

    def audit(self) -> dict[str, Any]:
        if self._pending:
            raise AssertionError("unreleased feedback remains")
        return self.audit_state.payload()


@dataclass(frozen=True)
class FormationRecord:
    temporal_features: tuple[float, ...]
    committed_action: int
    continuation_result: int
    identity: str


class TemporalBoundaryLearner:
    """Forms a non-negative temporal comparison field from delayed credit."""

    def __init__(self) -> None:
        self.records: list[FormationRecord] = []
        self.initial_weight_norm = 0.0
        self.scalar_credit_updates = 0
        self.direct_boundary_arguments = 0
        self.direct_relation_arguments = 0

    def receive(self, record: FormationRecord) -> None:
        if len(record.temporal_features) != PAIR_COUNT:
            raise ValueError("temporal feature size mismatch")
        self.records.append(record)
        self.scalar_credit_updates += 1

    @staticmethod
    def _projected_ridge(features: np.ndarray, targets: np.ndarray) -> np.ndarray:
        gram = features.T @ features
        system = (
            gram
            + TEMPORAL_RIDGE_LAMBDA * np.eye(PAIR_COUNT)
            + TEMPORAL_SMOOTHNESS_LAMBDA * PAIR_LAPLACIAN
        )
        rhs = features.T @ targets
        row_bound = max(float(np.max(np.sum(np.abs(system), axis=1))), EPS)
        step = 0.92 / row_bound
        weights = np.zeros(PAIR_COUNT, dtype=float)
        momentum = weights.copy()
        acceleration = 1.0
        for _ in range(260):
            gradient = system @ momentum - rhs
            updated = np.maximum(momentum - step * gradient, 0.0)
            next_acceleration = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * acceleration**2))
            momentum = updated + ((acceleration - 1.0) / next_acceleration) * (
                updated - weights
            )
            weights = updated
            acceleration = next_acceleration
        if float(np.linalg.norm(weights)) <= EPS:
            unconstrained = np.linalg.lstsq(system, rhs, rcond=None)[0]
            weights = np.maximum(unconstrained, 0.0)
        if float(np.linalg.norm(weights)) <= EPS:
            weights[int(np.argmax(np.abs(rhs)))] = 1.0
        threshold = float(np.quantile(weights, WEIGHT_PRUNE_FRACTION))
        weights = np.where(weights > threshold, weights, 0.0)
        scores = features @ weights
        denominator = float(scores @ scores)
        if denominator > EPS:
            positive_scale = max(float(scores @ targets) / denominator, EPS)
            weights *= positive_scale
        return weights

    def solve(self) -> tuple[np.ndarray, np.ndarray]:
        features = np.asarray([record.temporal_features for record in self.records], dtype=float)
        targets = np.asarray(
            [
                relation_signed_target(
                    record.committed_action,
                    record.continuation_result,
                )
                for record in self.records
            ],
            dtype=float,
        )
        learned = self._projected_ridge(features, targets)
        offset = max(1, len(targets) // 3)
        shuffled = self._projected_ridge(features, np.roll(targets, offset))
        return learned, shuffled

    def audit(self) -> dict[str, Any]:
        return {
            "formation_records": len(self.records),
            "scalar_credit_updates": self.scalar_credit_updates,
            "updates_equal_records": self.scalar_credit_updates == len(self.records),
            "initial_temporal_weight_norm": self.initial_weight_norm,
            "direct_boundary_arguments": self.direct_boundary_arguments,
            "direct_relation_arguments": self.direct_relation_arguments,
            "supplied_boundary_masks": 0,
            "supplied_boundary_indices": 0,
            "supplied_similarity_thresholds": 0,
            "supplied_correct_routes": 0,
        }


@dataclass(frozen=True)
class TemporalBoundarySnapshot:
    version: str
    stream_length: int
    raw_dim: int
    pair_count: int
    pair_vocabulary_hash: str
    temporal_pair_weights: tuple[float, ...]
    relation_codes: tuple[tuple[float, ...], ...]
    maintenance_basin: tuple[tuple[float, ...], ...]
    formation_hash: str
    substrate_hash: str

    def payload_without_hash(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "stream_length": self.stream_length,
            "raw_dim": self.raw_dim,
            "pair_count": self.pair_count,
            "pair_vocabulary_hash": self.pair_vocabulary_hash,
            "temporal_pair_weights": list(self.temporal_pair_weights),
            "relation_codes": [list(row) for row in self.relation_codes],
            "maintenance_basin": [list(row) for row in self.maintenance_basin],
            "formation_hash": self.formation_hash,
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self.payload_without_hash(), "substrate_hash": self.substrate_hash}

    def to_json(self) -> str:
        return canonical_json(self.to_payload())

    @classmethod
    def from_json(cls, text: str) -> "TemporalBoundarySnapshot":
        payload = json.loads(text)
        snapshot = cls(
            version=str(payload["version"]),
            stream_length=int(payload["stream_length"]),
            raw_dim=int(payload["raw_dim"]),
            pair_count=int(payload["pair_count"]),
            pair_vocabulary_hash=str(payload["pair_vocabulary_hash"]),
            temporal_pair_weights=tuple(float(v) for v in payload["temporal_pair_weights"]),
            relation_codes=tuple(
                tuple(float(v) for v in row) for row in payload["relation_codes"]
            ),
            maintenance_basin=tuple(
                tuple(float(v) for v in row) for row in payload["maintenance_basin"]
            ),
            formation_hash=str(payload["formation_hash"]),
            substrate_hash=str(payload["substrate_hash"]),
        )
        if snapshot.stream_length != STREAM_LENGTH or snapshot.raw_dim != RAW_DIM:
            raise AssertionError("snapshot geometry mismatch")
        if snapshot.pair_count != PAIR_COUNT:
            raise AssertionError("snapshot pair-count mismatch")
        if snapshot.pair_vocabulary_hash != payload_hash(PAIR_INDICES):
            raise AssertionError("snapshot pair vocabulary mismatch")
        if payload_hash(snapshot.payload_without_hash()) != snapshot.substrate_hash:
            raise AssertionError("snapshot hash mismatch")
        return snapshot


class LearnedTemporalBoundaryCircuit:
    def __init__(self, snapshot: TemporalBoundarySnapshot) -> None:
        self.snapshot = snapshot
        self.weights = np.asarray(snapshot.temporal_pair_weights, dtype=float)
        self.codes = np.asarray(snapshot.relation_codes, dtype=float)
        self.basin = np.asarray(snapshot.maintenance_basin, dtype=float)
        self.explicit_boundary_argument_count = 0
        self.explicit_relation_argument_count = 0
        self.planning_calls = 0
        self.stream_calls = 0

    def raw_score(self, stream: np.ndarray, *, weights: np.ndarray | None = None) -> float:
        self.stream_calls += 1
        selected = self.weights if weights is None else np.asarray(weights, dtype=float)
        return float(selected @ temporal_pair_features(stream))

    def predict(
        self,
        stream: np.ndarray,
        *,
        weights: np.ndarray | None = None,
        delay: bool = False,
        maintain: bool = True,
        state_noise: float = RELATION_DELAY_DRIFT,
        seed_parts: Sequence[Any] = (),
    ) -> tuple[int, float]:
        score = self.raw_score(stream, weights=weights)
        initial = int(score >= 0.0)
        if not delay:
            return initial, abs(score)
        rng = np.random.default_rng(stable_u32("ectbf-v1-relation-delay", *seed_parts))
        state = self.codes[initial].copy()
        for _ in range(RELATION_DELAY_STEPS):
            state = RELATION_DELAY_RETENTION * state + state_noise * rng.normal(
                size=RELATION_STATE_DIM
            ) / math.sqrt(float(RELATION_STATE_DIM))
            if maintain:
                state = normalize(0.22 * state + 0.78 * (self.basin @ state))
        scores = self.codes @ state
        return int(np.argmax(scores)), float(np.max(scores) - np.min(scores))

    def audit(self) -> dict[str, Any]:
        return {
            "explicit_boundary_argument_count": self.explicit_boundary_argument_count,
            "explicit_relation_argument_count": self.explicit_relation_argument_count,
            "runtime_planning_calls": self.planning_calls,
            "stream_calls": self.stream_calls,
        }


def perturb_stream(
    stream: np.ndarray,
    variant: str,
    *,
    seed_parts: Sequence[Any],
) -> np.ndarray:
    values = np.asarray(stream, dtype=float).copy()
    if variant == "clean":
        return values
    rng = np.random.default_rng(stable_u32("ectbf-v1-runtime-perturb", variant, *seed_parts))
    if variant in ("partial", "combined"):
        fraction = RUNTIME_PARTIAL_FRACTION
        mask = rng.random(size=values.shape) < fraction
        values[mask] = 0.0
    if variant == "noise":
        values += RUNTIME_NOISE_STD * rng.normal(size=values.shape)
    elif variant == "combined":
        values += RUNTIME_COMBINED_NOISE_STD * rng.normal(size=values.shape)
    return values


def snapshot_forbidden_token_count(snapshot: TemporalBoundarySnapshot) -> int:
    text = snapshot.to_json().lower()
    return sum(text.count(token.lower()) for token in FORBIDDEN_SNAPSHOT_TOKENS)


def boundary_recovery_metrics(
    weights: np.ndarray,
    support_frequency: np.ndarray,
    mean_pre_center: float,
    mean_post_center: float,
) -> dict[str, float]:
    positive = np.maximum(np.asarray(weights, dtype=float), 0.0)
    if float(np.sum(positive)) <= EPS:
        positive = np.abs(np.asarray(weights, dtype=float))
    mass = positive / max(float(np.sum(positive)), EPS)
    baseline = float(np.mean(support_frequency))
    weighted_support = float(mass @ support_frequency)
    top_count = max(8, PAIR_COUNT // 12)
    top_indices = np.argsort(positive)[-top_count:]
    top_support = float(np.mean(support_frequency[top_indices]))
    expected_pre = float(mass @ PAIR_I)
    expected_post = float(mass @ PAIR_J)
    center_mae = 0.5 * (
        abs(expected_pre - mean_pre_center) + abs(expected_post - mean_post_center)
    )
    dominant = int(np.argmax(positive))
    return {
        "support_weighted_frequency": weighted_support,
        "support_baseline_frequency": baseline,
        "support_enrichment": weighted_support / max(baseline, EPS),
        "top_pair_support_frequency": top_support,
        "top_pair_support_enrichment": top_support / max(baseline, EPS),
        "expected_pre_center": expected_pre,
        "expected_post_center": expected_post,
        "mean_true_pre_center": mean_pre_center,
        "mean_true_post_center": mean_post_center,
        "temporal_center_mae": center_mae,
        "dominant_pair_left": float(PAIR_INDICES[dominant][0]),
        "dominant_pair_right": float(PAIR_INDICES[dominant][1]),
        "dominant_pair_support_frequency": float(support_frequency[dominant]),
        "positive_pair_fraction": float(np.mean(positive > 0.0)),
    }


def form_temporal_boundary(
    family_seed: int,
    held_query: int,
    suite: dict[str, Any],
) -> tuple[
    TemporalBoundarySnapshot,
    ContinuousConsequenceWorld,
    dict[str, Any],
    np.ndarray,
]:
    world = ContinuousConsequenceWorld(family_seed)
    learner = TemporalBoundaryLearner()
    vault = DelayedContinuationVault()
    pending: dict[int, tuple[np.ndarray, int, str]] = {}
    support_sum = np.zeros(PAIR_COUNT, dtype=float)
    pre_centers: list[float] = []
    post_centers: list[float] = []
    formation_identities: list[str] = []
    step = 0

    observed_queries = [query for query in range(QUERY_COUNT) if query != held_query]
    for query in observed_queries:
        for episode in range(FORMATION_EPISODES_PER_OBSERVED_QUERY):
            identity = (
                f"formation:f{family_seed}:h{held_query}:q{query}:"
                f"s{suite['name']}:e{episode}"
            )
            stream, metadata = world.generate(
                query,
                identity,
                suite,
                episode_index=episode,
            )
            features = temporal_pair_features(stream)
            action = int(
                stable_u32(
                    "ectbf-v1-formation-action",
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
            pending[token] = (features, action, identity)
            support_sum += metadata.support_vector()
            pre_centers.append(metadata.pre_center)
            post_centers.append(metadata.post_center)
            formation_identities.append(identity)
            for released_token, scalar_result in vault.release_ready(step):
                released_features, released_action, released_identity = pending.pop(
                    released_token
                )
                learner.receive(
                    FormationRecord(
                        temporal_features=tuple(float(v) for v in released_features),
                        committed_action=released_action,
                        continuation_result=scalar_result,
                        identity=released_identity,
                    )
                )
            step += 1

    flush_step = step + FORMATION_CREDIT_DELAY
    while pending:
        released = vault.release_ready(flush_step)
        if not released:
            flush_step += 1
            continue
        for released_token, scalar_result in released:
            released_features, released_action, released_identity = pending.pop(
                released_token
            )
            learner.receive(
                FormationRecord(
                    temporal_features=tuple(float(v) for v in released_features),
                    committed_action=released_action,
                    continuation_result=scalar_result,
                    identity=released_identity,
                )
            )

    learned_weights, shuffled_weights = learner.solve()
    code_zero = unit_code(
        "ectbf-v1-relation-state", family_seed, held_query, suite["name"], 0, dim=RELATION_STATE_DIM
    )
    code_one_seed = unit_code(
        "ectbf-v1-relation-state", family_seed, held_query, suite["name"], 1, dim=RELATION_STATE_DIM
    )
    code_one = normalize(code_one_seed - float(code_zero @ code_one_seed) * code_zero)
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
        }
    )
    provisional = TemporalBoundarySnapshot(
        version="ectbf-v1",
        stream_length=STREAM_LENGTH,
        raw_dim=RAW_DIM,
        pair_count=PAIR_COUNT,
        pair_vocabulary_hash=payload_hash(PAIR_INDICES),
        temporal_pair_weights=tuple(float(v) for v in learned_weights),
        relation_codes=tuple(tuple(float(v) for v in row) for row in codes),
        maintenance_basin=tuple(tuple(float(v) for v in row) for row in basin),
        formation_hash=formation_hash,
        substrate_hash="",
    )
    snapshot = TemporalBoundarySnapshot(
        **{
            **provisional.payload_without_hash(),
            "temporal_pair_weights": provisional.temporal_pair_weights,
            "relation_codes": provisional.relation_codes,
            "maintenance_basin": provisional.maintenance_basin,
            "substrate_hash": payload_hash(provisional.payload_without_hash()),
        }
    )
    serialized = snapshot.to_json()
    restored = TemporalBoundarySnapshot.from_json(serialized)
    roundtrip_exact = restored.to_json() == serialized

    support_frequency = support_sum / float(len(formation_identities))
    recovery = boundary_recovery_metrics(
        learned_weights,
        support_frequency,
        float(np.mean(pre_centers)),
        float(np.mean(post_centers)),
    )
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
        "snapshot_roundtrip_exact": roundtrip_exact,
        "snapshot_forbidden_token_count": snapshot_forbidden_token_count(snapshot),
        "boundary_recovery": recovery,
        "held_query_feedback_updates": 0,
        "formation_and_evaluation_namespaces_disjoint": True,
    }
    return snapshot, world, report, shuffled_weights


def _method_weights(
    method: str,
    learned: np.ndarray,
    shuffled: np.ndarray,
    *,
    family_seed: int,
    held_query: int,
    suite_name: str,
) -> np.ndarray | None:
    norm = float(np.linalg.norm(learned))
    if method in ROBUST_METHODS or method == NO_MAINTENANCE:
        return learned
    if method == SHIFTED:
        return temporal_shift_weights(learned, TEMPORAL_SHIFT_CONTROL)
    if method == REVERSED:
        return temporal_reverse_weights(learned)
    if method == BOUNDARY_ABLATED:
        return np.zeros_like(learned)
    if method == FIRST_LAST:
        return fixed_first_last_weights(norm)
    if method == UNIFORM:
        return uniform_pair_weights(norm)
    if method == SHUFFLED:
        return shuffled
    if method == RANDOM:
        return random_pair_weights(norm, family_seed, held_query, suite_name)
    return None


def _method_variant(method: str) -> str:
    if method == PARTIAL:
        return "partial"
    if method == NOISE:
        return "noise"
    if method == COMBINED or method == NO_MAINTENANCE:
        return "combined"
    return "clean"


def _method_delay(method: str) -> tuple[bool, bool, float]:
    if method == DELAY:
        return True, True, RELATION_DELAY_DRIFT
    if method == NO_MAINTENANCE:
        return True, False, NO_MAINTENANCE_DRIFT
    return False, True, RELATION_DELAY_DRIFT


def evaluate_direct(
    *,
    family_seed: int,
    held_query: int,
    suite: dict[str, Any],
    world: ContinuousConsequenceWorld,
    circuit: LearnedTemporalBoundaryCircuit,
    shuffled_weights: np.ndarray,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    counts = {method: 0 for method in METHODS}
    margins = {method: [] for method in METHODS}
    evaluation_identities: list[str] = []
    support_sum = np.zeros(PAIR_COUNT, dtype=float)
    pre_centers: list[float] = []
    post_centers: list[float] = []
    decisions_before_truth = 0

    for episode in range(DIRECT_TEST_EPISODES):
        identity = (
            f"evaluation-direct:f{family_seed}:h{held_query}:"
            f"s{suite['name']}:e{episode}"
        )
        stream, metadata = world.generate(
            held_query,
            identity,
            suite,
            episode_index=episode,
        )
        support_sum += metadata.support_vector()
        pre_centers.append(metadata.pre_center)
        post_centers.append(metadata.post_center)
        evaluation_identities.append(identity)
        predictions: dict[str, tuple[int, float]] = {EXPLICIT_UPPER: (metadata.relation_channel, 1.0)}
        for method in (*ROBUST_METHODS, *CONTROL_METHODS):
            variant = _method_variant(method)
            perturbed = perturb_stream(
                stream,
                variant,
                seed_parts=(family_seed, held_query, suite["name"], episode, method),
            )
            weights = _method_weights(
                method,
                circuit.weights,
                shuffled_weights,
                family_seed=family_seed,
                held_query=held_query,
                suite_name=suite["name"],
            )
            delay, maintain, state_noise = _method_delay(method)
            predictions[method] = circuit.predict(
                perturbed,
                weights=weights,
                delay=delay,
                maintain=maintain,
                state_noise=state_noise,
                seed_parts=(family_seed, held_query, suite["name"], episode, method),
            )
        decisions_before_truth += 1
        truth = metadata.relation_channel
        for method, (prediction, margin) in predictions.items():
            counts[method] += int(prediction == truth)
            margins[method].append(float(margin))

    methods: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        methods[method] = {
            "relation_accuracy": counts[method] / float(DIRECT_TEST_EPISODES),
            "mean_margin": float(np.mean(margins[method])),
            "minimum_margin": float(np.min(margins[method])),
        }
    support_frequency = support_sum / float(DIRECT_TEST_EPISODES)
    recovery = boundary_recovery_metrics(
        circuit.weights,
        support_frequency,
        float(np.mean(pre_centers)),
        float(np.mean(post_centers)),
    )
    audit = {
        "evaluation_episode_count": DIRECT_TEST_EPISODES,
        "evaluation_identity_count": len(set(evaluation_identities)),
        "evaluation_identity_hash": payload_hash(sorted(set(evaluation_identities))),
        "route_decisions_before_truth_reads": decisions_before_truth,
        "truth_reads_after_route_decision": DIRECT_TEST_EPISODES,
        "explicit_boundary_arguments": 0,
        "explicit_relation_arguments": 0,
        "boundary_recovery_on_unseen_streams": recovery,
    }
    return methods, audit


def evaluate_plan(
    *,
    family_seed: int,
    held_query: int,
    suite: dict[str, Any],
    world: ContinuousConsequenceWorld,
    circuit: LearnedTemporalBoundaryCircuit,
    shuffled_weights: np.ndarray,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    complete = {method: 0 for method in METHODS}
    step_correct = {method: 0 for method in METHODS}
    proposal_identity = {method: 0 for method in METHODS}
    total_steps = PLAN_EPISODES * PLAN_DEPTH
    identities: list[str] = []

    for episode in range(PLAN_EPISODES):
        truths: list[int] = []
        streams: list[np.ndarray] = []
        for depth in range(PLAN_DEPTH):
            identity = (
                f"evaluation-plan:f{family_seed}:h{held_query}:"
                f"s{suite['name']}:e{episode}:d{depth}"
            )
            stream, metadata = world.generate(
                held_query,
                identity,
                suite,
                episode_index=episode * PLAN_DEPTH + depth,
            )
            truths.append(metadata.relation_channel)
            streams.append(stream)
            identities.append(identity)

        predictions: dict[str, list[int]] = {EXPLICIT_UPPER: list(truths)}
        for method in (*ROBUST_METHODS, *CONTROL_METHODS):
            sequence: list[int] = []
            for depth, stream in enumerate(streams):
                variant = _method_variant(method)
                perturbed = perturb_stream(
                    stream,
                    variant,
                    seed_parts=(family_seed, held_query, suite["name"], episode, depth, method),
                )
                weights = _method_weights(
                    method,
                    circuit.weights,
                    shuffled_weights,
                    family_seed=family_seed,
                    held_query=held_query,
                    suite_name=suite["name"],
                )
                delay, maintain, state_noise = _method_delay(method)
                prediction, _margin = circuit.predict(
                    perturbed,
                    weights=weights,
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
                sequence.append(prediction)
            predictions[method] = sequence

        upper_sequence = predictions[EXPLICIT_UPPER]
        for method, sequence in predictions.items():
            step_correct[method] += sum(
                int(prediction == truth)
                for prediction, truth in zip(sequence, truths)
            )
            complete[method] += int(sequence == truths)
            proposal_identity[method] += sum(
                int(prediction == upper)
                for prediction, upper in zip(sequence, upper_sequence)
            )

    methods: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        methods[method] = {
            "step_accuracy": step_correct[method] / float(total_steps),
            "complete_sequence_identity_rate": complete[method] / float(PLAN_EPISODES),
            "proposal_identity_rate_to_explicit_upper": proposal_identity[method]
            / float(total_steps),
        }
    audit = {
        "plan_episode_count": PLAN_EPISODES,
        "plan_depth": PLAN_DEPTH,
        "plan_identity_count": len(set(identities)),
        "plan_identity_hash": payload_hash(sorted(set(identities))),
        "explicit_boundary_arguments": 0,
        "explicit_relation_arguments": 0,
        "runtime_planning_calls": 0,
    }
    return methods, audit


def run_case(family_seed: int, held_query: int, suite: dict[str, Any]) -> dict[str, Any]:
    snapshot, world, formation, shuffled_weights = form_temporal_boundary(
        family_seed,
        held_query,
        suite,
    )
    restored = TemporalBoundarySnapshot.from_json(snapshot.to_json())
    circuit = LearnedTemporalBoundaryCircuit(restored)
    direct_methods, direct_audit = evaluate_direct(
        family_seed=family_seed,
        held_query=held_query,
        suite=suite,
        world=world,
        circuit=circuit,
        shuffled_weights=shuffled_weights,
    )
    plan_methods, plan_audit = evaluate_plan(
        family_seed=family_seed,
        held_query=held_query,
        suite=suite,
        world=world,
        circuit=circuit,
        shuffled_weights=shuffled_weights,
    )

    methods: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        methods[method] = {
            **direct_methods[method],
            **plan_methods[method],
        }

    learned_direct = methods[PRIMARY]["relation_accuracy"]
    control_gains = {
        "gain_over_shifted": learned_direct - methods[SHIFTED]["relation_accuracy"],
        "gain_over_reversed": learned_direct - methods[REVERSED]["relation_accuracy"],
        "gain_over_first_last": learned_direct - methods[FIRST_LAST]["relation_accuracy"],
        "gain_over_uniform": learned_direct - methods[UNIFORM]["relation_accuracy"],
        "gain_over_shuffled_credit": learned_direct - methods[SHUFFLED]["relation_accuracy"],
        "gain_over_random_boundary": learned_direct - methods[RANDOM]["relation_accuracy"],
    }
    causal_gaps = {
        "shifted_sequence_causal_gap": (
            methods[PRIMARY]["complete_sequence_identity_rate"]
            - methods[SHIFTED]["complete_sequence_identity_rate"]
        ),
        "reversed_sequence_causal_gap": (
            methods[PRIMARY]["complete_sequence_identity_rate"]
            - methods[REVERSED]["complete_sequence_identity_rate"]
        ),
        "boundary_ablation_causal_gap": (
            methods[PRIMARY]["complete_sequence_identity_rate"]
            - methods[BOUNDARY_ABLATED]["complete_sequence_identity_rate"]
        ),
        "maintenance_causal_gap": (
            methods[COMBINED]["complete_sequence_identity_rate"]
            - methods[NO_MAINTENANCE]["complete_sequence_identity_rate"]
        ),
        "first_last_sequence_causal_gap": (
            methods[PRIMARY]["complete_sequence_identity_rate"]
            - methods[FIRST_LAST]["complete_sequence_identity_rate"]
        ),
    }
    information_boundary = {
        "continuous_stream_only_at_runtime": True,
        "endpoint_segmentation_absent": True,
        "true_boundary_reads_by_core": 0,
        "true_relation_reads_before_decision": 0,
        "held_query_formation_updates": formation["held_query_feedback_updates"],
        "explicit_boundary_arguments": (
            direct_audit["explicit_boundary_arguments"]
            + plan_audit["explicit_boundary_arguments"]
        ),
        "explicit_relation_arguments": (
            direct_audit["explicit_relation_arguments"]
            + plan_audit["explicit_relation_arguments"]
        ),
        "runtime_planning_calls": plan_audit["runtime_planning_calls"],
        "consequence_basis_hidden_from_core": True,
        "generic_ordered_pair_vocabulary_supplied": True,
    }
    control_identity = {
        "formation_and_direct_namespaces_disjoint": (
            formation["formation_identity_hash"] != direct_audit["evaluation_identity_hash"]
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
        "temporal_pair_vocabulary_hash_exact": (
            snapshot.pair_vocabulary_hash == payload_hash(PAIR_INDICES)
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
        "control_gains": control_gains,
        "causal_gaps": causal_gaps,
        "information_boundary": information_boundary,
        "control_identity": control_identity,
        "runtime_audit": circuit.audit(),
        "snapshot_sha256": hashlib.sha256(snapshot.to_json().encode("utf-8")).hexdigest(),
    }
    case["case_sha256"] = payload_hash(case)
    return case


def _mean(rows: Iterable[float]) -> float:
    values = list(float(value) for value in rows)
    return float(np.mean(values)) if values else 0.0


def _minimum(rows: Iterable[float]) -> float:
    values = list(float(value) for value in rows)
    return float(np.min(values)) if values else 0.0


def _maximum(rows: Iterable[float]) -> float:
    values = list(float(value) for value in rows)
    return float(np.max(values)) if values else 0.0


def aggregate_cases(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    methods: dict[str, dict[str, float]] = {}
    for method in METHODS:
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
            "proposal_identity_rate_to_explicit_upper": _mean(
                case["methods"][method]["proposal_identity_rate_to_explicit_upper"]
                for case in cases
            ),
        }

    formation = {
        "case_count": len(cases),
        "mean_unseen_relation_accuracy": methods[PRIMARY]["relation_accuracy"],
        "minimum_unseen_relation_accuracy": methods[PRIMARY]["minimum_relation_accuracy"],
        "minimum_support_enrichment": _minimum(
            case["direct_audit"]["boundary_recovery_on_unseen_streams"][
                "support_enrichment"
            ]
            for case in cases
        ),
        "mean_support_enrichment": _mean(
            case["direct_audit"]["boundary_recovery_on_unseen_streams"][
                "support_enrichment"
            ]
            for case in cases
        ),
        "minimum_top_pair_support_enrichment": _minimum(
            case["direct_audit"]["boundary_recovery_on_unseen_streams"][
                "top_pair_support_enrichment"
            ]
            for case in cases
        ),
        "maximum_temporal_center_mae": _maximum(
            case["direct_audit"]["boundary_recovery_on_unseen_streams"][
                "temporal_center_mae"
            ]
            for case in cases
        ),
        "minimum_dominant_pair_support_frequency": _minimum(
            case["direct_audit"]["boundary_recovery_on_unseen_streams"][
                "dominant_pair_support_frequency"
            ]
            for case in cases
        ),
        "maximum_positive_pair_fraction": _maximum(
            case["direct_audit"]["boundary_recovery_on_unseen_streams"][
                "positive_pair_fraction"
            ]
            for case in cases
        ),
        "minimum_gain_over_shifted": _minimum(
            case["control_gains"]["gain_over_shifted"] for case in cases
        ),
        "minimum_gain_over_reversed": _minimum(
            case["control_gains"]["gain_over_reversed"] for case in cases
        ),
        "minimum_gain_over_first_last": _minimum(
            case["control_gains"]["gain_over_first_last"] for case in cases
        ),
        "minimum_gain_over_uniform": _minimum(
            case["control_gains"]["gain_over_uniform"] for case in cases
        ),
        "minimum_gain_over_shuffled_credit": _minimum(
            case["control_gains"]["gain_over_shuffled_credit"] for case in cases
        ),
        "minimum_gain_over_random_boundary": _minimum(
            case["control_gains"]["gain_over_random_boundary"] for case in cases
        ),
        "all_initial_fields_zero": all(
            case["formation"]["initial_temporal_weight_norm"] == 0.0 for case in cases
        ),
        "all_learned_fields_nonzero": all(
            case["formation"]["learned_temporal_weight_norm"] > 0.0 for case in cases
        ),
        "all_feedback_after_required_delay": all(
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
            max(case["formation"]["snapshot_forbidden_token_count"] for case in cases)
        ),
        "all_boundary_masks_unsupplied": all(
            case["formation"]["learner_audit"]["supplied_boundary_masks"] == 0
            for case in cases
        ),
        "all_boundary_indices_unsupplied": all(
            case["formation"]["learner_audit"]["supplied_boundary_indices"] == 0
            for case in cases
        ),
        "all_correct_routes_unsupplied": all(
            case["formation"]["learner_audit"]["supplied_correct_routes"] == 0
            for case in cases
        ),
    }
    persistence = {
        "case_count": len(cases),
        "clean_sequence_identity_rate": methods[PRIMARY][
            "complete_sequence_identity_rate"
        ],
        "delay_sequence_identity_rate": methods[DELAY][
            "complete_sequence_identity_rate"
        ],
        "partial_sequence_identity_rate": methods[PARTIAL][
            "complete_sequence_identity_rate"
        ],
        "noise_sequence_identity_rate": methods[NOISE][
            "complete_sequence_identity_rate"
        ],
        "combined_sequence_identity_rate": methods[COMBINED][
            "complete_sequence_identity_rate"
        ],
        "minimum_robust_relation_accuracy": min(
            methods[method]["minimum_relation_accuracy"] for method in ROBUST_METHODS
        ),
        "shifted_sequence_identity_rate": methods[SHIFTED][
            "complete_sequence_identity_rate"
        ],
        "reversed_sequence_identity_rate": methods[REVERSED][
            "complete_sequence_identity_rate"
        ],
        "boundary_ablated_sequence_identity_rate": methods[BOUNDARY_ABLATED][
            "complete_sequence_identity_rate"
        ],
        "no_maintenance_sequence_identity_rate": methods[NO_MAINTENANCE][
            "complete_sequence_identity_rate"
        ],
        "first_last_sequence_identity_rate": methods[FIRST_LAST][
            "complete_sequence_identity_rate"
        ],
        "shifted_sequence_causal_gap": (
            methods[PRIMARY]["complete_sequence_identity_rate"]
            - methods[SHIFTED]["complete_sequence_identity_rate"]
        ),
        "reversed_sequence_causal_gap": (
            methods[PRIMARY]["complete_sequence_identity_rate"]
            - methods[REVERSED]["complete_sequence_identity_rate"]
        ),
        "boundary_ablation_causal_gap": (
            methods[PRIMARY]["complete_sequence_identity_rate"]
            - methods[BOUNDARY_ABLATED]["complete_sequence_identity_rate"]
        ),
        "maintenance_causal_gap": (
            methods[COMBINED]["complete_sequence_identity_rate"]
            - methods[NO_MAINTENANCE]["complete_sequence_identity_rate"]
        ),
        "first_last_sequence_causal_gap": (
            methods[PRIMARY]["complete_sequence_identity_rate"]
            - methods[FIRST_LAST]["complete_sequence_identity_rate"]
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
        "temporal_boundary_formation": formation,
        "persistence": persistence,
        "structural_audits": structural,
    }


def decide(aggregate: dict[str, Any]) -> dict[str, Any]:
    if FROZEN_THRESHOLDS is None:
        raise RuntimeError("ECTBF v1 thresholds are not frozen")
    threshold = FROZEN_THRESHOLDS
    formation = aggregate["temporal_boundary_formation"]
    persistence = aggregate["persistence"]
    structural = aggregate["structural_audits"]
    audit_checks = {
        "all_structural_boolean_audits": all(
            value for value in structural.values() if isinstance(value, bool)
        ),
        "initial_fields_zero": formation["all_initial_fields_zero"],
        "learned_fields_nonzero": formation["all_learned_fields_nonzero"],
        "feedback_delayed": formation["all_feedback_after_required_delay"],
        "held_query_updates_zero": formation["all_held_query_updates_zero"],
        "snapshot_roundtrip": formation["all_snapshot_roundtrips_exact"],
        "snapshot_forbidden_zero": formation[
            "maximum_snapshot_forbidden_token_count"
        ]
        == 0,
        "boundary_masks_unsupplied": formation["all_boundary_masks_unsupplied"],
        "boundary_indices_unsupplied": formation[
            "all_boundary_indices_unsupplied"
        ],
        "correct_routes_unsupplied": formation["all_correct_routes_unsupplied"],
    }
    boundary_checks = {
        "minimum_unseen_relation_accuracy": (
            formation["minimum_unseen_relation_accuracy"]
            >= threshold["minimum_unseen_relation_accuracy"]
        ),
        "minimum_support_enrichment": (
            formation["minimum_support_enrichment"]
            >= threshold["minimum_support_enrichment"]
        ),
        "minimum_top_pair_support_enrichment": (
            formation["minimum_top_pair_support_enrichment"]
            >= threshold["minimum_top_pair_support_enrichment"]
        ),
        "maximum_temporal_center_mae": (
            formation["maximum_temporal_center_mae"]
            <= threshold["maximum_temporal_center_mae"]
        ),
        "minimum_gain_over_shifted": (
            formation["minimum_gain_over_shifted"]
            >= threshold["minimum_gain_over_shifted"]
        ),
        "minimum_gain_over_first_last": (
            formation["minimum_gain_over_first_last"]
            >= threshold["minimum_gain_over_first_last"]
        ),
        "minimum_gain_over_uniform": (
            formation["minimum_gain_over_uniform"]
            >= threshold["minimum_gain_over_uniform"]
        ),
        "minimum_gain_over_shuffled_credit": (
            formation["minimum_gain_over_shuffled_credit"]
            >= threshold["minimum_gain_over_shuffled_credit"]
        ),
        "minimum_gain_over_random_boundary": (
            formation["minimum_gain_over_random_boundary"]
            >= threshold["minimum_gain_over_random_boundary"]
        ),
        "maximum_positive_pair_fraction": (
            formation["maximum_positive_pair_fraction"]
            <= threshold["maximum_positive_pair_fraction"]
        ),
    }
    persistence_checks = {
        "clean_sequence_identity": (
            persistence["clean_sequence_identity_rate"]
            >= threshold["clean_sequence_identity_rate"]
        ),
        "delay_sequence_identity": (
            persistence["delay_sequence_identity_rate"]
            >= threshold["delay_sequence_identity_rate"]
        ),
        "partial_sequence_identity": (
            persistence["partial_sequence_identity_rate"]
            >= threshold["partial_sequence_identity_rate"]
        ),
        "noise_sequence_identity": (
            persistence["noise_sequence_identity_rate"]
            >= threshold["noise_sequence_identity_rate"]
        ),
        "combined_sequence_identity": (
            persistence["combined_sequence_identity_rate"]
            >= threshold["combined_sequence_identity_rate"]
        ),
        "minimum_robust_relation_accuracy": (
            persistence["minimum_robust_relation_accuracy"]
            >= threshold["minimum_robust_relation_accuracy"]
        ),
    }
    causal_checks = {
        "shifted_sequence_causal_gap": (
            persistence["shifted_sequence_causal_gap"]
            >= threshold["shifted_sequence_causal_gap"]
        ),
        "boundary_ablation_causal_gap": (
            persistence["boundary_ablation_causal_gap"]
            >= threshold["boundary_ablation_causal_gap"]
        ),
        "maintenance_causal_gap": (
            persistence["maintenance_causal_gap"]
            >= threshold["maintenance_causal_gap"]
        ),
        "first_last_sequence_causal_gap": (
            persistence["first_last_sequence_causal_gap"]
            >= threshold["first_last_sequence_causal_gap"]
        ),
    }
    audits_pass = all(audit_checks.values())
    boundary_pass = all(boundary_checks.values())
    persistence_pass = all(persistence_checks.values())
    causal_pass = all(causal_checks.values())
    robust_clean = (
        persistence_checks["clean_sequence_identity"]
        and persistence_checks["minimum_robust_relation_accuracy"]
    )
    if not audits_pass:
        status = STATUS_INVALID
    elif boundary_pass and persistence_pass and causal_pass:
        status = STATUS_POSITIVE
    elif persistence_pass and causal_pass and not boundary_pass:
        status = STATUS_PLAN_ONLY
    elif boundary_pass and not persistence_pass:
        status = STATUS_BOUNDARY_ONLY
    elif robust_clean:
        status = STATUS_FRAGILE
    else:
        status = STATUS_NEGATIVE
    return {
        "status": status,
        "candidate_promoted": status == STATUS_POSITIVE,
        "all_audits_pass": audits_pass,
        "boundary_pass": boundary_pass,
        "persistence_pass": persistence_pass,
        "causal_pass": causal_pass,
        "audit_checks": audit_checks,
        "boundary_checks": boundary_checks,
        "persistence_checks": persistence_checks,
        "causal_checks": causal_checks,
    }


def ancestry_record() -> dict[str, Any]:
    record: dict[str, Any] = {"branch": ANCESTRY_BRANCH}
    for label, path_text in (
        ("completion_receipt", ANCESTRY_COMPLETION_PATH),
        ("formal_summary", ANCESTRY_FORMAL_SUMMARY_PATH),
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
            }
        else:
            record[label] = {"path": path_text, "missing_in_local_probe": True}
    return record


def source_manifest() -> tuple[str, dict[str, str]]:
    paths = (
        "endogenous_temporal_consequence_boundary_formation_config_v1.py",
        "endogenous_temporal_consequence_boundary_formation_v1.py",
        "run_endogenous_temporal_consequence_boundary_formation_v1.py",
        "research/protocols/endogenous_temporal_consequence_boundary_formation_v1.md",
        ANCESTRY_COMPLETION_PATH,
        ANCESTRY_FORMAL_SUMMARY_PATH,
    )
    members: dict[str, str] = {}
    for path_text in paths:
        path = Path(path_text)
        if path.exists():
            members[path_text] = hashlib.sha256(path.read_bytes()).hexdigest()
    return payload_hash(members), members


def write_markdown(result: dict[str, Any], path: Path) -> None:
    aggregate = result["aggregate"]
    formation = aggregate["temporal_boundary_formation"]
    persistence = aggregate["persistence"]
    lines = [
        f"# {STUDY}",
        "",
        f"Phase: **{result['phase']}**",
        "",
        "## Temporal boundary formation",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Cases | {result['case_count']} |",
        f"| Mean unseen relation accuracy | {formation['mean_unseen_relation_accuracy']:.4f} |",
        f"| Minimum unseen relation accuracy | {formation['minimum_unseen_relation_accuracy']:.4f} |",
        f"| Minimum support enrichment | {formation['minimum_support_enrichment']:.4f} |",
        f"| Minimum top-pair support enrichment | {formation['minimum_top_pair_support_enrichment']:.4f} |",
        f"| Maximum temporal center MAE | {formation['maximum_temporal_center_mae']:.4f} |",
        f"| Minimum gain over shifted field | {formation['minimum_gain_over_shifted']:.4f} |",
        f"| Minimum gain over shuffled credit | {formation['minimum_gain_over_shuffled_credit']:.4f} |",
        "",
        "## Plan continuation",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Clean sequence identity | {persistence['clean_sequence_identity_rate']:.4f} |",
        f"| Delay sequence identity | {persistence['delay_sequence_identity_rate']:.4f} |",
        f"| Partial sequence identity | {persistence['partial_sequence_identity_rate']:.4f} |",
        f"| Noise sequence identity | {persistence['noise_sequence_identity_rate']:.4f} |",
        f"| Combined sequence identity | {persistence['combined_sequence_identity_rate']:.4f} |",
        f"| Shifted-field causal gap | {persistence['shifted_sequence_causal_gap']:.4f} |",
        f"| Boundary-ablation causal gap | {persistence['boundary_ablation_causal_gap']:.4f} |",
        f"| Maintenance causal gap | {persistence['maintenance_causal_gap']:.4f} |",
        "",
        "## Methods",
        "",
        "| Method | Relation accuracy | Step accuracy | Complete sequence |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        row = aggregate["methods"][method]
        lines.append(
            f"| {method} | {row['relation_accuracy']:.4f} | "
            f"{row['step_accuracy']:.4f} | "
            f"{row['complete_sequence_identity_rate']:.4f} |"
        )
    if result.get("decision") is not None:
        lines.extend(["", f"Formal status: `{result['decision']['status']}`"])
    lines.extend(["", f"Source manifest: `{result['source_manifest_sha256']}`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_case_csv(cases: Sequence[dict[str, Any]], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for case in cases:
        recovery = case["direct_audit"]["boundary_recovery_on_unseen_streams"]
        rows.append(
            {
                "family_seed": case["family_seed"],
                "held_query": case["held_query"],
                "suite": case["suite"],
                "primary_relation_accuracy": case["methods"][PRIMARY][
                    "relation_accuracy"
                ],
                "primary_complete_sequence": case["methods"][PRIMARY][
                    "complete_sequence_identity_rate"
                ],
                "support_enrichment": recovery["support_enrichment"],
                "top_pair_support_enrichment": recovery[
                    "top_pair_support_enrichment"
                ],
                "temporal_center_mae": recovery["temporal_center_mae"],
                "gain_over_shifted": case["control_gains"]["gain_over_shifted"],
                "gain_over_shuffled_credit": case["control_gains"][
                    "gain_over_shuffled_credit"
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
        for held_query in range(QUERY_COUNT):
            for suite in SUITES:
                cases.append(run_case(family_seed, held_query, dict(suite)))
    aggregate = aggregate_cases(cases)
    source_sha, source_members = source_manifest()
    result: dict[str, Any] = {
        "study": STUDY,
        "target_branch": TARGET_BRANCH,
        "phase": phase,
        "family_seeds": list(int(seed) for seed in family_seeds),
        "query_count": QUERY_COUNT,
        "suite_names": [suite["name"] for suite in SUITES],
        "case_count": len(cases),
        "pair_count": PAIR_COUNT,
        "formation_episodes_per_case": (
            (QUERY_COUNT - 1) * FORMATION_EPISODES_PER_OBSERVED_QUERY
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
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    output_path.with_suffix(".sha256").write_text(digest + "  " + output_path.name + "\n")
    write_markdown(result, output_path.with_suffix(".md"))
    write_case_csv(cases, output_path.with_name(output_path.stem + "_cases.csv"))
    return result
