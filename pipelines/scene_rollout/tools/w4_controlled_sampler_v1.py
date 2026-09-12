#!/usr/bin/env python3
"""Deterministic, rank-aware sampler contract for W4 controlled correction V1."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable


SAMPLER_VERSION = "w4_controlled_correction_v1"
STRATA = ("strict_zero", "boundary", "person_positive", "natural_anchor")
DEFAULT_QUOTAS = {"strict_zero": 3, "boundary": 3, "person_positive": 4, "natural_anchor": 2}


def stable_key(seed: int, *parts: object) -> str:
    value = "\0".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def record_id(row: dict[str, Any]) -> str:
    value = row.get("clip_id")
    if not value:
        raise ValueError("W4 rows require a non-empty clip_id")
    return str(value)


def record_uid(row: dict[str, Any]) -> str:
    """Stable identity across manifest reordering, including intentional cross-release overlaps."""
    identity = {
        "release": int(row.get("_training_release_index", 0)),
        "clip_id": record_id(row),
        "latent_cache": row.get("latent_cache"),
        "context_aug": row.get("context_aug"),
        "roles": row.get("map_memory_selection_roles"),
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def match_id(row: dict[str, Any]) -> str:
    value = row.get("map_memory_match_id") or row.get("game_id")
    if not value:
        raise ValueError(f"{record_id(row)}: W4 rows require map_memory_match_id or game_id")
    return str(value)


def roles(row: dict[str, Any]) -> tuple[str, ...]:
    value = tuple(str(item) for item in row.get("map_memory_selection_roles", ()))
    if len(value) != 21 or any(item not in {"context", "positive"} for item in value):
        raise ValueError(f"{record_id(row)}: expected exactly 21 context/positive roles")
    return value


def boundary_subtype(row: dict[str, Any]) -> str | None:
    value = roles(row)
    enter = any(a == "context" and b == "positive" for a, b in zip(value, value[1:]))
    exit_ = any(a == "positive" and b == "context" for a, b in zip(value, value[1:]))
    if enter and exit_:
        return "both"
    if enter:
        return "enter"
    if exit_:
        return "exit"
    return None


def positive_bin(row: dict[str, Any]) -> str:
    count = sum(item == "positive" for item in roles(row))
    if count <= 0:
        return "zero"
    if count <= 3:
        return "01_03"
    if count <= 7:
        return "04_07"
    if count <= 14:
        return "08_14"
    return "15_21"


def validate_quotas(quotas: dict[str, int], *, world: int, grad_accum: int) -> dict[str, int]:
    if set(quotas) != set(STRATA):
        missing = sorted(set(STRATA) - set(quotas))
        extra = sorted(set(quotas) - set(STRATA))
        raise ValueError(f"W4 quotas require each stratum exactly once; missing={missing} extra={extra}")
    normalized = {name: int(quotas[name]) for name in STRATA}
    if any(value <= 0 for value in normalized.values()):
        raise ValueError(f"W4 quotas must be positive: {normalized}")
    expected = world * grad_accum
    if sum(normalized.values()) != expected:
        raise ValueError(f"W4 quota total {sum(normalized.values())} != world*grad_accum {expected}")
    return normalized


def quota_schedule(quotas: dict[str, int], *, world: int, grad_accum: int) -> tuple[str, ...]:
    quotas = validate_quotas(quotas, world=world, grad_accum=grad_accum)
    # Interleave quotas so ranks do not specialize in one source while preserving exact global counts.
    remaining = dict(quotas)
    schedule: list[str] = []
    while any(remaining.values()):
        for name in STRATA:
            if remaining[name]:
                schedule.append(name)
                remaining[name] -= 1
    return tuple(schedule)


def global_slot(*, micro_step: int, rank: int, world: int, grad_accum: int) -> int:
    if not 0 <= rank < world or not 0 <= micro_step < grad_accum:
        raise ValueError(f"invalid W4 rank/micro_step rank={rank}/{world} micro={micro_step}/{grad_accum}")
    return micro_step * world + rank


@dataclass(frozen=True)
class Selection:
    stratum: str
    record_index: int
    record_id: str
    match_id: str
    subgroup: str
    stratum_ordinal: int


class W4SamplerV1:
    """Order-independent strata with subgroup/match round robin and hashed rows."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        strict_zero_release_indices: Iterable[int],
        seed: int,
        world: int,
        grad_accum: int,
        quotas: dict[str, int] | None = None,
    ) -> None:
        self.records = records
        self.seed = int(seed)
        self.world = int(world)
        self.grad_accum = int(grad_accum)
        self.quotas = validate_quotas(quotas or DEFAULT_QUOTAS, world=world, grad_accum=grad_accum)
        self.schedule = quota_schedule(self.quotas, world=world, grad_accum=grad_accum)
        strict_indices = {int(value) for value in strict_zero_release_indices}
        if not strict_indices:
            raise ValueError("W4 strict_zero_release_indices must not be empty")
        pools: dict[str, list[int]] = {name: [] for name in STRATA}
        subgroups: dict[tuple[str, int], str] = {}
        for index, row in enumerate(records):
            release_index = int(row.get("_training_release_index", 0))
            row_roles = roles(row)
            if release_index in strict_indices:
                if any(role != "context" for role in row_roles):
                    raise ValueError(f"{record_id(row)}: strict_zero row contains a positive role")
                pools["strict_zero"].append(index)
                subgroups[("strict_zero", index)] = "strict_zero"
                continue
            subtype = boundary_subtype(row)
            if subtype:
                pools["boundary"].append(index)
                subgroups[("boundary", index)] = subtype
            count = sum(role == "positive" for role in row_roles)
            if count:
                pools["person_positive"].append(index)
                subgroups[("person_positive", index)] = positive_bin(row)
            pools["natural_anchor"].append(index)
            subgroups[("natural_anchor", index)] = "natural"
        missing = [name for name, values in pools.items() if not values]
        if missing:
            raise ValueError(f"W4 sampler has empty required strata: {missing}")
        self.pools = {name: tuple(values) for name, values in pools.items()}
        self._cells: dict[str, tuple[tuple[str, str], ...]] = {}
        self._cell_rows: dict[tuple[str, str, str], tuple[int, ...]] = {}
        for stratum, indices in self.pools.items():
            grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
            for index in indices:
                subgroup = subgroups[(stratum, index)]
                grouped[(subgroup, match_id(records[index]))].append(index)
            cells = sorted(grouped, key=lambda cell: stable_key(seed, stratum, *cell))
            self._cells[stratum] = tuple(cells)
            for subgroup, match in cells:
                values = sorted(
                    grouped[(subgroup, match)],
                    key=lambda index: stable_key(seed, stratum, subgroup, match, record_uid(records[index])),
                )
                self._cell_rows[(stratum, subgroup, match)] = tuple(values)

    def stratum_for(self, *, micro_step: int, rank: int) -> str:
        return self.schedule[global_slot(micro_step=micro_step, rank=rank, world=self.world, grad_accum=self.grad_accum)]

    def select(self, *, sampler_step: int, micro_step: int, rank: int) -> Selection:
        if sampler_step < 1:
            raise ValueError("W4 sampler_step is one-based and must be >= 1")
        slot = global_slot(micro_step=micro_step, rank=rank, world=self.world, grad_accum=self.grad_accum)
        stratum = self.schedule[slot]
        prior_per_step = self.schedule[:slot].count(stratum)
        ordinal = (sampler_step - 1) * self.quotas[stratum] + prior_per_step
        cells = self._cells[stratum]
        subgroup, match = cells[ordinal % len(cells)]
        cell_cycle = ordinal // len(cells)
        choices = self._cell_rows[(stratum, subgroup, match)]
        index = choices[cell_cycle % len(choices)]
        row = self.records[index]
        return Selection(stratum, index, record_id(row), match_id(row), subgroup, ordinal)

    def metadata(self) -> dict[str, Any]:
        return {
            "version": SAMPLER_VERSION,
            "seed": self.seed,
            "world": self.world,
            "grad_accum": self.grad_accum,
            "quotas": self.quotas,
            "schedule": list(self.schedule),
            "pool_counts": {name: len(values) for name, values in self.pools.items()},
            "cell_counts": {name: len(values) for name, values in self._cells.items()},
        }
