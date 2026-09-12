#!/usr/bin/env python3
"""Interaction-channel token utilities for Memory dense adapter v1."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from interaction_channels_v0 import WEAPON_VOCABULARY


_EVENT_FIELDS = (
    "ego_fire_event",
    "ego_reload_event",
    "ego_weapon_switch_event",
    "ego_throw_event",
)
_ONEHOT_FIELDS = (
    "ego_current_weapon_onehot",
    "ego_fire_weapon_onehot",
    "ego_reload_weapon_onehot",
    "ego_switch_target_onehot",
    "ego_throw_item_onehot",
)
_EVENT_TARGET_FIELDS = (
    ("ego_fire_event", "ego_fire_weapon_onehot"),
    ("ego_reload_event", "ego_reload_weapon_onehot"),
    ("ego_weapon_switch_event", "ego_switch_target_onehot"),
    ("ego_throw_event", "ego_throw_item_onehot"),
)

INTERACTION_CHANNELS_V1 = [
    *_EVENT_FIELDS,
    *(
        f"{field}.{weapon}"
        for field in _ONEHOT_FIELDS
        for weapon in WEAPON_VOCABULARY
    ),
]

_FEATURE_DIM = 4 + 5 * len(WEAPON_VOCABULARY)
if len(INTERACTION_CHANNELS_V1) != _FEATURE_DIM or _FEATURE_DIM != 134:
    raise RuntimeError(
        "interaction feature protocol must be 4 + 5*26 = 134 dimensions, "
        f"got {len(INTERACTION_CHANNELS_V1)}"
    )


def _cache_error(cache_path: str, field: str, detail: str) -> ValueError:
    return ValueError(f"interaction cache {cache_path}: field={field}: {detail}")


def _validate_float32_array(
    value: np.ndarray,
    *,
    cache_path: str,
    field: str,
    expected_shape: tuple[int, ...],
) -> None:
    if value.shape != expected_shape:
        raise _cache_error(
            cache_path,
            field,
            f"shape={list(value.shape)}, expected={list(expected_shape)}",
        )
    if value.dtype != np.dtype(np.float32):
        raise _cache_error(
            cache_path,
            field,
            f"dtype={value.dtype}, expected=float32",
        )

    invalid = np.argwhere(~np.isfinite(value))
    if invalid.size:
        index = tuple(int(x) for x in invalid[0])
        raise _cache_error(
            cache_path,
            field,
            f"frame_index={index[0]} index={index} contains non-finite value",
        )

    non_binary = np.argwhere((value != 0.0) & (value != 1.0))
    if non_binary.size:
        index = tuple(int(x) for x in non_binary[0])
        raise _cache_error(
            cache_path,
            field,
            f"frame_index={index[0]} index={index} value={value[index]} is not binary",
        )


class InteractionTokenProjector(nn.Module):
    """Zero-init projection from per-frame interaction features to Wan tokens."""

    def __init__(
        self,
        interaction_channels: int = _FEATURE_DIM,
        cond_dim: int = 128,
    ):
        super().__init__()

        self.interaction_channels = int(interaction_channels)
        self.cond_dim = int(cond_dim)
        if self.interaction_channels != _FEATURE_DIM:
            raise ValueError(
                f"interaction_channels must match the {_FEATURE_DIM}-dim v1 "
                f"protocol, got {self.interaction_channels}"
            )

        self.proj = nn.Linear(_FEATURE_DIM, self.cond_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        interaction: torch.Tensor,
        *,
        target_token_hw: tuple[int, int],
    ) -> torch.Tensor:
        if interaction.ndim != 2:
            raise ValueError(
                "interaction must be [F,134], "
                f"got {tuple(interaction.shape)}"
            )

        frames, features = interaction.shape
        if features != _FEATURE_DIM:
            raise ValueError(
                f"expected {_FEATURE_DIM} interaction features, got {features}"
            )

        if len(target_token_hw) != 2:
            raise ValueError(f"target_token_hw must be (Ht,Wt), got {target_token_hw}")
        token_h, token_w = (int(value) for value in target_token_hw)
        if token_h <= 0 or token_w <= 0:
            raise ValueError(
                f"target_token_hw values must be positive, got {(token_h, token_w)}"
            )

        projected = self.proj(interaction.to(dtype=self.proj.weight.dtype))
        return (
            projected[:, None, None, :]
            .expand(frames, token_h, token_w, self.cond_dim)
            .reshape(1, frames * token_h * token_w, self.cond_dim)
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "interaction_token_projector_v1",
            "channels": INTERACTION_CHANNELS_V1,
            "feature_dim": _FEATURE_DIM,
            "feature_names": INTERACTION_CHANNELS_V1,
            "weapon_vocabulary": list(WEAPON_VOCABULARY),
            "cond_dim": self.cond_dim,
            "projection": f"linear_{_FEATURE_DIM}_to_{self.cond_dim}",
            "zero_init": True,
            "input_protocol": "[F,134] float interaction features",
            "output_protocol": "[1,F*Ht*Wt,cond_dim] Wan Dense Condition-aligned tokens",
        }


def load_interaction_tensor(
    row: dict[str, Any],
    *,
    frame_indices: list[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    cache_path = str(row.get("state_cache", ""))
    if not cache_path:
        raise ValueError("interaction cache path is missing from row field=state_cache")

    normalized_indices: list[int] = []
    for position, index in enumerate(frame_indices):
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
            raise _cache_error(
                cache_path,
                "frame_indices",
                f"frame_index_position={position} value={index!r} is not an integer",
            )
        normalized_indices.append(int(index))

    with np.load(cache_path, allow_pickle=False) as data:
        required = _EVENT_FIELDS + _ONEHOT_FIELDS
        missing = [field for field in required if field not in data.files]
        if missing:
            raise _cache_error(cache_path, missing[0], f"missing fields={missing}")

        first_event = data[_EVENT_FIELDS[0]]
        if first_event.ndim != 1:
            raise _cache_error(
                cache_path,
                _EVENT_FIELDS[0],
                f"shape={list(first_event.shape)}, expected=[F]",
            )
        frame_count = int(first_event.shape[0])
        vocabulary_size = len(WEAPON_VOCABULARY)
        arrays: dict[str, np.ndarray] = {}
        for field in _EVENT_FIELDS:
            value = data[field]
            _validate_float32_array(
                value,
                cache_path=cache_path,
                field=field,
                expected_shape=(frame_count,),
            )
            arrays[field] = value

        for field in _ONEHOT_FIELDS:
            value = data[field]
            _validate_float32_array(
                value,
                cache_path=cache_path,
                field=field,
                expected_shape=(frame_count, vocabulary_size),
            )
            arrays[field] = value

        for index in normalized_indices:
            if index < 0 or index >= frame_count:
                raise _cache_error(
                    cache_path,
                    "frame_indices",
                    f"frame_index={index} outside [0,{frame_count - 1}]",
                )

        current_sums = arrays["ego_current_weapon_onehot"].sum(axis=1)
        invalid_current = np.flatnonzero(current_sums != 1.0)
        if invalid_current.size:
            index = int(invalid_current[0])
            raise _cache_error(
                cache_path,
                "ego_current_weapon_onehot",
                f"frame_index={index} row_sum={current_sums[index]}, expected=1",
            )

        for event_field, target_field in _EVENT_TARGET_FIELDS:
            target_sums = arrays[target_field].sum(axis=1)
            mismatch = np.flatnonzero(target_sums != arrays[event_field])
            if mismatch.size:
                index = int(mismatch[0])
                raise _cache_error(
                    cache_path,
                    target_field,
                    f"frame_index={index} row_sum={target_sums[index]} does not "
                    f"match {event_field}={arrays[event_field][index]}",
                )

        selected = np.asarray(normalized_indices, dtype=np.int64)
        features = np.concatenate(
            [
                *(arrays[field][selected, None] for field in _EVENT_FIELDS),
                *(arrays[field][selected] for field in _ONEHOT_FIELDS),
            ],
            axis=1,
            dtype=np.float32,
        )

    if features.shape != (len(normalized_indices), _FEATURE_DIM):
        raise _cache_error(
            cache_path,
            "concatenated_features",
            f"shape={list(features.shape)}, expected=[{len(normalized_indices)},{_FEATURE_DIM}]",
        )
    return torch.from_numpy(features).to(device=device, dtype=dtype)
