#!/usr/bin/env python3
"""Latent-aligned CS:GO action features and a zero-init token projector.

This module is intentionally independent from the current state-channel trainer.
It provides the data contract needed for an opt-in action-conditioning experiment
without changing a running checkpoint or its no-op initialization behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn


ACTION_CONTROL_CHANNELS_V0 = (
    "forward_any",
    "back_any",
    "left_any",
    "right_any",
    "jump_any",
    "crouch_any",
    "walk_any",
    "fire_any",
    "reload_any",
    "use_any",
    "scope_any",
    "inspect_any",
    "plant_any",
    "defuse_any",
    "weapon_switch_any",
)

ACTION_CONTINUOUS_CHANNELS_V0 = (
    "look_dx_sum_tanh45",
    "look_dy_sum_tanh45",
    "fire_fraction",
    "reload_fraction",
)

WEAPON_CLASSES_V0 = (
    "none",
    "pistol",
    "rifle",
    "sniper",
    "smg",
    "shotgun",
    "heavy",
    "grenade",
    "knife",
    "c4",
    "other",
)

ACTION_STATE_CHANNELS_V0 = (
    "ammo_magazine_clip100",
    "ammo_reserve_clip300",
    *(f"weapon_class_{name}" for name in WEAPON_CLASSES_V0),
)

ACTION_CHANNELS_V0 = (
    *ACTION_CONTROL_CHANNELS_V0,
    *ACTION_CONTINUOUS_CHANNELS_V0,
    *ACTION_STATE_CHANNELS_V0,
)

RAW_CONTROL_FIELDS_V0 = ACTION_CONTROL_CHANNELS_V0[:-1]

_PISTOLS = {
    "CZ75-Auto", "Desert Eagle", "Dual Berettas", "Five-SeveN", "Glock-18",
    "P2000", "P250", "R8 Revolver", "Tec-9", "USP-S",
}
_RIFLES = {"AK-47", "AUG", "FAMAS", "Galil AR", "M4A1", "M4A4", "SG 553"}
_SNIPERS = {"AWP", "G3SG1", "SCAR-20", "SSG 08"}
_SMGS = {"MAC-10", "MP5-SD", "MP7", "MP9", "P90", "PP-Bizon", "UMP-45"}
_SHOTGUNS = {"MAG-7", "Nova", "Sawed-Off", "XM1014"}
_HEAVY = {"M249", "Negev"}
_GRENADES = {
    "Decoy Grenade", "Flashbang", "HE Grenade", "Incendiary Grenade",
    "Molotov", "Smoke Grenade",
}


def weapon_class_v0(name: str | None) -> str:
    value = str(name or "").strip()
    if not value:
        return "none"
    if value in _PISTOLS:
        return "pistol"
    if value in _RIFLES:
        return "rifle"
    if value in _SNIPERS:
        return "sniper"
    if value in _SMGS:
        return "smg"
    if value in _SHOTGUNS:
        return "shotgun"
    if value in _HEAVY:
        return "heavy"
    if value in _GRENADES or "Grenade" in value:
        return "grenade"
    if value == "Knife" or "Knife" in value:
        return "knife"
    if value == "C4":
        return "c4"
    return "other"


def _active_ammo(frame: dict[str, Any], weapon: str | None) -> tuple[float, float]:
    for item in frame.get("equipment") or []:
        if str(item.get("name") or "") == str(weapon or ""):
            mag = max(0.0, min(100.0, float(item.get("ammo_magazine") or 0.0))) / 100.0
            reserve = max(0.0, min(300.0, float(item.get("ammo_reserve") or 0.0))) / 300.0
            return mag, reserve
    return 0.0, 0.0


def aggregate_action_window_v0(
    frames: Iterable[dict[str, Any]],
    *,
    previous_weapon: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Aggregate raw 32 Hz actions into one causal latent-time feature vector."""
    rows = list(frames)
    if not rows:
        raise ValueError("action aggregation window is empty")
    actions = [row.get("action") or {} for row in rows]

    values: list[float] = []
    for field in RAW_CONTROL_FIELDS_V0:
        raw_field = field.removesuffix("_any")
        values.append(float(any(bool(action.get(raw_field)) for action in actions)))

    weapons = [str(action.get("weapon_slot") or "") for action in actions]
    weapon_trace = ([str(previous_weapon or "")] if previous_weapon is not None else []) + weapons
    weapon_switch = any(a != b for a, b in zip(weapon_trace, weapon_trace[1:]))
    values.append(float(weapon_switch))

    look_dx = sum(float(action.get("look_dx") or 0.0) for action in actions)
    look_dy = sum(float(action.get("look_dy") or 0.0) for action in actions)
    fire_fraction = sum(bool(action.get("fire")) for action in actions) / len(actions)
    reload_fraction = sum(bool(action.get("reload")) for action in actions) / len(actions)
    values.extend(
        [
            float(np.tanh(look_dx / 45.0)),
            float(np.tanh(look_dy / 45.0)),
            float(fire_fraction),
            float(reload_fraction),
        ]
    )

    active_weapon = weapons[-1]
    magazine, reserve = _active_ammo(rows[-1], active_weapon)
    values.extend([magazine, reserve])
    cls = weapon_class_v0(active_weapon)
    values.extend(float(name == cls) for name in WEAPON_CLASSES_V0)

    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (len(ACTION_CHANNELS_V0),):
        raise RuntimeError(f"action vector shape {vector.shape} != {(len(ACTION_CHANNELS_V0),)}")
    debug = {
        "raw_frame_start": int(rows[0].get("frame_count", -1)),
        "raw_frame_end": int(rows[-1].get("frame_count", -1)),
        "raw_frame_count": len(rows),
        "active_weapon": active_weapon,
        "weapon_class": cls,
        "fire_raw_frames": int(sum(bool(action.get("fire")) for action in actions)),
        "reload_raw_frames": int(sum(bool(action.get("reload")) for action in actions)),
        "weapon_switch": bool(weapon_switch),
    }
    return vector, debug


def load_action_manifest(path: str | Path) -> dict[str, dict[str, Any]]:
    manifest_path = Path(path)
    rows: dict[str, dict[str, Any]] = {}
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            clip_id = str(row["clip_id"])
            if clip_id in rows:
                if rows[clip_id].get("action_cache") != row.get("action_cache"):
                    raise ValueError(f"conflicting action cache rows for {clip_id}")
                continue
            rows[clip_id] = row
    return rows


def resolve_action_cache_path(row: dict[str, Any], manifest_path: str | Path) -> Path:
    path = Path(str(row["action_cache"]))
    if not path.is_absolute():
        path = Path(manifest_path).resolve().parent / path
    return path


def load_action_tensor(
    row: dict[str, Any],
    *,
    frame_indices: list[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    cache_path = Path(str(row["action_cache"]))
    with np.load(cache_path, allow_pickle=False) as data:
        vector = np.asarray(data["action_vector"], dtype=np.float32)
        channels = tuple(str(value) for value in data["channels"].tolist())
    if channels != ACTION_CHANNELS_V0:
        raise ValueError(f"{cache_path}: action channel contract mismatch")
    if max(frame_indices, default=-1) >= len(vector):
        raise ValueError(f"{cache_path}: frame index out of range: {frame_indices} vs {len(vector)}")
    return torch.from_numpy(vector[frame_indices]).to(device=device, dtype=dtype)


class ActionTokenProjector(nn.Module):
    """Project latent-time action vectors and broadcast them over Wan tokens."""

    def __init__(self, *, cond_dim: int = 128, hidden_dim: int = 128) -> None:
        super().__init__()
        c0 = len(ACTION_CONTROL_CHANNELS_V0)
        c1 = c0 + len(ACTION_CONTINUOUS_CHANNELS_V0)
        self.control_slice = slice(0, c0)
        self.continuous_slice = slice(c0, c1)
        self.state_slice = slice(c1, len(ACTION_CHANNELS_V0))
        self.control = nn.Sequential(nn.Linear(c0, hidden_dim), nn.SiLU())
        self.continuous = nn.Sequential(nn.Linear(c1 - c0, hidden_dim), nn.SiLU())
        self.state = nn.Sequential(nn.Linear(len(ACTION_CHANNELS_V0) - c1, hidden_dim), nn.SiLU())
        self.output = nn.Linear(hidden_dim * 3, cond_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.cond_dim = int(cond_dim)
        self.hidden_dim = int(hidden_dim)

    def forward(self, action: torch.Tensor, *, target_token_hw: tuple[int, int]) -> torch.Tensor:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3 or action.shape[-1] != len(ACTION_CHANNELS_V0):
            raise ValueError(
                f"action tensor must be [B,T,{len(ACTION_CHANNELS_V0)}], got {tuple(action.shape)}"
            )
        features = torch.cat(
            [
                self.control(action[..., self.control_slice]),
                self.continuous(action[..., self.continuous_slice]),
                self.state(action[..., self.state_slice]),
            ],
            dim=-1,
        )
        projected = self.output(features)
        h, w = target_token_hw
        return projected[:, :, None, None, :].expand(-1, -1, h, w, -1).reshape(
            action.shape[0], action.shape[1] * h * w, self.cond_dim
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "action_token_projector_v0",
            "channels": list(ACTION_CHANNELS_V0),
            "control_channels": list(ACTION_CONTROL_CHANNELS_V0),
            "continuous_channels": list(ACTION_CONTINUOUS_CHANNELS_V0),
            "state_channels": list(ACTION_STATE_CHANNELS_V0),
            "projection": f"three_branch_mlp_to_{self.cond_dim}",
            "hidden_dim": self.hidden_dim,
            "zero_init_output": True,
            "injection_candidate": "broadcast latent-time bias added to memory_dense_cond_tokens",
        }
