#!/usr/bin/env python3
"""State-channel token utilities for Memory dense adapter v0."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

STATE_CHANNELS_V0 = [
    "ego_alive_constant_plane",
    "ego_health_norm_constant_plane",
    "opponent_dead_marker_mask",
]


def load_state_manifest(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            clip_id = str(row["clip_id"])
            if clip_id in out:
                raise ValueError(f"duplicate clip_id in state manifest: {clip_id}")
            out[clip_id] = row
    return out


def resolve_state_cache_path(row: dict[str, Any], manifest_path: Path) -> Path:
    path = Path(row["state_cache"])
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


class StateTokenProjector(nn.Module):
    """Zero-init state token projection added after the existing dense encoder."""

    def __init__(self, state_channels: int = 3, cond_dim: int = 128):
        super().__init__()
        self.state_channels = int(state_channels)
        self.cond_dim = int(cond_dim)
        self.proj = nn.Linear(self.state_channels, self.cond_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, state: torch.Tensor, *, target_token_hw: tuple[int, int]) -> torch.Tensor:
        if state.ndim != 4:
            raise ValueError(f"state must be [F,C,H,W], got {tuple(state.shape)}")
        frames, channels, _, _ = state.shape
        if channels != self.state_channels:
            raise ValueError(f"expected {self.state_channels} state channels, got {channels}")
        x = F.interpolate(state.to(dtype=self.proj.weight.dtype), size=target_token_hw, mode="nearest")
        x = x.permute(0, 2, 3, 1).reshape(1, frames * target_token_hw[0] * target_token_hw[1], channels)
        return self.proj(x)

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "state_token_projector_v0",
            "channels": STATE_CHANNELS_V0,
            "projection": f"linear_{self.state_channels}_to_{self.cond_dim}",
            "zero_init": True,
        }


def load_state_tensor(row: dict[str, Any], *, frame_indices: list[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    data = np.load(row["state_cache"])
    alive = data["ego_alive"].astype("float32")
    health = data["ego_health"].astype("float32")
    dead = data["opponent_dead_mask"].astype("float32")
    if max(frame_indices, default=-1) >= len(alive):
        raise ValueError(f"state frame index out of range for {row.get('clip_id')}: {frame_indices} vs {len(alive)}")
    planes = []
    for idx in frame_indices:
        h, w = dead.shape[1], dead.shape[2]
        planes.append(np.stack([
            np.full((h, w), alive[idx], dtype=np.float32),
            np.full((h, w), health[idx], dtype=np.float32),
            dead[idx].astype(np.float32),
        ], axis=0))
    arr = np.stack(planes, axis=0) if planes else np.zeros((0, 3, 60, 104), dtype=np.float32)
    return torch.from_numpy(arr).to(device=device, dtype=dtype)
