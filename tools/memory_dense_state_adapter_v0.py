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


STATE_PROJECTION_CONTRACT_V2 = {"far": 3000.0, "pitch_sign": 1.0}

STATE_BUILD_REPORT_NAME = "state_cache_build_report_v0.json"


def find_state_cache_build_report(
    manifest_path: Path,
    rows: dict[str, dict[str, Any]] | None = None,
    *,
    max_depth: int = 6,
) -> Path | None:
    """Locate the build report that recorded a state cache's projection params.

    build_state_channels_v2.py writes far/pitch_sign into the sibling build
    report, not into the manifest rows, so the contract can only be checked by
    finding that report. Search order: the manifest's own directory, then the
    directories above the first row's ``source_state_cache`` (the v1 interaction
    sidecar points back at the v0 caches it was derived from), then above its
    own ``state_cache``.
    """
    manifest_path = Path(manifest_path)
    candidates: list[Path] = [manifest_path.parent]
    first_row = next(iter(rows.values()), None) if rows else None
    if first_row:
        for key in ("source_state_cache", "state_cache"):
            raw = first_row.get(key)
            if not raw:
                continue
            start = Path(raw)
            if not start.is_absolute():
                start = manifest_path.parent / start
            candidates.extend(list(start.parents)[:max_depth])
    seen: set[str] = set()
    for directory in candidates:
        key = str(directory)
        if key in seen:
            continue
        seen.add(key)
        report = directory / STATE_BUILD_REPORT_NAME
        if report.is_file():
            return report
    return None


def verify_state_projection_contract(
    manifest_path: Path,
    rows: dict[str, dict[str, Any]] | None = None,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Fail fast on state caches built with the old pitch/far convention.

    The v2 contract is far=3000.0 and pitch_sign=+1.0. The demo_10ego /
    demo_10ego10s caches are still -1 / 4096 products; feeding them to a
    current checkpoint manufactures a fresh train/inference mismatch worth
    ~1.36 latent rows per degree of pitch. A confirmed mismatch raises; a
    report that cannot be located only warns, so that eval loops running on
    manifests without a discoverable report keep working.
    """
    manifest_path = Path(manifest_path)
    report_path = find_state_cache_build_report(manifest_path, rows)
    result: dict[str, Any] = {
        "event": "state_projection_contract",
        "manifest": str(manifest_path),
        "expected": dict(STATE_PROJECTION_CONTRACT_V2),
    }
    if report_path is None:
        result["status"] = "unknown"
        result["reason"] = (
            f"no {STATE_BUILD_REPORT_NAME} found next to the manifest or its caches; "
            "projection params could not be verified")
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result
    params = (json.loads(report_path.read_text(encoding="utf-8")) or {}).get("projection_params") or {}
    result["build_report"] = str(report_path)
    result["found"] = {"far": params.get("far"), "pitch_sign": params.get("pitch_sign")}
    bad = [
        f"{name}={params.get(name)!r} (expected {expected})"
        for name, expected in STATE_PROJECTION_CONTRACT_V2.items()
        if params.get(name) is None or abs(float(params[name]) - expected) > 1e-6
    ]
    result["status"] = "ok" if not bad else "mismatch"
    if bad:
        result["mismatch"] = bad
        result["strict"] = bool(strict)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if bad and strict:
        raise ValueError(
            f"{manifest_path}: state cache violates the v2 projection contract "
            f"({'; '.join(bad)}) per {report_path}. Rebuild the cache with "
            "build_state_channels_v2.py, or pass --allow-legacy-state-projection "
            "to accept the mismatch deliberately.")
    return result


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
