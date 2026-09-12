from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResampledActions:
    keyboard: np.ndarray  # [T, 6]
    sticks: np.ndarray  # [T, 4]: left x/y, right x/y


def _overlap_weights(source_count: int, source_fps: float, target_count: int, target_fps: float) -> np.ndarray:
    src_lo = np.arange(source_count) / source_fps
    src_hi = (np.arange(source_count) + 1) / source_fps
    dst_lo = np.arange(target_count) / target_fps
    dst_hi = (np.arange(target_count) + 1) / target_fps
    return np.maximum(0.0, np.minimum(dst_hi[:, None], src_hi[None, :]) - np.maximum(dst_lo[:, None], src_lo[None, :]))


def resample(
    movement: np.ndarray,
    mouse_delta: np.ndarray,
    buttons: np.ndarray,
    source_fps: float,
    target_fps: float,
    target_count: int,
    mouse_gain: float | None,
) -> ResampledActions:
    """Timestamp-window resampling: state average, button OR, mouse integration."""
    movement = np.asarray(movement, dtype=np.float32)
    mouse_delta = np.asarray(mouse_delta, dtype=np.float32)
    buttons = np.asarray(buttons, dtype=np.float32)
    n = len(movement)
    if movement.shape != (n, 2) or mouse_delta.shape != (n, 2) or buttons.shape != (n, 6):
        raise ValueError("expected movement [N,2], mouse_delta [N,2], buttons [N,6]")
    weights = _overlap_weights(n, source_fps, target_count, target_fps)
    coverage = weights.sum(axis=1)
    if np.any(coverage <= 0):
        raise ValueError("source actions do not cover requested output duration")
    move = weights @ movement / coverage[:, None]
    norm = np.linalg.norm(move, axis=1, keepdims=True)
    move = move / np.maximum(1.0, norm)
    key = np.stack([(buttons[weights[i] > 0, j] > 0.5).any() for i in range(target_count) for j in range(6)]).reshape(target_count, 6).astype(np.float32)
    # Raw mouse fields are deltas per source interval, so overlap is a fraction of that interval.
    mouse = (weights * source_fps) @ mouse_delta
    if mouse_gain is not None:
        if not np.isfinite(mouse_gain) or mouse_gain <= 0:
            raise ValueError("mouse_gain must be finite and positive")
        mouse = np.clip(mouse / mouse_gain, -1.0, 1.0)
    return ResampledActions(keyboard=key, sticks=np.concatenate([move, mouse], axis=1).astype(np.float32))


def output_timestamps(count: int, fps: float) -> np.ndarray:
    return np.arange(count, dtype=np.float64) / fps
