#!/usr/bin/env python3
"""CPU smoke test for action cache loading and zero-init token projection."""

from __future__ import annotations

import argparse
import json

import torch

from memory_dense_action_adapter_v0 import (
    ACTION_CHANNELS_V0,
    ActionTokenProjector,
    load_action_manifest,
    load_action_tensor,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    rows = load_action_manifest(args.manifest)
    if not rows:
        raise RuntimeError("action manifest is empty")
    row = next(iter(rows.values()))
    tensor = load_action_tensor(
        row,
        frame_indices=list(range(21)),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    if tensor.shape != (21, len(ACTION_CHANNELS_V0)):
        raise RuntimeError(f"unexpected action shape: {tuple(tensor.shape)}")

    projector = ActionTokenProjector(cond_dim=128, hidden_dim=64)
    tokens = projector(tensor, target_token_hw=(3, 5))
    expected = (1, 21 * 3 * 5, 128)
    if tuple(tokens.shape) != expected:
        raise RuntimeError(f"unexpected token shape: {tuple(tokens.shape)} != {expected}")
    max_abs = float(tokens.abs().max())
    if max_abs != 0.0:
        raise RuntimeError(f"zero-init projector changed conditioning: max_abs={max_abs}")

    report = {
        "kind": "memory_dense_action_adapter_smoke_v0",
        "status": "pass",
        "clip_id": row["clip_id"],
        "action_shape": list(tensor.shape),
        "token_shape": list(tokens.shape),
        "zero_init_token_max_abs": max_abs,
        "fire_latent_frames": int((tensor[:, ACTION_CHANNELS_V0.index("fire_any")] > 0).sum()),
        "projector": projector.metadata(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
