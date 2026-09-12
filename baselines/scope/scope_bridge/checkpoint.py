from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

STRICT_KEYS = ("manifest_sha256", "action_calibration_sha256", "base_checkpoint_fingerprint", "scope_commit", "project_commit", "trainable_scope_sha256")


def capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: str | Path, dit, optimizer, scheduler, global_step: int, metadata: dict[str, Any]) -> None:
    trainable = {name: p.detach().cpu() for name, p in dit.named_parameters() if p.requires_grad}
    payload = {
        "format": "scope_actionmodule_ft_v1", "global_step": global_step,
        "trainable_state": trainable, "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "rng": capture_rng(), "metadata": metadata,
    }
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    path.with_suffix(path.suffix + ".metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def load_checkpoint(path: str | Path, dit, optimizer, scheduler, expected: dict[str, Any]) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "scope_actionmodule_ft_v1":
        raise ValueError("unsupported checkpoint format")
    actual = payload["metadata"]
    mismatch = {key: (actual.get(key), expected.get(key)) for key in STRICT_KEYS if actual.get(key) != expected.get(key)}
    if mismatch:
        raise ValueError(f"resume metadata mismatch: {mismatch}")
    result = dit.load_state_dict(payload["trainable_state"], strict=False)
    unexpected = list(result.unexpected_keys)
    if unexpected:
        raise ValueError(f"unexpected trainable keys: {unexpected}")
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler and payload.get("scheduler"):
        scheduler.load_state_dict(payload["scheduler"])
    restore_rng(payload["rng"])
    return int(payload["global_step"])
