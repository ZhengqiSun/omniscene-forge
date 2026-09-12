from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import CHECKPOINT_KIND, PAIR_KIND
from .checkpoint import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a LOW/HIGH LingBot LoRA pair descriptor without copying weights.")
    parser.add_argument("--low-checkpoint", type=Path, required=True)
    parser.add_argument("--high-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    low = torch.load(args.low_checkpoint, map_location="cpu", weights_only=False)
    high = torch.load(args.high_checkpoint, map_location="cpu", weights_only=False)
    for name, payload in (("low", low), ("high", high)):
        if payload.get("kind") != CHECKPOINT_KIND or payload.get("expert") != name:
            raise ValueError(f"{name} checkpoint has wrong kind/expert")
    for key in ("manifest_sha256", "lora_config"):
        low_value = low[key]
        high_value = high[key]
        if low_value != high_value:
            raise ValueError(f"LOW/HIGH {key} mismatch")
    if low["base_identity"]["root"] != high["base_identity"]["root"]:
        raise ValueError("LOW/HIGH base checkpoint roots differ")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "kind": PAIR_KIND, "version": 1,
        "base_checkpoint": low["base_identity"]["root"],
        "low_checkpoint": str(args.low_checkpoint.resolve()),
        "high_checkpoint": str(args.high_checkpoint.resolve()),
        "low_sha256": sha256_file(args.low_checkpoint),
        "high_sha256": sha256_file(args.high_checkpoint),
        "training_manifest_sha256": low["manifest_sha256"],
        "lora_config": low["lora_config"],
    }
    output.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(obj, indent=2))


if __name__ == "__main__":
    main()
