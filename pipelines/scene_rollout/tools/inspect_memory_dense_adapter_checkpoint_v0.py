#!/usr/bin/env python3
"""Inspect Memory dense adapter residual magnitude in a checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import torch

from memory_dense_wan_adapter_v0 import MemoryDenseAdapterConfig, ZeroInitMemoryDenseResidualAdapter


BLOCK_RE = re.compile(r"blocks\.(\d+)\.memory_dense_adapter\.")


def adapter_config_from_checkpoint(ckpt: dict[str, Any]) -> MemoryDenseAdapterConfig:
    cfg = ((ckpt.get("config") or {}).get("adapter_config") or {})
    return MemoryDenseAdapterConfig(
        dense_channels=int(cfg.get("dense_channels", 7)),
        cond_dim=int(cfg.get("cond_dim", 128)),
        encoder_hidden_dim=int(cfg.get("encoder_hidden_dim", 64)),
        adapter_hidden_dim=int(cfg.get("adapter_hidden_dim", 512)),
        vae_stride=int(cfg.get("vae_stride", 8)),
        wan_patch_size_hw=int(cfg.get("wan_patch_size_hw", 2)),
        cond_key=str(cfg.get("cond_key", "memory_dense_cond_tokens")),
        residual_mode=str(cfg.get("residual_mode", "cond_gated")),
    )


def split_block_state(adapter_state: dict[str, torch.Tensor]) -> dict[int, dict[str, torch.Tensor]]:
    blocks: dict[int, dict[str, torch.Tensor]] = {}
    for name, tensor in adapter_state.items():
        match = BLOCK_RE.search(name)
        if match is None:
            continue
        block_index = int(match.group(1))
        local_name = name[match.end() :]
        blocks.setdefault(block_index, {})[local_name] = tensor
    return blocks


def tensor_norm(tensor: torch.Tensor) -> float:
    return float(tensor.float().norm().detach().cpu())


def inspect_block(
    block_index: int,
    state: dict[str, torch.Tensor],
    *,
    config: MemoryDenseAdapterConfig,
    hidden_dim: int,
    seq_len: int,
    seed: int,
) -> dict[str, Any]:
    module = ZeroInitMemoryDenseResidualAdapter(
        hidden_dim=hidden_dim,
        cond_dim=config.cond_dim,
        adapter_hidden_dim=config.adapter_hidden_dim,
        residual_mode=config.residual_mode,
    ).float()
    module.load_state_dict(state, strict=True)
    module.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed + block_index)
    hidden = torch.randn(1, seq_len, hidden_dim, generator=generator)
    cond = torch.randn(1, seq_len, config.cond_dim, generator=generator)
    with torch.no_grad():
        out = module(hidden, cond)
        delta = out - hidden
    hidden_norm = tensor_norm(hidden)
    delta_norm = tensor_norm(delta)
    return {
        "block_index": block_index,
        "out_weight_norm": tensor_norm(state["out.weight"]),
        "out_bias_norm": tensor_norm(state["out.bias"]),
        "cond_proj_weight_norm": tensor_norm(state["cond_proj.weight"]),
        "cond_gate_weight_norm": tensor_norm(state["cond_gate.weight"]) if "cond_gate.weight" in state else None,
        "delta_norm_random_probe": delta_norm,
        "hidden_norm_random_probe": hidden_norm,
        "delta_over_hidden_random_probe": delta_norm / hidden_norm if hidden_norm else None,
        "max_abs_delta_random_probe": float(delta.abs().max().detach().cpu()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("state") or {}
    adapter_state = state.get("memory_dense_adapter") or {}
    if not adapter_state:
        raise ValueError(f"{args.checkpoint}: missing memory_dense_adapter state")
    config = adapter_config_from_checkpoint(ckpt)
    blocks = split_block_state(adapter_state)
    if not blocks:
        raise ValueError(f"{args.checkpoint}: no per-block adapter tensors found")
    hidden_dim = int(next(iter(blocks.values()))["hidden_proj.weight"].shape[1])
    block_reports = [
        inspect_block(
            block_index,
            blocks[block_index],
            config=config,
            hidden_dim=hidden_dim,
            seq_len=args.seq_len,
            seed=args.seed,
        )
        for block_index in sorted(blocks)
    ]
    ratios = [float(row["delta_over_hidden_random_probe"]) for row in block_reports]
    out_norms = [float(row["out_weight_norm"]) for row in block_reports]
    result = {
        "kind": "memory_dense_adapter_checkpoint_inspection_v0",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(ckpt.get("step", 0)),
        "block_count": len(block_reports),
        "adapter_config": config.to_dict(),
        "hidden_dim": hidden_dim,
        "seq_len": args.seq_len,
        "delta_over_hidden_random_probe": {
            "min": min(ratios),
            "mean": sum(ratios) / len(ratios),
            "max": max(ratios),
        },
        "out_weight_norm": {
            "min": min(out_norms),
            "mean": sum(out_norms) / len(out_norms),
            "max": max(out_norms),
        },
        "blocks": block_reports,
    }
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--seq-len", type=int, default=1560)
    parser.add_argument("--seed", type=int, default=20260606)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
