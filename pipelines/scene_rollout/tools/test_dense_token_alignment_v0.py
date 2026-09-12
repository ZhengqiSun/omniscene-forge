#!/usr/bin/env python3
"""Validate Memory dense tokens align with Wan hidden tokens.

Task 1 gates:
  1a: encoder image -> token spatial layout.
  1b: frame-major reshape used by dense_tokens_for_samples.
  1c: real WanModelFast patchify/unpatchify path with a forced nonzero adapter.

The probe initializes the encoder/adapter/head as deterministic local pass-through
components. That keeps the test focused on ordering: (f, h, w) in dense tokens
must match the Wan patch token order.
"""

from __future__ import annotations
from runtime_paths import source_path

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

from memory_dense_wan_adapter_v0 import (
    MemoryDenseAdapterConfig,
    MemoryDenseWanTokenEncoder,
    ZeroInitMemoryDenseResidualAdapter,
    wrap_wan_model_fast_with_memory_dense_adapter,
)


DENSE_H, DENSE_W = 176, 320
LATENT_HW = (60, 104)
TARGET_TOKEN_HW = (30, 52)
LATENT_CHANNELS = 16
COND_KEY = "memory_dense_cond_tokens"


@dataclass
class SyntheticDenseSample:
    dense: np.ndarray
    sample_id: str
    selection_role: str = "positive"


def quadrant_image(h: int, w: int, q: int) -> torch.Tensor:
    if q not in {0, 1, 2, 3}:
        raise ValueError(f"quadrant must be 0..3, got {q}")
    dense = torch.zeros(1, 7, h, w)
    hs = slice(0, h // 2) if q in (0, 1) else slice(h // 2, h)
    ws = slice(0, w // 2) if q in (0, 2) else slice(w // 2, w)
    dense[:, :, hs, ws] = 1.0
    return dense


def energy_grid(tokens: torch.Tensor, th: int, tw: int) -> torch.Tensor:
    if tokens.shape[:2] != (1, th * tw):
        raise RuntimeError(f"expected [1,{th * tw},C], got {tuple(tokens.shape)}")
    return tokens.pow(2).sum(-1).reshape(th, tw)


def quadrant_fraction(grid: torch.Tensor, q: int) -> float:
    th, tw = grid.shape
    hs = slice(0, th // 2) if q in (0, 1) else slice(th // 2, th)
    ws = slice(0, tw // 2) if q in (0, 2) else slice(tw // 2, tw)
    total = grid.sum().clamp_min(1e-9)
    return float((grid[hs, ws].sum() / total).detach().cpu())


def latent_quadrant_fraction(grid: torch.Tensor, q: int) -> float:
    if grid.ndim != 2:
        raise RuntimeError(f"expected [H,W] grid, got {tuple(grid.shape)}")
    return quadrant_fraction(grid, q)


def adapter_config() -> MemoryDenseAdapterConfig:
    return MemoryDenseAdapterConfig(
        dense_channels=7,
        cond_dim=64,
        encoder_hidden_dim=64,
        adapter_hidden_dim=64,
        vae_stride=8,
        wan_patch_size_hw=2,
        cond_key=COND_KEY,
        residual_mode="cond_gated",
    )


def make_probe_encoder(device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> MemoryDenseWanTokenEncoder:
    """Create a deterministic positive local averaging encoder.

    This uses the production encoder architecture, but removes random weights and
    biases so quadrant support is the only signal being tested.
    """

    cfg = adapter_config()
    encoder = MemoryDenseWanTokenEncoder(cfg).to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        for param in encoder.parameters():
            param.zero_()
        conv1 = encoder.net[0]
        conv2 = encoder.net[2]
        if not isinstance(conv1, nn.Conv2d) or not isinstance(conv2, nn.Conv2d):
            raise RuntimeError("unexpected MemoryDenseWanTokenEncoder layout")
        conv1.weight[:, :, 1, 1] = 1.0 / cfg.dense_channels
        conv2.weight[:, :, :, :] = 1.0 / (
            cfg.encoder_hidden_dim * cfg.image_to_wan_token_stride * cfg.image_to_wan_token_stride
        )
    return encoder


def force_adapter_passthrough(model: nn.Module) -> int:
    """Make every dense adapter produce a local hidden delta from cond tokens."""

    count = 0
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, ZeroInitMemoryDenseResidualAdapter):
                continue
            count += 1
            for param in module.parameters():
                param.zero_()
            hidden_dim = module.out.weight.shape[0]
            adapter_dim = module.out.weight.shape[1]
            cond_dim = module.cond_proj.weight.shape[1]
            diag = min(hidden_dim, adapter_dim, cond_dim)
            module.cond_proj.weight[:diag, :diag] = torch.eye(
                diag, device=module.cond_proj.weight.device, dtype=module.cond_proj.weight.dtype
            )
            if module.residual_mode == "cond_gated":
                module.cond_gate.weight[:diag, :diag] = torch.eye(
                    diag, device=module.cond_gate.weight.device, dtype=module.cond_gate.weight.dtype
                )
            module.out.weight[:diag, :diag] = torch.eye(
                diag, device=module.out.weight.device, dtype=module.out.weight.dtype
            )
    if count < 1:
        raise RuntimeError("no ZeroInitMemoryDenseResidualAdapter modules found")
    return count


def force_head_passthrough(model: nn.Module) -> None:
    """Make Wan head expose local hidden-token deltas in latent output space."""

    head = getattr(getattr(model, "head", None), "head", None)
    if not isinstance(head, nn.Linear):
        raise RuntimeError("unexpected WanModelFast head layout")
    with torch.no_grad():
        head.weight.zero_()
        if head.bias is not None:
            head.bias.zero_()
        diag = min(head.weight.shape)
        head.weight[:diag, :diag] = torch.eye(diag, device=head.weight.device, dtype=head.weight.dtype)


def test_1a_encoder_layout() -> None:
    encoder = make_probe_encoder()
    blank, _ = encoder(torch.zeros(1, 7, DENSE_H, DENSE_W), target_token_hw=TARGET_TOKEN_HW)
    for q in range(4):
        tokens, (th, tw) = encoder(quadrant_image(DENSE_H, DENSE_W, q), target_token_hw=TARGET_TOKEN_HW)
        frac = quadrant_fraction(energy_grid(tokens - blank, th, tw), q)
        print(f"[1a] quadrant {q}: differential token energy in-quadrant = {frac:.6f}")
        assert frac > 0.90, f"encoder layout broken at quadrant {q}: {frac:.6f}"
    print("[1a] PASS")


def test_1b_frame_order() -> None:
    encoder = make_probe_encoder()
    frames = torch.cat([quadrant_image(DENSE_H, DENSE_W, q) for q in range(3)], dim=0)
    blank, (th, tw) = encoder(torch.zeros(3, 7, DENSE_H, DENSE_W), target_token_hw=TARGET_TOKEN_HW)
    feat, _ = encoder(frames, target_token_hw=TARGET_TOKEN_HW)
    tokens_per_frame = th * tw
    stacked = (feat - blank).reshape(1, 3 * tokens_per_frame, feat.shape[-1])
    for frame_index, q in enumerate([0, 1, 2]):
        segment = stacked[0, frame_index * tokens_per_frame : (frame_index + 1) * tokens_per_frame].unsqueeze(0)
        frac = quadrant_fraction(energy_grid(segment, th, tw), q)
        print(f"[1b] frame {frame_index}, quadrant {q}: energy in expected quadrant = {frac:.6f}")
        assert frac > 0.90, f"frame-major reshape broken at frame {frame_index}: {frac:.6f}"
    print("[1b] PASS")


def build_tiny_wan_model(lingbot_repo: Path, device: torch.device) -> nn.Module:
    sys.path.insert(0, str(lingbot_repo.resolve()))
    from wan.modules.model_fast import WanModelFast  # type: ignore

    model = WanModelFast(
        model_type="t2v",
        control_type="cam",
        patch_size=(1, 2, 2),
        text_len=8,
        in_dim=LATENT_CHANNELS,
        dim=64,
        ffn_dim=128,
        freq_dim=16,
        text_dim=32,
        out_dim=LATENT_CHANNELS,
        num_heads=4,
        num_layers=1,
        local_attn_size=-1,
        sink_size=0,
        cross_attn_norm=True,
    ).to(device=device, dtype=torch.float32)
    model.requires_grad_(False)
    wrap_wan_model_fast_with_memory_dense_adapter(model, adapter_config(), freeze_base=True)
    model.to(device=device, dtype=torch.float32).eval()
    adapter_count = force_adapter_passthrough(model)
    force_head_passthrough(model)
    print(f"[1c] forced passthrough adapters: {adapter_count}")
    return model


def dense_tokens_via_training_helper(
    encoder: MemoryDenseWanTokenEncoder,
    quadrants: Iterable[int],
    *,
    device: torch.device,
) -> torch.Tensor:
    import train_memory_dense_adapter_v0 as train_mod

    samples = []
    for frame_index, q in enumerate(quadrants):
        dense = quadrant_image(DENSE_H, DENSE_W, q).squeeze(0).numpy().astype("float32")
        samples.append(SyntheticDenseSample(dense=dense, sample_id=f"synthetic_q{q}_f{frame_index}"))

    old_load_dense = train_mod.load_dense
    try:
        train_mod.load_dense = lambda sample: sample.dense  # type: ignore[assignment]
        tokens, token_hw = train_mod.dense_tokens_for_samples(
            encoder,
            samples,
            device=device,
            dtype=torch.float32,
            target_token_hw=TARGET_TOKEN_HW,
        )
    finally:
        train_mod.load_dense = old_load_dense  # type: ignore[assignment]
    if tuple(token_hw) != TARGET_TOKEN_HW:
        raise RuntimeError(f"dense token hw {token_hw} != expected {TARGET_TOKEN_HW}")
    return tokens


def test_1c_wan_patch_order(lingbot_repo: Path, device: torch.device) -> None:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    torch.manual_seed(20260606)
    encoder = make_probe_encoder(device=device)
    model = build_tiny_wan_model(lingbot_repo, device)
    quadrants = [0, 1, 2]
    dense_tokens = dense_tokens_via_training_helper(encoder, quadrants, device=device)
    blank_tokens = torch.zeros_like(dense_tokens)

    generator = torch.Generator(device=device).manual_seed(20260606)
    latent = torch.randn(LATENT_CHANNELS, len(quadrants), *LATENT_HW, generator=generator, device=device) * 0.01
    timestep = torch.tensor([100.0], dtype=torch.float32, device=device)
    context = torch.zeros(3, 32, dtype=torch.float32, device=device)
    seq_len = len(quadrants) * TARGET_TOKEN_HW[0] * TARGET_TOKEN_HW[1]

    with torch.no_grad():
        out_blank = model(
            x=[latent],
            t=timestep,
            context=[context],
            seq_len=seq_len,
            dit_cond_dict={COND_KEY: blank_tokens},
            kv_cache=None,
            crossattn_cache=None,
            current_start=0,
            max_attention_size=seq_len,
        )[0]
        out_dense = model(
            x=[latent],
            t=timestep,
            context=[context],
            seq_len=seq_len,
            dit_cond_dict={COND_KEY: dense_tokens},
            kv_cache=None,
            crossattn_cache=None,
            current_start=0,
            max_attention_size=seq_len,
        )[0]

    if tuple(out_dense.shape) != (LATENT_CHANNELS, len(quadrants), *LATENT_HW):
        raise RuntimeError(f"unexpected Wan output shape {tuple(out_dense.shape)}")
    delta_energy = (out_dense - out_blank).abs().sum(0)
    for frame_index, q in enumerate(quadrants):
        frac = latent_quadrant_fraction(delta_energy[frame_index], q)
        print(f"[1c] frame {frame_index}, quadrant {q}: latent delta energy in expected quadrant = {frac:.6f}")
        assert frac > 0.90, f"Wan patch token order mismatch at frame {frame_index}: {frac:.6f}"
    print("[1c] PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lingbot-repo", type=Path, default=Path(str(source_path('lingbot', ''))))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-1c", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_1a_encoder_layout()
    test_1b_frame_order()
    if not args.skip_1c:
        test_1c_wan_patch_order(args.lingbot_repo, torch.device(args.device))
    print("PASS: dense token (f,h,w) order matches Wan patch token order for the tested path.")


if __name__ == "__main__":
    main()
