#!/usr/bin/env python3
"""V2-native Dense projection and residual injection without LoRA."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


SCHEMA_VERSION = "lingbot-fast-v2-dense-direct-tune/v0"
CHECKPOINT_KIND = "lingbot-fast-v2-dense-direct-tune-checkpoint/v0"
CHECKPOINT_NAMESPACE = "lingbot_fast_v2_dense_direct_tune"
COND_KEY = "lingbot_fast_v2_dense_tokens"
VALID_MODES = ("true", "shuffled", "blank", "disabled")


@dataclass(frozen=True)
class DenseDirectTuneConfig:
    dense_channels: int = 7
    cond_dim: int = 128
    hidden_dim: int = 5120
    adapter_hidden_dim: int = 128
    block_indices: tuple[int, ...] = (0, 1, 2, 3)
    token_height: int = 30
    token_width: int = 52
    chunk_size: int = 4

    def validate(self, *, model_block_count: int | None = None) -> None:
        positive = (
            self.dense_channels,
            self.cond_dim,
            self.hidden_dim,
            self.adapter_hidden_dim,
            self.token_height,
            self.token_width,
            self.chunk_size,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("all Dense direct-tune dimensions must be positive")
        if not self.block_indices or len(set(self.block_indices)) != len(self.block_indices):
            raise ValueError("block_indices must be non-empty and unique")
        if tuple(sorted(self.block_indices)) != self.block_indices or min(self.block_indices) < 0:
            raise ValueError("block_indices must be sorted non-negative integers")
        if model_block_count is not None and max(self.block_indices) >= model_block_count:
            raise ValueError("Dense injection block index exceeds model block count")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["block_indices"] = list(self.block_indices)
        return value


def apply_dense_mode(
    dense: torch.Tensor,
    mode: str,
    *,
    shuffle_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply a condition arm without changing the validated tensor layout."""

    if mode not in VALID_MODES:
        raise ValueError(f"unsupported Dense mode: {mode!r}")
    if dense.ndim != 4:
        raise ValueError(f"Dense must be [T,7,H,W], got {tuple(dense.shape)}")
    frames = dense.shape[0]
    if mode in ("true", "disabled"):
        return dense, {"mode": mode, "shuffle_indices": None}
    if mode == "blank":
        return torch.zeros_like(dense), {"mode": mode, "shuffle_indices": None}
    if shuffle_indices is None:
        shuffle_indices = torch.arange(frames - 1, -1, -1, device=dense.device)
    shuffle_indices = shuffle_indices.to(device=dense.device, dtype=torch.long)
    if tuple(shuffle_indices.shape) != (frames,):
        raise ValueError("shuffle_indices must contain one index per Dense frame")
    if sorted(shuffle_indices.cpu().tolist()) != list(range(frames)):
        raise ValueError("shuffle_indices must be a permutation")
    if frames > 1 and torch.equal(shuffle_indices, torch.arange(frames, device=dense.device)):
        raise ValueError("shuffled Dense mode cannot use the identity permutation")
    return dense.index_select(0, shuffle_indices), {
        "mode": mode,
        "shuffle_indices": shuffle_indices.cpu().tolist(),
    }


class FastV2DenseProjection(nn.Module):
    """Resize `[T,7,H,W]` and learn a per-token V2 condition embedding."""

    def __init__(self, config: DenseDirectTuneConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.fast_v2_dense_projection = nn.Linear(config.dense_channels, config.cond_dim)

    def forward(
        self,
        dense: torch.Tensor,
        *,
        mode: str = "true",
        shuffle_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        if dense.ndim == 5:
            if dense.shape[0] != 1:
                raise ValueError("V2 Fast Dense baseline supports batch size one only")
            dense = dense[0]
        expected_prefix = (self.config.chunk_size, self.config.dense_channels)
        if dense.ndim != 4 or tuple(dense.shape[:2]) != expected_prefix:
            raise ValueError(
                f"Dense chunk must be [T,C,H,W] with T,C={expected_prefix}, "
                f"got {tuple(dense.shape)}"
            )
        if not bool(torch.isfinite(dense).all().item()):
            raise ValueError("Dense chunk contains non-finite values")
        dense, mode_audit = apply_dense_mode(
            dense, mode, shuffle_indices=shuffle_indices
        )
        if mode == "disabled":
            return None, {
                **mode_audit,
                "input_shape": list(dense.shape),
                "tokens_shape": None,
                "feature_norm": 0.0,
            }
        weight = self.fast_v2_dense_projection.weight
        dense = F.interpolate(
            dense.to(device=weight.device, dtype=weight.dtype),
            size=(self.config.token_height, self.config.token_width),
            mode="bilinear",
            align_corners=False,
        )
        values = dense.permute(0, 2, 3, 1).reshape(
            1,
            self.config.chunk_size * self.config.token_height * self.config.token_width,
            self.config.dense_channels,
        )
        tokens = self.fast_v2_dense_projection(values)
        expected = (
            1,
            self.config.chunk_size * self.config.token_height * self.config.token_width,
            self.config.cond_dim,
        )
        if tuple(tokens.shape) != expected:
            raise RuntimeError(f"projected Dense token shape drift: {tuple(tokens.shape)}")
        return tokens, {
            **mode_audit,
            "input_shape": list(dense.shape),
            "tokens_shape": list(tokens.shape),
            "feature_norm": float(tokens.detach().float().norm().cpu()),
        }


class FastV2DenseResidualAdapter(nn.Module):
    def __init__(self, config: DenseDirectTuneConfig):
        super().__init__()
        self.hidden_proj = nn.Linear(config.hidden_dim, config.adapter_hidden_dim)
        self.cond_proj = nn.Linear(config.cond_dim, config.adapter_hidden_dim)
        self.gate_proj = nn.Linear(config.cond_dim, config.adapter_hidden_dim, bias=False)
        self.out_proj = nn.Linear(config.adapter_hidden_dim, config.hidden_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.last_residual_norm = 0.0

    def forward(self, hidden: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or cond.ndim != 3 or hidden.shape[:2] != cond.shape[:2]:
            raise ValueError(
                f"hidden/condition token mismatch: {tuple(hidden.shape)} vs {tuple(cond.shape)}"
            )
        cond = cond.to(device=hidden.device, dtype=hidden.dtype)
        value = F.silu(self.hidden_proj(hidden)) + F.silu(self.cond_proj(cond))
        value = value * torch.tanh(self.gate_proj(cond))
        residual = self.out_proj(value)
        self.last_residual_norm = float(residual.detach().float().norm().cpu())
        return residual


class FastV2DenseInjectedBlock(nn.Module):
    def __init__(self, base_block: nn.Module, config: DenseDirectTuneConfig):
        super().__init__()
        self.fast_v2_dense_base_block = base_block
        self.fast_v2_dense_adapter = FastV2DenseResidualAdapter(config)
        self.fast_v2_dense_enabled = True

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        hidden = self.fast_v2_dense_base_block(*args, **kwargs)
        if not self.fast_v2_dense_enabled:
            return hidden
        conditions = kwargs.get("dit_cond_dict")
        if not isinstance(conditions, Mapping) or COND_KEY not in conditions:
            raise ValueError(f"enabled Dense block requires dit_cond_dict[{COND_KEY!r}]")
        return hidden + self.fast_v2_dense_adapter(hidden, conditions[COND_KEY]).to(hidden.dtype)


def inject_dense_direct_tune(
    model: nn.Module,
    config: DenseDirectTuneConfig,
    *,
    freeze_backbone: bool = True,
) -> dict[str, Any]:
    if not hasattr(model, "blocks") or not isinstance(model.blocks, nn.ModuleList):
        raise TypeError("V2 Fast model must expose blocks as ModuleList")
    config.validate(model_block_count=len(model.blocks))
    if int(getattr(model, "dim", -1)) != config.hidden_dim:
        raise ValueError("V2 Fast model hidden dimension does not match Dense config")
    if any(isinstance(block, FastV2DenseInjectedBlock) for block in model.blocks):
        raise RuntimeError("Dense direct-tune injection is already installed")
    if freeze_backbone:
        model.requires_grad_(False)
    wrapped = []
    for index in config.block_indices:
        base = model.blocks[index]
        reference = next(base.parameters())
        wrapper = FastV2DenseInjectedBlock(base, config)
        wrapper.fast_v2_dense_adapter.to(device=reference.device, dtype=reference.dtype)
        model.blocks[index] = wrapper
        wrapped.append(f"blocks.{index}")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(".fast_v2_dense_adapter." in name)
    zero = assert_zero_initialization(model)
    return {
        "schema_version": SCHEMA_VERSION,
        "config": config.to_dict(),
        "wrapped_blocks": wrapped,
        "zero_initialization": zero,
        "lora_enabled": False,
        "foreign_weights_loaded": False,
    }


def set_dense_enabled(model: nn.Module, enabled: bool) -> int:
    count = 0
    for module in model.modules():
        if isinstance(module, FastV2DenseInjectedBlock):
            module.fast_v2_dense_enabled = bool(enabled)
            count += 1
    if count == 0:
        raise RuntimeError("model has no Dense direct-tune blocks")
    return count


def assert_zero_initialization(model: nn.Module) -> dict[str, Any]:
    adapters = []
    for name, module in model.named_modules():
        if isinstance(module, FastV2DenseResidualAdapter):
            if torch.count_nonzero(module.out_proj.weight).item() != 0:
                raise RuntimeError(f"nonzero adapter output weight: {name}")
            if torch.count_nonzero(module.out_proj.bias).item() != 0:
                raise RuntimeError(f"nonzero adapter output bias: {name}")
            adapters.append(name)
    if not adapters:
        raise RuntimeError("no Dense residual adapters found")
    return {"exact": True, "adapter_count": len(adapters), "adapters": adapters}


def named_trainable_parameters(
    model: nn.Module,
    projection: nn.Module,
) -> list[tuple[str, nn.Parameter]]:
    values = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    values.extend(
        (f"dense_projection.{name}", parameter)
        for name, parameter in projection.named_parameters()
        if parameter.requires_grad
    )
    return values


def audit_trainables(model: nn.Module, projection: nn.Module) -> dict[str, Any]:
    trainable = named_trainable_parameters(model, projection)
    invalid = [
        name
        for name, _ in trainable
        if ".fast_v2_dense_adapter." not in name
        and not name.startswith("dense_projection.fast_v2_dense_projection.")
    ]
    if invalid:
        raise RuntimeError(f"unexpected trainable V2 parameters: {invalid[:10]}")
    frozen = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    return {
        "trainable_names": [name for name, _ in trainable],
        "trainable_shapes": {name: list(value.shape) for name, value in trainable},
        "trainable_parameter_count": sum(value.numel() for _, value in trainable),
        "frozen_model_parameter_count": frozen,
        "lora_parameter_count": 0,
    }


def gradient_audit(named: Sequence[tuple[str, nn.Parameter]]) -> dict[str, Any]:
    rows = []
    for name, parameter in named:
        gradient = parameter.grad
        finite = gradient is not None and bool(torch.isfinite(gradient).all().item())
        norm = None if gradient is None else float(gradient.detach().float().norm().cpu())
        rows.append({"name": name, "gradient_present": gradient is not None, "finite": finite, "norm": norm})
    projection = [row for row in rows if row["name"].startswith("dense_projection.")]
    adapters = [row for row in rows if ".fast_v2_dense_adapter." in row["name"]]
    return {
        "parameters": rows,
        "projection_any_nonzero": any((row["norm"] or 0.0) > 0 for row in projection),
        "adapter_any_nonzero": any((row["norm"] or 0.0) > 0 for row in adapters),
        "all_present_gradients_finite": all(row["finite"] for row in rows if row["gradient_present"]),
    }


def checkpoint_envelope(
    model: nn.Module,
    projection: FastV2DenseProjection,
    config: DenseDirectTuneConfig,
    *,
    optimizer_step: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    state = {}
    for name, parameter in named_trainable_parameters(model, projection):
        canonical = ".".join(
            part for part in name.split(".") if part != "_fsdp_wrapped_module"
        )
        if canonical in state:
            raise ValueError(f"canonical Dense checkpoint key collision: {canonical}")
        state[canonical] = parameter.detach().cpu().clone()
    return {
        "kind": CHECKPOINT_KIND,
        "namespace": CHECKPOINT_NAMESPACE,
        "schema_version": SCHEMA_VERSION,
        "config": config.to_dict(),
        "optimizer_step": int(optimizer_step),
        "state": state,
        "metadata": dict(metadata),
    }


def load_checkpoint_envelope(
    model: nn.Module,
    projection: FastV2DenseProjection,
    envelope: Mapping[str, Any],
    config: DenseDirectTuneConfig,
) -> None:
    if envelope.get("kind") != CHECKPOINT_KIND or envelope.get("namespace") != CHECKPOINT_NAMESPACE:
        raise ValueError("not a Dense direct-tune checkpoint")
    if envelope.get("schema_version") != SCHEMA_VERSION or envelope.get("config") != config.to_dict():
        raise ValueError("Dense direct-tune checkpoint config/schema mismatch")
    expected = dict(named_trainable_parameters(model, projection))
    expected = {
        ".".join(part for part in name.split(".") if part != "_fsdp_wrapped_module"): value
        for name, value in expected.items()
    }
    state = envelope.get("state")
    if not isinstance(state, Mapping) or set(state) != set(expected):
        raise ValueError("Dense direct-tune checkpoint keys are not exact")
    for name, target in expected.items():
        value = state[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(target.shape):
            raise ValueError(f"Dense checkpoint tensor mismatch: {name}")
        target.data.copy_(value.to(device=target.device, dtype=target.dtype))


def residual_norms(model: nn.Module) -> dict[str, float]:
    return {
        name: module.last_residual_norm
        for name, module in model.named_modules()
        if isinstance(module, FastV2DenseResidualAdapter)
    }
