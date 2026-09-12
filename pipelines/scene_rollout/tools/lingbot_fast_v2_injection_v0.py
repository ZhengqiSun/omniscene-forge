#!/usr/bin/env python3
"""Fast-v2-native residual and LoRA injection for direct fine-tuning.

This module deliberately has no compatibility path for the existing LOW/HIGH
Memory-dense checkpoints. Fast v2 parameters use a new namespace and the load
helper accepts only the versioned Fast v2 checkpoint envelope.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


INJECTION_SCHEMA_VERSION = "lingbot-fast-v2-native-injection/v1"
CHECKPOINT_KIND = "lingbot-fast-v2-direct-finetune-checkpoint/v1"
CHECKPOINT_NAMESPACE = "lingbot_fast_v2_direct"
FAST_V2_COND_KEY = "lingbot_fast_v2_cond_tokens"
FSDP_WRAPPED_MODULE_SEGMENT = "_fsdp_wrapped_module"

# These are the actual Linear paths under each official CausalWanAttentionBlock.
OFFICIAL_FAST_V2_LORA_TARGETS = (
    "self_attn.q",
    "self_attn.k",
    "self_attn.v",
    "self_attn.o",
    "cross_attn.q",
    "cross_attn.k",
    "cross_attn.v",
    "cross_attn.o",
    "ffn.0",
    "ffn.2",
)


@dataclass(frozen=True)
class FastV2InjectionConfig:
    cond_dim: int = 128
    adapter_hidden_dim: int = 128
    wrap_first_blocks: int | None = None
    residual_scale_init: float = 1.0
    lora_rank: int = 16
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_targets: tuple[str, ...] = OFFICIAL_FAST_V2_LORA_TARGETS
    cond_key: str = FAST_V2_COND_KEY

    def validate(self) -> None:
        if self.cond_dim <= 0 or self.adapter_hidden_dim <= 0:
            raise ValueError("cond_dim and adapter_hidden_dim must be positive")
        if self.wrap_first_blocks is not None and self.wrap_first_blocks <= 0:
            raise ValueError("wrap_first_blocks must be positive or None")
        if self.lora_rank < 0:
            raise ValueError("lora_rank cannot be negative")
        if self.lora_rank > 0 and self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be positive when LoRA is enabled")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.cond_key != FAST_V2_COND_KEY:
            raise ValueError(f"Fast v2 condition key is fixed to {FAST_V2_COND_KEY!r}")
        targets = tuple(self.lora_targets)
        if len(set(targets)) != len(targets):
            raise ValueError("lora_targets contains duplicates")
        unknown = sorted(set(targets) - set(OFFICIAL_FAST_V2_LORA_TARGETS))
        if unknown:
            raise ValueError(f"unsupported Fast v2 LoRA targets: {unknown}")

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["lora_targets"] = list(self.lora_targets)
        return out


class FastV2ZeroInitResidualAdapter(nn.Module):
    """Conditioned residual whose final projection is exactly zero at init."""

    def __init__(self, hidden_dim: int, cond_dim: int, adapter_hidden_dim: int):
        super().__init__()
        self.hidden_proj = nn.Linear(hidden_dim, adapter_hidden_dim)
        self.cond_proj = nn.Linear(cond_dim, adapter_hidden_dim)
        self.gate_proj = nn.Linear(cond_dim, adapter_hidden_dim, bias=False)
        self.out_proj = nn.Linear(adapter_hidden_dim, hidden_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, hidden: torch.Tensor, cond_tokens: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or cond_tokens.ndim != 3:
            raise ValueError("hidden and Fast v2 condition must both be [B,L,C]")
        if hidden.shape[:2] != cond_tokens.shape[:2]:
            raise ValueError(
                f"Fast v2 condition {tuple(cond_tokens.shape)} does not align with "
                f"block hidden state {tuple(hidden.shape)}"
            )
        cond_tokens = cond_tokens.to(device=hidden.device, dtype=hidden.dtype)
        features = F.silu(self.hidden_proj(hidden)) + F.silu(self.cond_proj(cond_tokens))
        features = features * torch.tanh(self.gate_proj(cond_tokens))
        return self.out_proj(features)


class FastV2LoRALinear(nn.Module):
    """Official Fast Linear plus a separately namespaced, zero-init LoRA delta."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("Fast v2 LoRA rank must be positive")
        self.fast_v2_base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout else nn.Identity()
        self.fast_v2_lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.fast_v2_lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.fast_v2_lora_A, a=5**0.5)
        nn.init.zeros_(self.fast_v2_lora_B)
        self.fast_v2_enabled = True
        self.fast_v2_base.requires_grad_(False)

    @property
    def weight(self) -> torch.Tensor:
        return self.fast_v2_base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.fast_v2_base.bias

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        base_output = self.fast_v2_base(value)
        if not self.fast_v2_enabled:
            return base_output
        delta = F.linear(
            F.linear(self.dropout(value), self.fast_v2_lora_A),
            self.fast_v2_lora_B,
        )
        return base_output + (delta * self.scaling).to(dtype=base_output.dtype)


class FastV2InjectedBlock(nn.Module):
    """Wrap one official Fast block with an optional conditioned residual."""

    def __init__(self, base_block: nn.Module, hidden_dim: int, config: FastV2InjectionConfig):
        super().__init__()
        self.fast_v2_base_block = base_block
        self.fast_v2_cond_key = config.cond_key
        self.fast_v2_adapter_scale = nn.Parameter(
            torch.tensor([float(config.residual_scale_init)])
        )
        self.fast_v2_adapter = FastV2ZeroInitResidualAdapter(
            hidden_dim=hidden_dim,
            cond_dim=config.cond_dim,
            adapter_hidden_dim=config.adapter_hidden_dim,
        )
        self.fast_v2_enabled = True

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        hidden = self.fast_v2_base_block(*args, **kwargs)
        if not self.fast_v2_enabled:
            return hidden
        dit_cond_dict = kwargs.get("dit_cond_dict")
        if not isinstance(dit_cond_dict, Mapping) or self.fast_v2_cond_key not in dit_cond_dict:
            raise ValueError(
                f"enabled Fast v2 block requires dit_cond_dict[{self.fast_v2_cond_key!r}]"
            )
        delta = self.fast_v2_adapter(hidden, dit_cond_dict[self.fast_v2_cond_key])
        scale = self.fast_v2_adapter_scale.to(device=hidden.device, dtype=hidden.dtype)
        return hidden + scale * delta.to(dtype=hidden.dtype)


def _official_block(model: nn.Module, block_index: int) -> nn.Module:
    block = model.blocks[block_index]
    if isinstance(block, FastV2InjectedBlock):
        return block.fast_v2_base_block
    return block


def _resolve_child(root: nn.Module, dotted: str) -> tuple[nn.Module, str, nn.Module]:
    parts = dotted.split(".")
    parent = root
    for part in parts[:-1]:
        if part.isdigit() and isinstance(parent, nn.Sequential):
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
        if not isinstance(parent, nn.Module):
            raise TypeError(f"{dotted!r} traverses a non-module child")
    leaf = parts[-1]
    child = parent[int(leaf)] if leaf.isdigit() and isinstance(parent, nn.Sequential) else getattr(parent, leaf)
    if not isinstance(child, nn.Module):
        raise TypeError(f"{dotted!r} does not resolve to a module")
    return parent, leaf, child


def _replace_child(parent: nn.Module, leaf: str, replacement: nn.Module) -> None:
    if leaf.isdigit() and isinstance(parent, nn.Sequential):
        parent[int(leaf)] = replacement
    else:
        setattr(parent, leaf, replacement)


def validate_official_fast_v2_topology(model: nn.Module) -> dict[str, Any]:
    """Fail closed unless every block exposes the official Fast v2 Linear paths."""

    if not hasattr(model, "blocks") or not isinstance(model.blocks, nn.ModuleList):
        raise TypeError("official Fast v2 model must expose blocks as nn.ModuleList")
    if not hasattr(model, "dim") or not model.blocks:
        raise TypeError("official Fast v2 model must expose non-empty blocks and .dim")
    checked: list[str] = []
    for block_index in range(len(model.blocks)):
        block = _official_block(model, block_index)
        for suffix in OFFICIAL_FAST_V2_LORA_TARGETS:
            _, _, child = _resolve_child(block, suffix)
            if not isinstance(child, (nn.Linear, FastV2LoRALinear)):
                raise TypeError(
                    f"blocks.{block_index}.{suffix} must be Linear, got {type(child).__name__}"
                )
            checked.append(f"blocks.{block_index}.{suffix}")
    return {"block_count": len(model.blocks), "checked_linear_names": checked}


def inject_fast_v2_native(
    model: nn.Module,
    config: FastV2InjectionConfig,
    *,
    freeze_base: bool = True,
) -> dict[str, Any]:
    """Inject fresh Fast v2 LoRA and residual modules into the official model."""

    config.validate()
    topology = validate_official_fast_v2_topology(model)
    if any(isinstance(block, FastV2InjectedBlock) for block in model.blocks):
        raise RuntimeError("Fast v2 injection is already installed")

    wrapped_lora: list[str] = []
    if config.lora_rank > 0:
        for block_index in range(len(model.blocks)):
            block = _official_block(model, block_index)
            for suffix in config.lora_targets:
                parent, leaf, child = _resolve_child(block, suffix)
                if not isinstance(child, nn.Linear):
                    raise TypeError(f"blocks.{block_index}.{suffix} is already wrapped")
                replacement = FastV2LoRALinear(
                    child,
                    rank=config.lora_rank,
                    alpha=config.lora_alpha,
                    dropout=config.lora_dropout,
                ).to(device=child.weight.device, dtype=child.weight.dtype)
                _replace_child(parent, leaf, replacement)
                wrapped_lora.append(f"blocks.{block_index}.{suffix}")

    block_limit = len(model.blocks)
    if config.wrap_first_blocks is not None:
        block_limit = min(config.wrap_first_blocks, block_limit)
    wrapped_blocks: list[str] = []
    for block_index in range(block_limit):
        base_block = model.blocks[block_index]
        reference = next(base_block.parameters())
        wrapper = FastV2InjectedBlock(base_block, int(model.dim), config)
        wrapper.fast_v2_adapter.to(device=reference.device, dtype=reference.dtype)
        wrapper.fast_v2_adapter_scale.data = wrapper.fast_v2_adapter_scale.data.to(
            device=reference.device, dtype=reference.dtype
        )
        model.blocks[block_index] = wrapper
        wrapped_blocks.append(f"blocks.{block_index}")

    if freeze_base:
        model.requires_grad_(False)
        for name, parameter in model.named_parameters():
            if _is_fast_v2_trainable_name(name):
                parameter.requires_grad_(True)

    zero_init = assert_fast_v2_zero_initialization(model)
    return {
        "schema_version": INJECTION_SCHEMA_VERSION,
        "config": config.to_dict(),
        "official_topology": topology,
        "wrapped_blocks": wrapped_blocks,
        "wrapped_lora_names": wrapped_lora,
        "zero_initialization": zero_init,
        "checkpoint_namespace": CHECKPOINT_NAMESPACE,
        "foreign_low_high_weights_loaded": False,
    }


def _is_fast_v2_trainable_name(name: str) -> bool:
    return (
        ".fast_v2_adapter." in name
        or name.endswith(".fast_v2_adapter_scale")
        or name.endswith(".fast_v2_lora_A")
        or name.endswith(".fast_v2_lora_B")
    )


def fast_v2_trainable_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if _is_fast_v2_trainable_name(name)
    ]


def canonicalize_fast_v2_parameter_name(name: str) -> str:
    """Remove only exact FSDP wrapper path segments from a parameter name."""

    if not isinstance(name, str) or not name:
        raise ValueError("Fast v2 parameter name must be a non-empty string")
    parts = name.split(".")
    canonical = ".".join(
        part for part in parts if part != FSDP_WRAPPED_MODULE_SEGMENT
    )
    if not canonical:
        raise ValueError(f"Fast v2 parameter name has no canonical segments: {name!r}")
    return canonical


def canonicalize_fast_v2_state_keys(state: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize FSDP paths and fail closed on any resulting key collision."""

    canonical_state: dict[str, Any] = {}
    source_names: dict[str, str] = {}
    for source_name, value in state.items():
        canonical_name = canonicalize_fast_v2_parameter_name(source_name)
        if canonical_name in canonical_state:
            raise ValueError(
                "Fast v2 canonical checkpoint key collision: "
                f"{source_names[canonical_name]!r} and {source_name!r} -> "
                f"{canonical_name!r}"
            )
        canonical_state[canonical_name] = value
        source_names[canonical_name] = source_name
    return canonical_state


def set_fast_v2_enabled(model: nn.Module, enabled: bool) -> dict[str, int | bool]:
    block_count = 0
    lora_count = 0
    for module in model.modules():
        if isinstance(module, FastV2InjectedBlock):
            module.fast_v2_enabled = bool(enabled)
            block_count += 1
        elif isinstance(module, FastV2LoRALinear):
            module.fast_v2_enabled = bool(enabled)
            lora_count += 1
    if block_count == 0:
        raise RuntimeError("model has no Fast v2 injection modules")
    mismatched = [
        type(module).__name__
        for module in model.modules()
        if isinstance(module, (FastV2InjectedBlock, FastV2LoRALinear))
        and module.fast_v2_enabled is not bool(enabled)
    ]
    if mismatched:
        raise RuntimeError(f"Fast v2 enable state did not propagate through wrapper: {mismatched[:10]}")
    return {
        "enabled": bool(enabled),
        "injected_block_count": block_count,
        "lora_linear_count": lora_count,
    }


def assert_fast_v2_zero_initialization(model: nn.Module) -> dict[str, Any]:
    adapter_scales: list[str] = []
    adapter_outputs: list[str] = []
    lora_outputs: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, FastV2InjectedBlock):
            scale = module.fast_v2_adapter_scale
            if scale.ndim != 1 or scale.numel() != 1:
                raise RuntimeError(
                    f"Fast v2 adapter scale must be FSDP-compatible [1], got "
                    f"shape={tuple(scale.shape)} at {name}.fast_v2_adapter_scale"
                )
            adapter_scales.append(name)
        elif isinstance(module, FastV2ZeroInitResidualAdapter):
            if torch.count_nonzero(module.out_proj.weight).item() != 0:
                raise RuntimeError(f"nonzero Fast v2 adapter output weight: {name}.out_proj.weight")
            if module.out_proj.bias is not None and torch.count_nonzero(module.out_proj.bias).item() != 0:
                raise RuntimeError(f"nonzero Fast v2 adapter output bias: {name}.out_proj.bias")
            adapter_outputs.append(name)
        elif isinstance(module, FastV2LoRALinear):
            if torch.count_nonzero(module.fast_v2_lora_B).item() != 0:
                raise RuntimeError(f"nonzero Fast v2 LoRA B: {name}.fast_v2_lora_B")
            lora_outputs.append(name)
    if not adapter_outputs:
        raise RuntimeError("no Fast v2 residual adapters found")
    return {
        "exact": True,
        "fsdp_compatible_adapter_scale_count": len(adapter_scales),
        "adapter_output_projection_count": len(adapter_outputs),
        "lora_B_count": len(lora_outputs),
    }


def fast_v2_checkpoint_envelope(
    model: nn.Module,
    config: FastV2InjectionConfig,
) -> dict[str, Any]:
    named_state = canonicalize_fast_v2_state_keys(
        dict(fast_v2_trainable_parameters(model))
    )
    state = {
        # FSDP full parameters may be views into a much larger flat storage.
        name: parameter.detach().cpu().clone()
        for name, parameter in named_state.items()
    }
    if any(
        FSDP_WRAPPED_MODULE_SEGMENT in name.split(".") for name in state
    ):
        raise RuntimeError("Fast v2 checkpoint export retained an FSDP wrapper segment")
    return {
        "kind": CHECKPOINT_KIND,
        "namespace": CHECKPOINT_NAMESPACE,
        "injection_schema_version": INJECTION_SCHEMA_VERSION,
        "config": config.to_dict(),
        "state": state,
    }


def load_fast_v2_checkpoint_envelope(
    model: nn.Module,
    envelope: Mapping[str, Any],
    config: FastV2InjectionConfig,
) -> None:
    """Strictly load one Fast v2 state; no aliases, expert merge, or partial load."""

    if envelope.get("kind") != CHECKPOINT_KIND:
        raise ValueError("checkpoint is not a Fast v2 direct-finetune checkpoint")
    if envelope.get("namespace") != CHECKPOINT_NAMESPACE:
        raise ValueError("checkpoint namespace is not the isolated Fast v2 namespace")
    if envelope.get("injection_schema_version") != INJECTION_SCHEMA_VERSION:
        raise ValueError("Fast v2 injection schema drift")
    if envelope.get("config") != config.to_dict():
        raise ValueError("Fast v2 checkpoint config does not exactly match the installed injection")
    state = envelope.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("Fast v2 checkpoint state must be a mapping")
    expected = dict(fast_v2_trainable_parameters(model))
    if set(state) != set(expected):
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        raise ValueError(
            f"Fast v2 checkpoint keys are not exact: missing={missing[:10]} "
            f"unexpected={unexpected[:10]}"
        )
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Fast v2 checkpoint value is not a tensor: {name}")
        target = expected[name]
        if tuple(value.shape) != tuple(target.shape):
            raise ValueError(f"Fast v2 checkpoint shape mismatch for {name}")
        target.data.copy_(value.to(device=target.device, dtype=target.dtype))
