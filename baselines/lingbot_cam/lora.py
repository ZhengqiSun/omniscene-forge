from __future__ import annotations

import types
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint



class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float | None = None, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(rank if alpha is None else alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
        self.base.requires_grad_(False)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = self.base(value)
        delta = F.linear(F.linear(self.dropout(value), self.lora_A), self.lora_B)
        return output + (delta * self.scaling).to(output.dtype)


def _targets(value: str) -> set[str]:
    aliases = {
        "attn": {"self_attn", "cross_attn"}, "self_attn": {"self_attn"},
        "cross_attn": {"cross_attn"}, "mlp": {"mlp"}, "ffn": {"mlp"},
        "all": {"self_attn", "cross_attn", "mlp"},
    }
    result: set[str] = set()
    for token in value.replace(",", " ").split():
        if token not in aliases:
            raise ValueError(f"unsupported LoRA target {token!r}")
        result.update(aliases[token])
    if not result:
        raise ValueError("at least one LoRA target is required")
    return result


def inject_lora(model: nn.Module, *, rank: int, targets: str, alpha: float | None, dropout: float) -> dict[str, Any]:
    selected = _targets(targets)
    wrapped: list[str] = []
    for index, block in enumerate(model.blocks):
        if "self_attn" in selected:
            for name in ("q", "k", "v", "o"):
                layer = getattr(block.self_attn, name)
                setattr(block.self_attn, name, LoRALinear(layer, rank, alpha, dropout))
                wrapped.append(f"blocks.{index}.self_attn.{name}")
        if "cross_attn" in selected:
            for name in ("q", "k", "v", "o"):
                layer = getattr(block.cross_attn, name)
                setattr(block.cross_attn, name, LoRALinear(layer, rank, alpha, dropout))
                wrapped.append(f"blocks.{index}.cross_attn.{name}")
        if "mlp" in selected:
            for name, layer in list(block.ffn.named_children()):
                if isinstance(layer, nn.Linear):
                    setattr(block.ffn, name, LoRALinear(layer, rank, alpha, dropout))
                    wrapped.append(f"blocks.{index}.ffn.{name}")
    if not wrapped:
        raise RuntimeError("LoRA injection wrapped no Linear layers")
    return {
        "rank": rank, "alpha": rank if alpha is None else alpha, "dropout": dropout,
        "targets": sorted(selected), "wrapped_linear_count": len(wrapped),
        "wrapped_linear_names": wrapped,
    }


def enable_block_checkpointing(model: nn.Module) -> None:
    def make_forward(original):
        def checkpointed(self, *args, **kwargs):
            return checkpoint(original, *args, use_reentrant=False, **kwargs)
        return checkpointed
    for block in model.blocks:
        block.forward = types.MethodType(make_forward(block.forward), block)


def named_lora_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [(name, parameter) for name, parameter in model.named_parameters() if ".lora_A" in name or ".lora_B" in name]


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu() for name, parameter in named_lora_parameters(model)}


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    current = dict(named_lora_parameters(model))
    if set(current) != set(state):
        raise ValueError(
            f"LoRA keys mismatch missing={sorted(set(current)-set(state))[:10]} "
            f"unexpected={sorted(set(state)-set(current))[:10]}"
        )
    for name, value in state.items():
        if current[name].shape != value.shape:
            raise ValueError(f"LoRA shape mismatch for {name}: {value.shape} != {current[name].shape}")
        current[name].data.copy_(value.to(current[name].device, current[name].dtype))


def assert_only_lora_trainable(model: nn.Module) -> dict[str, Any]:
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    bad = [name for name, _ in trainable if ".lora_A" not in name and ".lora_B" not in name]
    if bad:
        raise RuntimeError(f"non-LoRA parameters are trainable: {bad[:20]}")
    return {
        "names": [name for name, _ in trainable],
        "parameter_count": sum(p.numel() for _, p in trainable),
        "tensor_count": len(trainable),
    }


def assert_only_lora_gradients(model: nn.Module) -> dict[str, Any]:
    names = [name for name, p in model.named_parameters() if p.grad is not None]
    bad = [name for name in names if ".lora_A" not in name and ".lora_B" not in name]
    if bad:
        raise RuntimeError(f"non-LoRA gradients detected: {bad[:20]}")
    finite = all(torch.isfinite(p.grad).all() for _, p in model.named_parameters() if p.grad is not None)
    if not finite:
        raise RuntimeError("non-finite LoRA gradient")
    return {"gradient_tensor_count": len(names), "gradient_names": names}
