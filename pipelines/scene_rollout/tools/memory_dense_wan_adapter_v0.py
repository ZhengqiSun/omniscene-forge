#!/usr/bin/env python3
"""Reusable Memory dense residual adapter wrappers for WanModelFast.

This module is intentionally small and dependency-light so training scripts can
import it without pulling in the data-rendering tools. It does not edit Wan
source code; it wraps existing blocks in place.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class MemoryDenseAdapterConfig:
    dense_channels: int = 7
    cond_dim: int = 64
    encoder_hidden_dim: int = 32
    adapter_hidden_dim: int = 128
    vae_stride: int = 8
    wan_patch_size_hw: int = 2
    cond_key: str = "memory_dense_cond_tokens"
    residual_mode: str = "cond_gated"
    wrap_first_blocks: int | None = None
    residual_scale_init: float = 1.0
    activation_checkpoint_blocks: bool = False

    @property
    def image_to_wan_token_stride(self) -> int:
        return self.vae_stride * self.wan_patch_size_hw

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["image_to_wan_token_stride"] = self.image_to_wan_token_stride
        return out


class MemoryDenseWanTokenEncoder(nn.Module):
    """Encode first-person Memory dense maps to Wan spatial token features."""

    def __init__(self, config: MemoryDenseAdapterConfig):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Conv2d(config.dense_channels, config.encoder_hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(
                config.encoder_hidden_dim,
                config.cond_dim,
                kernel_size=config.image_to_wan_token_stride,
                stride=config.image_to_wan_token_stride,
            ),
        )

    def forward(
        self,
        dense: torch.Tensor,
        *,
        target_token_hw: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        if dense.ndim != 4:
            raise ValueError(f"dense must be [B,C,H,W], got {tuple(dense.shape)}")
        _, channels, height, width = dense.shape
        if channels != self.config.dense_channels:
            raise ValueError(f"expected {self.config.dense_channels} dense channels, got {channels}")
        stride = self.config.image_to_wan_token_stride
        if height % stride != 0 or width % stride != 0:
            raise ValueError(f"dense H/W {(height, width)} must be divisible by token stride {stride}")
        feat = self.net(dense)
        if target_token_hw is not None:
            target_h, target_w = target_token_hw
            if target_h < 1 or target_w < 1:
                raise ValueError(f"target_token_hw must be positive, got {target_token_hw}")
            if feat.shape[-2:] != target_token_hw:
                feat = F.interpolate(feat, size=target_token_hw, mode="bilinear", align_corners=False)
        _, _, token_h, token_w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2).contiguous()
        return tokens, (token_h, token_w)


class MemoryDenseFrozenVAEWanTokenEncoder(nn.Module):
    """Encode Memory dense maps through a frozen Wan2.1 VAE condition path.

    The Wan VAE wrapper is intentionally kept as a plain attribute instead of a
    registered submodule, so adapter checkpoints only contain the trainable
    projection layer and never duplicate the frozen VAE weights.
    """

    PACKING_SCHEME = {
        "img1": ["ch0 env_depth", "ch2 semantic", "ch3 player_mask"],
        "img2": ["ch4 player_depth", "ch5 yaw_sin", "ch6 yaw_cos"],
        "dropped": "ch1",
        "value_map": "x*2-1",
        "native_vae_hw": [22, 40],
    }

    def __init__(
        self,
        config: MemoryDenseAdapterConfig,
        *,
        vae: Any,
        vae_pth: str | Path,
        native_hw: tuple[int, int] = (22, 40),
        packing: str = "img1_mask_img2_player_v0",
    ):
        super().__init__()
        self.config = config
        self.vae = vae
        self.vae_pth = str(vae_pth)
        self.native_hw = tuple(int(v) for v in native_hw)
        self.packing = str(packing)
        self.proj = nn.Linear(32, config.cond_dim)
        self._assert_frozen_vae()

    def _assert_frozen_vae(self) -> None:
        model = getattr(self.vae, "model", None)
        if model is None or not hasattr(model, "parameters"):
            raise TypeError("Wan VAE object must expose frozen .model parameters")
        trainable = [name for name, param in model.named_parameters() if param.requires_grad]
        if trainable:
            raise RuntimeError(f"frozen VAE has trainable parameters: {trainable[:10]}")

    @property
    def metadata(self) -> dict[str, Any]:
        model = getattr(self.vae, "model", None)
        vae_param_count = int(sum(p.numel() for p in model.parameters())) if model is not None else None
        vae_trainable_param_count = (
            int(sum(p.numel() for p in model.parameters() if p.requires_grad)) if model is not None else None
        )
        return {
            "type": "frozen_vae",
            "vae_pth": self.vae_pth,
            "packing": self.packing,
            "packing_scheme": self.PACKING_SCHEME,
            "native_hw": list(self.native_hw),
            "projection": "linear_32_to_cond_dim",
            "vae_param_count": vae_param_count,
            "vae_trainable_param_count": vae_trainable_param_count,
            "projection_trainable_param_count": int(sum(p.numel() for p in self.proj.parameters() if p.requires_grad)),
            "checkpoint_state": "projection_only",
        }

    def _pack_pseudo_rgb(self, dense: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if dense.ndim != 4:
            raise ValueError(f"dense must be [B,C,H,W], got {tuple(dense.shape)}")
        _, channels, _, _ = dense.shape
        if channels != self.config.dense_channels:
            raise ValueError(f"expected {self.config.dense_channels} dense channels, got {channels}")
        dense_f = dense.float()
        img1 = torch.stack([dense_f[:, 0], dense_f[:, 2], dense_f[:, 3]], dim=1)
        img2 = torch.stack([dense_f[:, 4], dense_f[:, 5], dense_f[:, 6]], dim=1)
        return img1.mul(2.0).sub(1.0), img2.mul(2.0).sub(1.0)

    def forward(
        self,
        dense: torch.Tensor,
        *,
        target_token_hw: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        self._assert_frozen_vae()
        target_hw = tuple(int(v) for v in (target_token_hw or self.native_hw))
        if target_hw[0] < 1 or target_hw[1] < 1:
            raise ValueError(f"target_token_hw must be positive, got {target_hw}")

        img1, img2 = self._pack_pseudo_rgb(dense)
        device = dense.device
        with torch.no_grad():
            latents = self.vae.encode([
                img1[i].unsqueeze(1).to(device=device) for i in range(img1.shape[0])
            ] + [
                img2[i].unsqueeze(1).to(device=device) for i in range(img2.shape[0])
            ])
        lat1 = torch.stack([item.squeeze(1).float() for item in latents[: img1.shape[0]]], dim=0)
        lat2 = torch.stack([item.squeeze(1).float() for item in latents[img1.shape[0] :]], dim=0)
        if tuple(lat1.shape[-2:]) != self.native_hw or tuple(lat2.shape[-2:]) != self.native_hw:
            raise RuntimeError(
                f"frozen VAE native grid {tuple(lat1.shape[-2:])}/{tuple(lat2.shape[-2:])} != expected {self.native_hw}"
            )
        feat = torch.cat([lat1, lat2], dim=1)
        if feat.shape[1] != 32:
            raise RuntimeError(f"expected 32 VAE condition channels, got {feat.shape[1]}")
        if tuple(feat.shape[-2:]) != target_hw:
            feat = F.interpolate(feat, size=target_hw, mode="bilinear", align_corners=False)
        feat = feat.to(device=self.proj.weight.device, dtype=self.proj.weight.dtype)
        tokens_32 = feat.flatten(2).transpose(1, 2).contiguous()
        tokens = self.proj(tokens_32)
        expected_tokens = target_hw[0] * target_hw[1]
        if tokens.shape[1] != expected_tokens:
            raise RuntimeError(f"frozen VAE token count {tokens.shape[1]} != expected {expected_tokens}")
        return tokens, target_hw

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        return {f"proj.{key}": value for key, value in self.proj.state_dict(*args, **kwargs).items()}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True):
        proj_state = {
            key.replace("proj.", "", 1): value
            for key, value in state_dict.items()
            if key.startswith("proj.")
        }
        return self.proj.load_state_dict(proj_state, strict=strict)


class ZeroInitMemoryDenseResidualAdapter(nn.Module):
    """Same-shape hidden residual conditioned on Memory dense tokens."""

    def __init__(self, hidden_dim: int, cond_dim: int, adapter_hidden_dim: int, residual_mode: str):
        super().__init__()
        if residual_mode not in {"additive", "cond_gated"}:
            raise ValueError(f"unsupported residual_mode={residual_mode!r}")
        self.residual_mode = residual_mode
        self.hidden_proj = nn.Linear(hidden_dim, adapter_hidden_dim)
        self.cond_proj = nn.Linear(cond_dim, adapter_hidden_dim)
        if residual_mode == "cond_gated":
            self.cond_gate = nn.Linear(cond_dim, adapter_hidden_dim, bias=False)
        self.out = nn.Linear(adapter_hidden_dim, hidden_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, hidden: torch.Tensor, cond_tokens: torch.Tensor) -> torch.Tensor:
        if hidden.shape[:2] != cond_tokens.shape[:2]:
            raise RuntimeError(f"hidden {tuple(hidden.shape)} and cond {tuple(cond_tokens.shape)} do not align")
        if self.residual_mode == "additive":
            features = F.silu(self.hidden_proj(hidden) + self.cond_proj(cond_tokens))
        else:
            hidden_features = F.silu(self.hidden_proj(hidden))
            cond_features = F.silu(self.cond_proj(cond_tokens))
            gate = torch.tanh(self.cond_gate(cond_tokens))
            features = (hidden_features + cond_features) * gate
        delta = self.out(features)
        return hidden + delta


class LoRALinear(nn.Module):
    """Frozen Linear plus zero-init low-rank delta."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float | None = None, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha if alpha is not None else rank)
        self.scaling = self.alpha / float(rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
        for param in self.base.parameters():
            param.requires_grad_(False)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.rank <= 0:
            return out
        delta = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling
        return out + delta.to(dtype=out.dtype)


def _target_tokens_for_lora(targets: str | list[str] | tuple[str, ...]) -> set[str]:
    if isinstance(targets, str):
        raw = [item.strip() for item in targets.replace(",", " ").split() if item.strip()]
    else:
        raw = [str(item).strip() for item in targets if str(item).strip()]
    if not raw:
        raw = ["attn", "mlp"]
    aliases = {
        "all": {"self_attn", "cross_attn", "mlp"},
        "attn": {"self_attn", "cross_attn"},
        "self_attn": {"self_attn"},
        "cross_attn": {"cross_attn"},
        "mlp": {"mlp"},
        "ffn": {"mlp"},
    }
    out: set[str] = set()
    for item in raw:
        if item not in aliases:
            raise ValueError(f"unsupported LoRA target {item!r}; choose from {sorted(aliases)}")
        out.update(aliases[item])
    return out


def inject_lora_into_wan_model_fast(
    model: nn.Module,
    *,
    rank: int,
    targets: str | list[str] | tuple[str, ...] = "attn,mlp",
    alpha: float | None = None,
    dropout: float = 0.0,
) -> dict[str, Any]:
    """Replace selected Wan block Linear layers with LoRA wrappers in place."""

    if rank <= 0:
        return {"enabled": False, "rank": int(rank), "wrapped_linear_count": 0, "targets": []}
    if not hasattr(model, "blocks"):
        raise TypeError("model must expose a .blocks ModuleList")
    target_set = _target_tokens_for_lora(targets)
    wrapped: list[str] = []
    for block_index, block in enumerate(model.blocks):
        block_obj = block.block if isinstance(block, WanBlockMemoryDenseWrapper) else block
        if "self_attn" in target_set and hasattr(block_obj, "self_attn"):
            for name in ("q", "k", "v", "o"):
                layer = getattr(block_obj.self_attn, name, None)
                if isinstance(layer, nn.Linear):
                    setattr(block_obj.self_attn, name, LoRALinear(layer, rank=rank, alpha=alpha, dropout=dropout))
                    wrapped.append(f"blocks.{block_index}.self_attn.{name}")
        if "cross_attn" in target_set and hasattr(block_obj, "cross_attn"):
            for name in ("q", "k", "v", "o"):
                layer = getattr(block_obj.cross_attn, name, None)
                if isinstance(layer, nn.Linear):
                    setattr(block_obj.cross_attn, name, LoRALinear(layer, rank=rank, alpha=alpha, dropout=dropout))
                    wrapped.append(f"blocks.{block_index}.cross_attn.{name}")
        if "mlp" in target_set and hasattr(block_obj, "ffn"):
            for name, layer in list(block_obj.ffn.named_children()):
                if isinstance(layer, nn.Linear):
                    setattr(block_obj.ffn, name, LoRALinear(layer, rank=rank, alpha=alpha, dropout=dropout))
                    wrapped.append(f"blocks.{block_index}.ffn.{name}")
    if not wrapped:
        raise RuntimeError(f"LoRA rank={rank} requested but no target Linear layers were wrapped for targets={sorted(target_set)}")
    return {
        "enabled": True,
        "rank": int(rank),
        "alpha": float(alpha if alpha is not None else rank),
        "dropout": float(dropout),
        "targets": sorted(target_set),
        "wrapped_linear_count": len(wrapped),
        "wrapped_linear_names": wrapped,
    }


def lora_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [(name, param) for name, param in model.named_parameters() if ".lora_A" in name or ".lora_B" in name]


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: param.detach().cpu() for name, param in model.named_parameters() if ".lora_A" in name or ".lora_B" in name}


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
    if not state:
        return [], []
    current = dict(model.named_parameters())
    missing = [name for name in state if name not in current]
    unexpected = [name for name in current if (".lora_A" in name or ".lora_B" in name) and name not in state]
    if missing:
        return missing, unexpected
    for name, value in state.items():
        current[name].data.copy_(value.to(device=current[name].device, dtype=current[name].dtype))
    return missing, unexpected


class WanBlockMemoryDenseWrapper(nn.Module):
    """Wrap one Wan block and add zero-init Memory dense residual after it."""

    def __init__(self, block: nn.Module, hidden_dim: int, config: MemoryDenseAdapterConfig):
        super().__init__()
        self.block = block
        self.cond_key = config.cond_key
        self.activation_checkpoint_blocks = bool(config.activation_checkpoint_blocks)
        self.memory_dense_adapter_scale = nn.Parameter(torch.tensor(float(config.residual_scale_init)))
        self.memory_dense_adapter = ZeroInitMemoryDenseResidualAdapter(
            hidden_dim=hidden_dim,
            cond_dim=config.cond_dim,
            adapter_hidden_dim=config.adapter_hidden_dim,
            residual_mode=config.residual_mode,
        )

    def forward(self, *block_args: Any, **block_kwargs: Any) -> torch.Tensor:
        if (
            self.activation_checkpoint_blocks
            and self.training
            and torch.is_grad_enabled()
            and block_kwargs.get("kv_cache") is None
            and block_kwargs.get("crossattn_cache") is None
        ):
            def run_block(*args: Any) -> torch.Tensor:
                return self.block(*args, **block_kwargs)

            hidden = checkpoint(run_block, *block_args, use_reentrant=False)
        else:
            hidden = self.block(*block_args, **block_kwargs)
        dit_cond_dict = block_kwargs.get("dit_cond_dict")
        if dit_cond_dict is None or self.cond_key not in dit_cond_dict:
            return hidden
        cond_tokens = dit_cond_dict[self.cond_key].to(device=hidden.device, dtype=hidden.dtype)
        adapted = self.memory_dense_adapter(hidden, cond_tokens)
        return hidden + self.memory_dense_adapter_scale.to(device=hidden.device, dtype=hidden.dtype) * (adapted - hidden)


def freeze_non_memory_dense_adapter_params(model: nn.Module) -> None:
    """Freeze all parameters except Memory dense residual adapter parameters."""

    for name, param in model.named_parameters():
        param.requires_grad_("memory_dense_adapter" in name or ".lora_A" in name or ".lora_B" in name)


def memory_dense_adapter_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [(name, param) for name, param in model.named_parameters() if "memory_dense_adapter" in name]


def wrap_wan_model_fast_with_memory_dense_adapter(
    model: nn.Module,
    config: MemoryDenseAdapterConfig,
    *,
    freeze_base: bool = True,
) -> nn.Module:
    """Wrap all WanModelFast blocks in place and optionally freeze the base."""

    if not hasattr(model, "blocks"):
        raise TypeError("model must expose a .blocks ModuleList")
    if not hasattr(model, "dim"):
        raise TypeError("model must expose hidden dimension as .dim")
    max_blocks = len(model.blocks) if config.wrap_first_blocks is None else min(int(config.wrap_first_blocks), len(model.blocks))
    for index, block in enumerate(model.blocks):
        if index >= max_blocks:
            continue
        if isinstance(block, WanBlockMemoryDenseWrapper):
            continue
        model.blocks[index] = WanBlockMemoryDenseWrapper(block, int(model.dim), config)
    if freeze_base:
        freeze_non_memory_dense_adapter_params(model)
    return model


def assert_wan_token_grid_matches_dense(
    *,
    dense_hw: tuple[int, int],
    latent_hw: tuple[int, int],
    wan_token_hw: tuple[int, int],
    config: MemoryDenseAdapterConfig,
) -> None:
    expected_latent = (dense_hw[0] // config.vae_stride, dense_hw[1] // config.vae_stride)
    expected_token = (
        expected_latent[0] // config.wan_patch_size_hw,
        expected_latent[1] // config.wan_patch_size_hw,
    )
    if latent_hw != expected_latent:
        raise RuntimeError(f"latent grid {latent_hw} != expected {expected_latent} from dense {dense_hw}")
    if wan_token_hw != expected_token:
        raise RuntimeError(f"Wan token grid {wan_token_hw} != expected {expected_token} from dense {dense_hw}")
