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

import math

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
    activation_checkpoint_adapter: bool = False
    highfreq_branch: bool = False
    highfreq_kind: str = "laplacian"  # laplacian fixed high-pass then conv; rawconv learned 3x3 then conv
    highfreq_hidden_dim: int = 32
    # v2 dense contract (416x240 -> 30x52 native, no interpolation): 240/16=15, 416/16=26
    highfreq_stride: int = 16  # coarse grid before interp to the DiT token grid

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


class MemoryDenseHighFreqBranch(nn.Module):
    """VAE-free zero-init high-frequency cond residual on the RAW dense map."""

    def __init__(self, config: "MemoryDenseAdapterConfig"):
        super().__init__()
        self.config = config
        self.kind = str(config.highfreq_kind)
        c_in = int(config.dense_channels)
        h = int(config.highfreq_hidden_dim)
        s = int(config.highfreq_stride)
        if self.kind == "laplacian":
            k = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
            w = k.view(1, 1, 3, 3).repeat(c_in, 1, 1, 1)
            self.register_buffer("hp_kernel", w, persistent=False)
            self.hp_groups = c_in
            conv_in = c_in
        elif self.kind == "rawconv":
            self.hp_kernel = None
            self.hp_groups = None
            conv_in = c_in
        else:
            raise ValueError(f"unsupported highfreq_kind={self.kind!r}")
        num_groups = min(8, h)
        while h % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.body = nn.Sequential(
            nn.Conv2d(conv_in, h, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=h),
            nn.SiLU(),
            nn.Conv2d(h, config.cond_dim, kernel_size=s, stride=s),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, dense: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        x = dense.float()
        if self.kind == "laplacian":
            x = F.conv2d(x, self.hp_kernel.to(device=x.device, dtype=x.dtype), padding=1, groups=self.hp_groups)
        # match the conv body parameter dtype (bf16 under --precision bf16); high-pass stays fp32 above
        body_dtype = self.body[0].weight.dtype
        x = x.to(dtype=body_dtype)
        feat = self.body(x)
        if feat.shape[-2:] != tuple(target_hw):
            feat = F.interpolate(feat, size=tuple(target_hw), mode="bilinear", align_corners=False)
        return feat.flatten(2).transpose(1, 2).contiguous()


def resolve_dense_native_hw(ckpt_config: dict[str, Any]) -> tuple[int, int]:
    """Frozen-VAE native latent grid for a checkpoint, read from its own config.

    Inference-side helper (samplers only; no trainer calls it). The samplers used
    to fall back to a hardcoded ``[22, 40]``. That constant is *correct* for the
    176x320 base lineage and *wrong* for the v2 240x416 contract (30x52), so a
    hardcoded default silently mis-shapes one of the two lineages instead of
    failing. Read the grid the checkpoint actually recorded, in this order:

      1. ``dense_condition_encoder.native_hw``            (dense_encoder_metadata)
      2. ``dense_condition_encoder.packing_scheme.native_vae_hw``
      3. ``dense_native_hw`` // ``adapter_config.vae_stride``

    and raise when the checkpoint records none of them, rather than guessing.
    """
    cfg = ckpt_config or {}
    encoder_cfg = cfg.get("dense_condition_encoder") or {}
    packing_scheme = encoder_cfg.get("packing_scheme") or {}
    for source, value in (
        ("dense_condition_encoder.native_hw", encoder_cfg.get("native_hw")),
        ("dense_condition_encoder.packing_scheme.native_vae_hw",
         packing_scheme.get("native_vae_hw")),
    ):
        if value:
            hw = tuple(int(v) for v in value)
            if len(hw) != 2 or hw[0] < 1 or hw[1] < 1:
                raise ValueError(f"checkpoint {source}={value!r} is not a positive (H, W)")
            return hw
    dense_hw = cfg.get("dense_native_hw")
    stride = int((cfg.get("adapter_config") or {}).get("vae_stride", 0) or 0)
    if dense_hw and stride > 0:
        height, width = int(dense_hw[0]), int(dense_hw[1])
        if height % stride or width % stride:
            raise ValueError(
                f"checkpoint dense_native_hw={list(dense_hw)} is not divisible by "
                f"vae_stride={stride}")
        return (height // stride, width // stride)
    raise ValueError(
        "checkpoint config records no frozen-VAE native grid; need one of "
        "dense_condition_encoder.native_hw, "
        "dense_condition_encoder.packing_scheme.native_vae_hw, or "
        "dense_native_hw together with adapter_config.vae_stride"
    )


class MemoryDenseFrozenVAEWanTokenEncoder(nn.Module):
    """Encode Memory dense maps through a frozen Wan2.1 VAE condition path.

    The Wan VAE wrapper is intentionally kept as a plain attribute instead of a
    registered submodule, so adapter checkpoints only contain the trainable
    projection layer and never duplicate the frozen VAE weights.
    """

    PACKING_SCHEMES = {
        "img1_mask_img2_player_v0": {
            "img1": ["ch0 env_depth", "ch2 semantic", "ch3 player_mask"],
            "img2": ["ch4 player_depth", "ch5 yaw_sin", "ch6 yaw_cos"],
            "dropped": "ch1",
            "value_map": "x*2-1",
            "known_defects": "ch1 dropped; ch5/6 already [-1,1] -> x*2-1 lands [-3,1]",
            # v2 dense contract (416x240 -> 30x52 native, no interpolation)
            "native_vae_hw": [30, 52],
        },
        "full7_yawangle_v2": {
            "img1": ["ch0 env_depth", "ch2 semantic", "ch3 player_mask"],
            "img2": ["ch4 player_depth", "yaw_angle=atan2(ch5,ch6)/pi", "ch1 mesh_hit"],
            "dropped": None,
            "value_map": "x*2-1 for [0,1] channels; identity for yaw_angle (already [-1,1])",
            # v2 dense contract (416x240 -> 30x52 native, no interpolation)
            "native_vae_hw": [30, 52],
        },
    }
    # Backward-compat alias: v0 scheme under the legacy attribute name.
    PACKING_SCHEME = PACKING_SCHEMES["img1_mask_img2_player_v0"]

    def __init__(
        self,
        config: MemoryDenseAdapterConfig,
        *,
        vae: Any,
        vae_pth: str | Path,
        # v2 dense contract (416x240 -> 30x52 native, no interpolation)
        native_hw: tuple[int, int] = (30, 52),
        packing: str = "img1_mask_img2_player_v0",
    ):
        super().__init__()
        self.config = config
        self.vae = vae
        self.vae_pth = str(vae_pth)
        self.native_hw = tuple(int(v) for v in native_hw)
        self.packing = str(packing)
        if self.packing not in self.PACKING_SCHEMES:
            raise ValueError(
                f"unknown packing {self.packing!r}; known: {sorted(self.PACKING_SCHEMES)}")
        self.proj = nn.Linear(32, config.cond_dim)
        self.highfreq = MemoryDenseHighFreqBranch(config) if config.highfreq_branch else None
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
            # v2 dense contract (416x240 -> 30x52 native, no interpolation): report the
            # live instance grid so a ckpt config can never disagree with self.native_hw
            "packing_scheme": {
                **self.PACKING_SCHEMES[self.packing],
                "native_vae_hw": list(self.native_hw),
            },
            "native_hw": list(self.native_hw),
            "projection": "linear_32_to_cond_dim",
            "vae_param_count": vae_param_count,
            "vae_trainable_param_count": vae_trainable_param_count,
            "projection_trainable_param_count": int(sum(p.numel() for p in self.proj.parameters() if p.requires_grad)),
            "checkpoint_state": "projection_only",
            "highfreq": {
                "enabled": self.highfreq is not None,
                "kind": str(self.config.highfreq_kind),
                "hidden_dim": int(self.config.highfreq_hidden_dim),
                "stride": int(self.config.highfreq_stride),
            },
        }

    def _pack_pseudo_rgb(self, dense: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if dense.ndim != 4:
            raise ValueError(f"dense must be [B,C,H,W], got {tuple(dense.shape)}")
        _, channels, _, _ = dense.shape
        if channels != self.config.dense_channels:
            raise ValueError(f"expected {self.config.dense_channels} dense channels, got {channels}")
        dense_f = dense.float()
        img1 = torch.stack([dense_f[:, 0], dense_f[:, 2], dense_f[:, 3]], dim=1)
        if self.packing == "img1_mask_img2_player_v0":
            img2 = torch.stack([dense_f[:, 4], dense_f[:, 5], dense_f[:, 6]], dim=1)
            return img1.mul(2.0).sub(1.0), img2.mul(2.0).sub(1.0)
        # full7_yawangle_v2: recover ch1, keep every VAE input inside [-1,1].
        # yaw sin/cos are stored raw in [-1,1] inside the player mask and 0 outside;
        # atan2(0,0)=0 keeps the zero-outside convention for the merged channel.
        yaw_angle = torch.atan2(dense_f[:, 5], dense_f[:, 6]).div(math.pi)
        img2 = torch.stack([
            dense_f[:, 4].mul(2.0).sub(1.0),
            yaw_angle,
            dense_f[:, 1].mul(2.0).sub(1.0),
        ], dim=1)
        return img1.mul(2.0).sub(1.0), img2

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
        if self.highfreq is not None:
            hf = self.highfreq(dense.to(device=tokens.device), target_hw).to(dtype=tokens.dtype)
            tokens = tokens + hf  # zero-init -> no-op at W2 init
        expected_tokens = target_hw[0] * target_hw[1]
        if tokens.shape[1] != expected_tokens:
            raise RuntimeError(f"frozen VAE token count {tokens.shape[1]} != expected {expected_tokens}")
        return tokens, target_hw

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        out = {f"proj.{key}": value for key, value in self.proj.state_dict(*args, **kwargs).items()}
        if self.highfreq is not None:
            out.update({f"highfreq.{key}": value for key, value in self.highfreq.state_dict(*args, **kwargs).items()})
        return out

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True):
        proj_state = {
            key.replace("proj.", "", 1): value
            for key, value in state_dict.items()
            if key.startswith("proj.")
        }
        proj_result = self.proj.load_state_dict(proj_state, strict=strict)
        if self.highfreq is not None:
            hf_state = {
                key.replace("highfreq.", "", 1): value
                for key, value in state_dict.items()
                if key.startswith("highfreq.")
            }
            return self.highfreq.load_state_dict(hf_state, strict=False)
        # nn.Module contract: always return an unpackable (missing, unexpected).
        # The highfreq branch landed with `return None` on this path, which crashes
        # train_memory_dense_adapter_interaction_v1.py -- it unpacks the result
        # (_enc_missing, _enc_unexpected = encoder.load_state_dict(...)). state_v0
        # only survived because it discards the return value. highfreq is off in
        # both our args and the DLC package, so this path is the one always taken.
        return proj_result


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
        self.activation_checkpoint_adapter = bool(getattr(config, "activation_checkpoint_adapter", False))
        self.memory_dense_adapter_scale = nn.Parameter(torch.tensor(float(config.residual_scale_init)))
        self.memory_dense_adapter = ZeroInitMemoryDenseResidualAdapter(
            hidden_dim=hidden_dim,
            cond_dim=config.cond_dim,
            adapter_hidden_dim=config.adapter_hidden_dim,
            residual_mode=config.residual_mode,
        )

    def _adapter_residual(self, hidden: torch.Tensor, block_kwargs: dict[str, Any]) -> torch.Tensor:
        dit_cond_dict = block_kwargs.get("dit_cond_dict")
        if dit_cond_dict is None or self.cond_key not in dit_cond_dict:
            return hidden
        cond_tokens = dit_cond_dict[self.cond_key].to(device=hidden.device, dtype=hidden.dtype)
        adapted = self.memory_dense_adapter(hidden, cond_tokens)
        return hidden + self.memory_dense_adapter_scale.to(device=hidden.device, dtype=hidden.dtype) * (adapted - hidden)

    def forward(self, *block_args: Any, **block_kwargs: Any) -> torch.Tensor:
        checkpoint_ok = (
            self.training
            and torch.is_grad_enabled()
            and block_kwargs.get("kv_cache") is None
            and block_kwargs.get("crossattn_cache") is None
        )
        if self.activation_checkpoint_adapter and checkpoint_ok:
            # Recompute block+adapter jointly in backward: without this the adapter
            # residual activations of every wrapped block stay resident through the
            # whole step (~85GB at 21x30x52 tokens), which cannot fit a 96GB GPU.
            def run_block_and_adapter(*args: Any) -> torch.Tensor:
                return self._adapter_residual(self.block(*args, **block_kwargs), block_kwargs)

            return checkpoint(run_block_and_adapter, *block_args, use_reentrant=False)
        if self.activation_checkpoint_blocks and checkpoint_ok:
            def run_block(*args: Any) -> torch.Tensor:
                return self.block(*args, **block_kwargs)

            hidden = checkpoint(run_block, *block_args, use_reentrant=False)
        else:
            hidden = self.block(*block_args, **block_kwargs)
        return self._adapter_residual(hidden, block_kwargs)


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
