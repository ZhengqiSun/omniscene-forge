#!/usr/bin/env python3
"""Fast-v2-only dense/state conditioning for the Phase 3 smoke trainer."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CONDITIONING_SCHEMA_VERSION = "lingbot-fast-v2-phase3-conditioning/v1"
STATE_ACTION_PROJECTOR_KIND = "lingbot_fast_v2_state_action_projector/v1"
STATE_CHANNELS = (
    "ego_alive_constant_plane",
    "ego_health_norm_constant_plane",
    "opponent_dead_marker_mask",
)


_DIFFERENTIABLE_SP_GROUP: Any = None


def _resolve_sp_group(group: Any) -> Any:
    return _DIFFERENTIABLE_SP_GROUP if group is None else group


def differentiable_all_to_all(
    value: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: Any = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Match LingBot's Ulysses layout using autograd-aware collectives."""

    import torch.distributed as dist
    import torch.distributed.nn.functional as dist_nn

    if kwargs:
        raise ValueError(f"unsupported differentiable all_to_all options: {sorted(kwargs)}")
    group = _resolve_sp_group(group)
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return value
    if value.shape[scatter_dim] % world_size:
        raise ValueError("differentiable all_to_all scatter dimension is not divisible")
    inputs = [part.contiguous() for part in value.chunk(world_size, dim=scatter_dim)]
    outputs = [torch.empty_like(part) for part in inputs]
    received = dist_nn.all_to_all(outputs, inputs, group=group)
    return torch.cat(tuple(received), dim=gather_dim).contiguous()


def differentiable_gather_forward(
    value: torch.Tensor,
    dim: int,
    group: Any = None,
) -> torch.Tensor:
    """Gather sequence shards without detaching the autograd graph."""

    import torch.distributed as dist
    import torch.distributed.nn.functional as dist_nn

    group = _resolve_sp_group(group)
    if dist.get_world_size(group) == 1:
        return value
    gathered = dist_nn.all_gather(value, group=group)
    return torch.cat(tuple(gathered), dim=dim).contiguous()


def install_differentiable_sequence_parallel_collectives(
    sequence_parallel: Any,
    *,
    process_group: Any = None,
) -> dict[str, Any]:
    """Patch only the loaded Fast v2 SP module, leaving official files untouched."""

    if getattr(sequence_parallel, "__name__", None) != "wan.distributed.sequence_parallel":
        raise ValueError("unexpected sequence-parallel module")
    global _DIFFERENTIABLE_SP_GROUP

    original_all_to_all = getattr(sequence_parallel, "all_to_all", None)
    original_gather = getattr(sequence_parallel, "gather_forward", None)
    if not callable(original_all_to_all) or not callable(original_gather):
        raise RuntimeError("official sequence-parallel collective hooks are missing")
    if process_group is None:
        raise ValueError("Fast v2 training requires a dedicated sequence-parallel process group")
    _DIFFERENTIABLE_SP_GROUP = process_group
    sequence_parallel.all_to_all = differentiable_all_to_all
    sequence_parallel.gather_forward = differentiable_gather_forward
    if sequence_parallel.all_to_all is not differentiable_all_to_all:
        raise RuntimeError("failed to install differentiable all_to_all")
    if sequence_parallel.gather_forward is not differentiable_gather_forward:
        raise RuntimeError("failed to install differentiable gather_forward")
    return {
        "installed": True,
        "module": sequence_parallel.__name__,
        "all_to_all": "torch.distributed.nn.functional.all_to_all",
        "gather_forward": "torch.distributed.nn.functional.all_gather",
        "dedicated_process_group": True,
        "official_source_modified": False,
        "original_all_to_all_module": getattr(original_all_to_all, "__module__", None),
        "original_gather_forward_module": getattr(original_gather, "__module__", None),
    }


def prepare_fast_v2_camera_chunks(
    poses_path: str | Path,
    intrinsics_path: str | Path,
    *,
    official_source_root: str | Path,
    device: torch.device,
    dtype: torch.dtype,
    height: int = 480,
    width: int = 832,
    latent_height: int = 60,
    latent_width: int = 104,
    chunk_size: int = 4,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Build camera chunks using the official LingBot Fast v2 camera route."""

    if (height, width) != (480, 832):
        raise ValueError("Fast v2 Phase 3 camera input is fixed to 480x832")
    if height // latent_height != 8 or width // latent_width != 8:
        raise ValueError("Fast v2 camera packing requires an exact 8x8 latent stride")
    if chunk_size != 4:
        raise ValueError("Fast v2 camera chunk size is fixed to 4")

    source_root = Path(official_source_root).resolve(strict=True)
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from wan.utils.cam_utils import (  # type: ignore
        compute_relative_poses,
        get_Ks_transformed,
        get_plucker_embeddings,
        interpolate_camera_poses,
    )

    c2ws = np.load(Path(poses_path)).astype("float32")
    intrinsics = torch.from_numpy(np.load(Path(intrinsics_path)).astype("float32"))
    if c2ws.ndim != 3 or c2ws.shape[1:] != (4, 4) or len(c2ws) < 2:
        raise ValueError(f"poses must be [F,4,4] with F>=2, got {c2ws.shape}")
    if intrinsics.ndim != 2 or intrinsics.shape != (len(c2ws), 4):
        raise ValueError(
            f"intrinsics must be [{len(c2ws)},4], got {tuple(intrinsics.shape)}"
        )

    transformed = get_Ks_transformed(
        intrinsics,
        height_org=480,
        width_org=832,
        height_resize=height,
        width_resize=width,
        height_final=height,
        width_final=width,
    )
    first_intrinsic = transformed[0]
    latent_frames = ((len(c2ws) - 1) // 4) + 1
    latent_frames -= latent_frames % chunk_size
    if latent_frames != 20:
        raise ValueError(
            f"Fast v2 Phase 3 requires exactly 20 usable camera frames, got {latent_frames}"
        )
    c2ws_infer = interpolate_camera_poses(
        src_indices=np.linspace(0, len(c2ws) - 1, len(c2ws)),
        src_rot_mat=c2ws[:, :3, :3],
        src_trans_vec=c2ws[:, :3, 3],
        tgt_indices=np.linspace(0, len(c2ws) - 1, latent_frames),
    )
    c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True).to(device)
    repeated_intrinsics = first_intrinsic.repeat(latent_frames, 1).to(device)
    plucker = get_plucker_embeddings(
        c2ws_infer,
        repeated_intrinsics,
        height,
        width,
    )
    full = (
        plucker.view(latent_frames, latent_height, 8, latent_width, 8, 6)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(latent_frames, latent_height, latent_width, 384)
        .permute(3, 0, 1, 2)
        .unsqueeze(0)
        .contiguous()
        .to(dtype=dtype)
    )
    if tuple(full.shape) != (1, 384, 20, 60, 104):
        raise RuntimeError(f"Fast v2 camera tensor shape drifted: {tuple(full.shape)}")
    if not bool(torch.isfinite(full).all().item()):
        raise RuntimeError("Fast v2 camera tensor contains non-finite values")
    chunks = tuple(full[:, :, start : start + chunk_size] for start in range(0, 20, 4))
    if len(chunks) != 5 or any(tuple(chunk.shape) != (1, 384, 4, 60, 104) for chunk in chunks):
        raise RuntimeError("Fast v2 camera chunk contract drifted")
    return full, chunks


class FastV2StateActionProjector(nn.Module):
    """Add zero-init state tokens to a fixed, lossless dense-channel packing.

    Camera/action control remains on the official c2ws Plucker route. The seven
    validated dense values occupy the first seven condition dimensions; this
    introduces no extra trainable encoder outside the explicit allowlist.
    """

    def __init__(self, *, dense_channels: int = 7, state_channels: int = 3, cond_dim: int = 128):
        super().__init__()
        if dense_channels <= 0 or state_channels <= 0 or cond_dim < dense_channels:
            raise ValueError("invalid Fast v2 conditioning dimensions")
        self.dense_channels = int(dense_channels)
        self.state_channels = int(state_channels)
        self.cond_dim = int(cond_dim)
        self.fast_v2_state_proj = nn.Linear(self.state_channels, self.cond_dim)
        nn.init.zeros_(self.fast_v2_state_proj.weight)
        nn.init.zeros_(self.fast_v2_state_proj.bias)

    def forward(
        self,
        dense: torch.Tensor,
        state: torch.Tensor,
        *,
        target_token_hw: tuple[int, int],
    ) -> torch.Tensor:
        if dense.ndim != 4 or state.ndim != 4:
            raise ValueError("dense and state must both be [F,C,H,W]")
        if dense.shape[0] != state.shape[0]:
            raise ValueError("dense/state frame counts differ")
        if dense.shape[1] != self.dense_channels or state.shape[1] != self.state_channels:
            raise ValueError("dense/state channel contract mismatch")
        target_h, target_w = (int(value) for value in target_token_hw)
        if target_h <= 0 or target_w <= 0:
            raise ValueError("target token grid must be positive")

        dtype = self.fast_v2_state_proj.weight.dtype
        device = self.fast_v2_state_proj.weight.device
        dense = F.interpolate(
            dense.to(device=device, dtype=dtype),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        state = F.interpolate(
            state.to(device=device, dtype=dtype),
            size=(target_h, target_w),
            mode="nearest",
        )
        frames = dense.shape[0]
        dense_values = dense.permute(0, 2, 3, 1).reshape(
            1, frames * target_h * target_w, self.dense_channels
        )
        fixed_dense = dense_values.new_zeros(
            1, frames * target_h * target_w, self.cond_dim
        )
        fixed_dense[..., : self.dense_channels] = dense_values
        state_values = state.permute(0, 2, 3, 1).reshape(
            1, frames * target_h * target_w, self.state_channels
        )
        return fixed_dense + self.fast_v2_state_proj(state_values)

    def assert_zero_initialization(self) -> dict[str, Any]:
        weight_nonzero = int(torch.count_nonzero(self.fast_v2_state_proj.weight).item())
        bias_nonzero = int(torch.count_nonzero(self.fast_v2_state_proj.bias).item())
        if weight_nonzero or bias_nonzero:
            raise RuntimeError("Fast v2 state/action projector is not exactly zero initialized")
        return {"exact": True, "weight_nonzero": 0, "bias_nonzero": 0}

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": CONDITIONING_SCHEMA_VERSION,
            "kind": STATE_ACTION_PROJECTOR_KIND,
            "dense_channels": self.dense_channels,
            "state_channels": list(STATE_CHANNELS),
            "cond_dim": self.cond_dim,
            "dense_route": "fixed_first_dimensions_no_trainable_encoder",
            "state_projection": f"zero_init_linear_{self.state_channels}_to_{self.cond_dim}",
            "action_route": "official_c2ws_plucker_camera_condition",
        }


def partition_sequence_parallel(tokens: torch.Tensor, *, rank: int, world_size: int) -> torch.Tensor:
    """Partition already patch-aligned tokens exactly like the official SP path."""

    if tokens.ndim != 3 or tokens.shape[0] != 1:
        raise ValueError("Fast v2 condition tokens must be [1,L,C]")
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid sequence-parallel rank/world size")
    if tokens.shape[1] % world_size:
        raise ValueError("Fast v2 condition token count is not divisible by world size")
    return tokens.chunk(world_size, dim=1)[rank].contiguous()
