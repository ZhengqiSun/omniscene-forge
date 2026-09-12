#!/usr/bin/env python3
"""Formal training mainline for the Map Memory dense Wan adapter.

This is not a smoke script.  It is the intended entry point for dense-adapter
training once a release manifest and a LingBot latent-cache manifest have been
aligned.  Lightweight modes such as ``audit-data`` and ``check-noop`` use the
same data contract and adapter code as training, but they do not create
checkpoints or run optimizer steps. ``preflight-train`` uses the formal training
data contract and real cache payloads, then stops before loading the Wan
checkpoint or constructing an optimizer.

The training mode intentionally refuses to run from a Map Memory release alone:
LingBot training needs video latents, first-frame condition latents, text cache,
camera controls, and one Map Memory sample id per latent frame.
"""

from __future__ import annotations

from runtime_paths import ASSET_ROOT, LINGBOT_ROOT

import argparse
from contextlib import nullcontext
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from teacher_player_latent_mask_v0 import teacher_player_mask_latent
import map_memory_training_data_v0 as map_memory_data
from map_memory_training_data_v0 import (
    CHANNELS,
    MapMemoryRelease,
    MapMemorySample,
    load_dense,
    load_json,
    read_jsonl,
    sample_from_manifest,
    sha256_file,
    split_report,
    split_samples,
)
from build_memory_dense_aligned_cache_v0 import (
    expected_sample_ids,
    parse_clip_identity,
    validate_record_shapes,
)
from memory_dense_wan_adapter_v0 import (
    MemoryDenseAdapterConfig,
    MemoryDenseFrozenVAEWanTokenEncoder,
    MemoryDenseWanTokenEncoder,
    assert_wan_token_grid_matches_dense,
    inject_lora_into_wan_model_fast,
    load_lora_state_dict,
    lora_parameters,
    lora_state_dict,
    memory_dense_adapter_parameters,
    wrap_wan_model_fast_with_memory_dense_adapter,
)
from memory_dense_state_adapter_v0 import (
    STATE_CHANNELS_V0,
    StateTokenProjector,
    load_state_manifest,
    load_state_tensor,
    resolve_state_cache_path,
)
from memory_dense_interaction_adapter_v1 import (
    INTERACTION_CHANNELS_V1,
    InteractionTokenProjector,
    load_interaction_tensor,
)
from w4_controlled_sampler_v1 import DEFAULT_QUOTAS, SAMPLER_VERSION, STRATA, W4SamplerV1, validate_quotas


DEFAULT_MANIFEST = Path("docs/assets/memory_dense_dataset_v43_bsp_all5_match_large_h176/manifest.json")
COND_KEY = "memory_dense_cond_tokens"
RELEASE_INDEX_KEY = "_training_release_index"
TRAINING_INDEX_CACHE_KIND = "memory_dense_training_input_index_cache_v0"
TRAINING_INDEX_CACHE_VERSION = 1
INFERENCE_SIGMA_BAND_CENTERS = (1.0, 0.932, 0.843, 0.587)
INFERENCE_SIGMA_BAND_RADIUS = 0.02
MIN_FAIL_FAST_TRAIN_RECORDS = 100
W5_SAMPLER_VERSION = "w5_loss_balanced_v1"
W5_EXTENSION_SAMPLER_VERSION = "w5_loss_balanced_extension_v1"
W5_LOSS_MULTIPLIERS = {
    "strict_zero": 8.0,
    "boundary": 1.0,
    "person_positive": 1.0,
    "natural_anchor": 1.0,
}

INTERACTION_V1_FEATURE_DIM = 134


def validate_interaction_v1_protocol(
    projector: torch.nn.Module | None = None,
) -> dict[str, Any]:
    feature_dim = len(INTERACTION_CHANNELS_V1)
    if feature_dim != INTERACTION_V1_FEATURE_DIM:
        raise RuntimeError(
            "interaction V1 protocol feature_dim mismatch: "
            f"len(INTERACTION_CHANNELS_V1)={feature_dim}, "
            f"expected={INTERACTION_V1_FEATURE_DIM}"
        )

    projector_input_dim = None
    if projector is not None:
        projector_module = unwrap_module(projector)
        projection = getattr(projector_module, "proj", None)
        projector_input_dim = getattr(projection, "in_features", None)
        if projector_input_dim != feature_dim:
            raise RuntimeError(
                "interaction V1 projector/protocol mismatch: "
                f"projector_input_dim={projector_input_dim}, "
                f"feature_dim={feature_dim}"
            )
    return {
        "feature_dim": feature_dim,
        "projector_input_dim": projector_input_dim,
        "status": "pass",
    }


def interaction_chunk_audit(interaction: torch.Tensor) -> dict[str, Any]:
    validate_interaction_v1_protocol()
    if interaction.ndim != 2 or interaction.shape[1] != INTERACTION_V1_FEATURE_DIM:
        raise RuntimeError(
            "interaction chunk must be [F,134] for audit, "
            f"got {tuple(interaction.shape)}"
        )
    detached = interaction.detach()
    event_values = detached[:, :4].float().sum(dim=0).cpu().tolist()
    event_counts = {
        feature_name.removeprefix("ego_").removesuffix("_event"): int(value)
        for feature_name, value in zip(INTERACTION_CHANNELS_V1[:4], event_values)
    }
    return {
        "event_counts": event_counts,
        "nonzero_feature_count": int(torch.count_nonzero(detached).cpu()),
    }
W6_SAMPLER_VERSION = "w6_diverse_positive_guard_v1"
W6_QUOTAS = {
    "strict_zero": 3,
    "boundary": 2,
    "person_positive": 5,
    "natural_anchor": 2,
}
W6_LOSS_MULTIPLIERS = {
    "strict_zero": 6.0,
    "boundary": 1.0,
    "person_positive": 1.0,
    "natural_anchor": 1.0,
}
W7_SAMPLER_VERSION = "w7_negative_space_lora_only_v1"
W7_QUOTAS = dict(W6_QUOTAS)
W7_LOSS_MULTIPLIERS = {
    "strict_zero": 4.0,
    "boundary": 1.0,
    "person_positive": 1.0,
    "natural_anchor": 1.0,
}


def dense_hw_from_args(args: argparse.Namespace) -> tuple[int, int]:
    """Dense condition grid (H, W) for this run, from --dense-hw.

    Single source of truth for every dense-grid constant in this trainer.
    v2 dense contract (416x240 -> 30x52 native, no interpolation); the legacy
    v43 contract (320x176 -> 22x40 native) stays reachable via --dense-hw 176 320.
    """
    raw = getattr(args, "dense_hw", None) or list(map_memory_data.dense_hw())
    height, width = int(raw[0]), int(raw[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"--dense-hw must be positive, got {(height, width)}")
    return height, width


def add_lingbot_path(path: Path) -> None:
    root = str(path.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def setup_dist() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1 and not dist.is_initialized():
        torch.cuda.set_device(local)
        dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    return rank, world, local, device


def cleanup_dist() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if hasattr(module, "module") else module


def assert_only_memory_dense_trainable(
    model: torch.nn.Module,
    encoder: torch.nn.Module,
    state_projector: torch.nn.Module | None = None,
    interaction_projector: torch.nn.Module | None = None,
    *,
    freeze_lora: bool = False,
    freeze_state_projector: bool = False,
    lora_only: bool = False,
) -> dict[str, Any]:
    bad_model = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and "memory_dense_adapter" not in name and ".lora_A" not in name and ".lora_B" not in name
    ]
    if bad_model:
        raise RuntimeError(f"unexpected trainable base model parameters: {bad_model[:20]}")
    adapter_params = memory_dense_adapter_parameters(model)
    if not adapter_params:
        raise RuntimeError("no trainable memory_dense_adapter parameters after wrapping")
    bad_adapter = [name for name, param in adapter_params if param.requires_grad == lora_only]
    if bad_adapter:
        expected = "frozen" if lora_only else "trainable"
        raise RuntimeError(f"memory_dense_adapter parameters are not uniformly {expected}: {bad_adapter[:20]}")
    lora_params = lora_parameters(model)
    bad_lora = [name for name, param in lora_params if param.requires_grad == freeze_lora]
    if bad_lora:
        expected = "frozen" if freeze_lora else "trainable"
        raise RuntimeError(f"LoRA parameters are not uniformly {expected}: {bad_lora[:20]}")
    encoder_params = list(encoder.named_parameters())
    bad_encoder = [name for name, param in encoder_params if param.requires_grad == lora_only]
    if bad_encoder:
        expected = "frozen" if lora_only else "trainable"
        raise RuntimeError(f"memory_dense_encoder parameters are not uniformly {expected}: {bad_encoder[:20]}")
    if isinstance(encoder, MemoryDenseFrozenVAEWanTokenEncoder):
        metadata = encoder.metadata
        if metadata.get("vae_trainable_param_count") != 0:
            raise RuntimeError(f"frozen VAE encoder reports trainable VAE params: {metadata}")
        if not lora_only and metadata.get("projection_trainable_param_count", 0) <= 0:
            raise RuntimeError(f"frozen VAE encoder projection has no trainable params: {metadata}")
    state_params = list(state_projector.named_parameters()) if state_projector is not None else []
    bad_state = [name for name, param in state_params if param.requires_grad == freeze_state_projector]
    if bad_state:
        expected = "frozen" if freeze_state_projector else "trainable"
        raise RuntimeError(f"memory_dense_state_projector parameters are not uniformly {expected}: {bad_state[:20]}")
    interaction_params = (
    list(interaction_projector.named_parameters())
    if interaction_projector is not None
        else []
    )

    bad_interaction = [
        name
        for name, param in interaction_params
        if not param.requires_grad
    ]

    if bad_interaction:
        raise RuntimeError(
            "memory_dense_interaction_projector must be trainable, "
            f"but found frozen parameters: {bad_interaction[:20]}"
        )
    return {
        "adapter_trainable_param_count": int(sum(param.numel() for _, param in adapter_params)),
        "encoder_trainable_param_count": int(sum(param.numel() for _, param in encoder_params)),
        "lora_trainable_param_count": int(sum(param.numel() for _, param in lora_params if param.requires_grad)),
        "state_projector_trainable_param_count": int(sum(param.numel() for _, param in state_params if param.requires_grad)),
        "adapter_trainable_tensor_count": len(adapter_params),
        "encoder_trainable_tensor_count": len(encoder_params),
        "lora_trainable_tensor_count": sum(param.requires_grad for _, param in lora_params),
        "state_projector_trainable_tensor_count": sum(param.requires_grad for _, param in state_params),
        "lora_total_tensor_count": len(lora_params),
        "state_projector_total_tensor_count": len(state_params),
        "adapter_effective_trainable_tensor_count": sum(param.requires_grad for _, param in adapter_params),
        "encoder_effective_trainable_tensor_count": sum(param.requires_grad for _, param in encoder_params),
        "state_projector_effective_trainable_tensor_count": sum(param.requires_grad for _, param in state_params),
        "interaction_projector_trainable_param_count": int(sum(param.numel() for _, param in interaction_params if param.requires_grad)),
        "interaction_projector_trainable_tensor_count": int(sum(param.requires_grad for _, param in interaction_params)),
        "interaction_projector_total_tensor_count": len(interaction_params),
        "freeze_lora": freeze_lora,
        "freeze_state_projector": freeze_state_projector,
        "lora_only": lora_only,
        "base_trainable_param_count": 0,
    }


def parse_w4_quotas(values: list[str] | None) -> dict[str, int]:
    if values is None:
        return dict(DEFAULT_QUOTAS)
    result: dict[str, int] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"invalid --w4-quota {item!r}; expected stratum=count")
        name, raw = item.split("=", 1)
        if name in result:
            raise ValueError(f"duplicate --w4-quota stratum {name!r}")
        result[name] = int(raw)
    return result


def parse_w5_loss_multipliers(values: list[str] | None) -> dict[str, float]:
    if values is None:
        return {}
    result: dict[str, float] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"invalid --w5-loss-multiplier {item!r}; expected stratum=value")
        name, raw = item.split("=", 1)
        if name in result:
            raise ValueError(f"duplicate --w5-loss-multiplier stratum {name!r}")
        result[name] = float(raw)
    return result


def apply_w5_loss_multiplier(loss: torch.Tensor, *, stratum: str, multipliers: dict[str, float]) -> tuple[torch.Tensor, float]:
    if multipliers != W5_LOSS_MULTIPLIERS:
        raise ValueError(f"W5 loss multipliers must be exactly {W5_LOSS_MULTIPLIERS}, got {multipliers}")
    if stratum not in multipliers:
        raise ValueError(f"unknown W5 stratum {stratum!r}")
    multiplier = multipliers[stratum]
    return loss * multiplier, multiplier


def apply_w6_loss_multiplier(loss: torch.Tensor, *, stratum: str, multipliers: dict[str, float]) -> tuple[torch.Tensor, float]:
    if multipliers != W6_LOSS_MULTIPLIERS:
        raise ValueError(f"W6 loss multipliers must be exactly {W6_LOSS_MULTIPLIERS}, got {multipliers}")
    if stratum not in multipliers:
        raise ValueError(f"unknown W6 stratum {stratum!r}")
    multiplier = multipliers[stratum]
    return loss * multiplier, multiplier


def apply_w7_loss_multiplier(loss: torch.Tensor, *, stratum: str, multipliers: dict[str, float]) -> tuple[torch.Tensor, float]:
    if multipliers != W7_LOSS_MULTIPLIERS:
        raise ValueError(f"W7 loss multipliers must be exactly {W7_LOSS_MULTIPLIERS}, got {multipliers}")
    if stratum not in multipliers:
        raise ValueError(f"unknown W7 stratum {stratum!r}")
    multiplier = multipliers[stratum]
    return loss * multiplier, multiplier


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ddp_kwargs(*, local_rank: int, w4_enabled: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "device_ids": [local_rank],
        "output_device": local_rank,
        "find_unused_parameters": False,
    }
    if w4_enabled:
        kwargs["gradient_as_bucket_view"] = True
    return kwargs


def cuda_memory_telemetry(
    *,
    device: torch.device,
    phase: str,
    rank: int,
    trainable_gradient_bytes: int,
    step: int | None = None,
    micro_step: int | None = None,
) -> dict[str, Any]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    allocated_bytes = torch.cuda.memory_allocated(device)
    reserved_bytes = torch.cuda.memory_reserved(device)
    payload: dict[str, Any] = {
        "event": "w4_cuda_memory",
        "phase": phase,
        "rank": rank,
        "device": str(device),
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "allocated_bytes": allocated_bytes,
        "reserved_bytes": reserved_bytes,
        "allocator_slack_bytes": reserved_bytes - allocated_bytes,
        "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "trainable_gradient_bytes": trainable_gradient_bytes,
        "free_minus_trainable_gradient_bytes": free_bytes - trainable_gradient_bytes,
    }
    if step is not None:
        payload["step"] = step
    if micro_step is not None:
        payload["micro_step"] = micro_step
    return payload


def module_gradient_norm(named_params: list[tuple[str, torch.nn.Parameter]]) -> float:
    squared = 0.0
    for _, param in named_params:
        if param.grad is not None:
            squared += float(param.grad.detach().float().norm(2).cpu()) ** 2
    return squared ** 0.5


def import_wan_model_fast(lingbot_repo: Path):
    if not lingbot_repo.exists():
        raise FileNotFoundError(f"LingBot repo not found: {lingbot_repo}")
    add_lingbot_path(lingbot_repo)
    from wan.modules.model_fast import WanModelFast  # type: ignore

    return WanModelFast


def import_wan_model_base(lingbot_repo: Path):
    """Import the bidirectional base WanModel + the i2v-A14B config (two-expert base)."""
    if not lingbot_repo.exists():
        raise FileNotFoundError(f"LingBot repo not found: {lingbot_repo}")
    add_lingbot_path(lingbot_repo)
    from wan.modules.model import WanModel  # type: ignore
    from wan.configs import WAN_CONFIGS  # type: ignore

    return WanModel, WAN_CONFIGS["i2v-A14B"]


def resolve_dense_vae_path(args: argparse.Namespace) -> Path:
    if args.dense_vae_pth is not None:
        path = args.dense_vae_pth
    else:
        add_lingbot_path(args.lingbot_repo)
        from wan.configs import WAN_CONFIGS  # type: ignore

        cfg = WAN_CONFIGS["i2v-A14B"]
        path = args.ckpt_dir / cfg.vae_checkpoint
    if not path.exists():
        raise FileNotFoundError(f"missing frozen dense VAE checkpoint: {path}")
    return path


def build_dense_encoder(
    args: argparse.Namespace,
    config: MemoryDenseAdapterConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    if args.dense_condition_encoder == "conv":
        encoder = MemoryDenseWanTokenEncoder(config).to(device=device, dtype=dtype)
        encoder.only_player_dense_channels = bool(args.only_player_dense_channels)
        return encoder
    if args.dense_condition_encoder != "frozen_vae":
        raise ValueError(f"unsupported --dense-condition-encoder {args.dense_condition_encoder!r}")
    if args.only_player_dense_channels:
        raise ValueError("--only-player-dense-channels is incompatible with --dense-condition-encoder frozen_vae")
    add_lingbot_path(args.lingbot_repo)
    from wan.modules.vae2_1 import Wan2_1_VAE  # type: ignore

    vae_pth = resolve_dense_vae_path(args)
    vae = Wan2_1_VAE(vae_pth=str(vae_pth), device=device)
    # v2 dense contract (416x240 -> 30x52 native, no interpolation): the frozen VAE
    # emits its condition latent on the (H/vae_stride, W/vae_stride) grid.
    dense_h, dense_w = dense_hw_from_args(args)
    encoder = MemoryDenseFrozenVAEWanTokenEncoder(
        config,
        vae=vae,
        vae_pth=vae_pth,
        native_hw=(dense_h // args.vae_stride, dense_w // args.vae_stride),
        packing=args.dense_vae_packing,
    ).to(device=device, dtype=dtype)
    return encoder


def dense_encoder_metadata(encoder: torch.nn.Module) -> dict[str, Any]:
    if hasattr(encoder, "metadata"):
        return dict(getattr(encoder, "metadata"))
    return {
        "type": "conv",
        "native_stride": "dense_to_wan_token_stride",
        "checkpoint_state": "full_encoder",
    }


def resolve_record_sample_id(record: dict[str, Any]) -> str:
    for key in ["map_memory_sample_id", "memory_dense_sample_id", "sample_id"]:
        value = record.get(key)
        if value:
            return str(value)
    raise ValueError(f"cache record has no Map Memory sample id: {record.get('clip_id', '<unknown>')}")


def resolve_record_sample_ids(
    record: dict[str, Any],
    *,
    latent_frames: int,
    allow_static_dense_repeat: bool,
) -> list[str]:
    for key in ["map_memory_sample_ids", "memory_dense_sample_ids"]:
        value = record.get(key)
        if value is not None:
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError(f"{record.get('clip_id', '<unknown>')}: {key} must be a list[str]")
            if len(value) != latent_frames:
                raise ValueError(f"{record.get('clip_id', '<unknown>')}: {key} length {len(value)} != latent_frames {latent_frames}")
            return value
    single = resolve_record_sample_id(record)
    if not allow_static_dense_repeat:
        raise ValueError(
            f"{record.get('clip_id', '<unknown>')}: formal training requires map_memory_sample_ids per latent frame; "
            "pass --allow-static-dense-repeat only for an explicitly accepted debug run"
        )
    return [single for _ in range(latent_frames)]


def validate_record_paths(record: dict[str, Any], *, video_frames: int) -> list[str]:
    errors: list[str] = []
    clip_id = str(record.get("clip_id", ""))
    for key in ["latent_cache", "text_cache", "poses", "intrinsics"]:
        raw = record.get(key)
        if not raw:
            errors.append(f"missing {key}")
            continue
        path = Path(str(raw))
        if not path.exists():
            errors.append(f"{key} does not exist: {path}")
            continue
        if key in {"latent_cache", "poses", "intrinsics"} and clip_id and clip_id not in path.name and clip_id not in str(path.parent):
            errors.append(f"{key} path is not clip-bound to clip_id={clip_id}: {path}")
    poses_path = record.get("poses")
    if poses_path and Path(str(poses_path)).exists():
        poses = np.load(str(poses_path), mmap_mode="r")
        if poses.shape != (video_frames, 4, 4):
            errors.append(f"poses shape {tuple(poses.shape)} != {(video_frames, 4, 4)}")
    intrinsics_path = record.get("intrinsics")
    if intrinsics_path and Path(str(intrinsics_path)).exists():
        intrinsics = np.load(str(intrinsics_path), mmap_mode="r")
        if intrinsics.shape[0] != video_frames:
            errors.append(f"intrinsics frame count {intrinsics.shape[0]} != {video_frames}")
        if intrinsics.ndim != 2 or intrinsics.shape[1] not in {4, 9}:
            errors.append(f"intrinsics shape {tuple(intrinsics.shape)} is not [frames,4] or [frames,9]")
    return errors


def manifest_pairs_from_args(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    if args.cache_manifest is None:
        raise ValueError("training requires --cache-manifest with rows aligned to Map Memory sample ids")
    extra_map_manifests = list(args.extra_map_manifest or [])
    extra_cache_manifests = list(args.extra_cache_manifest or [])
    if len(extra_map_manifests) != len(extra_cache_manifests):
        raise ValueError(
            f"--extra-map-manifest count {len(extra_map_manifests)} must match "
            f"--extra-cache-manifest count {len(extra_cache_manifests)}"
        )
    return [(args.map_manifest, args.cache_manifest)] + list(zip(extra_map_manifests, extra_cache_manifests))


def training_index_cache_contract(
    args: argparse.Namespace,
    manifest_pairs: list[tuple[Path, Path]],
) -> dict[str, Any]:
    return {
        "kind": TRAINING_INDEX_CACHE_KIND,
        "version": TRAINING_INDEX_CACHE_VERSION,
        "inputs": [
            {
                "release_index": idx,
                "map_manifest": str(map_manifest.resolve()),
                "map_manifest_sha256": sha256_file(map_manifest),
                "cache_manifest": str(cache_manifest.resolve()),
                "cache_manifest_sha256": sha256_file(cache_manifest),
            }
            for idx, (map_manifest, cache_manifest) in enumerate(manifest_pairs)
        ],
        "params": {
            "latent_frames": int(args.latent_frames),
            "video_frames": int(args.video_frames),
            "raw_stride": int(args.raw_stride),
            "video_height": int(args.video_height),
            "video_width": int(args.video_width),
            "vae_stride": int(args.vae_stride),
            "patch_size_hw": int(args.patch_size_hw),
            "split_key": args.split_key,
            "train_split": args.train_split,
            "allow_static_dense_repeat": bool(args.allow_static_dense_repeat),
            "require_backend_id": str(args.require_backend_id),
            "limit": args.limit,
            "require_release_report": bool(args.require_release_report),
            "lightweight_training_index_rebuild": bool(args.lightweight_training_index_rebuild),
            "min_train_records": int(args.min_train_records),
            "min_train_positive_frames": int(args.min_train_positive_frames),
            "min_train_context_frames": int(args.min_train_context_frames),
            "min_train_matches": int(args.min_train_matches),
            "min_train_episodes": int(args.min_train_episodes),
        },
    }


def log_loader_stage(args: argparse.Namespace, event: str, **payload: Any) -> None:
    if getattr(args, "log_loader_stages", False):
        print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def sample_to_cache_row(sample: MapMemorySample) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "match_id": sample.match_id,
        "episode": sample.episode,
        "raw_episode": sample.raw_episode,
        "ego_stem": sample.ego_stem,
        "frame_index": int(sample.frame_index),
        "selection_role": sample.selection_role,
        "dense_path": str(sample.dense_path),
        "target_rgb_path": str(sample.target_rgb_path),
        "meta_path": str(sample.meta_path),
        "qa_path": str(sample.qa_path),
        "row": sample.row,
    }


def sample_from_cache_row(row: dict[str, Any]) -> MapMemorySample:
    return MapMemorySample(
        sample_id=str(row["sample_id"]),
        match_id=str(row["match_id"]),
        episode=str(row["episode"]),
        raw_episode=str(row["raw_episode"]),
        ego_stem=str(row["ego_stem"]),
        frame_index=int(row["frame_index"]),
        selection_role=str(row["selection_role"]),
        dense_path=Path(str(row["dense_path"])),
        target_rgb_path=Path(str(row["target_rgb_path"])),
        meta_path=Path(str(row["meta_path"])),
        qa_path=Path(str(row["qa_path"])),
        row=row.get("row") if isinstance(row.get("row"), dict) else row,
    )


def build_cached_sample_index(
    releases: list[MapMemoryRelease],
    records: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    wanted: list[set[str]] = [set() for _ in releases]
    for record in records:
        release_index = int(record.get(RELEASE_INDEX_KEY, 0))
        sample_ids = resolve_record_sample_ids(
            record,
            latent_frames=args.latent_frames,
            allow_static_dense_repeat=args.allow_static_dense_repeat,
        )
        wanted[release_index].update(sample_ids)
    out: list[dict[str, Any]] = []
    for release_index, release in enumerate(releases):
        missing = sorted(sid for sid in wanted[release_index] if sid not in release.by_id)
        if missing:
            raise ValueError(f"cannot cache release {release_index}: missing samples first={missing[:5]}")
        out.append(
            {
                "release_index": release_index,
                "manifest_path": str(release.manifest_path),
                "sample_count": len(wanted[release_index]),
                "samples": [sample_to_cache_row(release.by_id[sid]) for sid in sorted(wanted[release_index])],
            }
        )
    return out


def build_lightweight_release_from_manifest(
    manifest_path: Path,
    *,
    require_backend_id: str,
    sample_ids: set[str],
) -> MapMemoryRelease:
    manifest_path = manifest_path.resolve()
    manifest = load_json(manifest_path)
    # v2 dense contract (416x240 -> 30x52 native, no interpolation): read the loader
    # module attribute at call time so --dense-hw / set_dense_hw() stay authoritative.
    expected_shape = list(map_memory_data.SHAPE)
    if manifest.get("shape") != expected_shape:
        raise ValueError(f"{manifest_path}: manifest shape {manifest.get('shape')} != {expected_shape}")
    if manifest.get("channels") != CHANNELS:
        raise ValueError(f"{manifest_path}: channel order mismatch")
    if manifest.get("geometry_backend_id") != require_backend_id:
        raise ValueError(f"{manifest_path}: geometry_backend_id {manifest.get('geometry_backend_id')} != {require_backend_id}")
    samples_raw = manifest.get("samples")
    if not isinstance(samples_raw, list):
        raise ValueError(f"{manifest_path}: manifest samples must be a list")
    wanted = set(sample_ids)
    samples: list[MapMemorySample] = []
    for sample in samples_raw:
        sid = str(sample.get("sample_id", ""))
        if sid not in wanted:
            continue
        role = str(sample.get("selection_role", ""))
        if role not in {"positive", "context"}:
            raise ValueError(f"{manifest_path}: {sid}: invalid selection_role {role!r}")
        if sample.get("shape") != expected_shape:
            raise ValueError(f"{manifest_path}: {sid}: sample shape {sample.get('shape')} != {expected_shape}")
        if sample.get("channels") != CHANNELS:
            raise ValueError(f"{manifest_path}: {sid}: sample channel order mismatch")
        if sample.get("geometry_backend_id") != require_backend_id:
            raise ValueError(f"{manifest_path}: {sid}: backend {sample.get('geometry_backend_id')} != {require_backend_id}")
        # For Memory-mask light releases the QA row is not needed to construct
        # the training sample; keep row-local fields for mask surrogate lookup.
        samples.append(sample_from_manifest(manifest_path, sample, sample))
    found = {sample.sample_id for sample in samples}
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"{manifest_path}: lightweight manifest index missing sample ids first={missing[:5]}")
    return MapMemoryRelease(
        manifest_path,
        manifest,
        {"status": "cached_lightweight"},
        {"sample_count": len(samples), "rows": []},
        samples,
    )


def load_aligned_cache_records_lightweight(
    path: Path,
    *,
    args: argparse.Namespace,
    release_index: int,
    map_manifest_sha256: str,
) -> tuple[list[dict[str, Any]], set[str], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing aligned latent cache manifest: {path}")
    records_all = read_jsonl(path, limit=args.limit)
    if not records_all:
        raise ValueError(f"empty latent cache manifest: {path}")
    selected: list[dict[str, Any]] = []
    sample_ids_needed: set[str] = set()
    split_counts: dict[str, int] = {}
    errors: list[str] = []
    for record in records_all:
        split = str(record.get("map_memory_split", ""))
        split_counts[split] = split_counts.get(split, 0) + 1
        if args.train_split is not None and split != args.train_split:
            continue
        if record.get("alignment_kind") != "map_memory_dense_lingbot_latent_frame_exact_v0":
            errors.append(f"{record.get('clip_id','<unknown>')}: alignment_kind={record.get('alignment_kind')!r}")
        if record.get("alignment_exact_frame_match") is not True:
            errors.append(f"{record.get('clip_id','<unknown>')}: alignment_exact_frame_match is not true")
        if record.get("map_memory_manifest_sha256") != map_manifest_sha256:
            errors.append(f"{record.get('clip_id','<unknown>')}: map_memory_manifest_sha256 mismatch")
        if record.get("map_memory_split_key") != args.split_key:
            errors.append(f"{record.get('clip_id','<unknown>')}: map_memory_split_key={record.get('map_memory_split_key')!r} != {args.split_key!r}")
        if record.get("dtype") == "metadata_only":
            errors.append(f"{record.get('clip_id','<unknown>')}: metadata-only cache record cannot train")
        for required in ["latent_cache", "text_cache", "poses", "intrinsics"]:
            if not record.get(required):
                errors.append(f"{record.get('clip_id','<unknown>')}: missing {required}")
        for msg in validate_record_shapes(
            record,
            latent_frames=args.latent_frames,
            video_height=args.video_height,
            video_width=args.video_width,
        ):
            errors.append(f"{record.get('clip_id','<unknown>')}: {msg}")
        sample_ids = resolve_record_sample_ids(
            record,
            latent_frames=args.latent_frames,
            allow_static_dense_repeat=args.allow_static_dense_repeat,
        )
        record[RELEASE_INDEX_KEY] = release_index
        selected.append(record)
        sample_ids_needed.update(sample_ids)
    if errors:
        raise ValueError("lightweight aligned cache validation failed:\n" + "\n".join(f"- {msg}" for msg in errors[:80]))
    if args.train_split is not None and not selected:
        raise ValueError(f"aligned cache has no records for required split {args.train_split!r}; split_counts={split_counts}")
    return selected, sample_ids_needed, {"selected_record_count": len(selected), "split_counts": split_counts}


def releases_from_cached_sample_index(obj: dict[str, Any], manifest_pairs: list[tuple[Path, Path]]) -> list[MapMemoryRelease]:
    cached_releases = obj.get("sample_index")
    if not isinstance(cached_releases, list):
        raise ValueError("training index cache has no sample_index; rebuild it")
    by_index: dict[int, dict[str, Any]] = {}
    for item in cached_releases:
        if not isinstance(item, dict):
            raise ValueError("training index cache sample_index entries must be objects")
        by_index[int(item.get("release_index", -1))] = item
    releases: list[MapMemoryRelease] = []
    for release_index, (map_manifest, _cache_manifest) in enumerate(manifest_pairs):
        item = by_index.get(release_index)
        if item is None:
            raise ValueError(f"training index cache missing sample_index for release {release_index}")
        samples_raw = item.get("samples")
        if not isinstance(samples_raw, list) or not samples_raw:
            raise ValueError(f"training index cache release {release_index} has no cached samples")
        samples = [sample_from_cache_row(row) for row in samples_raw]
        releases.append(
            MapMemoryRelease(
                map_manifest.resolve(),
                {"sample_count": len(samples), "samples": [], "cached_lightweight": True},
                {"status": "cached_lightweight"},
                {"sample_count": len(samples), "rows": []},
                samples,
            )
        )
    return releases


def validate_training_index_cache_contract(
    obj: dict[str, Any],
    expected: dict[str, Any],
    *,
    path: Path,
) -> None:
    if obj.get("kind") != TRAINING_INDEX_CACHE_KIND:
        raise ValueError(f"{path}: kind={obj.get('kind')!r}, expected {TRAINING_INDEX_CACHE_KIND!r}")
    if int(obj.get("version", -1)) != TRAINING_INDEX_CACHE_VERSION:
        raise ValueError(f"{path}: version={obj.get('version')!r}, expected {TRAINING_INDEX_CACHE_VERSION}")
    actual = obj.get("contract")
    if actual != expected:
        raise ValueError(
            f"{path}: training index cache contract mismatch; rerun preflight with --rebuild-training-index-cache. "
            f"expected={expected} actual={actual}"
        )


def lightweight_validate_cached_records(
    records: list[dict[str, Any]],
    releases: list[MapMemoryRelease],
    *,
    args: argparse.Namespace,
    map_manifest_sha256_by_release: list[str],
) -> dict[str, Any]:
    """Cheap cache-hit guardrails: no full CPFS sidecar sweep."""
    errors: list[str] = []
    split_counts: dict[str, int] = {}
    for idx, record in enumerate(records):
        release_index = int(record.get(RELEASE_INDEX_KEY, -1))
        if release_index < 0 or release_index >= len(releases):
            errors.append(f"{record.get('clip_id', idx)}: invalid release index {release_index}")
            continue
        sample_ids = resolve_record_sample_ids(
            record,
            latent_frames=args.latent_frames,
            allow_static_dense_repeat=args.allow_static_dense_repeat,
        )
        release = releases[release_index]
        missing = [sid for sid in sample_ids if sid not in release.by_id]
        if missing:
            errors.append(f"{record.get('clip_id', idx)}: sample ids missing from release; first={missing[:5]}")
        if record.get("dtype") == "metadata_only":
            errors.append(f"{record.get('clip_id', idx)}: metadata-only cache record cannot train")
        if record.get("map_memory_manifest_sha256") != map_manifest_sha256_by_release[release_index]:
            errors.append(f"{record.get('clip_id', idx)}: map_memory_manifest_sha256 mismatch")
        for required in ["latent_cache", "text_cache", "poses", "intrinsics"]:
            if not record.get(required):
                errors.append(f"{record.get('clip_id', idx)}: cache record missing {required}")
        split = str(record.get("map_memory_split", ""))
        split_counts[split] = split_counts.get(split, 0) + 1
    if errors:
        raise ValueError("cached training index validation failed:\n" + "\n".join(f"- {msg}" for msg in errors[:80]))
    return {"checked_record_count": len(records), "split_counts": split_counts}


def probe_cached_record_paths(
    records: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    max_records: int,
) -> dict[str, Any]:
    """Gold-path probe for cache hits: sample path existence and npy shapes."""
    if max_records <= 0 or not records:
        return {"checked_record_count": 0, "errors": []}
    errors: list[str] = []
    checked: list[str] = []
    if len(records) <= max_records:
        indices = list(range(len(records)))
    else:
        indices = sorted({round(i * (len(records) - 1) / (max_records - 1)) for i in range(max_records)})
    for idx in indices:
        record = records[idx]
        checked.append(str(record.get("clip_id", idx)))
        errors.extend(f"{record.get('clip_id', idx)}: {msg}" for msg in validate_record_paths(record, video_frames=args.video_frames))
    if errors:
        raise ValueError("cached training index path probe failed:\n" + "\n".join(f"- {msg}" for msg in errors[:40]))
    return {"checked_record_count": len(indices), "checked_clip_ids": checked, "errors": []}


def load_training_index_cache(
    path: Path,
    *,
    args: argparse.Namespace,
    manifest_pairs: list[tuple[Path, Path]],
    expected_contract: dict[str, Any],
) -> tuple[list[MapMemoryRelease], list[dict[str, Any]], dict[str, Any]] | None:
    if not path.exists():
        return None
    t0 = time.time()
    log_loader_stage(args, "training_index_cache_read_start", path=str(path))
    obj = json.loads(path.read_text(encoding="utf-8"))
    validate_training_index_cache_contract(obj, expected_contract, path=path)
    log_loader_stage(args, "training_index_cache_read_done", elapsed_sec=round(time.time() - t0, 3), bytes=path.stat().st_size)
    releases = releases_from_cached_sample_index(obj, manifest_pairs)
    log_loader_stage(args, "training_index_cache_releases_restored", release_count=len(releases))
    records = obj.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: cache records must be a non-empty list")
    map_hashes = [item["map_manifest_sha256"] for item in expected_contract["inputs"]]
    light_report = lightweight_validate_cached_records(records, releases, args=args, map_manifest_sha256_by_release=map_hashes)
    path_probe = probe_cached_record_paths(records, args=args, max_records=args.training_index_cache_path_checks)
    log_loader_stage(args, "training_index_cache_hit_validated", record_count=len(records), elapsed_sec=round(time.time() - t0, 3))
    return releases, records, {
        "path": str(path),
        "status": "hit",
        "contract": expected_contract,
        "record_count": len(records),
        "created_by": obj.get("created_by"),
        "light_validation": light_report,
        "path_probe": path_probe,
    }


def write_training_index_cache(
    path: Path,
    *,
    args: argparse.Namespace,
    contract: dict[str, Any],
    releases: list[MapMemoryRelease],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "kind": TRAINING_INDEX_CACHE_KIND,
        "version": TRAINING_INDEX_CACHE_VERSION,
        "created_by": "tools/train_memory_dense_adapter_v0.py",
        "contract": contract,
        "record_count": len(records),
        "sample_index": build_cached_sample_index(releases, records, args=args),
        "records": records,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return {"path": str(path), "status": "built", "contract": contract, "record_count": len(records)}


def validate_aligned_record_contract(
    record: dict[str, Any],
    release: MapMemoryRelease,
    *,
    args: argparse.Namespace,
    map_manifest_sha256: str,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    sample_ids = resolve_record_sample_ids(
        record,
        latent_frames=args.latent_frames,
        allow_static_dense_repeat=args.allow_static_dense_repeat,
    )
    clip_id = str(record.get("clip_id", ""))
    if record.get("alignment_kind") != "map_memory_dense_lingbot_latent_frame_exact_v0":
        errors.append(f"{clip_id}: alignment_kind={record.get('alignment_kind')!r}")
    if record.get("alignment_exact_frame_match") is not True:
        errors.append(f"{clip_id}: alignment_exact_frame_match is not true")
    row_manifest_hash = record.get("map_memory_manifest_sha256")
    if row_manifest_hash != map_manifest_sha256:
        errors.append(f"{clip_id}: map_memory_manifest_sha256 mismatch")
    row_split_key = record.get("map_memory_split_key")
    if row_split_key != args.split_key:
        errors.append(f"{clip_id}: map_memory_split_key={row_split_key!r} != {args.split_key!r}")

    shape_errors = validate_record_shapes(
        record,
        latent_frames=args.latent_frames,
        video_height=args.video_height,
        video_width=args.video_width,
    )
    errors.extend(f"{clip_id}: {msg}" for msg in shape_errors)
    try:
        identity = parse_clip_identity(record, record)
        raw_frames, expected_ids = expected_sample_ids(
            identity,
            video_frames=args.video_frames,
            raw_stride=args.raw_stride,
            latent_frames=args.latent_frames,
        )
        if sample_ids != expected_ids:
            errors.append(f"{clip_id}: map_memory_sample_ids do not match clip_id/raw_start exact-frame formula")
        if record.get("map_memory_raw_frame_indices") != raw_frames:
            errors.append(f"{clip_id}: map_memory_raw_frame_indices do not match exact-frame formula")
    except Exception as exc:
        errors.append(f"{clip_id}: cannot validate clip identity/frame mapping: {exc}")

    missing = [sid for sid in sample_ids if sid not in release.by_id]
    if missing:
        errors.append(f"{clip_id}: {len(missing)} sample ids missing from release; first={missing[:5]}")
    else:
        expected_roles = [release.by_id[sid].selection_role for sid in sample_ids]
        if record.get("map_memory_selection_roles") != expected_roles:
            errors.append(f"{clip_id}: map_memory_selection_roles do not match release samples")
    errors.extend(f"{clip_id}: {msg}" for msg in validate_record_paths(record, video_frames=args.video_frames))
    return errors, sample_ids


def load_aligned_cache_records(
    path: Path,
    release: MapMemoryRelease,
    *,
    args: argparse.Namespace,
    latent_frames: int,
    allow_static_dense_repeat: bool,
    limit: int | None = None,
    required_split: str | None = "train",
    map_manifest_sha256: str | None = None,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing aligned latent cache manifest: {path}")
    records = read_jsonl(path, limit=limit)
    if not records:
        raise ValueError(f"empty latent cache manifest: {path}")
    map_manifest_sha256 = map_manifest_sha256 or sha256_file(release.manifest_path)
    contract_errors: list[str] = []
    metadata_only: list[str] = []
    split_counts: dict[str, int] = {}
    selected: list[dict[str, Any]] = []
    for record in records:
        record_errors, sample_ids = validate_aligned_record_contract(
            record,
            release,
            args=args,
            map_manifest_sha256=map_manifest_sha256,
        )
        contract_errors.extend(record_errors[:20])
        for sid in sample_ids:
            if sid not in release.by_id:
                continue
        if record.get("dtype") == "metadata_only":
            metadata_only.append(str(record.get("clip_id", sample_ids[0])))
        for required in ["latent_cache", "text_cache", "poses", "intrinsics"]:
            if not record.get(required):
                contract_errors.append(f"{record.get('clip_id', sample_ids[0])}: cache record missing {required}")
        split = str(record.get("map_memory_split", ""))
        split_counts[split] = split_counts.get(split, 0) + 1
        if required_split is None or split == required_split:
            selected.append(record)
    if contract_errors:
        raise ValueError("aligned cache contract validation failed:\n" + "\n".join(f"- {msg}" for msg in contract_errors[:80]))
    if metadata_only:
        raise ValueError(f"metadata-only cache records cannot train; first={metadata_only[:5]}")
    if required_split is not None and not selected:
        raise ValueError(f"aligned cache has no records for required split {required_split!r}; split_counts={split_counts}")
    return selected


def latent_hw_from_record(record: dict[str, Any]) -> tuple[int, int] | None:
    shape = record.get("shape")
    if isinstance(shape, list) and len(shape) == 4:
        return int(shape[2]), int(shape[3])
    return None


def validate_cache_grid(records: list[dict[str, Any]], args: argparse.Namespace) -> tuple[tuple[int, int], tuple[int, int]]:
    expected_latent_hw = (args.video_height // args.vae_stride, args.video_width // args.vae_stride)
    latent_hws: set[tuple[int, int]] = set()
    bad_records: list[str] = []
    for record in records:
        latent_hw = latent_hw_from_record(record)
        if latent_hw is not None:
            latent_hws.add(latent_hw)
            if latent_hw != expected_latent_hw:
                bad_records.append(f"{record.get('clip_id', '<unknown>')}: shape latent_hw={latent_hw}, expected={expected_latent_hw}")
        condition_shape = record.get("condition_shape")
        if isinstance(condition_shape, list) and len(condition_shape) == 4:
            cond_frames = int(condition_shape[1])
            cond_hw = (int(condition_shape[2]), int(condition_shape[3]))
            if cond_frames != args.latent_frames or cond_hw != expected_latent_hw:
                bad_records.append(
                    f"{record.get('clip_id', '<unknown>')}: condition_shape={condition_shape}, "
                    f"expected frames/hw={[args.latent_frames, *expected_latent_hw]}"
                )
        latent_frames_expected = record.get("latent_frames_expected")
        if latent_frames_expected is not None and int(latent_frames_expected) != args.latent_frames:
            bad_records.append(f"{record.get('clip_id', '<unknown>')}: latent_frames_expected={latent_frames_expected} != {args.latent_frames}")
    if len(latent_hws) > 1:
        raise ValueError(f"cache manifest mixes latent spatial grids: {sorted(latent_hws)}")
    if bad_records:
        raise ValueError("cache grid validation failed:\n" + "\n".join(f"- {msg}" for msg in bad_records[:20]))
    latent_hw = next(iter(latent_hws), expected_latent_hw)
    if latent_hw[0] % args.patch_size_hw != 0 or latent_hw[1] % args.patch_size_hw != 0:
        raise ValueError(f"latent_hw={latent_hw} must be divisible by patch_size_hw={args.patch_size_hw}")
    return latent_hw, (latent_hw[0] // args.patch_size_hw, latent_hw[1] // args.patch_size_hw)


def load_release_report(manifest_path: Path) -> dict[str, Any] | None:
    base = manifest_path.resolve().parent
    for name in ["combined_report_v0.json", "export_report_v0.json"]:
        path = base / name
        if path.exists():
            obj = json.loads(path.read_text(encoding="utf-8"))
            obj["_report_path"] = str(path)
            return obj
    return None


def validate_release_report_gate(
    release: MapMemoryRelease,
    records: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    report = load_release_report(release.manifest_path)
    role_counts: dict[str, int] = {}
    match_ids: set[str] = set()
    episode_ids: set[str] = set()
    train_positive_frames = 0
    train_context_frames = 0
    for record in records:
        roles = record.get("map_memory_selection_roles") or []
        train_positive_frames += sum(1 for role in roles if role == "positive")
        train_context_frames += sum(1 for role in roles if role == "context")
        if record.get("map_memory_match_id"):
            match_ids.add(str(record["map_memory_match_id"]))
        if record.get("map_memory_episode"):
            episode_ids.add(str(record["map_memory_episode"]))
    for sample in release.samples:
        role_counts[sample.selection_role] = role_counts.get(sample.selection_role, 0) + 1

    fail_fast_floor = int(getattr(args, "min_fail_fast_train_records", MIN_FAIL_FAST_TRAIN_RECORDS))
    if len(records) < fail_fast_floor:
        raise SystemExit(
            f"fail-fast: matched train records {len(records)} < {fail_fast_floor}; "
            "refusing to run because this usually means the aligned cache matched zero or too few records"
        )

    failures: list[str] = []
    if args.require_release_report:
        if report is None:
            failures.append("missing combined_report_v0.json or export_report_v0.json next to map manifest")
        elif report.get("status") != "pass":
            failures.append(f"release report {report.get('_report_path')} status is {report.get('status')!r}")
    if len(records) < args.min_train_records:
        failures.append(f"train records {len(records)} < min_train_records {args.min_train_records}")
    if train_positive_frames < args.min_train_positive_frames:
        failures.append(f"train positive latent frames {train_positive_frames} < min_train_positive_frames {args.min_train_positive_frames}")
    if train_context_frames < args.min_train_context_frames:
        failures.append(f"train context latent frames {train_context_frames} < min_train_context_frames {args.min_train_context_frames}")
    if len(match_ids) < args.min_train_matches:
        failures.append(f"train match count {len(match_ids)} < min_train_matches {args.min_train_matches}")
    if len(episode_ids) < args.min_train_episodes:
        failures.append(f"train episode count {len(episode_ids)} < min_train_episodes {args.min_train_episodes}")

    gate = {
        "release_report": report,
        "release_role_counts": role_counts,
        "train_record_count": len(records),
        "train_positive_latent_frames": train_positive_frames,
        "train_context_latent_frames": train_context_frames,
        "train_match_count": len(match_ids),
        "train_episode_count": len(episode_ids),
        "required": {
            "require_release_report": args.require_release_report,
            "min_train_records": args.min_train_records,
            "min_train_positive_frames": args.min_train_positive_frames,
            "min_train_context_frames": args.min_train_context_frames,
            "min_train_matches": args.min_train_matches,
            "min_train_episodes": args.min_train_episodes,
        },
        "failures": failures,
    }
    if failures:
        raise ValueError("formal training release gate failed:\n" + "\n".join(f"- {msg}" for msg in failures))
    return gate


def build_chunk_starts(latent_frames: int, chunk_size: int) -> list[int]:
    if chunk_size < 1 or chunk_size > latent_frames:
        raise ValueError(f"invalid chunk_size={chunk_size} for latent_frames={latent_frames}")
    starts = list(range(0, latent_frames - chunk_size + 1, chunk_size))
    tail = latent_frames - chunk_size
    if starts[-1] != tail:
        starts.append(tail)
    return starts


def pick_record(
    records: list[dict[str, Any]],
    *,
    step: int,
    micro_step: int,
    rank: int,
    world: int,
    grad_accum: int,
) -> tuple[dict[str, Any], int]:
    index = ((step - 1) * grad_accum + micro_step) * world + rank
    return records[index % len(records)], index % len(records)


def release_for_record(record: dict[str, Any], releases: list[MapMemoryRelease]) -> MapMemoryRelease:
    index = int(record.get(RELEASE_INDEX_KEY, 0))
    try:
        return releases[index]
    except IndexError as exc:
        raise ValueError(f"{record.get('clip_id', '<unknown>')}: invalid release index {index}") from exc


def chunk_roles_for_record(
    record: dict[str, Any],
    releases: list[MapMemoryRelease],
    *,
    chunk_start: int,
    chunk_size: int,
    latent_frames: int,
    allow_static_dense_repeat: bool,
) -> list[str]:
    sample_ids = resolve_record_sample_ids(
        record,
        latent_frames=latent_frames,
        allow_static_dense_repeat=allow_static_dense_repeat,
    )
    release = release_for_record(record, releases)
    return [release.by_id[sid].selection_role for sid in sample_ids[chunk_start : chunk_start + chunk_size]]


def build_positive_chunk_index(
    records: list[dict[str, Any]],
    releases: list[MapMemoryRelease],
    *,
    args: argparse.Namespace,
    chunk_starts: list[int],
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for record_index, record in enumerate(records):
        for chunk_start in chunk_starts:
            roles = chunk_roles_for_record(
                record,
                releases,
                chunk_start=chunk_start,
                chunk_size=args.chunk_size,
                latent_frames=args.latent_frames,
                allow_static_dense_repeat=args.allow_static_dense_repeat,
            )
            if "positive" in roles:
                out.append((record_index, chunk_start))
    return out


_RECORD_PERM_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def _record_permutation(n: int, seed: int, epoch: int) -> np.ndarray:
    key = (n, seed, epoch)
    perm = _RECORD_PERM_CACHE.get(key)
    if perm is None:
        rng = np.random.RandomState((seed * 1000003 + epoch * 7919 + 12345) % (2**31 - 1))
        perm = rng.permutation(n)
        if len(_RECORD_PERM_CACHE) > 8:
            _RECORD_PERM_CACHE.clear()
        _RECORD_PERM_CACHE[key] = perm
    return perm


def pick_training_record_chunk(
    records: list[dict[str, Any]],
    positive_chunks: list[tuple[int, int]],
    chunk_starts: list[int],
    *,
    step: int,
    micro_step: int,
    rank: int,
    world: int,
    grad_accum: int,
    positive_batch_fraction: float,
    seed: int,
    shuffle_records: bool = False,
) -> tuple[dict[str, Any], int, int, bool]:
    index = ((step - 1) * grad_accum + micro_step) * world + rank
    use_positive = False
    if positive_chunks and positive_batch_fraction > 0.0:
        draw = ((index * 1103515245 + seed) % 10000) / 10000.0
        use_positive = draw < positive_batch_fraction
    if use_positive:
        pos_n = len(positive_chunks)
        pos_slot = index % pos_n
        if shuffle_records:
            pos_slot = int(_record_permutation(pos_n, seed, index // pos_n)[pos_slot])
        record_index, chunk_start = positive_chunks[pos_slot]
        return records[record_index], record_index, chunk_start, True
    rec_n = len(records)
    rec_slot = index % rec_n
    if shuffle_records:
        rec_slot = int(_record_permutation(rec_n, seed + 1, index // rec_n)[rec_slot])
    record_index = rec_slot
    chunk_start = chunk_starts[(step + micro_step + record_index) % len(chunk_starts)]
    return records[record_index], record_index, chunk_start, False


def shuffled_chunk_samples(
    records: list[dict[str, Any]],
    releases: list[MapMemoryRelease],
    *,
    args: argparse.Namespace,
    record_index: int,
    chunk_start: int,
    offset: int,
) -> list[MapMemorySample]:
    if len(records) < 2:
        other_index = record_index
    else:
        other_index = (record_index + max(1, offset)) % len(records)
        if other_index == record_index:
            other_index = (other_index + 1) % len(records)
    other_record = records[other_index]
    sample_ids = resolve_record_sample_ids(
        other_record,
        latent_frames=args.latent_frames,
        allow_static_dense_repeat=args.allow_static_dense_repeat,
    )
    release = release_for_record(other_record, releases)
    return [release.by_id[sid] for sid in sample_ids[chunk_start : chunk_start + args.chunk_size]]


def load_text_context(record: dict[str, Any], device: torch.device) -> torch.Tensor:
    obj = torch.load(record["text_cache"], map_location="cpu")
    context = obj.get("context")
    if context is None:
        raise ValueError(f"{record['text_cache']}: missing context tensor")
    return context.to(device)


def load_latent_pair(record: dict[str, Any], device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    obj = torch.load(record["latent_cache"], map_location="cpu")
    if "latent" not in obj or "condition" not in obj:
        raise ValueError(f"{record['latent_cache']}: missing latent/condition tensors")
    return obj["latent"].to(device=device, dtype=dtype), obj["condition"].to(device=device, dtype=dtype)


def prepare_cam_chunk(
    record: dict[str, Any],
    *,
    latent_frames: int,
    chunk_start: int,
    chunk_size: int,
    height: int,
    width: int,
    vae_stride: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    from einops import rearrange
    from wan.utils.cam_utils import compute_relative_poses, get_Ks_transformed, get_plucker_embeddings, interpolate_camera_poses

    c2ws = np.load(record["poses"]).astype("float32")
    ks_full = torch.from_numpy(np.load(record["intrinsics"]).astype("float32"))
    if ks_full.shape[0] != len(c2ws):
        raise ValueError(f"{record.get('clip_id')}: intrinsics frame count {ks_full.shape[0]} != poses frame count {len(c2ws)}")
    latent_h, latent_w = height // vae_stride, width // vae_stride
    c2ws_infer = interpolate_camera_poses(
        src_indices=np.linspace(0, len(c2ws) - 1, len(c2ws)),
        src_rot_mat=c2ws[:, :3, :3],
        src_trans_vec=c2ws[:, :3, 3],
        tgt_indices=np.linspace(0, len(c2ws) - 1, latent_frames),
    )
    c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True).to(device)
    latent_positions = torch.from_numpy(np.rint(np.linspace(0, len(c2ws) - 1, latent_frames)).astype("int64"))
    ks = ks_full.index_select(0, latent_positions)
    ks = get_Ks_transformed(
        ks,
        height_org=height,
        width_org=width,
        height_resize=height,
        width_resize=width,
        height_final=height,
        width_final=width,
    ).to(device)
    emb = get_plucker_embeddings(c2ws_infer, ks, height, width, only_rays_d=False)
    emb = rearrange(
        emb,
        "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
        c1=height // latent_h,
        c2=width // latent_w,
    )
    emb = rearrange(emb[None, ...], "b (f h w) c -> b c f h w", f=latent_frames, h=latent_h, w=latent_w)
    return emb[:, :, chunk_start : chunk_start + chunk_size].to(dtype)


def dense_tokens_for_samples(
    encoder: torch.nn.Module,
    samples: list[MapMemorySample],
    *,
    device: torch.device,
    dtype: torch.dtype,
    target_token_hw: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, tuple[int, int]]:
    if not samples:
        raise ValueError("dense token chunk has no samples")
    dense_np = np.stack([load_dense(sample) for sample in samples], axis=0)
    if getattr(encoder, "only_player_dense_channels", False):
        dense_np[:, :3] = 0.0
    dense = torch.from_numpy(dense_np).to(device=device, dtype=dtype)
    tokens, token_hw = encoder(dense, target_token_hw=target_token_hw)
    tokens = tokens.reshape(1, len(samples) * tokens.shape[1], tokens.shape[2]).to(dtype)
    return tokens, token_hw


def region_loss_for_samples(
    pred: torch.Tensor,
    target: torch.Tensor,
    samples: list[MapMemorySample],
    *,
    latent_hw: tuple[int, int],
    min_pixels: int,
    normalize: str = "region",
) -> tuple[torch.Tensor | None, int]:
    masks = [teacher_player_mask_latent(sample, latent_hw=latent_hw) for sample in samples]
    mask_np = np.stack(masks, axis=0).astype(bool)
    mask = torch.from_numpy(mask_np).to(device=pred.device)
    region_pixels = int(mask.sum().detach().cpu())
    if region_pixels < min_pixels:
        return None, region_pixels
    mask_c = mask.unsqueeze(0).expand(pred.shape[0], -1, -1, -1)
    if normalize == "total":
        sq_err = (pred.float() - target.float()) ** 2
        return (sq_err * mask_c.float()).sum() / pred.numel(), region_pixels
    return F.mse_loss(pred.float()[mask_c], target.float()[mask_c]), region_pixels


def sigma_band_label(sigma_value: float) -> str:
    for center in INFERENCE_SIGMA_BAND_CENTERS:
        if abs(sigma_value - center) <= INFERENCE_SIGMA_BAND_RADIUS:
            return f"infer_{center:.3f}"
    return "outside_infer_bands"


def sample_training_sigma(args: argparse.Namespace, *, device: torch.device) -> torch.Tensor:
    prob = float(getattr(args, "sigma_curriculum_prob", 0.0))
    if prob <= 0.0:
        return torch.rand((), device=device, dtype=torch.float32)
    if prob > 1.0:
        raise ValueError(f"--sigma-curriculum-prob must be in [0,1], got {prob}")
    if torch.rand((), device=device, dtype=torch.float32).item() >= prob:
        return torch.rand((), device=device, dtype=torch.float32)
    centers = torch.tensor(INFERENCE_SIGMA_BAND_CENTERS, device=device, dtype=torch.float32)
    index = torch.randint(0, len(INFERENCE_SIGMA_BAND_CENTERS), (), device=device)
    jitter = (torch.rand((), device=device, dtype=torch.float32) * 2.0 - 1.0) * float(args.sigma_curriculum_radius)
    sigma = centers[index] + jitter
    return sigma.clamp(0.0, 1.0)


def min_snr_flow_loss_weight(args: argparse.Namespace, sigma: torch.Tensor) -> torch.Tensor:
    gamma = float(getattr(args, "min_snr_gamma", 0.0))
    if gamma <= 0.0:
        return torch.ones((), device=sigma.device, dtype=torch.float32)
    eps = float(getattr(args, "min_snr_eps", 1e-4))
    sigma_f = sigma.float().clamp(eps, 1.0 - eps)
    alpha = 1.0 - sigma_f
    snr = (alpha * alpha) / (sigma_f * sigma_f)
    gamma_t = torch.tensor(gamma, device=sigma.device, dtype=torch.float32)
    return torch.minimum(snr, gamma_t) / (snr + 1.0)


def add_region_loss_stat(
    stats: dict[str, Any],
    key: str,
    *,
    region_loss: torch.Tensor,
    region_pixels: int,
    loss_weight: float = 1.0,
) -> None:
    bucket = stats.setdefault(key, {"loss_sum": 0.0, "chunks": 0, "pixels": 0})
    bucket["loss_sum"] += float(region_loss.detach().cpu())
    bucket["weighted_loss_sum"] = bucket.get("weighted_loss_sum", 0.0) + float(region_loss.detach().cpu()) * float(loss_weight)
    bucket["weight_sum"] = bucket.get("weight_sum", 0.0) + float(loss_weight)
    bucket["chunks"] += 1
    bucket["pixels"] += int(region_pixels)


def finalize_region_loss_stats(stats: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in sorted(stats.items()):
        chunks = int(value.get("chunks", 0))
        loss_sum = float(value.get("loss_sum", 0.0))
        item = {
            "loss_sum": loss_sum,
            "chunks": chunks,
            "pixels": int(value.get("pixels", 0)),
        }
        if "weighted_loss_sum" in value:
            weighted_loss_sum = float(value.get("weighted_loss_sum", 0.0))
            item["weighted_loss_sum"] = weighted_loss_sum
            item["weighted_loss_mean"] = weighted_loss_sum / chunks if chunks > 0 else None
            item["weight_mean"] = float(value.get("weight_sum", 0.0)) / chunks if chunks > 0 else None
        item["loss_mean"] = loss_sum / chunks if chunks > 0 else None
        out[key] = item
    return out


def trainable_state_dict(
    model: torch.nn.Module,
    encoder: torch.nn.Module,
    state_projector: torch.nn.Module | None = None,
    interaction_projector: torch.nn.Module | None = None,
) -> dict[str, Any]:
    model_to_save = unwrap_module(model)
    encoder_to_save = unwrap_module(encoder)
    out = {
        "memory_dense_adapter": {
            name: param.detach().cpu()
            for name, param in model_to_save.named_parameters()
            if "memory_dense_adapter" in name
        },
        "memory_dense_lora": lora_state_dict(model_to_save),
        "memory_dense_encoder": encoder_to_save.state_dict(),
    }
    if state_projector is not None:
        out["memory_dense_state_projector"] = unwrap_module(state_projector).state_dict()
    if interaction_projector is not None:
        out["memory_dense_interaction_projector"] = (
            unwrap_module(interaction_projector).state_dict()
    )
    return out


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    encoder: torch.nn.Module,
    state_projector: torch.nn.Module | None,
    interaction_projector: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "memory_dense_adapter_checkpoint_v0",
            "step": int(step),
            "state": trainable_state_dict(model, encoder, state_projector, interaction_projector),
            "optimizer": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def validate_interaction_checkpoint_compatibility(
    path: Path,
    *,
    checkpoint_state: dict[str, Any],
    interaction_projector: torch.nn.Module | None,
    reset_optimizer: bool,
) -> dict[str, Any]:
    protocol = validate_interaction_v1_protocol(interaction_projector)
    if interaction_projector is None:
        return {**protocol, "checkpoint_protocol": "disabled"}
    if not checkpoint_state:
        if not reset_optimizer:
            raise ValueError(
                f"{path}: checkpoint has no interaction projector; "
                "an interaction V1 warm start requires --reset-optimizer"
            )
        return {**protocol, "checkpoint_protocol": "missing_zero_init"}

    expected_state = unwrap_module(interaction_projector).state_dict()
    for name, expected in expected_state.items():
        if name not in checkpoint_state:
            continue
        actual = checkpoint_state[name]
        actual_shape = tuple(actual.shape) if hasattr(actual, "shape") else None
        if actual_shape != tuple(expected.shape):
            raise ValueError(
                f"{path}: incompatible interaction projector checkpoint field={name} "
                f"shape={actual_shape}, expected={tuple(expected.shape)} for the "
                f"{INTERACTION_V1_FEATURE_DIM}-dim interaction V1 protocol"
            )
    return {**protocol, "checkpoint_protocol": "interaction_v1_134"}


def load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    encoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    reset_optimizer: bool,
    state_projector: torch.nn.Module | None = None,
    interaction_projector: torch.nn.Module | None = None,
) -> int:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("state", {})
    adapter_state = state.get("memory_dense_adapter", {})
    lora_state = state.get("memory_dense_lora", {})
    encoder_state = state.get("memory_dense_encoder", {})
    state_projector_state = state.get("memory_dense_state_projector", {})
    interaction_projector_state = state.get("memory_dense_interaction_projector", {})
    if not adapter_state:
        raise ValueError(f"{path}: checkpoint has no memory_dense_adapter state")
    interaction_checkpoint_audit = validate_interaction_checkpoint_compatibility(
        path,
        checkpoint_state=interaction_projector_state,
        interaction_projector=interaction_projector,
        reset_optimizer=reset_optimizer,
    )
    missing, unexpected = model.load_state_dict(adapter_state, strict=False)
    unexpected_adapter = [name for name in unexpected if "memory_dense_adapter" in name]
    missing_adapter = [
        name
        for name in missing
        if "memory_dense_adapter" in name and not name.endswith("memory_dense_adapter_scale")
    ]
    if unexpected_adapter or missing_adapter:
        raise ValueError(
            f"{path}: adapter checkpoint mismatch missing={missing_adapter[:10]} unexpected={unexpected_adapter[:10]}"
        )
    missing_lora, unexpected_lora = load_lora_state_dict(model, lora_state)
    if missing_lora or unexpected_lora:
        raise ValueError(f"{path}: LoRA checkpoint mismatch missing={missing_lora[:10]} unexpected={unexpected_lora[:10]}")
    _enc_missing, _enc_unexpected = encoder.load_state_dict(encoder_state, strict=False)
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(json.dumps({"event": "encoder_load_state_dict",
            "missing": list(_enc_missing), "unexpected": list(_enc_unexpected),
            "note": "strict=False: architecture evolution from proj->net is expected"}),
            flush=True)
    state_projector_missing = []
    state_projector_unexpected = []
    interaction_projector_missing = []
    interaction_projector_unexpected = []
    if state_projector is not None:
        if state_projector_state:
            state_projector_missing, state_projector_unexpected = state_projector.load_state_dict(state_projector_state)
        else:
            state_projector_missing = list(state_projector.state_dict())
            state_projector.load_state_dict(state_projector.state_dict())
    if interaction_projector is not None:
        if interaction_projector_state:
            (
                interaction_projector_missing,
                interaction_projector_unexpected,
            ) = interaction_projector.load_state_dict(
                interaction_projector_state
            )     
        else:
            interaction_projector_missing = list(
                interaction_projector.state_dict()
            )
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(json.dumps({
            "event": "resume_state_dict_audit",
            "path": str(path),
            "adapter_missing": missing_adapter,
            "adapter_unexpected": unexpected_adapter,
            "lora_missing": list(missing_lora),
            "lora_unexpected": list(unexpected_lora),
            "state_projector_missing": list(state_projector_missing),
            "state_projector_unexpected": list(state_projector_unexpected),
            "state_projector_init": "checkpoint" if state_projector_state else ("zero_init" if state_projector is not None else "disabled"),
            "interaction_projector_missing": list(interaction_projector_missing),
            "interaction_projector_unexpected": list(interaction_projector_unexpected),
            "interaction_projector_init": "checkpoint" if interaction_projector_state else ("zero_init" if interaction_projector is not None else "disabled"),
            "interaction_checkpoint_protocol": interaction_checkpoint_audit,
        }, ensure_ascii=False), flush=True)
    if not reset_optimizer:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("step", 0))


def build_adapter_config(args: argparse.Namespace) -> MemoryDenseAdapterConfig:
    return MemoryDenseAdapterConfig(
        dense_channels=7,
        cond_dim=args.cond_dim,
        encoder_hidden_dim=args.encoder_hidden_dim,
        adapter_hidden_dim=args.adapter_hidden_dim,
        vae_stride=args.vae_stride,
        wan_patch_size_hw=args.patch_size_hw,
        cond_key=COND_KEY,
        residual_mode=args.adapter_residual_mode,
        wrap_first_blocks=args.adapter_wrap_first_blocks if args.adapter_wrap_first_blocks > 0 else None,
        residual_scale_init=args.adapter_residual_scale_init,
        activation_checkpoint_blocks=bool(args.activation_checkpoint_blocks),
        activation_checkpoint_adapter=bool(args.activation_checkpoint_adapter),
    )


def build_state_projector(args: argparse.Namespace, *, device: torch.device, dtype: torch.dtype) -> StateTokenProjector | None:
    if args.state_cache_manifest is None:
        return None
    return StateTokenProjector(state_channels=len(STATE_CHANNELS_V0), cond_dim=args.cond_dim).to(device=device, dtype=dtype)

def build_interaction_projector(
    args: argparse.Namespace,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> InteractionTokenProjector | None:
    validate_interaction_v1_protocol()
    # interaction 和 state 共用同一个 state-cache manifest。
    if not args.state_cache_manifest:
        return None

    projector = InteractionTokenProjector(
        interaction_channels=len(INTERACTION_CHANNELS_V1),
        cond_dim=args.cond_dim,
    ).to(
        device=device,
        dtype=dtype,
    )
    validate_interaction_v1_protocol(projector)
    return projector

def load_state_manifest_for_args(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    if args.state_cache_manifest is None:
        return {}
    rows = load_state_manifest(args.state_cache_manifest)
    for row in rows.values():
        row["state_cache"] = str(resolve_state_cache_path(row, args.state_cache_manifest))
    return rows


def state_tokens_for_record_chunk(
    state_projector: StateTokenProjector | None,
    state_rows: dict[str, dict[str, Any]],
    record: dict[str, Any],
    *,
    chunk_start: int,
    chunk_size: int,
    device: torch.device,
    dtype: torch.dtype,
    target_token_hw: tuple[int, int],
) -> torch.Tensor | None:
    if state_projector is None:
        return None
    clip_id = str(record.get("clip_id"))
    if clip_id not in state_rows:
        raise KeyError(f"{clip_id}: missing state cache row in --state-cache-manifest")
    frame_indices = list(range(chunk_start, chunk_start + chunk_size))
    state = load_state_tensor(state_rows[clip_id], frame_indices=frame_indices, device=device, dtype=dtype)
    return state_projector(state, target_token_hw=target_token_hw).to(dtype)

def interaction_tokens_for_record_chunk(
    interaction_projector: InteractionTokenProjector | None,
    state_rows: dict[str, dict[str, Any]],
    record: dict[str, Any],
    *,
    chunk_start: int,
    chunk_size: int,
    device: torch.device,
    dtype: torch.dtype,
    target_token_hw: tuple[int, int],
    return_features: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
    if interaction_projector is None:
        return None

    clip_id = str(record.get("clip_id"))
    if clip_id not in state_rows:
        raise KeyError(
            f"{clip_id}: missing state cache row "
            "in --state-cache-manifest"
        )

    frame_indices = list(
        range(chunk_start, chunk_start + chunk_size)
    )

    interaction = load_interaction_tensor(
        state_rows[clip_id],
        frame_indices=frame_indices,
        device=device,
        dtype=dtype,
    )

    tokens = interaction_projector(
        interaction,
        target_token_hw=target_token_hw,
    ).to(dtype=dtype)
    if return_features:
        return tokens, interaction
    return tokens


def validate_training_geometry(args: argparse.Namespace, latent_hw: tuple[int, int], wan_token_hw: tuple[int, int]) -> dict[str, Any]:
    # v2 dense contract (416x240 -> 30x52 native, no interpolation)
    dense_h, dense_w = dense_hw_from_args(args)
    dense_aspect = dense_w / dense_h
    video_aspect = args.video_width / args.video_height
    aspect_delta = abs(dense_aspect - video_aspect) / video_aspect
    if aspect_delta > args.max_aspect_ratio_delta:
        raise ValueError(
            f"dense aspect {dense_aspect:.6f} and video aspect {video_aspect:.6f} differ by {aspect_delta:.4f}, "
            f"above --max-aspect-ratio-delta {args.max_aspect_ratio_delta}; regenerate dense maps at a compatible aspect"
        )
    return {
        "dense_hw": [dense_h, dense_w],
        "video_hw": [args.video_height, args.video_width],
        "latent_hw": list(latent_hw),
        "wan_token_grid_hw": list(wan_token_hw),
        "dense_aspect": dense_aspect,
        "video_aspect": video_aspect,
        "aspect_delta_fraction": aspect_delta,
        "max_aspect_ratio_delta": args.max_aspect_ratio_delta,
    }


def audit_data(args: argparse.Namespace) -> dict[str, Any]:
    release = MapMemoryRelease.load(args.map_manifest, require_backend_id=args.require_backend_id)
    splits = split_samples(
        release.samples,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        split_key=args.split_key,
        seed=args.seed,
    )
    report = release.audit(dense_limit=args.dense_audit_limit, hash_payloads=args.hash_dense)
    report["split_key"] = args.split_key
    report["splits"] = split_report(splits)
    report["channel_contract"] = CHANNELS
    report["map_manifest_sha256"] = sha256_file(args.map_manifest) if args.hash_manifest else None
    return report


def load_and_validate_training_inputs(
    args: argparse.Namespace,
) -> tuple[list[MapMemoryRelease], list[dict[str, Any]], tuple[int, int], tuple[int, int], dict[str, Any], dict[str, Any]]:
    t_all = time.time()
    manifest_pairs = manifest_pairs_from_args(args)
    cache_contract = training_index_cache_contract(args, manifest_pairs)
    cache_report: dict[str, Any] = {"status": "disabled"}
    releases: list[MapMemoryRelease]
    records: list[dict[str, Any]]
    release_gates: list[dict[str, Any]]
    cached = None
    if args.training_index_cache and not args.rebuild_training_index_cache:
        cached = load_training_index_cache(
            args.training_index_cache,
            args=args,
            manifest_pairs=manifest_pairs,
            expected_contract=cache_contract,
        )
    if cached is not None:
        releases, records, cache_report = cached
        release_records_by_index: list[list[dict[str, Any]]] = [[] for _ in releases]
        for record in records:
            release_records_by_index[int(record[RELEASE_INDEX_KEY])].append(record)
        release_gates = [
            validate_release_report_gate(release, release_records_by_index[idx], args=args)
            for idx, release in enumerate(releases)
        ]
    else:
        releases = []
        records = []
        release_gates = []
        if args.lightweight_training_index_rebuild:
            for release_index, (map_manifest, cache_manifest) in enumerate(manifest_pairs):
                map_sha = sha256_file(map_manifest)
                t_records = time.time()
                log_loader_stage(
                    args,
                    "cache_records_lightweight_validate_start",
                    release_index=release_index,
                    cache_manifest=str(cache_manifest),
                )
                release_records, needed_sample_ids, cache_light_report = load_aligned_cache_records_lightweight(
                    cache_manifest,
                    args=args,
                    release_index=release_index,
                    map_manifest_sha256=map_sha,
                )
                log_loader_stage(
                    args,
                    "cache_records_lightweight_validate_done",
                    release_index=release_index,
                    record_count=len(release_records),
                    needed_sample_count=len(needed_sample_ids),
                    elapsed_sec=round(time.time() - t_records, 3),
                )
                t_release = time.time()
                log_loader_stage(
                    args,
                    "release_lightweight_index_start",
                    release_index=release_index,
                    map_manifest=str(map_manifest),
                    needed_sample_count=len(needed_sample_ids),
                )
                release = build_lightweight_release_from_manifest(
                    map_manifest,
                    require_backend_id=args.require_backend_id,
                    sample_ids=needed_sample_ids,
                )
                log_loader_stage(
                    args,
                    "release_lightweight_index_done",
                    release_index=release_index,
                    sample_count=len(release.samples),
                    elapsed_sec=round(time.time() - t_release, 3),
                )
                releases.append(release)
                records.extend(release_records)
                release_gate = validate_release_report_gate(release, release_records, args=args)
                release_gate["lightweight_loader"] = cache_light_report
                release_gates.append(release_gate)
        else:
            for release_index, (map_manifest, cache_manifest) in enumerate(manifest_pairs):
                t_release = time.time()
                log_loader_stage(args, "release_full_load_start", release_index=release_index, map_manifest=str(map_manifest))
                release = MapMemoryRelease.load(map_manifest, require_backend_id=args.require_backend_id)
                log_loader_stage(
                    args,
                    "release_full_load_done",
                    release_index=release_index,
                    sample_count=len(release.samples),
                    elapsed_sec=round(time.time() - t_release, 3),
                )
                t_records = time.time()
                log_loader_stage(args, "cache_records_full_validate_start", release_index=release_index, cache_manifest=str(cache_manifest))
                release_records = load_aligned_cache_records(
                    cache_manifest,
                    release,
                    latent_frames=args.latent_frames,
                    args=args,
                    allow_static_dense_repeat=args.allow_static_dense_repeat,
                    limit=args.limit,
                    required_split=args.train_split,
                    map_manifest_sha256=sha256_file(release.manifest_path),
                )
                log_loader_stage(
                    args,
                    "cache_records_full_validate_done",
                    release_index=release_index,
                    record_count=len(release_records),
                    elapsed_sec=round(time.time() - t_records, 3),
                )
                for record in release_records:
                    record[RELEASE_INDEX_KEY] = release_index
                releases.append(release)
                records.extend(release_records)
                release_gates.append(validate_release_report_gate(release, release_records, args=args))
        if args.training_index_cache:
            cache_report = write_training_index_cache(
                args.training_index_cache,
                args=args,
                contract=cache_contract,
                releases=releases,
                records=records,
            )
            cache_report["path_probe"] = probe_cached_record_paths(
                records,
                args=args,
                max_records=args.training_index_cache_path_checks,
            )
    if not records:
        raise ValueError("no train records after loading release/cache inputs")
    release_gate = {
        "mode": "multi_release" if len(releases) > 1 else "single_release",
        "release_count": len(releases),
        "total_train_record_count": len(records),
        "releases": release_gates,
        "training_index_cache": cache_report,
    }
    latent_hw, wan_token_hw = validate_cache_grid(records, args)
    geometry_report = validate_training_geometry(args, latent_hw, wan_token_hw)
    if args.dense_resize_mode != "bilinear":
        # v2 dense contract (416x240 -> 30x52 native, no interpolation): the frozen-VAE
        # condition path emits tokens on the VAE latent grid (H/stride, W/stride); the
        # conv path patchifies once more (H/stride/patch, W/stride/patch).
        dense_h, dense_w = dense_hw_from_args(args)
        if args.dense_condition_encoder == "frozen_vae":
            dense_native_hw = (dense_h // args.vae_stride, dense_w // args.vae_stride)
        else:
            dense_native_hw = (
                dense_h // args.vae_stride // args.patch_size_hw,
                dense_w // args.vae_stride // args.patch_size_hw,
            )
        if wan_token_hw != dense_native_hw:
            raise ValueError(
                f"cache Wan grid {wan_token_hw} differs from native dense grid {dense_native_hw}; "
                "use --dense-resize-mode bilinear to train a cache whose Wan grid differs from --dense-hw"
            )
    log_loader_stage(
        args,
        "training_inputs_ready",
        record_count=len(records),
        release_count=len(releases),
        cache_status=release_gate["training_index_cache"].get("status"),
        elapsed_sec=round(time.time() - t_all, 3),
    )
    return releases, records, latent_hw, wan_token_hw, geometry_report, release_gate


def prepare_index(args: argparse.Namespace) -> dict[str, Any]:
    release = MapMemoryRelease.load(args.map_manifest, require_backend_id=args.require_backend_id)
    splits = split_samples(
        release.samples,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        split_key=args.split_key,
        seed=args.seed,
    )
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    index_paths: dict[str, str] = {}
    for split, rows in splits.items():
        path = out_dir / f"map_memory_{split}_index.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for sample in rows:
                f.write(json.dumps({
                    "sample_id": sample.sample_id,
                    "match_id": sample.match_id,
                    "episode": sample.episode,
                    "raw_episode": sample.raw_episode,
                    "ego_stem": sample.ego_stem,
                    "frame_index": sample.frame_index,
                    "selection_role": sample.selection_role,
                    "dense_path": str(sample.dense_path),
                    "target_rgb_path": str(sample.target_rgb_path),
                }, ensure_ascii=False) + "\n")
        index_paths[split] = str(path)
    return {
        "kind": "map_memory_training_index_v0",
        "manifest": str(args.map_manifest),
        "split_key": args.split_key,
        "splits": split_report(splits),
        "index_paths": index_paths,
    }


def check_noop(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    release = MapMemoryRelease.load(args.map_manifest, require_backend_id=args.require_backend_id)
    sample = release.samples[0]
    config = build_adapter_config(args)
    encoder = build_dense_encoder(args, config, device=device, dtype=torch.float32)
    # v2 dense contract (416x240 -> 30x52 native, no interpolation)
    dense_h, dense_w = dense_hw_from_args(args)
    latent_h = dense_h // args.vae_stride
    latent_w = dense_w // args.vae_stride
    wan_h = latent_h // args.patch_size_hw
    wan_w = latent_w // args.patch_size_hw
    WanModelFast = import_wan_model_fast(args.lingbot_repo)
    model_kwargs = {
        "model_type": "t2v",
        "control_type": "cam",
        "patch_size": (1, args.patch_size_hw, args.patch_size_hw),
        "text_len": args.text_len,
        "in_dim": args.latent_channels,
        "dim": args.hidden_dim,
        "ffn_dim": args.ffn_dim,
        "freq_dim": args.freq_dim,
        "text_dim": args.text_dim,
        "out_dim": args.latent_channels,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "local_attn_size": -1,
        "sink_size": 0,
        "cross_attn_norm": True,
    }
    base = WanModelFast(**model_kwargs).to(device)
    adapted = copy.deepcopy(base).to(device)
    for param in base.parameters():
        param.requires_grad_(False)
    wrap_wan_model_fast_with_memory_dense_adapter(adapted, config, freeze_base=True)
    adapted.to(device)

    with torch.no_grad():
        dense_tokens, token_hw = dense_tokens_for_samples(
            encoder,
            [sample for _ in range(args.noop_frames)],
            device=device,
            dtype=torch.float32,
            target_token_hw=(wan_h, wan_w),
        )
    assert_wan_token_grid_matches_dense(
        dense_hw=(dense_h, dense_w),
        latent_hw=(latent_h, latent_w),
        wan_token_hw=(wan_h, wan_w),
        config=config,
    )
    if tuple(token_hw) != (wan_h, wan_w):
        raise RuntimeError(f"dense token grid {token_hw} != Wan token grid {(wan_h, wan_w)}")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    latent = torch.randn(args.latent_channels, args.noop_frames, latent_h, latent_w, generator=generator, device=device)
    context = torch.randn(args.context_tokens, args.text_dim, generator=generator, device=device) * 0.02
    timestep = torch.tensor([args.timestep], dtype=torch.float32, device=device)
    seq_len = args.noop_frames * wan_h * wan_w

    base.eval()
    adapted.eval()
    with torch.no_grad():
        base_out = base([latent], timestep, [context], seq_len)[0]
        adapted_out = adapted(
            [latent],
            timestep,
            [context],
            seq_len,
            dit_cond_dict={COND_KEY: dense_tokens},
        )[0]
    diff = (base_out - adapted_out).abs()
    adapter_params = memory_dense_adapter_parameters(adapted)
    return {
        "kind": "memory_dense_adapter_noop_gate_v0",
        "status": "pass" if float(diff.max()) <= args.noop_tolerance else "fail",
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "tolerance": args.noop_tolerance,
        "sample_id": sample.sample_id,
        "dense_channels": config.dense_channels,
        "dense_hw": [dense_h, dense_w],
        "latent_hw": [latent_h, latent_w],
        "wan_token_grid_hw": [wan_h, wan_w],
        "tokens_per_frame": wan_h * wan_w,
        "seq_len": seq_len,
        "adapter_trainable_param_count": int(sum(param.numel() for _, param in adapter_params)),
        "encoder_param_count": int(sum(param.numel() for param in encoder.parameters())),
        "dense_condition_encoder": dense_encoder_metadata(encoder),
        "token_order_modified": False,
        "rope_modified": False,
        "current_start_modified": False,
        "native_kv_cache_modified": False,
    }


def preflight_train(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    releases, records, latent_hw, wan_token_hw, geometry_report, release_gate = load_and_validate_training_inputs(args)
    state_rows = load_state_manifest_for_args(args)
    if state_rows:
        missing_state = [str(record.get("clip_id")) for record in records if str(record.get("clip_id")) not in state_rows]
        if missing_state:
            raise KeyError(f"--state-cache-manifest missing {len(missing_state)} training clips; first={missing_state[:5]}")
    add_lingbot_path(args.lingbot_repo)
    config = build_adapter_config(args)
    encoder = build_dense_encoder(args, config, device=device, dtype=dtype)
    state_projector = build_state_projector(args, device=device, dtype=dtype)
    interaction_projector = build_interaction_projector(
        args,
        device=device,
        dtype=dtype,
    )
    chunk_starts = build_chunk_starts(args.latent_frames, args.chunk_size)
    checked_records: list[dict[str, Any]] = []
    max_records = min(len(records), max(1, args.preflight_records))
    for idx, record in enumerate(records[:max_records]):
        sample_ids = resolve_record_sample_ids(
            record,
            latent_frames=args.latent_frames,
            allow_static_dense_repeat=args.allow_static_dense_repeat,
        )
        chunk_start = chunk_starts[idx % len(chunk_starts)]
        chunk_sample_ids = sample_ids[chunk_start : chunk_start + args.chunk_size]
        release = release_for_record(record, releases)
        chunk_samples = [release.by_id[sid] for sid in chunk_sample_ids]
        x0, cond = load_latent_pair(record, device, dtype)
        if tuple(x0.shape) != (args.latent_channels, args.latent_frames, *latent_hw):
            raise RuntimeError(f"{record.get('clip_id')}: latent shape {tuple(x0.shape)} != {(args.latent_channels, args.latent_frames, *latent_hw)}")
        if cond.shape[1] != args.latent_frames or tuple(cond.shape[-2:]) != latent_hw:
            raise RuntimeError(f"{record.get('clip_id')}: condition shape {tuple(cond.shape)} does not match latent frames/hw")
        text_context = load_text_context(record, device)
        cam_chunk = prepare_cam_chunk(
            record,
            latent_frames=args.latent_frames,
            chunk_start=chunk_start,
            chunk_size=args.chunk_size,
            height=args.video_height,
            width=args.video_width,
            vae_stride=args.vae_stride,
            device=device,
            dtype=dtype,
        )
        dense_tokens, dense_hw = dense_tokens_for_samples(
            encoder,
            chunk_samples,
            device=device,
            dtype=dtype,
            target_token_hw=wan_token_hw,
        )
        state_tokens = state_tokens_for_record_chunk(
            state_projector,
            state_rows,
            record,
            chunk_start=chunk_start,
            chunk_size=args.chunk_size,
            device=device,
            dtype=dtype,
            target_token_hw=wan_token_hw,
        )
        interaction_result = interaction_tokens_for_record_chunk(
            interaction_projector, 
            state_rows, 
            record, 
            chunk_start=chunk_start, 
            chunk_size=args.chunk_size, 
            device=device, 
            dtype=dtype, 
            target_token_hw=wan_token_hw,
            return_features=True,
        )
        if interaction_result is None:
            interaction_tokens = None
            interaction_features = None
            interaction_audit = None
        else:
            interaction_tokens, interaction_features = interaction_result
            interaction_audit = interaction_chunk_audit(interaction_features)
        if state_tokens is not None:
            if state_tokens.shape != dense_tokens.shape:
                raise RuntimeError(f"{record.get('clip_id')}: state/dense token shape mismatch: state={tuple(state_tokens.shape)} dense={tuple(dense_tokens.shape)}")
            dense_tokens = dense_tokens + state_tokens
        if interaction_tokens is not None:
            if interaction_tokens.shape != dense_tokens.shape:
                raise RuntimeError(f"{record.get('clip_id')}: interaction/dense token shape mismatch: interaction={tuple(interaction_tokens.shape)} dense={tuple(dense_tokens.shape)}")
            dense_tokens = dense_tokens + interaction_tokens
        expected_tokens = args.chunk_size * wan_token_hw[0] * wan_token_hw[1]
        if dense_tokens.shape[1] != expected_tokens:
            raise RuntimeError(f"{record.get('clip_id')}: dense tokens {dense_tokens.shape[1]} != expected {expected_tokens}")
        if tuple(cam_chunk.shape[-3:]) != (args.chunk_size, *latent_hw):
            raise RuntimeError(f"{record.get('clip_id')}: camera chunk shape {tuple(cam_chunk.shape)} does not end with {(args.chunk_size, *latent_hw)}")
        checked_records.append(
            {
                "clip_id": record.get("clip_id"),
                "chunk_start": chunk_start,
                "sample_ids": chunk_sample_ids,
                "latent_shape": list(x0.shape),
                "condition_shape": list(cond.shape),
                "text_context_shape": list(text_context.shape),
                "camera_chunk_shape": list(cam_chunk.shape),
                "dense_token_shape": list(dense_tokens.shape),
                "dense_token_hw": list(dense_hw),
                "state_token_shape": list(state_tokens.shape) if state_tokens is not None else None,
                "interaction_token_shape": list(interaction_tokens.shape) if interaction_tokens is not None else None,
                "interaction_feature_shape": list(interaction_features.shape) if interaction_features is not None else None,
                "interaction_event_counts": interaction_audit["event_counts"] if interaction_audit is not None else None,
                "interaction_nonzero_feature_count": interaction_audit["nonzero_feature_count"] if interaction_audit is not None else None,
                "selection_roles": record.get("map_memory_selection_roles"),
            }
        )
        del x0, cond, text_context, cam_chunk, dense_tokens, state_tokens, interaction_tokens, interaction_features

    return {
        "kind": "memory_dense_adapter_train_preflight_v0",
        "status": "pass",
        "map_manifest": str(args.map_manifest),
        "cache_manifest": str(args.cache_manifest),
        "extra_map_manifest": [str(path) for path in (args.extra_map_manifest or [])],
        "extra_cache_manifest": [str(path) for path in (args.extra_cache_manifest or [])],
        "map_manifest_sha256": sha256_file(args.map_manifest),
        "release_count": len(releases),
        "train_split": args.train_split,
        "record_count": len(records),
        "checked_record_count": len(checked_records),
        "training_index_cache": release_gate.get("training_index_cache"),
        "release_gate": release_gate,
        "geometry_report": geometry_report,
        "latent_hw": list(latent_hw),
        "wan_token_grid_hw": list(wan_token_hw),
        "dense_resize_mode": args.dense_resize_mode,
        "dense_condition_encoder": dense_encoder_metadata(encoder),
        "state_channels_v0": {
            "enabled": state_projector is not None,
            "state_cache_manifest": str(args.state_cache_manifest) if args.state_cache_manifest else None,
            "channels": STATE_CHANNELS_V0 if state_projector is not None else [],
            "projector": state_projector.metadata() if state_projector is not None else None,
        },
        "interaction_channels_v1": {
            "enabled": interaction_projector is not None,
            "state_cache_manifest": str(args.state_cache_manifest) if args.state_cache_manifest else None,
            "channels": INTERACTION_CHANNELS_V1 if interaction_projector is not None else [],
            "projector": interaction_projector.metadata() if interaction_projector is not None else None,
            "protocol_validation": validate_interaction_v1_protocol(interaction_projector),
        },
        "checked_records": checked_records,
        "optimizer_steps_run": 0,
    }


def state_zero_preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.state_cache_manifest is None:
        raise ValueError("state-zero-preflight requires --state-cache-manifest")
    if args.resume_from is None:
        raise ValueError("state-zero-preflight requires --resume-from 70h adapter checkpoint")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    releases, records, latent_hw, wan_token_hw, geometry_report, release_gate = load_and_validate_training_inputs(args)
    state_rows = load_state_manifest_for_args(args)
    missing_state = [str(record.get("clip_id")) for record in records if str(record.get("clip_id")) not in state_rows]
    if missing_state:
        raise KeyError(f"--state-cache-manifest missing {len(missing_state)} training clips; first={missing_state[:5]}")
    config = build_adapter_config(args)
    encoder = build_dense_encoder(args, config, device=device, dtype=dtype)
    state_projector = build_state_projector(args, device=device, dtype=dtype)
    if state_projector is None:
        raise RuntimeError("internal error: state projector was not constructed")

    ckpt = torch.load(args.resume_from, map_location="cpu")
    ckpt_encoder_state = ckpt.get("state", {}).get("memory_dense_encoder", {})
    if ckpt_encoder_state:
        encoder.load_state_dict(ckpt_encoder_state)
    max_weight_abs = max((float(p.detach().abs().max().cpu()) for p in state_projector.parameters()), default=0.0)

    chunk_starts = build_chunk_starts(args.latent_frames, args.chunk_size)
    checked_records: list[dict[str, Any]] = []
    max_abs_diffs: list[float] = []
    mean_abs_diffs: list[float] = []
    for idx, record in enumerate(records[: max(1, args.preflight_records)]):
        sample_ids = resolve_record_sample_ids(
            record,
            latent_frames=args.latent_frames,
            allow_static_dense_repeat=args.allow_static_dense_repeat,
        )
        chunk_start = chunk_starts[idx % len(chunk_starts)]
        chunk_sample_ids = sample_ids[chunk_start : chunk_start + args.chunk_size]
        release = release_for_record(record, releases)
        chunk_samples = [release.by_id[sid] for sid in chunk_sample_ids]
        with torch.no_grad():
            dense_tokens, dense_hw = dense_tokens_for_samples(
                encoder,
                chunk_samples,
                device=device,
                dtype=dtype,
                target_token_hw=wan_token_hw,
            )
            state_tokens = state_tokens_for_record_chunk(
                state_projector,
                state_rows,
                record,
                chunk_start=chunk_start,
                chunk_size=args.chunk_size,
                device=device,
                dtype=dtype,
                target_token_hw=wan_token_hw,
            )
            combined = dense_tokens + state_tokens
            diff = (combined.float() - dense_tokens.float()).abs()
        max_abs = float(diff.max().detach().cpu())
        mean_abs = float(diff.mean().detach().cpu())
        max_abs_diffs.append(max_abs)
        mean_abs_diffs.append(mean_abs)
        checked_records.append(
            {
                "clip_id": record.get("clip_id"),
                "chunk_start": chunk_start,
                "dense_token_shape": list(dense_tokens.shape),
                "state_token_shape": list(state_tokens.shape),
                "dense_token_hw": list(dense_hw),
                "max_abs_token_diff": max_abs,
                "mean_abs_token_diff": mean_abs,
            }
        )
        del dense_tokens, state_tokens, combined, diff

    return {
        "kind": "state_channels_v0_zero_init_preflight",
        "status": "pass" if max(max_abs_diffs, default=0.0) == 0.0 and max_weight_abs == 0.0 else "fail",
        "resume_from": str(args.resume_from),
        "state_cache_manifest": str(args.state_cache_manifest),
        "checked_record_count": len(checked_records),
        "max_abs_token_diff": max(max_abs_diffs, default=0.0),
        "mean_abs_token_diff_mean": float(np.mean(mean_abs_diffs)) if mean_abs_diffs else None,
        "loss_difference_bound_pct": 0.0,
        "loss_note": "State projector is exactly zero-initialized, so DiT inputs are identical to the warm-start path before optimizer updates; full Wan forward loss delta is therefore 0 under deterministic evaluation.",
        "state_projector_max_abs_weight": max_weight_abs,
        "dense_condition_encoder": dense_encoder_metadata(encoder),
        "state_projector": state_projector.metadata(),
        "geometry_report": geometry_report,
        "release_gate": release_gate,
        "checked_records": checked_records,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.cache_manifest is None:
        raise ValueError("train mode requires --cache-manifest with rows aligned to Map Memory sample ids")
    rank, world, local, device = setup_dist()
    w4_quotas: dict[str, int] | None = None
    w5_enabled = args.w5_sampler_version is not None
    w5_extension_enabled = args.w5_sampler_version == W5_EXTENSION_SAMPLER_VERSION
    w6_enabled = args.w6_sampler_version is not None
    w7_enabled = args.w7_sampler_version is not None
    w5_loss_multipliers: dict[str, float] | None = None
    w6_loss_multipliers: dict[str, float] | None = None
    w7_loss_multipliers: dict[str, float] | None = None
    if not w5_enabled and (args.w5_loss_multiplier or args.w5_authorization_report or args.w5_authorization_sha256):
        raise ValueError("W5 loss/authorization flags require --w5-sampler-version")
    if not w6_enabled and (args.w6_loss_multiplier or args.w6_authorization_report or args.w6_authorization_sha256):
        raise ValueError("W6 loss/authorization flags require --w6-sampler-version")
    if not w7_enabled and (args.w7_loss_multiplier or args.w7_authorization_report or args.w7_authorization_sha256):
        raise ValueError("W7 loss/authorization flags require --w7-sampler-version")
    if sum((w5_enabled, w6_enabled, w7_enabled)) > 1:
        raise ValueError("W5, W6, and W7 identities are mutually exclusive")
    if w5_enabled:
        if args.w4_sampler_version is not None:
            raise ValueError("W5 and W4 sampler flags are mutually exclusive")
        if args.w5_sampler_version not in {W5_SAMPLER_VERSION, W5_EXTENSION_SAMPLER_VERSION}:
            raise ValueError(f"unsupported --w5-sampler-version {args.w5_sampler_version!r}")
        w5_loss_multipliers = parse_w5_loss_multipliers(args.w5_loss_multiplier)
        if w5_loss_multipliers != W5_LOSS_MULTIPLIERS:
            raise ValueError(f"W5 requires exact loss multipliers {W5_LOSS_MULTIPLIERS}")
        args.w4_sampler_version = SAMPLER_VERSION
    if w6_enabled:
        if args.w4_sampler_version is not None:
            raise ValueError("W6 and W4 sampler flags are mutually exclusive")
        if args.w6_sampler_version != W6_SAMPLER_VERSION:
            raise ValueError(f"unsupported --w6-sampler-version {args.w6_sampler_version!r}")
        w6_loss_multipliers = parse_w5_loss_multipliers(args.w6_loss_multiplier)
        if w6_loss_multipliers != W6_LOSS_MULTIPLIERS:
            raise ValueError(f"W6 requires exact loss multipliers {W6_LOSS_MULTIPLIERS}")
        args.w4_sampler_version = SAMPLER_VERSION
    if w7_enabled:
        if args.w4_sampler_version is not None:
            raise ValueError("W7 and W4 sampler flags are mutually exclusive")
        if args.w7_sampler_version != W7_SAMPLER_VERSION:
            raise ValueError(f"unsupported --w7-sampler-version {args.w7_sampler_version!r}")
        w7_loss_multipliers = parse_w5_loss_multipliers(args.w7_loss_multiplier)
        if w7_loss_multipliers != W7_LOSS_MULTIPLIERS:
            raise ValueError(f"W7 requires exact loss multipliers {W7_LOSS_MULTIPLIERS}")
        args.w4_sampler_version = SAMPLER_VERSION
    if args.w4_sampler_version is not None:
        if args.w4_sampler_version != SAMPLER_VERSION:
            raise ValueError(f"unsupported --w4-sampler-version {args.w4_sampler_version!r}")
        w4_quotas = parse_w4_quotas(args.w4_quota)
        validate_quotas(w4_quotas, world=world, grad_accum=args.grad_accum)
        expected_quotas = W7_QUOTAS if w7_enabled else (W6_QUOTAS if w6_enabled else DEFAULT_QUOTAS)
        if (world, args.grad_accum) != (6, 2) or w4_quotas != expected_quotas:
            raise ValueError(f"controlled sampler requires world=6 grad_accum=2 quotas={expected_quotas}")
        if w7_enabled:
            if args.w4_freeze_lora or not args.w4_freeze_state_projector:
                raise ValueError("W7 requires trainable LoRA and frozen state projector")
        elif not args.w4_freeze_lora or not args.w4_freeze_state_projector:
            raise ValueError("W4 V1 requires --w4-freeze-lora and --w4-freeze-state-projector")
        if args.w4_audit_report is None or args.w4_audit_sha256 is None:
            raise ValueError("W4 V1 requires --w4-audit-report and --w4-audit-sha256")
        actual_audit_sha = file_sha256(args.w4_audit_report)
        if actual_audit_sha != args.w4_audit_sha256:
            raise ValueError(f"W4 audit SHA mismatch: expected={args.w4_audit_sha256} actual={actual_audit_sha}")
        audit_payload = json.loads(args.w4_audit_report.read_text(encoding="utf-8"))
        if audit_payload.get("status") != "pass" or audit_payload.get("training_started") is not False:
            raise ValueError("W4 audit must have status=pass and training_started=false")
        required_w4 = {
            "use_base_model": (args.use_base_model, True),
            "reset_optimizer": (args.reset_optimizer, False if w5_extension_enabled else True),
            "lr": (args.lr, 1e-6 if w7_enabled else (5e-6 if w6_enabled else 1e-5)),
            "lora_lr": (args.lora_lr, 1e-6 if w7_enabled else (5e-6 if w6_enabled else 1e-5)),
            "region_loss_weight": (args.region_loss_weight, 10.0),
            "max_grad_norm": (args.max_grad_norm, 0.5 if w7_enabled else 1.0),
            "precision": (args.precision, "bf16"),
            "video_frames": (args.video_frames, 81),
            "latent_frames": (args.latent_frames, 21),
            "lora_rank": (args.lora_rank, 64),
            "negative_contrast_weight": (args.negative_contrast_weight, 0.0),
        }
        wrong = [f"{name}={actual!r}, expected {expected!r}" for name, (actual, expected) in required_w4.items() if actual != expected]
        if wrong:
            raise ValueError("W4 V1 trainer contract mismatch: " + "; ".join(wrong))
        if w5_enabled:
            allowed_steps = {102, 200} if w5_extension_enabled else {2, 100}
            if args.max_steps not in allowed_steps:
                raise ValueError(f"W5 identity permits exactly max_steps={sorted(allowed_steps)}")
            if args.w5_authorization_report is None or args.w5_authorization_sha256 is None:
                raise ValueError("W5 requires --w5-authorization-report and --w5-authorization-sha256")
            if file_sha256(args.w5_authorization_report) != args.w5_authorization_sha256:
                raise ValueError("W5 authorization SHA mismatch")
            authorization = json.loads(args.w5_authorization_report.read_text(encoding="utf-8"))
            if w5_extension_enabled:
                phrases = authorization.get("exact_user_phrases")
                if (authorization.get("scope") != "paired_w5_step100_to_step200_only" or phrases != [
                    "背景替代 我觉得无所谓 可能是训练不足的问题",
                    "可以 按照你的构思来 总之我想要更好的视觉demo",
                ]):
                    raise ValueError("W5 extension authorization scope or exact user phrases mismatch")
                if args.sampler_step_origin != "resume_relative" or args.sampler_step_offset != 100:
                    raise ValueError("W5 extension requires resume_relative sampler origin with offset 100")
                source = authorization.get("bindings", {}).get("step100_checkpoints", {}).get(args.base_expert, {})
                if str(args.resume_from) != source.get("path") or file_sha256(args.resume_from) != source.get("sha256"):
                    raise ValueError("W5 extension source path/SHA binding mismatch")
                expected_save = 102 if args.max_steps == 102 else 25
                if args.save_every != expected_save:
                    raise ValueError(f"W5 extension save interval must be {expected_save}")
            elif authorization.get("scope") != "paired_w5_smoke_then_100_only" or authorization.get("exact_user_phrase") != "拿你得判断一下训练方法 然后启动训练 我觉得要不就先训练100-step看看？你觉得怎么样合适":
                raise ValueError("W5 authorization scope or exact user phrase mismatch")
        if w6_enabled:
            if args.max_steps not in {2, 25}:
                raise ValueError("W6 identity permits exactly max_steps=[2, 25]")
            if args.w6_authorization_report is None or args.w6_authorization_sha256 is None:
                raise ValueError("W6 requires --w6-authorization-report and --w6-authorization-sha256")
            if file_sha256(args.w6_authorization_report) != args.w6_authorization_sha256:
                raise ValueError("W6 authorization SHA mismatch")
            authorization = json.loads(args.w6_authorization_report.read_text(encoding="utf-8"))
            if (authorization.get("scope") != "paired_w6_branch_a_smoke_then_25_only"
                    or authorization.get("exact_user_phrase") != "可以 继续"):
                raise ValueError("W6 authorization scope or exact user phrase mismatch")
            if args.sampler_step_origin != "resume_relative" or args.sampler_step_offset != 75:
                raise ValueError("W6 requires resume_relative sampler origin with offset 75")
            source = authorization.get("bindings", {}).get("step75_checkpoints", {}).get(args.base_expert, {})
            if str(args.resume_from) != source.get("path") or file_sha256(args.resume_from) != source.get("sha256"):
                raise ValueError("W6 source path/SHA binding mismatch")
            expected_save = 2 if args.max_steps == 2 else 25
            if args.save_every != expected_save:
                raise ValueError(f"W6 save interval must be {expected_save}")
        if w7_enabled:
            if args.max_steps not in {2, 25}:
                raise ValueError("W7 permits exactly max_steps=2 smoke or max_steps=25 formal")
            if args.w7_authorization_report is None or args.w7_authorization_sha256 is None:
                raise ValueError("W7 requires --w7-authorization-report and --w7-authorization-sha256")
            if file_sha256(args.w7_authorization_report) != args.w7_authorization_sha256:
                raise ValueError("W7 authorization SHA mismatch")
            authorization = json.loads(args.w7_authorization_report.read_text(encoding="utf-8"))
            if args.sampler_step_origin != "resume_relative" or args.sampler_step_offset != 75:
                raise ValueError("W7 requires resume_relative sampler origin with offset 75")
            source = authorization.get("bindings", {}).get("step75_checkpoints", {}).get(args.base_expert, {})
            if str(args.resume_from) != source.get("path") or file_sha256(args.resume_from) != source.get("sha256"):
                raise ValueError("W7 source path/SHA binding mismatch")
            if args.max_steps == 2:
                if (authorization.get("scope") != "paired_w7_real_2step_smoke_only"
                        or authorization.get("authorizes_formal_25") is not False):
                    raise ValueError("W7 smoke authorization must be smoke-only and deny formal 25-step")
                if args.save_every != 2:
                    raise ValueError("W7 smoke save interval must be 2")
            else:
                if (authorization.get("scope") != "paired_w7_formal_25_only"
                        or authorization.get("exact_user_phrase") != "对 卡别空着 继续训练"
                        or authorization.get("authorizes_formal_25") is not True
                        or authorization.get("authorizes_50_step") is not False):
                    raise ValueError("W7 formal authorization identity/scope mismatch")
                if args.save_every != 25:
                    raise ValueError("W7 formal save interval must be 25")
                bindings = authorization.get("bindings", {})
                required_bindings = {
                    "smoke_gate_report": "output/w7_negative_space_lora_only_20260720_v0/SMOKE_GATE_REPORT_v1.json",
                    "formal_launcher_contract": "configs/w7_negative_space_lora_only_formal25_launcher_contract_v1.json",
                    "w4_audit": "output/w4_controlled_correction_20260718_v0/SOURCE_AUDIT_v0.json",
                }
                for name, expected_path in required_bindings.items():
                    binding = bindings.get(name, {})
                    bound_path = Path(str(binding.get("path", "")))
                    if not bound_path.is_absolute():
                        bound_path = Path(__file__).resolve().parents[1] / bound_path
                    if (binding.get("path") != expected_path or not bound_path.is_file()
                            or file_sha256(bound_path) != binding.get("sha256")):
                        raise ValueError(f"W7 formal {name} path/SHA binding mismatch")
                smoke = json.loads((Path(__file__).resolve().parents[1] / required_bindings["smoke_gate_report"]).read_text(encoding="utf-8"))
                if smoke.get("kind") != "w7_lora_only_2step_smoke_gate_report_v1" or smoke.get("status") != "pass":
                    raise ValueError("W7 formal requires the bound passed paired smoke gate")
                contract_binding = bindings["formal_launcher_contract"]
                formal_contract = json.loads((Path(__file__).resolve().parents[1] / contract_binding["path"]).read_text(encoding="utf-8"))
                exact_formal = {
                    "kind": "w7_negative_space_lora_only_formal25_launcher_contract_v1",
                    "identity": W7_SAMPLER_VERSION, "sampler_version": W7_SAMPLER_VERSION,
                    "world": 6, "grad_accum": 2, "quotas": W7_QUOTAS,
                    "loss_multipliers": W7_LOSS_MULTIPLIERS, "allowed_steps": [25], "save_every": 25,
                    "authorizes_formal_25": True, "authorizes_50_step": False, "warm_start_step": 75,
                    "optimizer_reset": True, "sampler_step_origin": "resume_relative", "sampler_step_offset": 75,
                    "sampler_steps": [76, 100], "learning_rate": 1e-6, "region_loss_weight": 10.0,
                    "max_grad_norm": 0.5, "trainable_namespace": "memory_dense_lora",
                    "frozen_namespaces": ["memory_dense_adapter", "memory_dense_encoder", "memory_dense_state_projector"],
                    "base_wan_frozen": True, "precision": "bf16", "video_frames": 81, "records": 129342,
                    "authorization_scope": "paired_w7_formal_25_only",
                }
                if any(formal_contract.get(key) != value for key, value in exact_formal.items()):
                    raise ValueError("W7 formal launcher contract method drift")
                formal_source = formal_contract.get("checkpoints", {}).get(args.base_expert, {})
                if (formal_source.get("path") != source.get("path") or formal_source.get("sha256") != source.get("sha256")
                        or formal_source.get("step") != 75 or formal_source.get("lora_tensor_count") != 800):
                    raise ValueError("W7 formal launcher source binding drift")
                audit_binding = bindings["w4_audit"]
                if audit_binding != {"path": str(args.w4_audit_report), "sha256": args.w4_audit_sha256}:
                    raise ValueError("W7 formal W4 audit binding mismatch")
        if w5_extension_enabled:
            if args.resume_from is None or "000100" not in args.resume_from.name:
                raise ValueError("W5 extension requires an explicit paired step100 source")
        elif w6_enabled or w7_enabled:
            if args.resume_from is None or "000075" not in args.resume_from.name:
                raise ValueError("W6/W7 requires an explicit paired W5 step75 source")
        elif args.resume_from is None or "008500" not in args.resume_from.name or "008800" in args.resume_from.name:
            raise ValueError("W4 V1 requires an explicit step8500 warm start and refuses step8800")
        if args.w4_region_loss_cap_fraction < 0.0 or args.w4_gradient_telemetry_every < 1:
            raise ValueError("W4 region cap must be >=0 and gradient telemetry interval must be >=1")
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    try:
        torch.manual_seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        random.seed(args.seed + rank)
        releases, records, latent_hw, wan_token_hw, geometry_report, release_gate = load_and_validate_training_inputs(args)
        state_rows = load_state_manifest_for_args(args)
        if state_rows:
            missing_state = [str(record.get("clip_id")) for record in records if str(record.get("clip_id")) not in state_rows]
            if missing_state:
                raise KeyError(f"--state-cache-manifest missing {len(missing_state)} training clips; first={missing_state[:5]}")
        base_sigma_lo, base_sigma_hi = 0.0, 1.0
        if args.use_base_model:
            if args.negative_contrast_weight > 0.0:
                raise ValueError("--use-base-model does not support negative-contrast forwards yet; pass --negative-contrast-weight 0")
            WanModelBase, base_cfg = import_wan_model_base(args.lingbot_repo)
            expert_sub = base_cfg.low_noise_checkpoint if args.base_expert == "low" else base_cfg.high_noise_checkpoint
            model = WanModelBase.from_pretrained(
                str(args.ckpt_dir), subfolder=expert_sub, torch_dtype=dtype, control_type="cam"
            ).to(device)
            model.requires_grad_(False)
            boundary_sigma = float(base_cfg.boundary)
            if args.base_expert == "low":
                base_sigma_lo = 0.0 if args.base_sigma_min is None else float(args.base_sigma_min)
                base_sigma_hi = boundary_sigma if args.base_sigma_max is None else float(args.base_sigma_max)
            else:
                base_sigma_lo = boundary_sigma if args.base_sigma_min is None else float(args.base_sigma_min)
                base_sigma_hi = 1.0 if args.base_sigma_max is None else float(args.base_sigma_max)
            args.chunk_size = args.latent_frames
            if rank == 0:
                print(json.dumps({"event": "base_model_loaded", "expert": args.base_expert,
                                  "subfolder": expert_sub, "sigma_band": [base_sigma_lo, base_sigma_hi],
                                  "full_seq_chunk_size": args.chunk_size}, ensure_ascii=False), flush=True)
        else:
            WanModelFast = import_wan_model_fast(args.lingbot_repo)
            model_dir = args.fast_model_dir or args.ckpt_dir / "lingbot_world_fast"
            from_pretrained_kwargs: dict[str, Any] = {"torch_dtype": dtype, "control_type": "cam"}
            if args.wan_low_cpu_mem_usage != "auto":
                from_pretrained_kwargs["low_cpu_mem_usage"] = args.wan_low_cpu_mem_usage == "true"
            model = WanModelFast.from_pretrained(str(model_dir), **from_pretrained_kwargs).to(device)
            model.requires_grad_(False)
        config = build_adapter_config(args)
        wrap_wan_model_fast_with_memory_dense_adapter(model, config, freeze_base=True)
        lora_report = inject_lora_into_wan_model_fast(
            model,
            rank=args.lora_rank,
            targets=args.lora_targets,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
        model.to(device=device, dtype=dtype)
        encoder = build_dense_encoder(args, config, device=device, dtype=dtype)
        state_projector = build_state_projector(args, device=device, dtype=dtype)
        interaction_projector = build_interaction_projector(args, device=device, dtype=dtype)
        if args.w4_freeze_lora:
            for _, param in lora_parameters(model):
                param.requires_grad_(False)
        if args.w4_freeze_state_projector and state_projector is not None:
            state_projector.requires_grad_(False)
        if w7_enabled:
            for _, param in memory_dense_adapter_parameters(model):
                param.requires_grad_(False)
            encoder.requires_grad_(False)
        trainable_audit = assert_only_memory_dense_trainable(
            model,
            encoder,
            state_projector,
            interaction_projector=interaction_projector,
            freeze_lora=args.w4_freeze_lora,
            freeze_state_projector=args.w4_freeze_state_projector,
            lora_only=w7_enabled,
        )
        train_encoder: torch.nn.Module = encoder
        train_state_projector: torch.nn.Module | None = state_projector
        train_interaction_projector: torch.nn.Module | None = (
            interaction_projector
        )
        if world > 1 and not w7_enabled:
            train_encoder = torch.nn.parallel.DistributedDataParallel(
                encoder,
                **ddp_kwargs(local_rank=local, w4_enabled=args.w4_sampler_version is not None),
            )
            if state_projector is not None and not args.w4_freeze_state_projector:
                train_state_projector = torch.nn.parallel.DistributedDataParallel(
                    state_projector,
                    device_ids=[local],
                    output_device=local,
                    find_unused_parameters=False,
                )
            if interaction_projector is not None:
                train_interaction_projector = (
                    torch.nn.parallel.DistributedDataParallel(
                        interaction_projector,
                        device_ids=[local],
                        output_device=local,
                        find_unused_parameters=False,
                    )
                )
        
        adapter_all_params = memory_dense_adapter_parameters(model)
        lora_all_params = lora_parameters(model)
        encoder_all_params = list(encoder.named_parameters())
        state_all_params = list(state_projector.named_parameters()) if state_projector is not None else []
        interaction_all_params = (
            list(interaction_projector.named_parameters())
            if interaction_projector is not None
            else []
        )
        interaction_named_params = [
            (name, param)
            for name, param in interaction_all_params
            if param.requires_grad
        ]
        adapter_named_params = [(n, p) for n, p in adapter_all_params if p.requires_grad]
        lora_named_params = [(n, p) for n, p in lora_all_params if p.requires_grad]
        encoder_named_params = [(n, p) for n, p in encoder_all_params if p.requires_grad]
        state_named_params = [(n, p) for n, p in state_all_params if p.requires_grad]
        if args.w4_sampler_version is not None:
            if w7_enabled:
                if adapter_named_params or encoder_named_params or state_named_params or not lora_named_params:
                    raise RuntimeError("W7 exact trainable groups violated: only the nonempty LoRA namespace may be trainable")
                if interaction_named_params:
                    raise RuntimeError(
                        "interaction V1 cannot use W7 LoRA-only mode: "
                        "the interaction projector must remain trainable"
                    )
            elif lora_named_params or state_named_params or not adapter_named_params or not encoder_named_params or not interaction_named_params:
                raise RuntimeError(
                    "W4 exact trainable groups violated: adapter and dense encoder projection, and interaction projector must be trainable; "
                    "LoRA and state projector must be frozen"
                )
        params = (
            [p for _, p in adapter_named_params]
            + [p for _, p in lora_named_params]
            + [p for _, p in encoder_named_params]
            + [p for _, p in state_named_params]
            + [p for _, p in interaction_named_params]
        )
        if not params:
            raise RuntimeError("no trainable adapter, LoRA, encoder, state, or interaction parameters")
        trainable_names = (
            [name for name, _ in adapter_named_params]
            + [(f"memory_dense_lora.{name}" if w7_enabled else name) for name, _ in lora_named_params]
            + [f"memory_dense_encoder.{n}" for n, _ in encoder_named_params]
            + [f"memory_dense_state_projector.{n}" for n, _ in state_named_params]
            + [f"memory_dense_interaction_projector.{n}" for n, _ in interaction_named_params]
        )
        optimizer_groups: list[dict[str, Any]] = []
        non_lora_params = [p for _, p in adapter_named_params] + [p for _, p in encoder_named_params] + [p for _, p in state_named_params] + [p for _, p in interaction_named_params]
        if non_lora_params:
            optimizer_groups.append({"params": non_lora_params, "lr": args.lr, "weight_decay": args.weight_decay})
        if lora_named_params:
            optimizer_groups.append({"params": [p for _, p in lora_named_params], "lr": args.lora_lr, "weight_decay": args.weight_decay})
        fp32_master_params: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
        if args.fp32_master:
            master_groups: list[dict[str, Any]] = []
            for _group in optimizer_groups:
                _masters = []
                for _p in _group["params"]:
                    _m = _p.detach().clone().float()
                    _m.requires_grad_(True)
                    fp32_master_params.append((_p, _m))
                    _masters.append(_m)
                master_groups.append({**_group, "params": _masters})
            optimizer = torch.optim.AdamW(master_groups)
        else:
            optimizer = torch.optim.AdamW(optimizer_groups)
        resume_checkpoint_step = 0
        start_step = 0
        if args.resume_from:
            resume_checkpoint_step = load_checkpoint(
                args.resume_from,
                model=model,
                encoder=encoder,
                state_projector=state_projector,
                interaction_projector=interaction_projector,
                optimizer=optimizer,
                reset_optimizer=args.reset_optimizer,
            )
            start_step = 0 if args.reset_optimizer else resume_checkpoint_step
        if fp32_master_params:
            with torch.no_grad():
                for _p, _m in fp32_master_params:
                    _m.data.copy_(_p.data.float())
        expected_resume_step = 100 if w5_extension_enabled else 8500
        if w6_enabled or w7_enabled:
            expected_resume_step = 75
        if args.w4_sampler_version is not None and resume_checkpoint_step != expected_resume_step:
            raise ValueError(f"W4/W5 checkpoint payload step={resume_checkpoint_step}, expected {expected_resume_step}")
        if args.sampler_step_offset < 0:
            raise ValueError("--sampler-step-offset must be non-negative")
        if args.sampler_step_origin == "resume_relative" and args.resume_from is None:
            raise ValueError("--sampler-step-origin resume_relative requires --resume-from")
        if args.sampler_step_origin == "absolute" and args.sampler_step_offset != 0:
            raise ValueError("--sampler-step-offset is only valid with --sampler-step-origin resume_relative")
        planned_steps = args.max_steps - start_step
        if planned_steps <= 0:
            raise RuntimeError(
                f"no optimizer steps would run: start_step={start_step} max_steps={args.max_steps} "
                f"resume_checkpoint_step={resume_checkpoint_step} reset_optimizer={args.reset_optimizer}"
            )
        train_model: torch.nn.Module = model
        if world > 1:
            train_model = torch.nn.parallel.DistributedDataParallel(
                model,
                **ddp_kwargs(local_rank=local, w4_enabled=args.w4_sampler_version is not None),
            )

        # [OOM诊断+修复] WanBlockMemoryDenseWrapper.forward 中 activation checkpoint 依赖 self.training=True;
        # from_pretrained 常返回 eval 模式导致其静默失效->全序列激活累积 OOM。显式置 train 并打点。
        if rank == 0:
            _trainable_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
            print(json.dumps({"event": "oom_diag_before_train", "model_training": bool(model.training),
                              "trainable_params_M": round(_trainable_m, 3),
                              "mem_alloc_gb": round(torch.cuda.memory_allocated() / 1e9, 2)}), flush=True)
        model.train()
        train_model.train()
        if rank == 0:
            print(json.dumps({"event": "oom_diag_after_train", "model_training": bool(model.training),
                              "mem_alloc_gb": round(torch.cuda.memory_allocated() / 1e9, 2)}), flush=True)

        chunk_starts = build_chunk_starts(args.latent_frames, args.chunk_size)
        positive_chunks = build_positive_chunk_index(records, releases, args=args, chunk_starts=chunk_starts)
        w4_sampler = (
            W4SamplerV1(
                records,
                strict_zero_release_indices={0},
                seed=args.w4_sampler_seed,
                world=world,
                grad_accum=args.grad_accum,
                quotas=w4_quotas,
            )
            if args.w4_sampler_version is not None
            else None
        )
        if len(records) < args.min_runtime_train_records:
            raise RuntimeError(
                f"runtime train records {len(records)} < min_runtime_train_records {args.min_runtime_train_records}; "
                "refusing to silently train on an empty or tiny matched set"
            )
        if not positive_chunks:
            raise RuntimeError("no positive chunks matched training records; refusing state-channel smoke with no positive supervision")
        losses: list[float] = []
        args.out_dir.mkdir(parents=True, exist_ok=True)
        release_inputs = [
            {
                "map_manifest": str(release.manifest_path),
                "cache_manifest": str(cache_manifest),
                "map_manifest_sha256": sha256_file(release.manifest_path),
            }
            for release, cache_manifest in zip(releases, [args.cache_manifest, *(args.extra_cache_manifest or [])])
        ]
        run_config = {
            "map_manifest": str(args.map_manifest),
            "cache_manifest": str(args.cache_manifest),
            "extra_map_manifest": [str(path) for path in (args.extra_map_manifest or [])],
            "extra_cache_manifest": [str(path) for path in (args.extra_cache_manifest or [])],
            "map_manifest_sha256": sha256_file(args.map_manifest),
            "release_inputs": release_inputs,
            "release_count": len(releases),
            "adapter_config": config.to_dict(),
            "lora_config": {
                "rank": int(args.lora_rank),
                "targets": args.lora_targets,
                "alpha": args.lora_alpha,
                "dropout": float(args.lora_dropout),
                "lr": float(args.lora_lr),
            },
            "lora_report": lora_report,
            "activation_checkpoint_blocks": bool(args.activation_checkpoint_blocks),
            "activation_checkpoint_adapter": bool(args.activation_checkpoint_adapter),
            "wan_low_cpu_mem_usage": args.wan_low_cpu_mem_usage,
            "latent_frames": args.latent_frames,
            "chunk_size": args.chunk_size,
            # v2 dense contract (416x240 -> 30x52 native, no interpolation)
            "dense_native_hw": list(dense_hw_from_args(args)),
            "video_hw": [args.video_height, args.video_width],
            "latent_hw": list(latent_hw),
            "wan_token_grid_hw": list(wan_token_hw),
            "geometry_report": geometry_report,
            "tokens_per_frame": wan_token_hw[0] * wan_token_hw[1],
            "dense_resize_mode": args.dense_resize_mode,
            "dense_condition_encoder": dense_encoder_metadata(encoder),
            "state_channels_v0": {
                "enabled": state_projector is not None,
                "state_cache_manifest": str(args.state_cache_manifest) if args.state_cache_manifest else None,
                "channels": STATE_CHANNELS_V0 if state_projector is not None else [],
                "projector": state_projector.metadata() if state_projector is not None else None,
            },
            "interaction_channels_v1": {
                "enabled": interaction_projector is not None,
                "state_cache_manifest": str(args.state_cache_manifest) if args.state_cache_manifest else None,
                "channels": INTERACTION_CHANNELS_V1 if interaction_projector is not None else [],
                "projector": interaction_projector.metadata() if interaction_projector is not None else None,
                "protocol_validation": validate_interaction_v1_protocol(interaction_projector),
            },
            "world_size": world,
            "train_split": args.train_split,
            "min_train_records": int(args.min_train_records),
            "min_train_positive_frames": int(args.min_train_positive_frames),
            "min_runtime_train_records": int(args.min_runtime_train_records),
            "release_gate": release_gate,
            "trainable_audit": trainable_audit,
            "resume_from": str(args.resume_from) if args.resume_from else None,
            "resume_checkpoint_step": resume_checkpoint_step,
            "reset_optimizer": bool(args.reset_optimizer),
            "start_step": start_step,
            "planned_optimizer_steps": planned_steps,
            "sampler_step_origin": args.sampler_step_origin,
            "sampler_step_offset": args.sampler_step_offset,
            "sampler_first_step": (
                args.sampler_step_offset + 1 if args.sampler_step_origin == "resume_relative" else start_step + 1
            ),
            "trainable_names": trainable_names,
            "optimizer_parameter_names": trainable_names,
            "positive_chunk_count": len(positive_chunks),
            "positive_batch_fraction": args.positive_batch_fraction,
            "negative_contrast_weight": args.negative_contrast_weight,
            "negative_contrast_margin": args.negative_contrast_margin,
            "negative_contrast_modes": args.negative_contrast_modes,
            "region_loss_weight": args.region_loss_weight,
            "region_loss_min_pixels": args.region_loss_min_pixels,
            "chunk0_region_loss_weight": args.chunk0_region_loss_weight,
            "sigma_curriculum_prob": args.sigma_curriculum_prob,
            "sigma_curriculum_radius": args.sigma_curriculum_radius,
            "sigma_curriculum_centers": list(INFERENCE_SIGMA_BAND_CENTERS),
            "min_snr_gamma": args.min_snr_gamma,
            "min_snr_eps": args.min_snr_eps,
            "min_snr_weight_form": "flow_v_min_snr_over_snr_plus_one",
            "only_player_dense_channels": bool(args.only_player_dense_channels),
            "region_mask_note": "dense-space teacher player mask, area-pooled to latent grid; ~5% aspect offset, relative-only",
            "w4_sampler": w4_sampler.metadata() if w4_sampler is not None else None,
            "w4_freeze_lora": bool(args.w4_freeze_lora),
            "w4_freeze_state_projector": bool(args.w4_freeze_state_projector),
            "w4_audit_report": str(args.w4_audit_report) if args.w4_audit_report else None,
            "w4_audit_sha256": args.w4_audit_sha256,
            "w4_region_loss_cap_fraction": args.w4_region_loss_cap_fraction,
            "w4_gradient_telemetry_every": args.w4_gradient_telemetry_every,
            "w4_ddp_gradient_as_bucket_view": args.w4_sampler_version is not None,
            "w5_sampler_version": args.w5_sampler_version if w5_enabled else None,
            "w5_extension_identity": W5_EXTENSION_SAMPLER_VERSION if w5_extension_enabled else None,
            "w5_extension_config": ({
                "source_step": 100, "target_step": args.max_steps,
                "optimizer_restored": True, "sampler_steps": [101, args.max_steps],
            } if w5_extension_enabled else None),
            "w5_loss_multipliers": w5_loss_multipliers,
            "w5_authorization_report": str(args.w5_authorization_report) if args.w5_authorization_report else None,
            "w5_authorization_sha256": args.w5_authorization_sha256,
            "w6_sampler_version": args.w6_sampler_version if w6_enabled else None,
            "w6_branch_config": ({
                "source_step": 75, "target_step": args.max_steps,
                "optimizer_reset": True, "sampler_steps": [76, 75 + args.max_steps],
            } if w6_enabled else None),
            "w6_loss_multipliers": w6_loss_multipliers,
            "w6_authorization_report": str(args.w6_authorization_report) if args.w6_authorization_report else None,
            "w6_authorization_sha256": args.w6_authorization_sha256,
            "w7_sampler_version": args.w7_sampler_version if w7_enabled else None,
            "w7_branch_config": ({
                "source_step": 75, "target_step": args.max_steps, "optimizer_reset": True,
                "sampler_steps": [76, 75 + args.max_steps],
                "authorization_scope": authorization.get("scope"),
                "formal_25": args.max_steps == 25,
                "authorizes_formal_25": False if args.max_steps == 2 else True,
                "trainable_namespace": "memory_dense_lora",
                "trainable_namespaces": ["memory_dense_lora"],
                "frozen_namespaces": ["memory_dense_adapter", "memory_dense_encoder", "memory_dense_state_projector", "base_wan"],
            } if w7_enabled else None),
            "w7_loss_multipliers": w7_loss_multipliers,
            "w7_authorization_report": str(args.w7_authorization_report) if args.w7_authorization_report else None,
            "w7_authorization_sha256": args.w7_authorization_sha256,
        }
        if rank == 0:
            (args.out_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"event": "train_start", "records": len(records), "positive_chunk_count": len(positive_chunks), "world": world, "out_dir": str(args.out_dir), "resume_checkpoint_step": resume_checkpoint_step, "start_step": start_step, "planned_optimizer_steps": planned_steps}, ensure_ascii=False), flush=True)

        condition_probe_batch = None
        if args.condition_probe_every > 0:
            if args.condition_probe_batch is None:
                raise ValueError("--condition-probe-every requires --condition-probe-batch")
            condition_probe_batch = load_condition_probe_batch(args.condition_probe_batch)
            if rank == 0:
                print(json.dumps({
                    "event": "condition_probe_batch_loaded",
                    "path": str(args.condition_probe_batch),
                    "n_windows": len(condition_probe_batch["windows"]),
                    "split": condition_probe_batch.get("split"),
                    "probe_every": args.condition_probe_every,
                }, ensure_ascii=False), flush=True)

        text_context_cache: dict[str, torch.Tensor] = {}
        w4_cumulative_records = {name: set() for name in STRATA}
        w4_cumulative_matches = {name: set() for name in STRATA}
        w4_trainable_gradient_bytes = sum(param.numel() * param.element_size() for param in params)
        if w4_sampler is not None:
            # Initialization and checkpoint loading can leave unused cached blocks. Release
            # those blocks before first-backward DDP bucket allocation.
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            print(json.dumps(cuda_memory_telemetry(
                device=device,
                phase="pre_loop_after_empty_cache",
                rank=rank,
                trainable_gradient_bytes=w4_trainable_gradient_bytes,
            )), flush=True)
        for step in range(start_step + 1, args.max_steps + 1):
            sampler_step = (
                step - start_step + args.sampler_step_offset
                if args.sampler_step_origin == "resume_relative"
                else step
            )
            optimizer.zero_grad(set_to_none=True)
            if fp32_master_params:
                for _p, _m in fp32_master_params:
                    _p.grad = None
            micro_losses: list[float] = []
            step_role_counts = {"positive": 0, "context": 0}
            step_interaction_event_counts = {
                feature_name.removeprefix("ego_").removesuffix("_event"): 0
                for feature_name in INTERACTION_CHANNELS_V1[:4]
            }
            step_interaction_nonzero_feature_count = 0
            step_region_loss_by_sigma: dict[str, Any] = {}
            step_region_loss_by_chunk: dict[str, Any] = {}
            step_state_debug: dict[str, Any] | None = None
            step_w4_telemetry: list[dict[str, Any]] = []
            for micro_step in range(args.grad_accum):
                w4_selection = None
                if w4_sampler is not None:
                    w4_selection = w4_sampler.select(sampler_step=sampler_step, micro_step=micro_step, rank=rank)
                    record_index = w4_selection.record_index
                    record = records[record_index]
                    chunk_start = chunk_starts[(sampler_step + w4_selection.stratum_ordinal) % len(chunk_starts)]
                    positive_sampled = w4_selection.stratum in {"boundary", "person_positive"}
                else:
                    record, record_index, chunk_start, positive_sampled = pick_training_record_chunk(
                        records,
                        positive_chunks,
                        chunk_starts,
                        step=sampler_step,
                        micro_step=micro_step,
                        rank=rank,
                        world=world,
                        grad_accum=args.grad_accum,
                        positive_batch_fraction=args.positive_batch_fraction,
                        seed=args.seed,
                        shuffle_records=args.shuffle_records,
                    )
                sample_ids = resolve_record_sample_ids(
                    record,
                    latent_frames=args.latent_frames,
                    allow_static_dense_repeat=args.allow_static_dense_repeat,
                )
                chunk_sample_ids = sample_ids[chunk_start : chunk_start + args.chunk_size]
                release = release_for_record(record, releases)
                chunk_samples = [release.by_id[sid] for sid in chunk_sample_ids]
                if positive_sampled:
                    step_role_counts["positive_sampled_chunks"] = step_role_counts.get("positive_sampled_chunks", 0) + 1
                for sample in chunk_samples:
                    role = sample.selection_role
                    if role in step_role_counts:
                        step_role_counts[role] += 1
                    else:
                        step_role_counts[role] = step_role_counts.get(role, 0) + 1
                if os.environ.get("SOLARIS_MEM_DEBUG") == "1" and device.type == "cuda":
                    print(json.dumps({
                        "event": "mem_debug_micro_start", "step": step, "micro": micro_step,
                        "clip_id": str(record.get("clip_id"))[:80],
                        "positive_sampled": bool(positive_sampled),
                        "chunk_start": int(chunk_start),
                        "alloc_gb": round(torch.cuda.memory_allocated(device) / 2**30, 2),
                    }), flush=True)
                sync_context = train_model.no_sync() if world > 1 and micro_step < args.grad_accum - 1 else nullcontext()
                with sync_context:
                    x0, cond = load_latent_pair(record, device, dtype)
                    if x0.shape[1] != args.latent_frames or cond.shape[1] != args.latent_frames:
                        raise RuntimeError(f"{record.get('clip_id')}: latent frame count mismatch")
                    x0_chunk = x0[:, chunk_start : chunk_start + args.chunk_size]
                    cond_chunk = cond[:, chunk_start : chunk_start + args.chunk_size]
                    cam_chunk = prepare_cam_chunk(
                        record,
                        latent_frames=args.latent_frames,
                        chunk_start=chunk_start,
                        chunk_size=args.chunk_size,
                        height=args.video_height,
                        width=args.video_width,
                        vae_stride=args.vae_stride,
                        device=device,
                        dtype=dtype,
                    )
                    dense_tokens, dense_hw = dense_tokens_for_samples(
                        train_encoder,
                        chunk_samples,
                        device=device,
                        dtype=dtype,
                        target_token_hw=wan_token_hw,
                    )
                    state_tokens = state_tokens_for_record_chunk(
                        train_state_projector,
                        state_rows,
                        record,
                        chunk_start=chunk_start,
                        chunk_size=args.chunk_size,
                        device=device,
                        dtype=dtype,
                        target_token_hw=wan_token_hw,
                    )

                    interaction_result = interaction_tokens_for_record_chunk(
                        train_interaction_projector,
                        state_rows,
                        record,
                        chunk_start=chunk_start,
                        chunk_size=args.chunk_size,
                        device=device,
                        dtype=dtype,
                        target_token_hw=wan_token_hw,
                        return_features=True,
                    )
                    if interaction_result is None:
                        interaction_tokens = None
                        interaction_features = None
                        chunk_interaction_audit = None
                    else:
                        interaction_tokens, interaction_features = interaction_result
                        chunk_interaction_audit = interaction_chunk_audit(
                            interaction_features
                        )
                        for event_name, count in chunk_interaction_audit[
                            "event_counts"
                        ].items():
                            step_interaction_event_counts[event_name] += int(count)
                        step_interaction_nonzero_feature_count += int(
                            chunk_interaction_audit["nonzero_feature_count"]
                        )

                    if step_state_debug is None:
                        step_state_debug = {
                            "clip_id": str(record.get("clip_id")),
                            "chunk_start": int(chunk_start),
                            "chunk_roles": [
                                sample.selection_role
                                for sample in chunk_samples
                            ],
                            "state_cache_hit": state_tokens is not None,
                            "interaction_cache_hit": interaction_tokens is not None,
                            "state_token_shape": (
                                list(state_tokens.shape)
                                if state_tokens is not None
                                else None
                            ),
                            "interaction_token_shape": (
                                list(interaction_tokens.shape)
                                if interaction_tokens is not None
                                else None
                            ),
                            "interaction_feature_shape": (
                                list(interaction_features.shape)
                                if interaction_features is not None
                                else None
                            ),
                            "interaction_event_counts": (
                                chunk_interaction_audit["event_counts"]
                                if chunk_interaction_audit is not None
                                else None
                            ),
                            "interaction_nonzero_feature_count": (
                                chunk_interaction_audit["nonzero_feature_count"]
                                if chunk_interaction_audit is not None
                                else None
                            ),
                            "dense_token_shape_before_condition": list(
                                dense_tokens.shape
                            ),
                        }

                    if state_tokens is not None:
                        if state_tokens.shape != dense_tokens.shape:
                            raise RuntimeError(
                                f"{record.get('clip_id')}: "
                                "state/dense token shape mismatch: "
                                f"state={tuple(state_tokens.shape)} "
                                f"dense={tuple(dense_tokens.shape)}"
                            )

                        dense_tokens = dense_tokens + state_tokens

                    if interaction_tokens is not None:
                        if interaction_tokens.shape != dense_tokens.shape:
                            raise RuntimeError(
                                f"{record.get('clip_id')}: "
                                "interaction/dense token shape mismatch: "
                                f"interaction={tuple(interaction_tokens.shape)} "
                                f"dense={tuple(dense_tokens.shape)}"
                            )

                        dense_tokens = dense_tokens + interaction_tokens

                    latent_h, latent_w = x0_chunk.shape[-2:]
                    wan_h = latent_h // args.patch_size_hw
                    wan_w = latent_w // args.patch_size_hw
                    if (latent_h, latent_w) != latent_hw or (wan_h, wan_w) != wan_token_hw:
                        raise RuntimeError(
                            f"{record.get('clip_id')}: runtime grid latent={(latent_h, latent_w)} Wan={(wan_h, wan_w)} "
                            f"does not match validated latent={latent_hw} Wan={wan_token_hw}"
                        )
                    if dense_tokens.shape[1] != args.chunk_size * wan_h * wan_w:
                        raise RuntimeError(
                            f"{chunk_sample_ids[0]}: dense tokens {dense_tokens.shape[1]} do not match Wan seq {args.chunk_size * wan_h * wan_w}"
                        )
                    text_key = str(record["text_cache"])
                    if text_key not in text_context_cache:
                        text_context_cache[text_key] = load_text_context(record, device)
                    text_context = text_context_cache[text_key]
                    noise = torch.randn_like(x0_chunk, dtype=torch.float32).to(dtype)
                    if args.use_base_model:
                        sigma = torch.empty((), device=device, dtype=torch.float32).uniform_(base_sigma_lo, base_sigma_hi)
                    else:
                        sigma = sample_training_sigma(args, device=device)
                    sigma_value = float(sigma.detach().cpu())
                    timestep = (sigma * 1000.0).reshape(1)
                    xt = ((1.0 - sigma) * x0_chunk.float() + sigma * noise.float()).to(dtype)
                    target = noise.float() - x0_chunk.float()
                    min_snr_weight = min_snr_flow_loss_weight(args, sigma)
                    seq_len = args.chunk_size * wan_h * wan_w
                    current_start = chunk_start * wan_h * wan_w
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda" and dtype == torch.bfloat16):
                        main_dit_cond = {
                            "c2ws_plucker_emb": cam_chunk.chunk(1, dim=0),
                            COND_KEY: dense_tokens,
                        }
                        if args.use_base_model:
                            pred = train_model(
                                x=[xt],
                                t=timestep,
                                context=[text_context],
                                seq_len=seq_len,
                                y=[cond_chunk],
                                dit_cond_dict=main_dit_cond,
                            )[0]
                        else:
                            pred = train_model(
                                x=[xt],
                                t=timestep,
                                context=[text_context],
                                seq_len=seq_len,
                                y=[cond_chunk],
                                dit_cond_dict=main_dit_cond,
                                kv_cache=None,
                                crossattn_cache=None,
                                current_start=current_start,
                                max_attention_size=seq_len,
                            )[0]
                        true_loss = F.mse_loss(pred.float(), target)
                        loss = true_loss
                        region_loss = None
                        weighted_region_loss_raw = None
                        weighted_region_loss_applied = None
                        region_pixels = 0
                        if args.region_loss_weight > 0.0:
                            region_loss, region_pixels = region_loss_for_samples(
                                pred,
                                target,
                                chunk_samples,
                                latent_hw=latent_hw,
                                min_pixels=args.region_loss_min_pixels,
                                normalize=args.region_loss_normalize,
                            )
                            if region_loss is not None:
                                chunk_region_loss_weight = (
                                    float(args.chunk0_region_loss_weight) if chunk_start == 0 else 1.0
                                )
                                weighted_region_loss_raw = args.region_loss_weight * chunk_region_loss_weight * region_loss
                                weighted_region_loss_applied = weighted_region_loss_raw
                                if args.w4_region_loss_cap_fraction > 0.0:
                                    cap = true_loss.detach() * args.w4_region_loss_cap_fraction
                                    scale = torch.clamp(cap / weighted_region_loss_raw.detach().clamp_min(1e-12), max=1.0)
                                    weighted_region_loss_applied = weighted_region_loss_raw * scale
                                loss = loss + weighted_region_loss_applied
                                add_region_loss_stat(
                                    step_region_loss_by_sigma,
                                    sigma_band_label(sigma_value),
                                    region_loss=region_loss,
                                    region_pixels=region_pixels,
                                    loss_weight=chunk_region_loss_weight,
                                )
                                add_region_loss_stat(
                                    step_region_loss_by_chunk,
                                    f"chunk_start_{chunk_start}",
                                    region_loss=region_loss,
                                    region_pixels=region_pixels,
                                    loss_weight=chunk_region_loss_weight,
                                )
                        negative_losses: list[torch.Tensor] = []
                        apply_negative_contrast = args.negative_contrast_weight > 0.0 and (
                            not args.negative_contrast_positive_only
                            or any(sample.selection_role == "positive" for sample in chunk_samples)
                        )
                        if apply_negative_contrast:
                            if "blank" in args.negative_contrast_modes:
                                blank_tokens = torch.zeros_like(dense_tokens)
                                if state_tokens is not None:
                                    blank_tokens = blank_tokens + state_tokens
                                if interaction_tokens is not None:
                                    blank_tokens = blank_tokens + interaction_tokens

                                blank_pred = train_model(
                                    x=[xt],
                                    t=timestep,
                                    context=[text_context],
                                    seq_len=seq_len,
                                    y=[cond_chunk],
                                    dit_cond_dict={
                                        "c2ws_plucker_emb": cam_chunk.chunk(1, dim=0),
                                        COND_KEY: blank_tokens,
                                    },
                                    kv_cache=None,
                                    crossattn_cache=None,
                                    current_start=current_start,
                                    max_attention_size=seq_len,
                                )[0]
                                negative_losses.append(F.mse_loss(blank_pred.float(), target))
                                del blank_tokens, blank_pred
                            if "shuffled" in args.negative_contrast_modes:
                                shuffled_samples = shuffled_chunk_samples(
                                    records,
                                    releases,
                                    args=args,
                                    record_index=record_index,
                                    chunk_start=chunk_start,
                                    offset=args.negative_shuffle_offset,
                                )
                                shuffled_tokens, _ = dense_tokens_for_samples(
                                    train_encoder,
                                    shuffled_samples,
                                    device=device,
                                    dtype=dtype,
                                    target_token_hw=wan_token_hw,
                                )
                                if state_tokens is not None:
                                    shuffled_tokens = shuffled_tokens + state_tokens
                                if interaction_tokens is not None:
                                    shuffled_tokens = shuffled_tokens + interaction_tokens
                                shuffled_pred = train_model(
                                    x=[xt],
                                    t=timestep,
                                    context=[text_context],
                                    seq_len=seq_len,
                                    y=[cond_chunk],
                                    dit_cond_dict={
                                        "c2ws_plucker_emb": cam_chunk.chunk(1, dim=0),
                                        COND_KEY: shuffled_tokens,
                                    },
                                    kv_cache=None,
                                    crossattn_cache=None,
                                    current_start=current_start,
                                    max_attention_size=seq_len,
                                )[0]
                                negative_losses.append(F.mse_loss(shuffled_pred.float(), target))
                                del shuffled_tokens, shuffled_pred
                            if negative_losses:
                                ranking = [
                                    F.relu(true_loss - neg_loss + args.negative_contrast_margin)
                                    for neg_loss in negative_losses
                                ]
                                contrast_loss = torch.stack(ranking).mean()
                                loss = loss + args.negative_contrast_weight * contrast_loss
                            for item in negative_losses:
                                del item
                        unweighted_loss = loss
                        if args.min_snr_gamma > 0.0:
                            loss = loss * min_snr_weight
                        raw_loss = loss
                        loss_multiplier = 1.0
                        if w5_enabled:
                            if w4_selection is None or w5_loss_multipliers is None:
                                raise RuntimeError("W5 loss weighting requires a selected stratum")
                            loss, loss_multiplier = apply_w5_loss_multiplier(
                                raw_loss, stratum=w4_selection.stratum, multipliers=w5_loss_multipliers
                            )
                        elif w6_enabled:
                            if w4_selection is None or w6_loss_multipliers is None:
                                raise RuntimeError("W6 loss weighting requires a selected stratum")
                            loss, loss_multiplier = apply_w6_loss_multiplier(
                                raw_loss, stratum=w4_selection.stratum, multipliers=w6_loss_multipliers
                            )
                        elif w7_enabled:
                            if w4_selection is None or w7_loss_multipliers is None:
                                raise RuntimeError("W7 loss weighting requires a selected stratum")
                            loss, loss_multiplier = apply_w7_loss_multiplier(
                                raw_loss, stratum=w4_selection.stratum, multipliers=w7_loss_multipliers
                            )
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"non-finite loss at step={step} micro_step={micro_step}: {float(loss.detach().cpu())}")
                    if w4_sampler is not None:
                        print(json.dumps(cuda_memory_telemetry(
                            device=device,
                            phase="pre_backward",
                            rank=rank,
                            trainable_gradient_bytes=w4_trainable_gradient_bytes,
                            step=step,
                            micro_step=micro_step,
                        )), flush=True)
                    if os.environ.get("SOLARIS_MEM_DEBUG") == "1" and device.type == "cuda":
                        print(json.dumps({
                            "event": "mem_debug_post_forward", "step": step, "micro": micro_step,
                            "alloc_gb": round(torch.cuda.memory_allocated(device) / 2**30, 2),
                            "peak_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 2),
                        }), flush=True)
                    (loss / args.grad_accum).backward()
                    if os.environ.get("SOLARIS_MEM_DEBUG") == "1" and device.type == "cuda":
                        print(json.dumps({
                            "event": "mem_debug_post_backward", "step": step, "micro": micro_step,
                            "alloc_gb": round(torch.cuda.memory_allocated(device) / 2**30, 2),
                            "peak_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 2),
                        }), flush=True)
                        torch.cuda.reset_peak_memory_stats(device)
                micro_losses.append(float(loss.detach().cpu()))
                if w4_selection is not None:
                    step_w4_telemetry.append({
                        "stratum": w4_selection.stratum,
                        "subgroup": w4_selection.subgroup,
                        "record_index": record_index,
                        "record_id": w4_selection.record_id,
                        "match_id": w4_selection.match_id,
                        "episode": record.get("map_memory_episode"),
                        "track_id": record.get("map_memory_track_id"),
                        "roles": list(record.get("map_memory_selection_roles", [])),
                        "positive_frames": int(record.get("map_memory_positive_frames", 0)),
                        "loss": float(loss.detach().cpu()),
                        "region_contribution_raw": (
                            float(weighted_region_loss_raw.detach().cpu()) if weighted_region_loss_raw is not None else 0.0
                        ),
                        "region_contribution_applied": (
                            float(weighted_region_loss_applied.detach().cpu()) if weighted_region_loss_applied is not None else 0.0
                        ),
                        **({
                            "raw_loss": float(raw_loss.detach().cpu()),
                            "loss_multiplier": loss_multiplier,
                            "applied_loss": float(loss.detach().cpu()),
                        } if (w5_enabled or w6_enabled or w7_enabled) else {}),
                    })
                if args.min_snr_gamma > 0.0:
                    step_role_counts["min_snr_weight_sum"] = step_role_counts.get("min_snr_weight_sum", 0.0) + float(
                        min_snr_weight.detach().cpu()
                    )
                    step_role_counts["min_snr_weight_count"] = step_role_counts.get("min_snr_weight_count", 0) + 1
                    step_role_counts["unweighted_loss_sum"] = step_role_counts.get("unweighted_loss_sum", 0.0) + float(
                        unweighted_loss.detach().cpu()
                    )
                    step_role_counts["weighted_loss_sum"] = step_role_counts.get("weighted_loss_sum", 0.0) + float(
                        loss.detach().cpu()
                    )
                if region_loss is not None:
                    chunk_region_loss_weight = float(args.chunk0_region_loss_weight) if chunk_start == 0 else 1.0
                    step_role_counts["region_loss_sum"] = step_role_counts.get("region_loss_sum", 0.0) + float(region_loss.detach().cpu())
                    step_role_counts["region_loss_weighted_sum"] = step_role_counts.get("region_loss_weighted_sum", 0.0) + (
                        float(region_loss.detach().cpu()) * chunk_region_loss_weight
                    )
                    step_role_counts["region_loss_chunks"] = step_role_counts.get("region_loss_chunks", 0) + 1
                    step_role_counts["region_loss_weight_sum"] = step_role_counts.get("region_loss_weight_sum", 0.0) + chunk_region_loss_weight
                step_role_counts["region_pixels"] = step_role_counts.get("region_pixels", 0) + int(region_pixels)
                del x0, cond, x0_chunk, cond_chunk, cam_chunk, dense_tokens, state_tokens, interaction_tokens, interaction_features, noise, timestep, xt, target, pred, loss, region_loss, min_snr_weight
            pre_clip_module_gradient_norms = None
            if w4_sampler is not None and (step == start_step + 1 or step % args.w4_gradient_telemetry_every == 0):
                pre_clip_module_gradient_norms = {
                    "adapter": module_gradient_norm(adapter_all_params),
                    "encoder": module_gradient_norm(encoder_all_params),
                    "lora": module_gradient_norm(lora_all_params),
                    "state_projector": module_gradient_norm(state_all_params),
                    "interaction_projector": module_gradient_norm(interaction_all_params),
                }
            interaction_projector_gradient_norm = (
                module_gradient_norm(interaction_all_params)
                if interaction_all_params
                else None
            )
            grad_norm = None
            if args.max_grad_norm > 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm).detach().cpu())
                if not np.isfinite(grad_norm):
                    raise RuntimeError(f"non-finite grad_norm at step={step}: {grad_norm}")
            if fp32_master_params:
                for _p, _m in fp32_master_params:
                    _m.grad = None if _p.grad is None else _p.grad.detach().float()
            optimizer.step()
            if fp32_master_params:
                with torch.no_grad():
                    for _p, _m in fp32_master_params:
                        _p.data.copy_(_m.data)
            do_condition_probe = (
                condition_probe_batch is not None
                and args.condition_probe_every > 0
                and (step == 1 or step % args.condition_probe_every == 0)
            )
            if do_condition_probe:
                # Probe runs after optimizer.step()+master copy-back (params are current)
                # and after backward has freed the main graph (peak-memory precedent :2758).
                if rank == 0:
                    probe_payload = run_condition_probe(
                        args=args,
                        model=unwrap_module(train_model),
                        encoder=unwrap_module(train_encoder),
                        state_projector=(
                            unwrap_module(train_state_projector)
                            if train_state_projector is not None else None
                        ),
                        interaction_projector=(
                            unwrap_module(train_interaction_projector)
                            if train_interaction_projector is not None else None
                        ),
                        state_rows=state_rows,
                        probe=condition_probe_batch,
                        device=device,
                        dtype=dtype,
                        wan_token_hw=wan_token_hw,
                        step=step,
                    )
                    head = {k: v for k, v in probe_payload.items() if k != "windows"}
                    print(json.dumps(head, ensure_ascii=False), flush=True)
                    with (args.out_dir / "condition_probe.jsonl").open("a", encoding="utf-8") as probe_f:
                        probe_f.write(json.dumps(probe_payload, ensure_ascii=False) + "\n")
                if dist.is_initialized():
                    dist.barrier()
            loss_value = sum(micro_losses) / len(micro_losses)
            if not np.isfinite(loss_value):
                raise RuntimeError(f"non-finite averaged loss at step={step}: {loss_value}")
            losses.append(loss_value)
            all_w4: list[dict[str, Any]] = []
            if w4_sampler is not None and (step == 1 or step % args.log_every == 0):
                gathered: list[Any] = [None for _ in range(world)]
                if dist.is_initialized():
                    dist.all_gather_object(gathered, step_w4_telemetry)
                    all_w4 = [item for rank_items in gathered for item in rank_items]
                else:
                    all_w4 = step_w4_telemetry
            interaction_step_audit = None
            if step == 1 or step % args.log_every == 0:
                event_names = list(step_interaction_event_counts)
                audit_values = torch.tensor(
                    [
                        *(step_interaction_event_counts[name] for name in event_names),
                        step_interaction_nonzero_feature_count,
                    ],
                    device=device,
                    dtype=torch.int64,
                )
                if dist.is_initialized():
                    dist.all_reduce(audit_values, op=dist.ReduceOp.SUM)
                audit_counts = audit_values.detach().cpu().tolist()
                interaction_step_audit = {
                    "event_counts": {
                        name: int(audit_counts[index])
                        for index, name in enumerate(event_names)
                    },
                    "nonzero_feature_count": int(audit_counts[-1]),
                    "scope": "all_ranks_all_micro_steps",
                }
            if rank == 0 and (step == 1 or step % args.log_every == 0):
                payload = {
                    "event": "train_step",
                    "step": step,
                    "sampler_step": sampler_step,
                    "loss": loss_value,
                    "cuda_mem_alloc_gb": (
                        round(torch.cuda.memory_allocated(device) / 2**30, 2)
                        if device.type == "cuda" else None
                    ),
                    "cuda_mem_reserved_gb": (
                        round(torch.cuda.memory_reserved(device) / 2**30, 2)
                        if device.type == "cuda" else None
                    ),
                    "chunk_role_counts": step_role_counts,
                    "interaction_event_coverage": interaction_step_audit["event_counts"] if interaction_step_audit is not None else None,
                    "interaction_nonzero_feature_count": interaction_step_audit["nonzero_feature_count"] if interaction_step_audit is not None else None,
                    "interaction_event_coverage_scope": interaction_step_audit["scope"] if interaction_step_audit is not None else None,
                }
                if step_state_debug is not None:
                    payload["state_channel_debug"] = step_state_debug
                if args.region_loss_weight > 0.0:
                    payload["region_loss_by_sigma_band"] = finalize_region_loss_stats(step_region_loss_by_sigma)
                    payload["region_loss_by_chunk_index"] = finalize_region_loss_stats(step_region_loss_by_chunk)
                    if step_role_counts.get("region_loss_chunks", 0) > 0 and (
                        not step_region_loss_by_sigma or not step_region_loss_by_chunk
                    ):
                        raise RuntimeError(
                            "region loss chunks were counted but sigma/chunk stratified stats are empty; "
                            "V0 diagnostics must stay on the actual backward path"
                        )
                if grad_norm is not None:
                    payload["grad_norm"] = grad_norm
                if interaction_projector_gradient_norm is not None:
                    payload["interaction_projector_gradient_norm"] = interaction_projector_gradient_norm
                if w4_sampler is not None:
                    stratum_loss = {}
                    stratum_role_counts = {}
                    for name in STRATA:
                        values = [item["loss"] for item in all_w4 if item["stratum"] == name]
                        stratum_loss[name] = {"count": len(values), "mean": sum(values) / len(values) if values else None}
                        selected = [item for item in all_w4 if item["stratum"] == name]
                        stratum_role_counts[name] = {
                            "context": sum(item["roles"].count("context") for item in selected),
                            "positive": sum(item["roles"].count("positive") for item in selected),
                        }
                        w4_cumulative_records[name].update(item["record_id"] for item in selected)
                        w4_cumulative_matches[name].update(item["match_id"] for item in selected)
                    payload["w4_source_telemetry"] = all_w4
                    payload["w4_stratum_loss"] = stratum_loss
                    if w5_enabled:
                        payload["w5_stratum_loss"] = {
                            name: {
                                "count": len(selected),
                                "raw_mean": sum(item["raw_loss"] for item in selected) / len(selected) if selected else None,
                                "applied_mean": sum(item["applied_loss"] for item in selected) / len(selected) if selected else None,
                                "multiplier": W5_LOSS_MULTIPLIERS[name],
                            }
                            for name in STRATA
                            for selected in [[item for item in all_w4 if item["stratum"] == name]]
                        }
                    elif w6_enabled:
                        payload["w6_stratum_loss"] = {
                            name: {
                                "count": len(selected),
                                "raw_mean": sum(item["raw_loss"] for item in selected) / len(selected) if selected else None,
                                "applied_mean": sum(item["applied_loss"] for item in selected) / len(selected) if selected else None,
                                "multiplier": W6_LOSS_MULTIPLIERS[name],
                            }
                            for name in STRATA
                            for selected in [[item for item in all_w4 if item["stratum"] == name]]
                        }
                    elif w7_enabled:
                        payload["w7_stratum_loss"] = {
                            name: {
                                "count": len(selected),
                                "raw_mean": sum(item["raw_loss"] for item in selected) / len(selected) if selected else None,
                                "applied_mean": sum(item["applied_loss"] for item in selected) / len(selected) if selected else None,
                                "multiplier": W7_LOSS_MULTIPLIERS[name],
                            }
                            for name in STRATA
                            for selected in [[item for item in all_w4 if item["stratum"] == name]]
                        }
                    payload["w4_stratum_role_counts"] = stratum_role_counts
                    payload["w4_source_counts"] = {name: sum(item["stratum"] == name for item in all_w4) for name in STRATA}
                    payload["w4_cumulative_unique_record_coverage"] = {
                        name: len(w4_cumulative_records[name]) for name in STRATA
                    }
                    payload["w4_cumulative_unique_match_coverage"] = {
                        name: len(w4_cumulative_matches[name]) for name in STRATA
                    }
                    if pre_clip_module_gradient_norms is not None:
                        payload["pre_clip_module_gradient_norms"] = pre_clip_module_gradient_norms
                print(json.dumps(payload), flush=True)
            if rank == 0 and (step % args.save_every == 0 or step == args.max_steps):
                ckpt = args.out_dir / "checkpoints" / f"memory_dense_adapter_step_{step:06d}.pt"
                save_checkpoint(ckpt, model=train_model, encoder=train_encoder, state_projector=train_state_projector, interaction_projector=train_interaction_projector, optimizer=optimizer, step=step, config=run_config)
                print(json.dumps({"event": "checkpoint", "step": step, "path": str(ckpt)}), flush=True)
        if not losses:
            raise RuntimeError(
                f"training loop produced zero losses: start_step={start_step} max_steps={args.max_steps} "
                f"resume_checkpoint_step={resume_checkpoint_step} reset_optimizer={args.reset_optimizer}"
            )
        return {
            "kind": "memory_dense_adapter_train_v0",
            "status": "complete",
            "steps": args.max_steps,
            "optimizer_steps_run": len(losses),
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "out_dir": str(args.out_dir),
        }
    finally:
        cleanup_dist()


def write_report(result: dict[str, Any], out_json: Path | None) -> None:
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(text, encoding="utf-8")
    print(text)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["audit-data", "prepare-index", "check-noop", "preflight-train", "state-zero-preflight", "train"])
    ap.add_argument("--map-manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--require-backend-id", default="bsp_faces_disp_gpu")
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("output/memory_dense_adapter_v0"))
    ap.add_argument("--seed", type=int, default=20260531)

    ap.add_argument("--split-key", choices=["episode", "track", "match"], default="episode")
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--test-fraction", type=float, default=0.05)
    ap.add_argument("--dense-audit-limit", type=int, default=128)
    ap.add_argument("--hash-dense", action="store_true")
    ap.add_argument("--hash-manifest", action="store_true")

    ap.add_argument("--lingbot-repo", type=Path, default=LINGBOT_ROOT)
    ap.add_argument("--ckpt-dir", type=Path, default=ASSET_ROOT / "lingbot-world-base-cam")
    ap.add_argument("--fast-model-dir", type=Path, default=None)
    ap.add_argument(
        "--wan-low-cpu-mem-usage",
        choices=["auto", "true", "false"],
        default="auto",
        help="Override diffusers low_cpu_mem_usage when loading WanModelFast. Use false on high-RAM hosts if meta tensors remain for newly initialized modules.",
    )
    ap.add_argument("--cache-manifest", type=Path, default=None)
    ap.add_argument(
        "--training-index-cache",
        type=Path,
        default=None,
        help="Validated training input index cache for faster startup. Contract is checked with manifest/cache sha and key loader args.",
    )
    ap.add_argument(
        "--rebuild-training-index-cache",
        action="store_true",
        help="Ignore any existing --training-index-cache and rebuild it after full validation.",
    )
    ap.add_argument(
        "--lightweight-training-index-rebuild",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When rebuilding --training-index-cache, avoid full sidecar/teacher QA load and build only the train sample index from manifest/cache rows.",
    )
    ap.add_argument(
        "--training-index-cache-path-checks",
        type=int,
        default=4,
        help="On cache hit, probe this many cached records for path existence and pose/intrinsics shape.",
    )
    ap.add_argument(
        "--log-loader-stages",
        action="store_true",
        help="Print JSON timing events while loading releases/cache records. Useful for CPFS startup diagnosis.",
    )
    ap.add_argument(
        "--extra-map-manifest",
        type=Path,
        action="append",
        default=[],
        help="Additional Map Memory release manifest. Repeat with matching --extra-cache-manifest for loader-level concat.",
    )
    ap.add_argument(
        "--extra-cache-manifest",
        type=Path,
        action="append",
        default=[],
        help="Additional aligned cache manifest. Must be repeated in the same order as --extra-map-manifest.",
    )
    ap.add_argument(
        "--state-cache-manifest",
        type=Path,
        default=None,
        help="State channel v0 manifest keyed by clip_id. Enables zero-init latent-post state token injection.",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--video-frames", type=int, default=81)
    ap.add_argument("--raw-stride", type=int, default=2)
    ap.add_argument("--train-split", default="train", help="Aligned cache split to use for optimizer steps; set to empty string to disable filtering.")
    ap.add_argument(
        "--allow-static-dense-repeat",
        action="store_true",
        help="Permit one Map Memory dense sample to be repeated across a latent chunk. Disabled for formal training by default.",
    )

    ap.add_argument("--vae-stride", type=int, default=8)
    ap.add_argument("--patch-size-hw", type=int, default=2)
    ap.add_argument("--latent-channels", type=int, default=16)
    ap.add_argument("--cond-dim", type=int, default=128)
    ap.add_argument("--encoder-hidden-dim", type=int, default=64)
    ap.add_argument(
        "--dense-condition-encoder",
        choices=["conv", "frozen_vae"],
        default="conv",
        help="Condition encoder source. Default conv preserves previous behavior; frozen_vae uses default-off frozen Wan2.1 VAE features.",
    )
    ap.add_argument(
        "--dense-vae-pth",
        type=Path,
        default=None,
        help="Wan2.1 VAE checkpoint for --dense-condition-encoder frozen_vae. Defaults to ckpt-dir/WAN_CONFIGS['i2v-A14B'].vae_checkpoint.",
    )
    ap.add_argument(
        "--dense-vae-packing",
        default="img1_mask_img2_player_v0",
        help="Recorded packing id for frozen_vae: img1=[ch0,ch2,ch3], img2=[ch4,ch5,ch6], ch1 dropped, x*2-1.",
    )
    ap.add_argument("--adapter-hidden-dim", type=int, default=512)
    ap.add_argument("--adapter-residual-mode", choices=["cond_gated", "additive"], default="cond_gated")
    ap.add_argument("--adapter-wrap-first-blocks", type=int, default=0, help="0 wraps all blocks; B2b probe uses first third only.")
    ap.add_argument("--adapter-residual-scale-init", type=float, default=1.0)
    ap.add_argument(
        "--activation-checkpoint-blocks",
        action="store_true",
        help="Checkpoint wrapped Wan blocks during training to reduce activation memory. Does not change adapter structure.",
    )
    ap.add_argument(
        "--activation-checkpoint-adapter",
        action="store_true",
        help="Checkpoint each wrapped Wan block together with its dense-adapter residual so adapter activations are recomputed in backward instead of staying resident. Required to fit 96GB GPUs; numerically equivalent to --activation-checkpoint-blocks.",
    )
    ap.add_argument("--lora-rank", type=int, default=0, help="Enable DiT LoRA with this rank. 0 disables LoRA and preserves previous behavior.")
    ap.add_argument("--lora-targets", default="attn,mlp", help="Comma/space list: attn,self_attn,cross_attn,mlp,all.")
    ap.add_argument("--lora-alpha", type=float, default=None, help="LoRA alpha; defaults to rank.")
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--lora-lr", type=float, default=1e-5)

    ap.add_argument("--hidden-dim", type=int, default=64, help="check-noop tiny model hidden dim")
    ap.add_argument("--ffn-dim", type=int, default=128, help="check-noop tiny model FFN dim")
    ap.add_argument("--freq-dim", type=int, default=16, help="check-noop tiny model frequency dim")
    ap.add_argument("--text-dim", type=int, default=32, help="check-noop tiny model text dim")
    ap.add_argument("--text-len", type=int, default=8, help="check-noop tiny model text length")
    ap.add_argument("--context-tokens", type=int, default=3)
    ap.add_argument("--num-heads", type=int, default=4, help="check-noop tiny model heads")
    ap.add_argument("--num-layers", type=int, default=2, help="check-noop tiny model layers")
    ap.add_argument("--noop-frames", type=int, default=1)
    ap.add_argument("--noop-tolerance", type=float, default=1e-6)
    ap.add_argument("--timestep", type=float, default=100.0)

    ap.add_argument("--video-width", type=int, default=832)
    ap.add_argument("--video-height", type=int, default=480)
    ap.add_argument(
        "--dense-hw",
        nargs=2,
        type=int,
        metavar=("H", "W"),
        default=[240, 416],
        help=(
            "Dense condition map grid H W. Default is the v2 dense contract 240 416 "
            "(-> 30x52 native VAE grid == DiT token grid, no interpolation). Pass the "
            "legacy pre-v2 grid to train or audit an older release."
        ),
    )
    ap.add_argument("--latent-frames", type=int, default=21)
    ap.add_argument("--chunk-size", type=int, default=3)
    ap.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument(
        "--dense-resize-mode",
        choices=["bilinear", "native-only"],
        default="bilinear",
        help="Resize encoded dense tokens to the cache Wan token grid, or require the native dense grid implied by --dense-hw.",
    )
    ap.add_argument("--max-aspect-ratio-delta", type=float, default=0.08)
    ap.add_argument("--require-release-report", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--min-train-records", type=int, default=1)
    ap.add_argument("--min-runtime-train-records", type=int, default=100)
    ap.add_argument("--min-fail-fast-train-records", type=int, default=MIN_FAIL_FAST_TRAIN_RECORDS,
                    help="per-release floor for the fail-fast record check; lower it only for a deliberately small set")
    ap.add_argument("--min-train-positive-frames", type=int, default=1)
    ap.add_argument("--min-train-context-frames", type=int, default=0)
    ap.add_argument("--min-train-matches", type=int, default=1)
    ap.add_argument("--min-train-episodes", type=int, default=1)
    ap.add_argument("--preflight-records", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--positive-batch-fraction", type=float, default=0.5)
    ap.add_argument("--negative-contrast-weight", type=float, default=0.05)
    ap.add_argument("--negative-contrast-margin", type=float, default=0.005)
    ap.add_argument("--negative-contrast-modes", nargs="+", choices=["blank", "shuffled"], default=["blank", "shuffled"])
    ap.add_argument("--negative-contrast-positive-only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--negative-shuffle-offset", type=int, default=7)
    ap.add_argument("--region-loss-weight", type=float, default=0.0)
    ap.add_argument("--region-loss-min-pixels", type=int, default=1)
    ap.add_argument("--region-loss-normalize", choices=["total", "region"], default="total",
                    help="total: sum(masked sq err)/pred.numel(), per-element weight = 1+w, area-independent; region: legacy mean over mask pixels (unbounded 1+w*N_tot/N_reg).")
    ap.add_argument("--shuffle-records", action=argparse.BooleanOptionalAction, default=True,
                    help="Deterministic per-epoch permutation over records/positive chunks; breaks manifest-order scanning and intra-batch adjacency.")
    ap.add_argument("--fp32-master", action=argparse.BooleanOptionalAction, default=True,
                    help="AdamW on fp32 master copies; bf16 params refreshed from masters each step (fixes bf16 rounding freeze).")
    ap.add_argument("--condition-probe-every", type=int, default=0,
                    help="Every N optimizer steps run the frozen-val condition-sensitivity probe on rank 0 (0=off). Requires --condition-probe-batch. Base-model path only.")
    ap.add_argument("--condition-probe-batch", type=Path, default=None,
                    help="Precomputed probe batch .pt (kind=condition_probe_batch_v0): frozen val windows with per-variant condition tokens (true / shuffled_xmatch / blank_dense) plus xt/timestep/target/cond/cam tensors.")
    ap.add_argument("--chunk0-region-loss-weight", type=float, default=1.0)
    ap.add_argument("--sigma-curriculum-prob", type=float, default=0.0)
    ap.add_argument("--sigma-curriculum-radius", type=float, default=INFERENCE_SIGMA_BAND_RADIUS)
    ap.add_argument(
        "--min-snr-gamma",
        type=float,
        default=0.0,
        help="Enable flow/v Min-SNR loss weighting with min(SNR,gamma)/(SNR+1). Default 0 disables weighting.",
    )
    ap.add_argument(
        "--min-snr-eps",
        type=float,
        default=1e-4,
        help="Numerical clamp for Min-SNR sigma when --min-snr-gamma > 0.",
    )
    ap.add_argument("--only-player-dense-channels", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--resume-from", type=Path, default=None)
    ap.add_argument("--reset-optimizer", action="store_true")
    ap.add_argument(
        "--sampler-step-origin",
        choices=["absolute", "resume_relative"],
        default="absolute",
        help=(
            "Use absolute optimizer steps for record sampling, or restart only the data sampler at step 1 "
            "when resuming while preserving checkpoint weights and optimizer state."
        ),
    )
    ap.add_argument(
        "--sampler-step-offset",
        type=int,
        default=0,
        help="Continuation sampler steps already consumed before this resume segment (0 for +300, 300 for +300->+600).",
    )
    ap.add_argument("--w4-sampler-version", choices=[SAMPLER_VERSION], default=None)
    ap.add_argument("--w4-quota", action="append", default=None, help="Repeat stratum=count for all four W4 strata.")
    ap.add_argument("--w4-sampler-seed", type=int, default=20260718)
    ap.add_argument("--w4-freeze-lora", action="store_true")
    ap.add_argument("--w4-freeze-state-projector", action="store_true")
    ap.add_argument("--w4-audit-report", type=Path, default=None)
    ap.add_argument("--w4-audit-sha256", default=None)
    ap.add_argument(
        "--w4-region-loss-cap-fraction",
        type=float,
        default=0.0,
        help="Cap weighted region contribution to this fraction of detached base loss; 0 keeps existing behavior.",
    )
    ap.add_argument("--w4-gradient-telemetry-every", type=int, default=1)
    ap.add_argument("--w5-sampler-version", choices=[W5_SAMPLER_VERSION, W5_EXTENSION_SAMPLER_VERSION], default=None)
    ap.add_argument("--w5-loss-multiplier", action="append", default=None)
    ap.add_argument("--w5-authorization-report", type=Path, default=None)
    ap.add_argument("--w5-authorization-sha256", default=None)
    ap.add_argument("--w6-sampler-version", choices=[W6_SAMPLER_VERSION], default=None)
    ap.add_argument("--w6-loss-multiplier", action="append", default=None)
    ap.add_argument("--w6-authorization-report", type=Path, default=None)
    ap.add_argument("--w6-authorization-sha256", default=None)
    ap.add_argument("--w7-sampler-version", choices=[W7_SAMPLER_VERSION], default=None)
    ap.add_argument("--w7-loss-multiplier", action="append", default=None)
    ap.add_argument("--w7-authorization-report", type=Path, default=None)
    ap.add_argument("--w7-authorization-sha256", default=None)
    ap.add_argument("--use-base-model", action="store_true",
                    help="Train the dense adapter+LoRA on ONE WanI2V base expert (non-causal full-attention, no kv-cache) instead of WanModelFast.")
    ap.add_argument("--base-expert", choices=["low", "high"], default="low",
                    help="Which base expert to train: low (sigma<boundary, most steps/detail) or high (sigma>=boundary, layout).")
    ap.add_argument("--base-sigma-min", type=float, default=None, help="Override base training sigma band low end (default derived from boundary+expert).")
    ap.add_argument("--base-sigma-max", type=float, default=None, help="Override base training sigma band high end (default derived from boundary+expert).")
    return ap


PROBE_VARIANTS = ("true", "shuffled_xmatch", "blank_dense", "disabled")


def load_condition_probe_batch(path: Path) -> dict[str, Any]:
    """Record-based probe batch (v1). Stores records + pickled MapMemorySample lists
    and weight-independent tensors only; condition tokens are built at probe time
    through the CURRENT encoder/projectors. A token-precomputing contract would
    freeze the (trainable, zero-init) projection weights and make from-scratch
    probes read all-zero conditions."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data.get("kind") != "condition_probe_batch_v1":
        raise ValueError(f"unexpected probe batch kind: {data.get('kind')!r} at {path}")
    if data.get("split") != "val":
        raise ValueError(
            f"probe batch split must be 'val' (got {data.get('split')!r}); "
            "train-batch probes are not citable (frozen R4-val contract)"
        )
    windows = data.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError(f"probe batch has no windows: {path}")
    if "text_context" not in data:
        raise ValueError(f"probe batch missing text_context: {path}")
    required = ("window_id", "match_id", "partner_match_id", "record", "partner_record",
                "chunk_samples", "partner_chunk_samples", "xt", "timestep", "target",
                "cond_chunk", "cam_chunk")
    for win in windows:
        for key in required:
            if key not in win:
                raise ValueError(f"probe window {win.get('window_id')!r} missing key: {key}")
        if win["match_id"] == win["partner_match_id"]:
            raise ValueError(
                f"{win['window_id']}: shuffled partner is same-match; "
                "cross-match sampling is mandatory (offset-style neighbours are weak negatives)"
            )
        for key in ("chunk_samples", "partner_chunk_samples"):
            if not isinstance(win[key], list) or not win[key]:
                raise ValueError(f"{win['window_id']}: {key} is empty")
        if len(win["chunk_samples"]) != len(win["partner_chunk_samples"]):
            raise ValueError(f"{win['window_id']}: chunk_samples length mismatch with partner")
    return data


def _probe_tokens_for(
    encoder: torch.nn.Module,
    state_projector: torch.nn.Module | None,
    interaction_projector: torch.nn.Module | None,
    state_rows: dict[str, Any] | None,
    record: dict[str, Any],
    chunk_samples: list[Any],
    *,
    chunk_size: int,
    device: torch.device,
    dtype: torch.dtype,
    wan_token_hw: tuple[int, int],
) -> torch.Tensor:
    """True-path token assembly for a probe window, mirroring the training loop:
    dense through the CURRENT encoder, plus state/interaction through the CURRENT
    projectors when the sidecar row exists (missing row -> stream absent, same as
    training)."""
    tokens, _ = dense_tokens_for_samples(
        encoder, chunk_samples, device=device, dtype=dtype, target_token_hw=wan_token_hw,
    )
    if state_projector is not None and state_rows is not None:
        state_tokens = state_tokens_for_record_chunk(
            state_projector, state_rows, record,
            chunk_start=0, chunk_size=chunk_size,
            device=device, dtype=dtype, target_token_hw=wan_token_hw,
        )
        if state_tokens is not None:
            tokens = tokens + state_tokens
    if interaction_projector is not None and state_rows is not None:
        interaction_result = interaction_tokens_for_record_chunk(
            interaction_projector, state_rows, record,
            chunk_start=0, chunk_size=chunk_size,
            device=device, dtype=dtype, target_token_hw=wan_token_hw,
        )
        if interaction_result is not None:
            tokens = tokens + interaction_result
    return tokens


def run_condition_probe(
    args: argparse.Namespace,
    model: torch.nn.Module,
    encoder: torch.nn.Module,
    state_projector: torch.nn.Module | None,
    interaction_projector: torch.nn.Module | None,
    state_rows: dict[str, Any] | None,
    probe: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    wan_token_hw: tuple[int, int],
    step: int,
) -> dict[str, Any]:
    """Paired condition-sensitivity probe on the frozen val batch.

    Tokens are built at probe time through the CURRENT encoder/projectors (they are
    trainable and zero-init at fresh starts; precomputed tokens would be stale or
    all-zero). Variants: shuffled_xmatch = full condition set from a different-match
    partner window; blank_dense = zero dense image through the current encoder,
    state/interaction streams absent; disabled = COND_KEY omitted (adapter no-op)
    with LoRA still active, which is NOT the base model.
    """
    if not args.use_base_model:
        raise RuntimeError("--condition-probe-every currently supports the base-model path only")
    was_training = model.training
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    t0 = time.time()
    text_context = probe["text_context"].to(device=device, dtype=dtype)
    rows: list[dict[str, Any]] = []
    sums = {variant: 0.0 for variant in PROBE_VARIANTS}
    with torch.no_grad():
        for win in probe["windows"]:
            xt = win["xt"].to(device=device, dtype=dtype)
            timestep = win["timestep"].to(device=device, dtype=torch.float32)
            target = win["target"].to(device=device, dtype=torch.float32)
            cond_chunk = win["cond_chunk"].to(device=device, dtype=dtype)
            cam_chunk = win["cam_chunk"].to(device=device, dtype=dtype)
            n_samples = len(win["chunk_samples"])
            variant_tokens: dict[str, torch.Tensor] = {}
            variant_tokens["true"] = _probe_tokens_for(
                encoder, state_projector, interaction_projector, state_rows,
                win["record"], win["chunk_samples"],
                chunk_size=n_samples, device=device, dtype=dtype, wan_token_hw=wan_token_hw,
            )
            variant_tokens["shuffled_xmatch"] = _probe_tokens_for(
                encoder, state_projector, interaction_projector, state_rows,
                win["partner_record"], win["partner_chunk_samples"],
                chunk_size=n_samples, device=device, dtype=dtype, wan_token_hw=wan_token_hw,
            )
            sample_dense = load_dense(win["chunk_samples"][0])
            zero_dense = torch.zeros(
                (n_samples, *sample_dense.shape), device=device, dtype=dtype,
            )
            blank_tokens, _ = encoder(zero_dense, target_token_hw=wan_token_hw)
            variant_tokens["blank_dense"] = blank_tokens.reshape(
                1, n_samples * blank_tokens.shape[1], blank_tokens.shape[2]
            ).to(dtype)
            if variant_tokens["true"].shape != variant_tokens["shuffled_xmatch"].shape:
                raise RuntimeError(f"{win['window_id']}: true/shuffled token shapes differ")
            if variant_tokens["true"].shape != variant_tokens["blank_dense"].shape:
                raise RuntimeError(f"{win['window_id']}: true/blank token shapes differ")
            seq_len = int(variant_tokens["true"].shape[1])
            per: dict[str, float] = {}
            for variant in PROBE_VARIANTS:
                dit_cond: dict[str, Any] = {"c2ws_plucker_emb": cam_chunk.chunk(1, dim=0)}
                if variant != "disabled":
                    dit_cond[COND_KEY] = variant_tokens[variant]
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda" and dtype == torch.bfloat16):
                    pred = model(
                        x=[xt],
                        t=timestep,
                        context=[text_context],
                        seq_len=seq_len,
                        y=[cond_chunk],
                        dit_cond_dict=dit_cond,
                    )[0]
                loss = float(F.mse_loss(pred.float(), target).detach().cpu())
                per[variant] = loss
                sums[variant] += loss
            rows.append({
                "window_id": win["window_id"],
                "losses": per,
                "true_lt_shuffled": per["true"] < per["shuffled_xmatch"],
                "true_lt_blank": per["true"] < per["blank_dense"],
            })
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    if model.training != was_training:
        raise RuntimeError("condition probe altered model.training; activation-checkpoint gate depends on it")
    n = len(rows)
    return {
        "event": "condition_probe",
        "step": step,
        "n_windows": n,
        "true_lt_shuffled_count": sum(1 for r in rows if r["true_lt_shuffled"]),
        "true_lt_blank_count": sum(1 for r in rows if r["true_lt_blank"]),
        "mean_loss": {variant: sums[variant] / n for variant in PROBE_VARIANTS},
        "variant_semantics": {
            "shuffled_xmatch": "full condition set from a different-match partner window (current modules)",
            "blank_dense": "zero dense image through current encoder; state/interaction streams absent",
            "disabled": "COND_KEY omitted -> adapter no-op; LoRA active; NOT the base model",
        },
        "sign_test_hint": "n=32: true_lt_shuffled_count>=22 -> one-sided p<0.025 under H0",
        "wall_seconds": round(time.time() - t0, 2),
        "max_mem_gb": (
            round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
            if device.type == "cuda" else None
        ),
        "windows": rows,
    }


def main() -> None:
    args = build_parser().parse_args()
    # v2 dense contract (416x240 -> 30x52 native, no interpolation): align the loader
    # validators with --dense-hw before any manifest or npz is touched.
    map_memory_data.set_dense_hw(*dense_hw_from_args(args))
    validate_interaction_v1_protocol()
    if args.train_split == "":
        args.train_split = None
    if args.mode == "audit-data":
        result = audit_data(args)
    elif args.mode == "prepare-index":
        result = prepare_index(args)
    elif args.mode == "check-noop":
        result = check_noop(args)
    elif args.mode == "preflight-train":
        result = preflight_train(args)
    elif args.mode == "state-zero-preflight":
        result = state_zero_preflight(args)
    else:
        result = train(args)
    write_report(result, args.out_json)


if __name__ == "__main__":
    if os.environ.get("QXQ_MEM_HISTORY"):
        torch.cuda.memory._record_memory_history(max_entries=200000)
    try:
        main()
    except torch.cuda.OutOfMemoryError:
        snapshot_path = os.environ.get("QXQ_MEM_SNAPSHOT_PATH")
        if snapshot_path:
            torch.cuda.memory._dump_snapshot(snapshot_path)
        raise
