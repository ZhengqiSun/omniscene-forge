#!/usr/bin/env python3
"""Phase2a-lite sampler for LingBot Fast + Memory dense adapter.

This is a qualitative smoke/visualization tool. It keeps LingBot's native
WanI2VFast rollout path intact and only wraps the already-trained dense adapter
around the DiT model, then injects per-chunk dense condition tokens.
"""

from __future__ import annotations
from runtime_paths import source_path

import argparse
import copy
import json
import math
import shutil
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from PIL import Image


COND_KEY = "memory_dense_cond_tokens"


def add_path(path: Path) -> None:
    text = str(path.resolve())
    if text not in sys.path:
        sys.path.insert(0, text)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_dense_npz(path: Path) -> np.ndarray:
    with np.load(path) as data:
        if "dense" in data:
            arr = data["dense"]
        elif "condition" in data:
            arr = data["condition"]
        else:
            arr = data[data.files[0]]
    # Guard against bypass-produced npz (e.g. bare float16 exports without meta):
    # the training loader rejects dtype != float32, so a silent convert here hides
    # a schema mismatch. Convert, but make it visible and check basic sanity.
    if arr.dtype != np.float32:
        print(f"[load_dense_npz] WARNING dtype={arr.dtype} != float32 "
              f"(bypass export?) path={path}", flush=True)
    arr = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError(f"non-finite values in dense npz: {path}")
    return arr


class DensePathSample:
    def __init__(self, sample_id: str, dense_path: Path):
        self.sample_id = sample_id
        self.dense_path = dense_path


def load_dense_path_sample(sample: DensePathSample) -> np.ndarray:
    return load_dense_npz(sample.dense_path)


def resolve_clip_dir(record: dict[str, Any], fallback_root: Path | None) -> Path:
    for key in ["clip_dir", "action_path"]:
        value = record.get(key)
        if value and Path(value).exists():
            return Path(value)
    if fallback_root is not None:
        path = fallback_root / str(record["clip_id"])
        if path.exists():
            return path
    raise FileNotFoundError(f"cannot resolve clip_dir for {record.get('clip_id')}")


def choose_record(records: list[dict[str, Any]], *, split: str, clip_index: int, clip_id: str | None) -> dict[str, Any]:
    candidates = [r for r in records if str(r.get("map_memory_split", r.get("split", ""))) == split]
    if not candidates:
        raise ValueError(f"no records for split={split!r}")
    if clip_id:
        for row in candidates:
            if str(row.get("clip_id")) == clip_id:
                return row
        raise ValueError(f"clip_id {clip_id!r} not found in split={split!r}")
    if clip_index < 0 or clip_index >= len(candidates):
        raise IndexError(f"clip_index {clip_index} outside split size {len(candidates)}")
    return candidates[clip_index]


def choose_phase2a_source_row(path: Path, *, window_id: str | None, ego_stem: str | None, clip_index: int) -> dict[str, Any]:
    rows = read_jsonl(path)
    if window_id or ego_stem:
        rows = [
            row for row in rows
            if (not window_id or row.get("phase2a_window_id") == window_id)
            and (not ego_stem or row.get("player_stem") == ego_stem)
        ]
    if not rows:
        raise ValueError(f"no phase2a source rows match window_id={window_id!r} ego_stem={ego_stem!r}")
    if clip_index < 0 or clip_index >= len(rows):
        raise IndexError(f"clip_index {clip_index} outside phase2a source row count {len(rows)}")
    return rows[clip_index]


def choose_phase2a_shuffled_source_row(path: Path, *, current_clip_id: str) -> dict[str, Any]:
    rows = read_jsonl(path)
    for row in rows:
        if str(row.get("clip_id")) != str(current_clip_id) and row.get("phase2a_dense_sequence_manifest"):
            return row
    raise ValueError(f"no alternate phase2a row available for shuffled_dense current_clip_id={current_clip_id!r}")


def choose_shuffled_record(records: list[dict[str, Any]], *, current: dict[str, Any], split: str) -> dict[str, Any]:
    candidates = [r for r in records if str(r.get("map_memory_split", r.get("split", ""))) == split]
    current_clip_id = str(current.get("clip_id"))
    for row in candidates:
        if str(row.get("clip_id")) != current_clip_id:
            return row
    raise ValueError(f"no alternate aligned-cache row available for shuffled_dense split={split!r}")


def load_phase2a_dense_samples(path: Path, latent_frames: int) -> tuple[list[DensePathSample], list[str]]:
    rows = read_jsonl(path)
    rows = rows[:latent_frames]
    samples = [
        DensePathSample(str(row.get("sample_id", f"dense_{idx:03d}")), Path(row["dense_path"]))
        for idx, row in enumerate(rows)
    ]
    return samples, [sample.sample_id for sample in samples]


def pose_rotation_from_action_frame(frame: dict[str, Any]) -> np.ndarray:
    rotation = frame.get("camera_rotation") or [0.0, 0.0, 0.0]
    pitch = math.radians(float(rotation[1]))
    yaw = math.radians(float(rotation[2]))
    sy = math.sin(yaw)
    cy = math.cos(yaw)
    sp = math.sin(pitch)
    cp = math.cos(pitch)
    return np.asarray(
        [
            [sy, -cy * sp, cy * cp],
            [-cy, -sy * sp, sy * cp],
            [0.0, -cp, -sp],
        ],
        dtype=np.float32,
    )


def poses_from_phase2a_action(row: dict[str, Any], frame_num: int) -> np.ndarray:
    action_path = Path(row["action_json"])
    frames = json.loads(action_path.read_text(encoding="utf-8"))
    raw_indices = [int(x) for x in row["raw_indices"][:frame_num]]
    if len(raw_indices) != frame_num:
        raise ValueError(f"phase2a row has {len(raw_indices)} raw indices, need {frame_num}")
    poses = np.tile(np.eye(4, dtype=np.float32), (frame_num, 1, 1))
    for out_idx, raw_idx in enumerate(raw_indices):
        frame = frames[raw_idx]
        cam_pos = frame.get("camera_position") or [
            float(frame["x"]),
            float(frame["y"]),
            float(frame.get("z", 0.0)) + 64.0,
        ]
        poses[out_idx, :3, :3] = pose_rotation_from_action_frame(frame)
        poses[out_idx, :3, 3] = np.asarray(cam_pos, dtype=np.float32)
    return poses


def intrinsics_for_frame_count(src: Path, frame_num: int) -> np.ndarray:
    intrinsics = np.load(src / "intrinsics.npy")
    if len(intrinsics) >= frame_num:
        return intrinsics[:frame_num].astype(np.float32, copy=True)
    first = intrinsics[0].astype(np.float32)
    return np.repeat(first[None, :], frame_num, axis=0)


def materialize_phase2a_clip_dir(
    *,
    row: dict[str, Any],
    out_root: Path,
    latent_frames: int,
    chunk_size: int,
    source_clip_root: Path,
) -> Path:
    """Create a LingBot action_path from Phase2a handoff rows."""
    existing = row.get("phase2a_existing_5s_cache_clip_id")
    if not existing:
        raise ValueError("phase2a source row has no phase2a_existing_5s_cache_clip_id")
    src = source_clip_root / str(existing)
    if not src.exists():
        raise FileNotFoundError(f"existing phase2a source clip does not exist: {src}")
    out = out_root / str(row["clip_id"])
    out.mkdir(parents=True, exist_ok=True)
    for name in ["image.jpg", "prompt.txt"]:
        shutil.copy2(src / name, out / name)
    for optional in ["action.npy", "meta.json", "video.mp4"]:
        src_path = src / optional
        if src_path.exists():
            shutil.copy2(src_path, out / optional)

    requested_frame_num = frame_num_for_latent_frames(latent_frames)
    effective_latent_frames = effective_latent_frames_for_chunking(latent_frames, chunk_size=chunk_size)
    frame_num = frame_num_for_latent_frames(effective_latent_frames)
    poses = poses_from_phase2a_action(row, frame_num)
    intrinsics = intrinsics_for_frame_count(src, frame_num)
    np.save(out / "poses.npy", poses)
    np.save(out / "intrinsics.npy", intrinsics)

    validation: dict[str, Any] = {}
    src_poses_path = src / "poses.npy"
    if src_poses_path.exists():
        src_poses = np.load(src_poses_path)
        n = min(len(src_poses), len(poses))
        validation = {
            "compared_existing_pose_frames": int(n),
            "max_abs_pose_diff_vs_existing": float(np.max(np.abs(src_poses[:n] - poses[:n]))) if n else None,
        }
        if n and validation["max_abs_pose_diff_vs_existing"] > 1e-3:
            raise ValueError(f"phase2a pose reconstruction mismatch: {validation}")

    meta = {
        "kind": "phase2a_materialized_clip_from_action_json_v0",
        "source_clip_dir": str(src),
        "source_clip_id": str(existing),
        "phase2a_clip_id": row.get("clip_id"),
        "phase2a_window_id": row.get("phase2a_window_id"),
        "phase2a_dense_sequence_manifest": row.get("phase2a_dense_sequence_manifest"),
        "requested_latent_frames": int(latent_frames),
        "effective_latent_frames": int(effective_latent_frames),
        "requested_frame_num": int(requested_frame_num),
        "effective_frame_num": int(frame_num),
        "frame_num": int(frame_num),
        "pose_validation": validation,
        "note": "Image/prompt come from existing 5s cache; poses are rebuilt from phase2a raw action JSON.",
    }
    write_json(out / "phase2a_materialize_meta.json", meta)
    return out


def load_adapter_checkpoint(path: Path, model: torch.nn.Module, encoder: torch.nn.Module) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("state", {})
    adapter_state = state.get("memory_dense_adapter", {})
    lora_state = state.get("memory_dense_lora", {})
    encoder_state = state.get("memory_dense_encoder", {})
    if not adapter_state or not encoder_state:
        raise ValueError(f"{path}: missing adapter or encoder state")
    missing, unexpected = model.load_state_dict(adapter_state, strict=False)
    missing_adapter = [
        name for name in missing
        if "memory_dense_adapter" in name and not name.endswith("memory_dense_adapter_scale")
    ]
    unexpected_adapter = [name for name in unexpected if "memory_dense_adapter" in name]
    if missing_adapter or unexpected_adapter:
        raise ValueError(
            f"adapter mismatch missing={missing_adapter[:10]} unexpected={unexpected_adapter[:10]}"
        )
    if lora_state:
        model_state = dict(model.named_parameters())
        missing_lora = [name for name in lora_state if name not in model_state]
        if missing_lora:
            raise ValueError(f"{path}: LoRA state cannot be loaded; missing={missing_lora[:10]}")
        for name, value in lora_state.items():
            model_state[name].data.copy_(value.to(device=model_state[name].device, dtype=model_state[name].dtype))
    encoder.load_state_dict(encoder_state)
    return ckpt


def build_blank_release_samples(samples: list[Any], load_dense_fn) -> list[Any]:
    out = []
    for sample in samples:
        blank = np.zeros_like(load_dense_fn(sample), dtype=np.float32)
        cloned = copy.copy(sample)
        object.__setattr__(cloned, "_memory_dense_override", blank)
        out.append(cloned)
    return out


def dense_tokens_for_sequence(
    *,
    encoder: torch.nn.Module,
    samples: list[Any],
    load_dense_fn,
    device: torch.device,
    dtype: torch.dtype,
    target_token_hw: tuple[int, int],
) -> tuple[torch.Tensor, tuple[int, int]]:
    dense_np = []
    for sample in samples:
        if hasattr(sample, "_memory_dense_override"):
            dense_np.append(np.asarray(getattr(sample, "_memory_dense_override"), dtype=np.float32))
        else:
            dense_np.append(np.asarray(load_dense_fn(sample), dtype=np.float32))
    dense = torch.from_numpy(np.stack(dense_np, axis=0)).to(device=device, dtype=dtype)
    tokens, token_hw = encoder(dense, target_token_hw=target_token_hw)
    tokens = tokens.reshape(1, len(samples) * tokens.shape[1], tokens.shape[2]).to(dtype)
    return tokens, token_hw


def patch_model_forward_with_dense_tokens(model: torch.nn.Module, dense_tokens: torch.Tensor, tokens_per_frame: int) -> None:
    original_forward = model.forward

    def forward_with_dense(*args: Any, **kwargs: Any):
        dit_cond = dict(kwargs.get("dit_cond_dict") or {})
        x_arg = kwargs.get("x")
        if x_arg is None and args:
            # WanModelFast forward is usually called with keyword x, but keep a fallback.
            x_arg = args[0]
        if x_arg is None:
            raise RuntimeError("cannot infer current latent chunk for dense adapter injection")
        current = x_arg[0] if isinstance(x_arg, list) else x_arg
        chunk_frames = int(current.shape[1])
        current_start = int(kwargs.get("current_start", 0) or 0)
        frame_start = current_start // int(tokens_per_frame)
        token_start = frame_start * int(tokens_per_frame)
        token_end = token_start + chunk_frames * int(tokens_per_frame)
        cond = dense_tokens[:, token_start:token_end]
        expected = chunk_frames * int(tokens_per_frame)
        if cond.shape[1] != expected:
            raise RuntimeError(
                f"dense token slice {tuple(cond.shape)} does not match chunk_frames={chunk_frames}, "
                f"tokens_per_frame={tokens_per_frame}, frame_start={frame_start}"
            )
        dit_cond[COND_KEY] = cond.to(device=current.device, dtype=current.dtype)
        kwargs["dit_cond_dict"] = dit_cond
        return original_forward(*args, **kwargs)

    model.forward = forward_with_dense  # type: ignore[method-assign]


def frame_num_for_latent_frames(latent_frames: int) -> int:
    return (int(latent_frames) - 1) * 4 + 1


def effective_latent_frames_for_chunking(latent_frames: int, chunk_size: int) -> int:
    # LingBot Fast processes latent time in complete chunks. For lf=41, chunk_size=3,
    # rollout uses 39 latent frames and saves 153 video frames.
    latent_frames = int(latent_frames)
    chunk_size = int(chunk_size)
    if chunk_size <= 1:
        return latent_frames
    effective = (latent_frames // chunk_size) * chunk_size
    return max(chunk_size, effective)


def make_side_by_side(out_path: Path, labels_and_paths: list[tuple[str, Path]]) -> None:
    inputs: list[str] = []
    filter_parts: list[str] = []
    for idx, (label, path) in enumerate(labels_and_paths):
        inputs += ["-i", str(path)]
        safe = label.replace("'", "")
        filter_parts.append(
            f"[{idx}:v]drawtext=text='{safe}':x=12:y=12:fontsize=24:fontcolor=white:"
            "box=1:boxcolor=black@0.55[v{idx}]".format(idx=idx)
        )
    stacked = "".join(f"[v{i}]" for i in range(len(labels_and_paths))) + f"hstack=inputs={len(labels_and_paths)}[v]"
    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(filter_parts + [stacked]), "-map", "[v]", "-an", str(out_path)]
    import subprocess
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map-manifest", type=Path, required=True)
    ap.add_argument("--cache-manifest", type=Path, required=True)
    ap.add_argument("--phase2a-source-manifest", type=Path, default=None)
    ap.add_argument("--phase2a-window-id", default=None)
    ap.add_argument("--phase2a-ego-stem", default=None)
    ap.add_argument("--phase2a-materialized-root", type=Path, default=Path("output/memory_dense_adapter_v0/phase2a_materialized_clips_v0"))
    ap.add_argument("--adapter-checkpoint", type=Path, required=True)
    ap.add_argument("--lingbot-repo", type=Path, default=Path(str(source_path('lingbot', ''))))
    ap.add_argument("--ckpt-dir", type=Path, default=Path(str(source_path('assets', 'lingbot-world-base-cam'))))
    ap.add_argument("--clip-root", type=Path, default=Path(str(source_path('assets', 'datasets/memory_light_dust2_pilot_1200_v0/clips'))))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--clip-index", type=int, default=0)
    ap.add_argument("--clip-id", default=None)
    ap.add_argument("--variant", choices=["base", "true_dense", "blank_dense", "shuffled_dense"], default="true_dense")
    ap.add_argument("--make-triptych", action="store_true")
    ap.add_argument("--latent-frames", type=int, default=3)
    ap.add_argument("--chunk-size", type=int, default=3)
    ap.add_argument("--sample-shift", type=float, default=10.0)
    ap.add_argument("--timesteps-index", type=int, nargs="+", default=[0, 358, 679])
    ap.add_argument("--seed", type=int, default=20260609)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--offload-model", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--max-attention-size", type=int, default=None)
    ap.add_argument("--target-height", type=int, default=480,
                    help="Force generation/input height. Defaults to 480 because the bare "
                         "MAX_AREA path floors 480 to 464 and yields a 29x52 token grid "
                         "that training never saw; pass explicitly to override.")
    ap.add_argument("--target-width", type=int, default=832,
                    help="Force generation/input width. Set together with --target-height; "
                         "both must be divisible by 16.")
    args = ap.parse_args()

    if args.target_height is not None or args.target_width is not None:
        if args.target_height is None or args.target_width is None:
            raise ValueError("--target-height and --target-width must be set together")
        if args.target_height % 16 or args.target_width % 16:
            raise ValueError("--target-height/--target-width must be divisible by 16")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    add_path(Path.cwd() / "tools")
    add_path(args.lingbot_repo)

    from map_memory_training_data_v0 import MapMemoryRelease, load_dense
    from memory_dense_wan_adapter_v0 import (
        MemoryDenseAdapterConfig,
        MemoryDenseFrozenVAEWanTokenEncoder,
        MemoryDenseWanTokenEncoder,
        inject_lora_into_wan_model_fast,
        resolve_dense_native_hw,
        wrap_wan_model_fast_with_memory_dense_adapter,
    )
    from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
    from wan.image2video_fast import WanI2VFast
    from wan.utils.utils import save_video

    source_mode = "aligned_cache"
    dense_load_fn = load_dense
    condition_sample_ids: list[str] | None = None
    shuffled_condition_source: dict[str, Any] | None = None
    if args.phase2a_source_manifest:
        source_mode = "phase2a_window"
        record = choose_phase2a_source_row(
            args.phase2a_source_manifest,
            window_id=args.phase2a_window_id,
            ego_stem=args.phase2a_ego_stem,
            clip_index=args.clip_index,
        )
        dense_sequence_manifest = Path(record["phase2a_dense_sequence_manifest"])
        samples, sample_ids = load_phase2a_dense_samples(dense_sequence_manifest, args.latent_frames)
        condition_sample_ids = list(sample_ids)
        dense_load_fn = load_dense_path_sample
        clip_dir = materialize_phase2a_clip_dir(
            row=record,
            out_root=args.phase2a_materialized_root,
            latent_frames=args.latent_frames,
            chunk_size=args.chunk_size,
            source_clip_root=args.clip_root,
        )
        if args.variant == "shuffled_dense":
            shuffled_row = choose_phase2a_shuffled_source_row(
                args.phase2a_source_manifest,
                current_clip_id=str(record["clip_id"]),
            )
            shuffled_manifest = Path(shuffled_row["phase2a_dense_sequence_manifest"])
            samples, condition_sample_ids = load_phase2a_dense_samples(shuffled_manifest, args.latent_frames)
            shuffled_condition_source = {
                "clip_id": shuffled_row.get("clip_id"),
                "phase2a_window_id": shuffled_row.get("phase2a_window_id"),
                "player_stem": shuffled_row.get("player_stem"),
                "dense_sequence_manifest": str(shuffled_manifest),
            }
    else:
        records = read_jsonl(args.cache_manifest)
        record = choose_record(records, split=args.split, clip_index=args.clip_index, clip_id=args.clip_id)
        sample_ids = list(record["map_memory_sample_ids"])[: args.latent_frames]
        condition_sample_ids = list(sample_ids)
        release = MapMemoryRelease.load(args.map_manifest, validate_sidecars=False)
        samples = [release.by_id[sid] for sid in sample_ids]
        clip_dir = resolve_clip_dir(record, args.clip_root)
        if args.variant == "shuffled_dense":
            shuffled_record = choose_shuffled_record(records, current=record, split=args.split)
            condition_sample_ids = list(shuffled_record["map_memory_sample_ids"])[: args.latent_frames]
            samples = [release.by_id[sid] for sid in condition_sample_ids]
            shuffled_condition_source = {
                "clip_id": shuffled_record.get("clip_id"),
                "split": shuffled_record.get("map_memory_split", shuffled_record.get("split")),
            }
    if args.variant == "blank_dense":
        samples = build_blank_release_samples(samples, dense_load_fn)

    image_path = clip_dir / "image.jpg"
    prompt_path = clip_dir / "prompt.txt"
    prompt = prompt_path.read_text(encoding="utf-8").strip() if prompt_path.exists() else record.get("prompt", "")
    image = Image.open(image_path).convert("RGB")
    source_image_size = [int(image.width), int(image.height)]
    gen_max_area = None
    if args.target_height is not None and args.target_width is not None:
        if image.size != (args.target_width, args.target_height):
            image = image.resize((args.target_width, args.target_height), Image.LANCZOS)
        # LingBot internally recomputes height from max_area and aspect ratio.
        # The epsilon avoids 480 becoming 479.999... and then floor-dividing to 464.
        gen_max_area = math.ceil(args.target_height * args.target_width * 1.0001)

    cfg = WAN_CONFIGS["i2v-A14B"]
    device_id = int(str(args.device).split(":")[-1]) if str(args.device).startswith("cuda") else 0
    pipe = WanI2VFast(
        config=cfg,
        checkpoint_dir=str(args.ckpt_dir),
        device_id=device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        convert_model_dtype=False,
    )

    requested_frame_num = frame_num_for_latent_frames(args.latent_frames)
    effective_latent_frames = effective_latent_frames_for_chunking(args.latent_frames, args.chunk_size)
    effective_frame_num = frame_num_for_latent_frames(effective_latent_frames)

    adapter_loaded = False
    if args.variant != "base":
        ckpt = torch.load(args.adapter_checkpoint, map_location="cpu")
        ckpt_config = ckpt.get("config") or {}
        adapter_cfg = ckpt_config.get("adapter_config") or {}
        config = MemoryDenseAdapterConfig(
            dense_channels=int(adapter_cfg.get("dense_channels", 7)),
            cond_dim=int(adapter_cfg.get("cond_dim", 128)),
            encoder_hidden_dim=int(adapter_cfg.get("encoder_hidden_dim", 64)),
            adapter_hidden_dim=int(adapter_cfg.get("adapter_hidden_dim", 512)),
            vae_stride=int(adapter_cfg.get("vae_stride", 8)),
            wan_patch_size_hw=int(adapter_cfg.get("wan_patch_size_hw", 2)),
            cond_key=str(adapter_cfg.get("cond_key", COND_KEY)),
            residual_mode=str(adapter_cfg.get("residual_mode", "cond_gated")),
            wrap_first_blocks=adapter_cfg.get("wrap_first_blocks"),
            residual_scale_init=float(adapter_cfg.get("residual_scale_init", 1.0)),
            activation_checkpoint_blocks=False,
            highfreq_branch=bool(adapter_cfg.get("highfreq_branch", False)),
            highfreq_kind=str(adapter_cfg.get("highfreq_kind", "laplacian")),
            highfreq_hidden_dim=int(adapter_cfg.get("highfreq_hidden_dim", 32)),
            highfreq_stride=int(adapter_cfg.get("highfreq_stride", 16)),
        )
        pipe.model = wrap_wan_model_fast_with_memory_dense_adapter(pipe.model, config, freeze_base=True)
        lora_cfg = ckpt_config.get("lora_config") or {}
        if int(lora_cfg.get("rank", 0) or 0) > 0:
            inject_lora_into_wan_model_fast(
                pipe.model,
                rank=int(lora_cfg.get("rank", 0)),
                targets=str(lora_cfg.get("targets", "attn,mlp")),
                alpha=lora_cfg.get("alpha"),
                dropout=float(lora_cfg.get("dropout", 0.0)),
            )
        dense_encoder_cfg = ckpt_config.get("dense_condition_encoder") or {"type": "conv"}
        if dense_encoder_cfg.get("type") == "frozen_vae":
            from wan.modules.vae2_1 import Wan2_1_VAE

            vae_pth = Path(dense_encoder_cfg.get("vae_pth") or (args.ckpt_dir / cfg.vae_checkpoint))
            if not vae_pth.exists():
                raise FileNotFoundError(f"missing frozen dense VAE checkpoint: {vae_pth}")
            vae = Wan2_1_VAE(vae_pth=str(vae_pth), device=pipe.device)
            encoder = MemoryDenseFrozenVAEWanTokenEncoder(
                config,
                vae=vae,
                vae_pth=vae_pth,
                native_hw=resolve_dense_native_hw(ckpt_config),
                packing=str(dense_encoder_cfg.get("packing", "img1_mask_img2_player_v0")),
            ).to(device=pipe.device, dtype=pipe.param_dtype)
        elif dense_encoder_cfg.get("type", "conv") == "conv":
            encoder = MemoryDenseWanTokenEncoder(config).to(device=pipe.device, dtype=pipe.param_dtype)
        else:
            raise ValueError(f"unsupported dense condition encoder in checkpoint: {dense_encoder_cfg}")
        load_adapter_checkpoint(args.adapter_checkpoint, pipe.model, encoder)
        pipe.model.to(pipe.device).eval()
        encoder.eval()

        if args.target_height is not None and args.target_width is not None:
            lat_h = args.target_height // pipe.vae_stride[1]
            lat_w = args.target_width // pipe.vae_stride[2]
        else:
            h0, w0 = image.height, image.width
            aspect_ratio = h0 / w0
            max_area = MAX_AREA_CONFIGS["832*480"]
            lat_h = round(
                math.sqrt(max_area * aspect_ratio) // pipe.vae_stride[1] //
                pipe.patch_size[1] * pipe.patch_size[1]
            )
            lat_w = round(
                math.sqrt(max_area / aspect_ratio) // pipe.vae_stride[2] //
                pipe.patch_size[2] * pipe.patch_size[2]
            )
        token_hw = (lat_h // pipe.patch_size[1], lat_w // pipe.patch_size[2])
        dense_tokens, dense_token_hw = dense_tokens_for_sequence(
            encoder=encoder,
            samples=samples,
            load_dense_fn=dense_load_fn,
            device=pipe.device,
            dtype=pipe.param_dtype,
            target_token_hw=token_hw,
        )
        tokens_per_frame = token_hw[0] * token_hw[1]
        if dense_token_hw != token_hw:
            raise RuntimeError(f"dense token hw {dense_token_hw} != target {token_hw}")
        patch_model_forward_with_dense_tokens(pipe.model, dense_tokens, tokens_per_frame)
        adapter_loaded = True

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    out_video = args.out_dir / f"{record['clip_id']}_{args.variant}_lf{args.latent_frames}_seed{args.seed}.mp4"
    video = pipe.generate(
        prompt,
        image,
        action_path=str(clip_dir),
        chunk_size=args.chunk_size,
        max_area=gen_max_area if gen_max_area is not None else MAX_AREA_CONFIGS["832*480"],
        frame_num=effective_frame_num,
        timesteps_index=args.timesteps_index,
        shift=args.sample_shift,
        seed=args.seed,
        offload_model=args.offload_model,
        max_attention_size=args.max_attention_size,
    )
    save_video(video[None], save_file=str(out_video), fps=cfg.sample_fps, nrow=1, normalize=True, value_range=(-1, 1))

    report = {
        "kind": "phase2a_memory_dense_sampler_smoke_v0",
        "status": "complete",
        "source_mode": source_mode,
        "variant": args.variant,
        "adapter_loaded": adapter_loaded,
        "clip_id": record["clip_id"],
        "split": record.get("map_memory_split", record.get("split")),
        "clip_dir": str(clip_dir),
        "image": str(image_path),
        "source_image_size": source_image_size,
        "input_image_size": [int(image.width), int(image.height)],
        "target_height": args.target_height,
        "target_width": args.target_width,
        "gen_max_area": gen_max_area,
        "prompt": prompt,
        "sample_ids": sample_ids,
        "condition_sample_ids": condition_sample_ids,
        "shuffled_condition_source": shuffled_condition_source,
        "requested_latent_frames": args.latent_frames,
        "effective_latent_frames": effective_latent_frames,
        "requested_frame_num": requested_frame_num,
        "effective_frame_num": effective_frame_num,
        "latent_frames": args.latent_frames,
        "frame_num": effective_frame_num,
        "chunk_size": args.chunk_size,
        "timesteps_index": args.timesteps_index,
        "sample_shift": args.sample_shift,
        "seed": args.seed,
        "out_video": str(out_video),
    }
    if args.variant != "base":
        report.update({
            "latent_hw": [int(lat_h), int(lat_w)],
            "token_hw": [int(token_hw[0]), int(token_hw[1])],
        })
    write_json(args.out_dir / f"{record['clip_id']}_{args.variant}_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
