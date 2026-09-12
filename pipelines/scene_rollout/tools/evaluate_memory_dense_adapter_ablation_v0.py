#!/usr/bin/env python3
"""Evaluate whether Memory dense conditioning improves GT-latent denoising.

This is a held-out latent-space ablation, not a video-generation metric.  It
compares base LingBot against the trained Memory dense adapter with true,
blank, and shuffled dense tokens under identical noise/timestep draws.
"""

from __future__ import annotations
from runtime_paths import source_path

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from train_memory_dense_adapter_v0 import (
    COND_KEY,
    build_adapter_config,
    build_chunk_starts,
    dense_tokens_for_samples,
    import_wan_model_fast,
    load_and_validate_training_inputs,
    load_latent_pair,
    load_text_context,
    prepare_cam_chunk,
    resolve_record_sample_ids,
)
from teacher_player_latent_mask_v0 import teacher_player_mask_latent
from memory_dense_wan_adapter_v0 import (
    MemoryDenseWanTokenEncoder,
    wrap_wan_model_fast_with_memory_dense_adapter,
)
from map_memory_training_data_v0 import MapMemorySample, load_dense
from map_memory_training_data_v0 import sha256_file


VARIANTS = ["base", "true_dense", "blank_dense", "shuffled_dense", "player_shuffle"]
PLAYER_CHANNELS = (3, 4, 5, 6)
REGION_MASK_NOTE = "dense-space teacher player mask, area-pooled to latent grid; ~5% aspect offset, relative-only"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_adapter_checkpoint(path: Path, model: torch.nn.Module, encoder: torch.nn.Module) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("state", {})
    adapter_state = state.get("memory_dense_adapter", {})
    encoder_state = state.get("memory_dense_encoder", {})
    if not adapter_state:
        raise ValueError(f"{path}: checkpoint has no memory_dense_adapter state")
    if not encoder_state:
        raise ValueError(f"{path}: checkpoint has no memory_dense_encoder state")
    missing, unexpected = model.load_state_dict(adapter_state, strict=False)
    missing_adapter = [
        name
        for name in missing
        if "memory_dense_adapter" in name and not name.endswith("memory_dense_adapter_scale")
    ]
    unexpected_adapter = [name for name in unexpected if "memory_dense_adapter" in name]
    if missing_adapter or unexpected_adapter:
        raise ValueError(
            f"{path}: adapter checkpoint mismatch missing={missing_adapter[:10]} unexpected={unexpected_adapter[:10]}"
        )
    encoder.load_state_dict(encoder_state)
    return {"checkpoint": str(path), "step": int(ckpt.get("step", 0)), "kind": ckpt.get("kind")}


def make_eval_items(records: list[dict[str, Any]], release, args: argparse.Namespace) -> list[dict[str, Any]]:
    chunk_starts = build_chunk_starts(args.latent_frames, args.chunk_size)
    items: list[dict[str, Any]] = []
    max_records = min(len(records), args.max_records if args.max_records > 0 else len(records))
    for record_index, record in enumerate(records[:max_records]):
        sample_ids = resolve_record_sample_ids(
            record,
            latent_frames=args.latent_frames,
            allow_static_dense_repeat=args.allow_static_dense_repeat,
        )
        for chunk_ord, chunk_start in enumerate(chunk_starts):
            chunk_ids = sample_ids[chunk_start : chunk_start + args.chunk_size]
            roles = [release.by_id[sid].selection_role for sid in chunk_ids]
            if args.positive_chunks_only and "positive" not in roles:
                continue
            items.append(
                {
                    "record_index": record_index,
                    "record": record,
                    "chunk_ord": chunk_ord,
                    "chunk_start": chunk_start,
                    "sample_ids": chunk_ids,
                    "roles": roles,
                }
            )
            if args.max_chunks > 0 and len(items) >= args.max_chunks:
                return items
    return items


def choose_shuffled_samples(items: list[dict[str, Any]], index: int, release, args: argparse.Namespace):
    if len(items) < 2:
        return [release.by_id[sid] for sid in items[index]["sample_ids"]]
    other = items[(index + args.shuffle_offset) % len(items)]
    if other is items[index] and len(items) > 1:
        other = items[(index + args.shuffle_offset + 1) % len(items)]
    return [release.by_id[sid] for sid in other["sample_ids"]]


def make_player_shuffle_samples(true_samples: list[MapMemorySample], shuffled_samples: list[MapMemorySample]) -> list[MapMemorySample]:
    if len(true_samples) != len(shuffled_samples):
        raise ValueError("true and shuffled sample chunks must have equal length")
    out: list[MapMemorySample] = []
    for true_sample, shuffled_sample in zip(true_samples, shuffled_samples):
        true_dense = load_dense(true_sample)
        shuffled_dense = load_dense(shuffled_sample)
        mixed_dense = np.array(true_dense, copy=True)
        mixed_dense[list(PLAYER_CHANNELS)] = shuffled_dense[list(PLAYER_CHANNELS)]
        row = dict(true_sample.row)
        row["_player_shuffle_source_sample_id"] = shuffled_sample.sample_id
        sample = MapMemorySample(
            sample_id=f"{true_sample.sample_id}__player_from__{shuffled_sample.sample_id}",
            match_id=true_sample.match_id,
            episode=true_sample.episode,
            raw_episode=true_sample.raw_episode,
            ego_stem=true_sample.ego_stem,
            frame_index=true_sample.frame_index,
            selection_role=true_sample.selection_role,
            dense_path=true_sample.dense_path,
            target_rgb_path=true_sample.target_rgb_path,
            meta_path=true_sample.meta_path,
            qa_path=true_sample.qa_path,
            row=row,
        )
        object.__setattr__(sample, "_memory_dense_override", mixed_dense)
        out.append(sample)
    return out


def dense_tokens_for_samples_with_overrides(
    encoder: torch.nn.Module,
    samples: list[MapMemorySample],
    *,
    device: torch.device,
    dtype: torch.dtype,
    target_token_hw: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, tuple[int, int]]:
    if not any(hasattr(sample, "_memory_dense_override") for sample in samples):
        return dense_tokens_for_samples(
            encoder,
            samples,
            device=device,
            dtype=dtype,
            target_token_hw=target_token_hw,
        )
    dense_items = []
    for sample in samples:
        if hasattr(sample, "_memory_dense_override"):
            dense_items.append(np.asarray(getattr(sample, "_memory_dense_override"), dtype=np.float32))
        else:
            dense_items.append(load_dense(sample))
    dense_np = np.stack(dense_items, axis=0)
    if getattr(encoder, "only_player_dense_channels", False):
        dense_np[:, :3] = 0.0
    dense = torch.from_numpy(dense_np).to(device=device, dtype=dtype)
    tokens, token_hw = encoder(dense, target_token_hw=target_token_hw)
    tokens = tokens.reshape(1, len(samples) * tokens.shape[1], tokens.shape[2]).to(dtype)
    return tokens, token_hw


def region_loss_for_chunk(
    pred: torch.Tensor,
    target: torch.Tensor,
    samples: list[MapMemorySample],
    latent_hw: tuple[int, int],
) -> tuple[torch.Tensor, int]:
    masks = [teacher_player_mask_latent(sample, latent_hw=latent_hw) for sample in samples]
    mask_np = np.stack(masks, axis=0).astype(bool)
    mask = torch.from_numpy(mask_np).to(device=pred.device)
    region_pixels = int(mask.sum().detach().cpu())
    if region_pixels <= 0:
        return torch.tensor(float("nan"), device=pred.device), 0
    mask_c = mask.unsqueeze(0).expand(pred.shape[0], -1, -1, -1)
    return F.mse_loss(pred.float()[mask_c], target.float()[mask_c]), region_pixels


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[str(row["variant"])].append(row)
    out: dict[str, Any] = {}
    for variant, items in sorted(by_variant.items()):
        losses = np.asarray([float(row["loss"]) for row in items], dtype=np.float64)
        pos_losses = np.asarray([float(row["loss"]) for row in items if int(row["positive_frames"]) > 0], dtype=np.float64)
        ctx_losses = np.asarray([float(row["loss"]) for row in items if int(row["positive_frames"]) == 0], dtype=np.float64)
        region_items = [
            row
            for row in items
            if int(row.get("region_pixels", 0) or 0) > 0 and np.isfinite(float(row.get("region_loss", float("nan"))))
        ]
        region_losses = np.asarray([float(row["region_loss"]) for row in region_items], dtype=np.float64)
        out[variant] = {
            "count": int(len(items)),
            "loss_mean": float(losses.mean()) if len(losses) else None,
            "loss_median": float(np.median(losses)) if len(losses) else None,
            "loss_std": float(losses.std(ddof=0)) if len(losses) else None,
            "whole_frame_loss_mean": float(losses.mean()) if len(losses) else None,
            "whole_frame_loss_median": float(np.median(losses)) if len(losses) else None,
            "whole_frame_loss_std": float(losses.std(ddof=0)) if len(losses) else None,
            "positive_chunk_count": int(len(pos_losses)),
            "positive_loss_mean": float(pos_losses.mean()) if len(pos_losses) else None,
            "context_only_chunk_count": int(len(ctx_losses)),
            "context_only_loss_mean": float(ctx_losses.mean()) if len(ctx_losses) else None,
            "region_chunk_count": int(len(region_losses)),
            "region_loss_mean": float(region_losses.mean()) if len(region_losses) else None,
            "region_loss_median": float(np.median(region_losses)) if len(region_losses) else None,
            "region_loss_std": float(region_losses.std(ddof=0)) if len(region_losses) else None,
            "region_pixels_total": int(sum(int(row.get("region_pixels", 0) or 0) for row in items)),
        }
    pooled_region_losses = [
        float(row["region_loss"])
        for row in rows
        if str(row["variant"]) in {"true_dense", "blank_dense", "shuffled_dense", "player_shuffle"}
        and int(row.get("region_pixels", 0) or 0) > 0
        and np.isfinite(float(row.get("region_loss", float("nan"))))
    ]
    if pooled_region_losses:
        region_array = np.asarray(pooled_region_losses, dtype=np.float64)
        out["region_loss_pooled_std"] = float(region_array.std(ddof=0))
    return out


def gate_summary(summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    failures: list[str] = []
    checks: dict[str, Any] = {}
    true = summary.get("true_dense") or {}
    if true.get("region_loss_mean") is None:
        failures.append("missing true_dense region_loss_mean")
        return {"status": "fail", "checks": checks, "failures": failures}
    true_loss = float(true["region_loss_mean"])
    pooled_std = float(summary.get("region_loss_pooled_std") or 0.0)
    required_gap = 0.5 * pooled_std
    for baseline in ["blank_dense", "shuffled_dense"]:
        base = summary.get(baseline) or {}
        if base.get("region_loss_mean") is None:
            failures.append(f"missing {baseline} region_loss_mean")
            continue
        base_loss = float(base["region_loss_mean"])
        gap = base_loss - true_loss
        checks[baseline] = {
            "baseline_region_loss": base_loss,
            "true_region_loss": true_loss,
            "gap_baseline_minus_true": gap,
            "region_loss_pooled_std": pooled_std,
            "required_gap_0_5x_std": required_gap,
            "pass": gap >= required_gap,
        }
        if gap < required_gap:
            failures.append(
                f"true_dense region gap over {baseline} {gap:.6f} < required {required_gap:.6f}"
            )
    if args.require_positive_chunks and int(true.get("region_chunk_count", 0) or 0) <= 0:
        failures.append("no positive region chunks evaluated")
    return {"status": "pass" if not failures else "fail", "checks": checks, "failures": failures}


def eval_model(
    *,
    variant: str,
    model: torch.nn.Module,
    encoder: MemoryDenseWanTokenEncoder | None,
    release,
    items: list[dict[str, Any]],
    latent_hw: tuple[int, int],
    wan_token_hw: tuple[int, int],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text_context_cache: dict[str, torch.Tensor] = {}
    model.eval()
    if encoder is not None:
        encoder.eval()
    for idx, item in enumerate(items):
        record = item["record"]
        chunk_start = int(item["chunk_start"])
        chunk_samples = [release.by_id[sid] for sid in item["sample_ids"]]
        x0, cond = load_latent_pair(record, device, dtype)
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
        text_key = str(record["text_cache"])
        if text_key not in text_context_cache:
            text_context_cache[text_key] = load_text_context(record, device)
        text_context = text_context_cache[text_key]
        generator = torch.Generator(device=device).manual_seed(args.noise_seed + idx)
        noise = torch.randn(x0_chunk.shape, generator=generator, device=device, dtype=torch.float32).to(dtype)
        sigma_value = ((idx * 1103515245 + args.sigma_seed) % 10000 + 0.5) / 10000.0
        sigma = torch.tensor(float(sigma_value), device=device, dtype=torch.float32)
        timestep = (sigma * 1000.0).reshape(1)
        xt = ((1.0 - sigma) * x0_chunk.float() + sigma * noise.float()).to(dtype)
        target = noise.float() - x0_chunk.float()
        latent_h, latent_w = x0_chunk.shape[-2:]
        wan_h = latent_h // args.patch_size_hw
        wan_w = latent_w // args.patch_size_hw
        if (latent_h, latent_w) != latent_hw or (wan_h, wan_w) != wan_token_hw:
            raise RuntimeError(f"{record.get('clip_id')}: validated/runtime grid mismatch")
        dit_cond_dict = {"c2ws_plucker_emb": cam_chunk.chunk(1, dim=0)}
        dense_policy = "none"
        if variant != "base":
            if encoder is None:
                raise RuntimeError(f"{variant}: encoder is required")
            if variant == "true_dense":
                dense_samples = chunk_samples
                dense_tokens, _ = dense_tokens_for_samples_with_overrides(
                    encoder,
                    dense_samples,
                    device=device,
                    dtype=dtype,
                    target_token_hw=wan_token_hw,
                )
                dense_policy = "true"
            elif variant == "blank_dense":
                dense_tokens, _ = dense_tokens_for_samples_with_overrides(
                    encoder,
                    chunk_samples,
                    device=device,
                    dtype=dtype,
                    target_token_hw=wan_token_hw,
                )
                dense_tokens = torch.zeros_like(dense_tokens)
                dense_policy = "blank_zero_tokens"
            elif variant == "shuffled_dense":
                dense_samples = choose_shuffled_samples(items, idx, release, args)
                dense_tokens, _ = dense_tokens_for_samples_with_overrides(
                    encoder,
                    dense_samples,
                    device=device,
                    dtype=dtype,
                    target_token_hw=wan_token_hw,
                )
                dense_policy = "shuffled"
            elif variant == "player_shuffle":
                shuffled_samples = choose_shuffled_samples(items, idx, release, args)
                dense_samples = make_player_shuffle_samples(chunk_samples, shuffled_samples)
                dense_tokens, _ = dense_tokens_for_samples_with_overrides(
                    encoder,
                    dense_samples,
                    device=device,
                    dtype=dtype,
                    target_token_hw=wan_token_hw,
                )
                dense_policy = "player_channels_shuffled_env_true"
            else:
                raise ValueError(f"unknown variant {variant!r}")
            dit_cond_dict[COND_KEY] = dense_tokens
        seq_len = args.chunk_size * wan_h * wan_w
        current_start = chunk_start * wan_h * wan_w
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda" and dtype == torch.bfloat16):
            pred = model(
                x=[xt],
                t=timestep,
                context=[text_context],
                seq_len=seq_len,
                y=[cond_chunk],
                dit_cond_dict=dit_cond_dict,
                kv_cache=None,
                crossattn_cache=None,
                current_start=current_start,
                max_attention_size=seq_len,
            )[0]
            loss = F.mse_loss(pred.float(), target.float())
            region_loss, region_pixels = region_loss_for_chunk(pred, target, chunk_samples, latent_hw)
        roles = list(item["roles"])
        rows.append(
            {
                "variant": variant,
                "dense_policy": dense_policy,
                "clip_id": record.get("clip_id"),
                "record_index": int(item["record_index"]),
                "chunk_ord": int(item["chunk_ord"]),
                "chunk_start": chunk_start,
                "sample_ids": item["sample_ids"],
                "roles": roles,
                "positive_frames": int(sum(1 for role in roles if role == "positive")),
                "context_frames": int(sum(1 for role in roles if role == "context")),
                "sigma": float(sigma.detach().cpu()),
                "loss": float(loss.detach().cpu()),
                "whole_frame_loss": float(loss.detach().cpu()),
                "region_loss": float(region_loss.detach().cpu()),
                "region_pixels": int(region_pixels),
            }
        )
        del x0, cond, x0_chunk, cond_chunk, cam_chunk, noise, timestep, xt, target, pred, loss, region_loss
        if variant != "base":
            del dense_tokens
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.train_split = args.split
    release, records, latent_hw, wan_token_hw, geometry_report, release_gate = load_and_validate_training_inputs(args)
    items = make_eval_items(records, release, args)
    if not items:
        raise ValueError(f"no eval chunks selected for split={args.split!r}")
    WanModelFast = import_wan_model_fast(args.lingbot_repo)
    model_dir = args.fast_model_dir or args.ckpt_dir / "lingbot_world_fast"
    all_rows: list[dict[str, Any]] = []

    if "base" in args.variants:
        base_model = WanModelFast.from_pretrained(str(model_dir), torch_dtype=dtype, control_type="cam").to(device)
        base_model.requires_grad_(False)
        all_rows.extend(
            eval_model(
                variant="base",
                model=base_model,
                encoder=None,
                release=release,
                items=items,
                latent_hw=latent_hw,
                wan_token_hw=wan_token_hw,
                args=args,
                device=device,
                dtype=dtype,
            )
        )
        del base_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    adapter_variants = [variant for variant in args.variants if variant != "base"]
    checkpoint_info = None
    if adapter_variants:
        if args.adapter_checkpoint is None:
            raise ValueError("--adapter-checkpoint is required for dense variants")
        adapter_model = WanModelFast.from_pretrained(str(model_dir), torch_dtype=dtype, control_type="cam").to(device)
        adapter_model.requires_grad_(False)
        config = build_adapter_config(args)
        wrap_wan_model_fast_with_memory_dense_adapter(adapter_model, config, freeze_base=True)
        adapter_model.to(device=device, dtype=dtype)
        encoder = MemoryDenseWanTokenEncoder(config).to(device=device, dtype=dtype)
        encoder.only_player_dense_channels = bool(args.only_player_dense_channels)
        checkpoint_info = load_adapter_checkpoint(args.adapter_checkpoint, adapter_model, encoder)
        for variant in adapter_variants:
            all_rows.extend(
                eval_model(
                    variant=variant,
                    model=adapter_model,
                    encoder=encoder,
                    release=release,
                    items=items,
                    latent_hw=latent_hw,
                    wan_token_hw=wan_token_hw,
                    args=args,
                    device=device,
                    dtype=dtype,
                )
            )
        del adapter_model, encoder

    summary = summarize(all_rows)
    gate = gate_summary(summary, args)
    result = {
        "kind": "memory_dense_adapter_ablation_eval_v0",
        "status": gate["status"],
        "split": args.split,
        "variants": args.variants,
        "map_manifest": str(args.map_manifest),
        "cache_manifest": str(args.cache_manifest),
        "map_manifest_sha256": sha256_file(args.map_manifest),
        "adapter_checkpoint": str(args.adapter_checkpoint) if args.adapter_checkpoint else None,
        "checkpoint_info": checkpoint_info,
        "record_count": len(records),
        "eval_chunk_count": len(items),
        "positive_chunks_only": bool(args.positive_chunks_only),
        "only_player_dense_channels": bool(args.only_player_dense_channels),
        "region_mask_note": REGION_MASK_NOTE,
        "geometry_report": geometry_report,
        "release_gate": release_gate,
        "summary": summary,
        "gate": gate,
        "rows": all_rows,
    }
    if args.out_json is not None:
        write_json(args.out_json, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map-manifest", type=Path, required=True)
    ap.add_argument("--cache-manifest", type=Path, required=True)
    ap.add_argument("--adapter-checkpoint", type=Path, default=None)
    ap.add_argument("--lingbot-repo", type=Path, default=Path(str(source_path('lingbot', ''))))
    ap.add_argument("--ckpt-dir", type=Path, default=Path(str(source_path('assets', 'lingbot-world-base-cam'))))
    ap.add_argument("--fast-model-dir", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260531)
    ap.add_argument("--noise-seed", type=int, default=20260603)
    ap.add_argument("--sigma-seed", type=int, default=314159)
    ap.add_argument("--split", default="test")
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--positive-chunks-only", action="store_true")
    ap.add_argument("--shuffle-offset", type=int, default=7)
    ap.add_argument("--min-true-relative-improvement", type=float, default=0.02)
    ap.add_argument("--require-positive-chunks", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--require-backend-id", default="bsp_faces_disp_gpu")
    ap.add_argument("--split-key", choices=["episode", "track", "match"], default="episode")
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--test-fraction", type=float, default=0.05)
    ap.add_argument("--allow-static-dense-repeat", action="store_true")
    ap.add_argument("--video-frames", type=int, default=81)
    ap.add_argument("--raw-stride", type=int, default=2)
    ap.add_argument("--vae-stride", type=int, default=8)
    ap.add_argument("--patch-size-hw", type=int, default=2)
    ap.add_argument("--latent-channels", type=int, default=16)
    ap.add_argument("--cond-dim", type=int, default=128)
    ap.add_argument("--encoder-hidden-dim", type=int, default=64)
    ap.add_argument("--adapter-hidden-dim", type=int, default=512)
    ap.add_argument("--adapter-residual-mode", choices=["cond_gated", "additive"], default="cond_gated")
    ap.add_argument("--adapter-wrap-first-blocks", type=int, default=0, help="0 wraps all blocks; B2b probe uses first third only.")
    ap.add_argument("--adapter-residual-scale-init", type=float, default=1.0)
    ap.add_argument(
        "--activation-checkpoint-blocks",
        action="store_true",
        help="Enable activation checkpointing for wrapped Wan blocks when constructing the eval model.",
    )
    ap.add_argument("--video-width", type=int, default=832)
    ap.add_argument("--video-height", type=int, default=480)
    ap.add_argument("--latent-frames", type=int, default=21)
    ap.add_argument("--chunk-size", type=int, default=3)
    ap.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--dense-resize-mode", choices=["bilinear", "native-only"], default="bilinear")
    ap.add_argument("--only-player-dense-channels", action="store_true")
    ap.add_argument("--max-aspect-ratio-delta", type=float, default=0.08)
    ap.add_argument("--require-release-report", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--min-train-records", type=int, default=1)
    ap.add_argument("--min-train-positive-frames", type=int, default=0)
    ap.add_argument("--min-train-context-frames", type=int, default=0)
    ap.add_argument("--min-train-matches", type=int, default=1)
    ap.add_argument("--min-train-episodes", type=int, default=1)
    return ap


def main() -> None:
    result = run(build_parser().parse_args())
    text = json.dumps({k: result[k] for k in ["kind", "status", "split", "eval_chunk_count", "summary", "gate"]}, ensure_ascii=False, indent=2)
    print(text)
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
