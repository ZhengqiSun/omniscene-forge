#!/usr/bin/env python3
"""Candidate1 sequential I2V continuation wrapper.

QXQ sampler sources stay read-only. This keeps one WanI2V pipe resident, iterates
candidate1 windows in temporal order, and from window 1 onward uses the previous
window generated tail frame as the next window image condition.
"""
from __future__ import annotations
from runtime_paths import source_path

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch
from PIL import Image

ROOT = Path(str(source_path('scene', '')))
QXQ = Path(str(source_path('project', '')))
ZH = ROOT


def add_path(p: Path) -> None:
    s = str(Path(p).resolve())
    if s in sys.path:
        sys.path.remove(s)
    sys.path.insert(0, s)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def raw_span(row: dict[str, Any]) -> tuple[int, int]:
    """Return (raw_start, window_len_raw) for a source-manifest window.

    window_len_raw is the raw-frame span from the window's first to last conditioned
    frame. For a proper non-overlapping AR chain the next window must begin exactly
    where the previous one ended, i.e. next.raw_start == prev.raw_start + window_len_raw.
    Derived (in priority order) from raw_indices/positive_latent_frames span, then
    frame_count_end-frame_count_start, then 8*(positive_latent_frame_count-1) using the
    canonical raw stride of 8 (frame_contract: raw_frame = raw_start + 8*latent_index).
    """
    ri = row.get("raw_indices") or row.get("positive_latent_frames")
    if isinstance(ri, list) and len(ri) >= 2:
        return int(ri[0]), int(ri[-1]) - int(ri[0])
    raw_start = int(row["raw_start"])
    if row.get("frame_count_start") is not None and row.get("frame_count_end") is not None:
        return raw_start, int(row["frame_count_end"]) - int(row["frame_count_start"])
    n_lat = int(row.get("positive_latent_frame_count") or row.get("latent_frames"))
    return raw_start, 8 * (n_lat - 1)


def frames_from_video_tensor(video: torch.Tensor) -> np.ndarray:
    v = video.detach().cpu().float().numpy()
    v = np.transpose(v, (1, 2, 3, 0))
    v = np.clip((v + 1.0) / 2.0, 0, 1)
    return (v * 255.0 + 0.5).astype(np.uint8)


def save_mp4(path: Path, frames: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8,
                                macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"])
    try:
        for i in range(frames.shape[0]):
            writer.append_data(frames[i])
    finally:
        writer.close()


def stitch(ffmpeg: str, mp4s: list[Path], out_mp4: Path) -> dict[str, Any]:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    concat = out_mp4.parent / "candidate1_ar_concat_list.txt"
    concat.write_text("".join(f"file {str(p.resolve())!r}\n" for p in mp4s), encoding="utf-8")
    subprocess.check_call([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
        "-i", str(concat), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "16", str(out_mp4)
    ])
    return {"concat_list": str(concat), "mp4": str(out_mp4), "size": out_mp4.stat().st_size}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-manifest", type=Path, default=ROOT / "output/demo_1min_20260706_v0/manifests/candidate1_true_dense_gen_source_manifest_v0.jsonl")
    ap.add_argument("--out-root", type=Path, default=ROOT / "output/demo_1min_ar_20260706_v0")
    ap.add_argument("--adapter-checkpoint-low", type=Path, default=ROOT / "output/memory_dense_adapter_v0/runs/zq_base_low_tier69h_lora64_w6_v1/checkpoints/memory_dense_adapter_step_004500.pt")
    ap.add_argument("--adapter-checkpoint-high", type=Path, default=ROOT / "output/memory_dense_adapter_v0/runs/zq_base_high_tier69h_lora64_dsw2_w6_v1/checkpoints/memory_dense_adapter_step_004650.pt")
    ap.add_argument("--state-cache-manifest", type=Path, default=None)
    ap.add_argument("--allow-legacy-state-projection", action="store_true",
                    help="accept a state cache built with the pre-v2 projection contract "
                         "(pitch_sign=-1 / far=4096) instead of failing; only for deliberate "
                         "reruns of the old demo caches.")
    ap.add_argument("--lingbot-repo", type=Path, default=Path(str(source_path('lingbot', ''))))
    ap.add_argument("--ckpt-dir", type=Path, default=Path(str(source_path('assets', 'lingbot-world-base-cam'))))
    ap.add_argument("--variant", choices=["true_dense"], default="true_dense")
    ap.add_argument("--latent-frames", type=int, default=21)
    ap.add_argument("--chunk-size", type=int, default=3)
    ap.add_argument("--steps", type=int, default=70)
    ap.add_argument("--shift", type=float, default=10.0)  # official wan_i2v_A14B sample_shift; 3.0 was a legacy wrapper default
    ap.add_argument("--guide", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=20260629)
    ap.add_argument("--size", default="832*480")
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--ffmpeg", default='ffmpeg')
    ap.add_argument("--stop-after", type=int, default=0, help="debug: stop after N windows; 0 means all")
    ap.add_argument("--independent-context", action="store_true", help="use each row's original image instead of AR tail chaining")
    ap.add_argument("--target-height", type=int, default=480, help="Force generation height. default 480 to fix the 464 floor bug (MAX_AREA_CONFIGS['832*480'] float sqrt//8 floors 480->464); pass explicitly to override.")
    ap.add_argument("--target-width", type=int, default=832, help="Force generation width. default 832 to fix the 464 floor bug; pass explicitly to override. Both target dims must be divisible by 16.")
    ap.add_argument("--skip-ar-timestamp-check", action="store_true", help="disable the AR-chain raw_start continuity assertion (default False). Only the chained path is guarded; --independent-context is never checked.")
    ap.add_argument("--skip-resolution-check", action="store_true", help="disable the post-generate assertion that frames match --target-height/--target-width (default False).")
    args = ap.parse_args()

    if args.target_height % 16 or args.target_width % 16:
        raise ValueError("--target-height/--target-width must be divisible by 16")
    # Epsilon avoids 480 becoming 479.999... and then floor-dividing to 464 (token grid 29x52 instead of 30x52).
    gen_max_area = math.ceil(args.target_height * args.target_width * 1.0001)
    expected_lat_h = args.target_height // 8
    expected_lat_w = args.target_width // 8

    add_path(ZH / "tools")
    add_path(QXQ / "tools")
    add_path(args.lingbot_repo)
    add_path(QXQ / "tools")

    from qxq_sample_base_dense_v0 import patch_base_forward_full_seq, CLEAN_NEG_PROMPT
    from sample_memory_dense_adapter_phase2a_v0 import (
        load_phase2a_dense_samples,
        dense_tokens_for_sequence,
        load_adapter_checkpoint as load_dense_adapter_checkpoint,
        load_dense_path_sample,
        frame_num_for_latent_frames,
        effective_latent_frames_for_chunking,
    )
    from sample_memory_dense_adapter_state_v0 import (
        load_adapter_checkpoint as load_state_adapter_checkpoint,
        state_tokens_for_sequence,
    )
    from memory_dense_state_adapter_v0 import (
        STATE_CHANNELS_V0,
        StateTokenProjector,
        load_state_manifest,
        resolve_state_cache_path,
        verify_state_projection_contract,
    )
    from memory_dense_wan_adapter_v0 import (
        MemoryDenseAdapterConfig,
        MemoryDenseFrozenVAEWanTokenEncoder,
        inject_lora_into_wan_model_fast,
        resolve_dense_native_hw,
        wrap_wan_model_fast_with_memory_dense_adapter,
    )
    import wan
    from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS
    from wan.modules.vae2_1 import Wan2_1_VAE

    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", args.ffmpeg)
    gen_dir = args.out_root / "gen_true_dense_ar"
    out_dir = gen_dir / "out"
    frame_dir = gen_dir / "context_frames"
    log_dir = gen_dir / "logs"
    for d in (out_dir, frame_dir, log_dir, args.out_root / "stitched"):
        d.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.source_manifest)
    rows.sort(key=lambda r: int(str(r["clip_id"]).split("_")[-1]))
    if args.stop_after and args.stop_after > 0:
        rows = rows[:args.stop_after]

    cfg = WAN_CONFIGS["i2v-A14B"]
    eff_lf = effective_latent_frames_for_chunking(args.latent_frames, args.chunk_size)
    eff_frame_num = frame_num_for_latent_frames(eff_lf)
    load_dense_fn = load_dense_path_sample
    active_samples: list[Any] = []
    active_sample_ids: list[str] = []
    active_dense_manifest: str | None = None
    active_state_row: dict[str, Any] | None = None
    state_rows = load_state_manifest(args.state_cache_manifest) if args.state_cache_manifest else {}
    if args.state_cache_manifest:
        verify_state_projection_contract(
            args.state_cache_manifest, state_rows,
            strict=not args.allow_legacy_state_projection)

    def set_active_condition(row: dict[str, Any]) -> None:
        nonlocal active_samples, active_sample_ids, active_dense_manifest, active_state_row
        manifest = Path(row["phase2a_dense_sequence_manifest"])
        samples, sample_ids = load_phase2a_dense_samples(manifest, eff_lf)
        if len(samples) != eff_lf:
            raise ValueError(f"dense manifest has {len(samples)} rows but eff_lf={eff_lf}: {manifest}")
        active_samples = samples
        active_sample_ids = sample_ids
        active_dense_manifest = str(manifest)
        active_state_row = None
        if args.state_cache_manifest:
            clip_id = str(row["clip_id"])
            if clip_id not in state_rows:
                raise KeyError(f"{clip_id}: missing state cache row in {args.state_cache_manifest}")
            active_state_row = dict(state_rows[clip_id])
            active_state_row["state_cache"] = str(
                resolve_state_cache_path(active_state_row, args.state_cache_manifest)
            )

    pipe = wan.WanI2V(
        config=cfg,
        checkpoint_dir=str(args.ckpt_dir),
        device_id=args.device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        convert_model_dtype=False,
    )

    def build_expert(which: str, ckpt_path: Path) -> dict[str, Any]:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cc = ckpt.get("config") or {}
        ac = cc.get("adapter_config") or {}
        config = MemoryDenseAdapterConfig(
            dense_channels=int(ac.get("dense_channels", 7)),
            cond_dim=int(ac.get("cond_dim", 128)),
            encoder_hidden_dim=int(ac.get("encoder_hidden_dim", 64)),
            adapter_hidden_dim=int(ac.get("adapter_hidden_dim", 512)),
            vae_stride=int(ac.get("vae_stride", 8)),
            wan_patch_size_hw=int(ac.get("wan_patch_size_hw", 2)),
            cond_key=str(ac.get("cond_key", "memory_dense_cond_tokens")),
            residual_mode=str(ac.get("residual_mode", "cond_gated")),
            wrap_first_blocks=ac.get("wrap_first_blocks"),
            residual_scale_init=float(ac.get("residual_scale_init", 1.0)),
            activation_checkpoint_blocks=False,
            highfreq_branch=bool(ac.get("highfreq_branch", False)),
        )
        expert = getattr(pipe, which)
        wrap_wan_model_fast_with_memory_dense_adapter(expert, config, freeze_base=True)
        lc = cc.get("lora_config") or {}
        if int(lc.get("rank", 0) or 0) > 0:
            inject_lora_into_wan_model_fast(
                expert, rank=int(lc["rank"]), targets=str(lc.get("targets", "attn,mlp")),
                alpha=lc.get("alpha"), dropout=float(lc.get("dropout", 0.0)),
            )
        de = cc.get("dense_condition_encoder") or {}
        vae_pth = Path(de.get("vae_pth") or (args.ckpt_dir / cfg.vae_checkpoint))
        vae = Wan2_1_VAE(vae_pth=str(vae_pth), device=pipe.device)
        encoder = MemoryDenseFrozenVAEWanTokenEncoder(
            config, vae=vae, vae_pth=vae_pth,
            native_hw=resolve_dense_native_hw(cc),
            packing=str(de.get("packing", "img1_mask_img2_player_v0")),
        ).to(device=pipe.device, dtype=pipe.param_dtype)
        state_projector = None
        if args.state_cache_manifest:
            state_projector = StateTokenProjector(
                state_channels=len(STATE_CHANNELS_V0),
                cond_dim=int(ac.get("cond_dim", 128)),
            ).to(device=pipe.device, dtype=pipe.param_dtype)
            load_state_adapter_checkpoint(ckpt_path, expert, encoder, state_projector)
            state_projector.eval()
        else:
            load_dense_adapter_checkpoint(ckpt_path, expert, encoder)
        expert.to(pipe.device).eval()
        encoder.eval()

        original_forward = expert.forward
        cache: dict[tuple[str, int, int], torch.Tensor] = {}

        def forward_with_active_dense(*f_args: Any, **kwargs: Any):
            if not active_samples or active_dense_manifest is None:
                raise RuntimeError("active dense samples are not set")
            dit_cond = dict(kwargs.get("dit_cond_dict") or {})
            x_arg = kwargs.get("x")
            if x_arg is None and f_args:
                x_arg = f_args[0]
            current = x_arg[0] if isinstance(x_arg, list) else x_arg
            lat_h, lat_w = int(current.shape[-2]), int(current.shape[-1])
            thw = (lat_h // int(pipe.patch_size[1]), lat_w // int(pipe.patch_size[1]))
            key = (active_dense_manifest, thw[0], thw[1])
            if key not in cache:
                toks, _ = dense_tokens_for_sequence(
                    encoder=encoder, samples=active_samples, load_dense_fn=load_dense_fn,
                    device=current.device, dtype=pipe.param_dtype, target_token_hw=thw,
                )
                if state_projector is not None:
                    if active_state_row is None:
                        raise RuntimeError("active state row is not set")
                    state_tokens = state_tokens_for_sequence(
                        state_projector=state_projector,
                        state_row=active_state_row,
                        latent_frames=eff_lf,
                        device=current.device,
                        dtype=pipe.param_dtype,
                        target_token_hw=thw,
                    )
                    if state_tokens.shape != toks.shape:
                        raise RuntimeError(
                            f"state token shape {tuple(state_tokens.shape)} != dense token shape {tuple(toks.shape)}"
                        )
                    toks = toks + state_tokens
                cache[key] = toks.detach()
            dit_cond[config.cond_key] = cache[key].to(device=current.device, dtype=current.dtype)
            kwargs["dit_cond_dict"] = dit_cond
            return original_forward(*f_args, **kwargs)

        expert.forward = forward_with_active_dense  # type: ignore[method-assign]
        return {
            "which": which,
            "ckpt": str(ckpt_path),
            "step": ckpt.get("step"),
            "state_projector_loaded": state_projector is not None,
        }

    expert_reports = [
        build_expert("low_noise_model", args.adapter_checkpoint_low),
        build_expert("high_noise_model", args.adapter_checkpoint_high),
    ]

    run_report = {
        "kind": "candidate1_ar_wrapper_v0",
        "status": "running",
        "started_at": time.strftime("%F %T %z"),
        "source_manifest": str(args.source_manifest),
        "state_cache_manifest": str(args.state_cache_manifest) if args.state_cache_manifest else None,
        "window_count_target": len(rows),
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_id": args.device_id,
        "target_height": args.target_height,
        "target_width": args.target_width,
        "shift": args.shift,
        "steps": args.steps,
        "guide": args.guide,
        "gen_max_area": gen_max_area,
        "expected_lat_h": expected_lat_h,
        "expected_lat_w": expected_lat_w,
        "fixed480_note": "max_area = ceil(target_h*target_w*1.0001) to avoid the 480->464 floor bug; lat = target//8 (expect 60x104, token grid 30x52).",
        "condition_image_resize": f"condition image resized to {args.target_width}x{args.target_height} before generate(); WanI2V derives lat_h/lat_w from the image aspect ratio, so a 16:9 clip image silently produced 464 regardless of max_area.",
        "resolution_check": not args.skip_resolution_check,
        "ar_timestamp_check": (not args.independent_context) and (not args.skip_ar_timestamp_check),
        "experts": expert_reports,
        "context_mode": "window0 original clip_dir/image.jpg; windowN uses previous generated tail_frame.png as WanI2V image condition",
        "windows": [],
    }
    write_json(args.out_root / "candidate1_ar_run_report_v0.json", run_report)

    prev_tail: Path | None = None
    prev_row: dict[str, Any] | None = None
    mp4s: list[Path] = []
    failures: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        t0 = time.time()
        clip_id = row["clip_id"]
        clip_dir = Path(row["clip_dir"])
        for need in ["image.jpg", "poses.npy", "intrinsics.npy"]:
            if not (clip_dir / need).exists():
                raise FileNotFoundError(f"existing clip_dir missing {need}: {clip_dir}")
        set_active_condition(row)
        # AR-chain timestamp continuity guard: only when we are about to feed the previous
        # window's generated tail frame as this window's I2V image condition (chained path).
        # Independent-context mode (each window uses its own GT first frame) is never checked.
        if (not args.independent_context) and (prev_row is not None) and (not args.skip_ar_timestamp_check):
            if ("raw_start" not in row) or ("raw_start" not in prev_row):
                print(json.dumps({"event": "ar_timestamp_check_skipped", "reason": "raw_start missing in manifest schema", "index": idx}, ensure_ascii=False), flush=True)
            else:
                prev_start, window_len_raw = raw_span(prev_row)
                cur_start = int(row["raw_start"])
                if prev_start + window_len_raw != cur_start:
                    raise RuntimeError(
                        "AR-chain timestamp discontinuity: previous window raw_start="
                        f"{prev_start} + window_len_raw={window_len_raw} = {prev_start + window_len_raw} "
                        f"!= current window raw_start={cur_start} (index={idx}, clip_id={clip_id}). "
                        "The chained tail frame therefore leads this window's dense/pose start by "
                        f"{prev_start + window_len_raw - cur_start} raw frames. Manifest suspected 50% "
                        "overlap layout (stride < window_len); rebuild with stride==window_len via "
                        "build_ar_gen_source_manifest_v1.py. Pass --skip-ar-timestamp-check to bypass."
                    )
        original_image = clip_dir / "image.jpg"
        context_image = original_image if args.independent_context or prev_tail is None else prev_tail
        image = Image.open(context_image).convert("RGB")
        # WanI2V derives the generation resolution from the CONDITION IMAGE aspect ratio,
        # not from max_area alone (wan/image2video.py:329-337):
        #     lat_h = round(sqrt(max_area * h/w) // 8 // 2 * 2)
        # A 16:9 clip image (1280x720, aspect .5625) yields lat 58x104 -> 464x832, while
        # training used clips whose image.jpg is already 832x480 (aspect .5769) -> 60x104 -> 480x832.
        # No max_area can reach (60,104) from 16:9 (409600 gives 848x480), so resize the
        # condition image to the target aspect here. Verified against training clips_tier20h:
        # plain resize matches their image.jpg at MAE 1.28 vs 5.99/8.25 for center-crop variants.
        context_image_original_size = list(image.size)
        if image.size != (args.target_width, args.target_height):
            image = image.resize((args.target_width, args.target_height), Image.BICUBIC)
        prompt_path = clip_dir / "prompt.txt"
        prompt = prompt_path.read_text(encoding="utf-8").strip() if prompt_path.exists() else (row.get("prompt") or "first-person gameplay video in Counter-Strike, de_dust2 map")
        # Derive a distinct seed per window; resetting to the same seed every
        # window made window 1+ reuse the window-0 noise sequence.
        window_seed = int(args.seed) + idx
        torch.manual_seed(window_seed)
        np.random.seed(window_seed)
        random.seed(window_seed)
        mp4 = out_dir / f"{clip_id}_true_dense_ar_lf{eff_lf}_seed{args.seed}.mp4"
        tail_png = frame_dir / f"{clip_id}_tail_frame.png"
        status = "started"
        rec: dict[str, Any] = {
            "index": idx,
            "window_seed": window_seed,
            "clip_id": clip_id,
            "window_id": row.get("phase2a_window_id"),
            "clip_dir": str(clip_dir),
            "dense_sequence_manifest": active_dense_manifest,
            "sample_ids": active_sample_ids,
            "context_image_source": str(context_image),
            "context_image_kind": "generated_tail_frame" if prev_tail is not None else "original_clip_image",
            "context_image_original_size": context_image_original_size,
            "context_image_fed_size": [args.target_width, args.target_height],
            "original_image": str(original_image),
            "mp4": str(mp4),
            "tail_frame": str(tail_png),
            "started_at": time.strftime("%F %T %z"),
        }
        try:
            print(json.dumps({"event": "window_start", **rec}, ensure_ascii=False), flush=True)
            video = pipe.generate(
                prompt, image,
                action_path=str(clip_dir),
                allow_act2cam=False, action_string=None, vis_ui=False,
                max_area=gen_max_area,
                frame_num=eff_frame_num,
                shift=args.shift,
                sample_solver="unipc",
                sampling_steps=args.steps,
                guide_scale=(args.guide, args.guide),
                n_prompt=CLEAN_NEG_PROMPT,
                seed=args.seed,
                offload_model=False,
            )
            frames = frames_from_video_tensor(video)
            # Fail loud on a resolution mismatch. The report used to echo --target-height
            # while the actual frames were 464; every W4-W7 demo shipped at the wrong
            # resolution because nothing ever compared the request against the result.
            got_h, got_w = int(frames.shape[1]), int(frames.shape[2])
            if not args.skip_resolution_check and (got_h, got_w) != (args.target_height, args.target_width):
                raise RuntimeError(
                    f"{clip_id}: generated {got_h}x{got_w} but --target-height/--target-width "
                    f"asked for {args.target_height}x{args.target_width}. WanI2V derives the "
                    f"resolution from the condition image aspect ratio; check that the image fed "
                    f"to generate() was resized to the target. Pass --skip-resolution-check to bypass."
                )
            save_mp4(mp4, frames, args.fps)
            Image.fromarray(frames[-1]).save(tail_png)
            if not args.independent_context:
                prev_tail = tail_png
            prev_row = row
            mp4s.append(mp4)
            rec.update({
                "status": "complete",
                "frames": int(frames.shape[0]),
                "frame_shape": list(frames.shape),
                "mp4_size": mp4.stat().st_size,
                "tail_frame_size": tail_png.stat().st_size,
                "elapsed_s": round(time.time() - t0, 3),
                "completed_at": time.strftime("%F %T %z"),
                "eff_latent_frames": int(eff_lf),
                "steps": args.steps,
                "shift": args.shift,
                "guide": args.guide,
                "seed": args.seed,
            })
            write_json(out_dir / f"{clip_id}_true_dense_ar_report.json", rec)
            print(json.dumps({"event": "window_complete", "index": idx, "clip_id": clip_id, "mp4_size": rec["mp4_size"], "elapsed_s": rec["elapsed_s"]}, ensure_ascii=False), flush=True)
        except Exception as exc:
            rec.update({"status": "failed", "error": repr(exc), "elapsed_s": round(time.time() - t0, 3), "completed_at": time.strftime("%F %T %z")})
            failures.append(rec)
            write_json(out_dir / f"{clip_id}_true_dense_ar_report.json", rec)
            write_json(gen_dir / "failures.json", failures)
            raise
        finally:
            run_report["windows"].append(rec)
            run_report["completed_windows"] = sum(1 for w in run_report["windows"] if w.get("status") == "complete")
            run_report["failed_windows"] = len(failures)
            write_json(args.out_root / "candidate1_ar_run_report_v0.json", run_report)

    stitched = stitch(args.ffmpeg, mp4s, args.out_root / "stitched/candidate1_ar_1min.mp4")
    run_report.update({
        "status": "complete",
        "completed_at": time.strftime("%F %T %z"),
        "completed_windows": len(mp4s),
        "failed_windows": len(failures),
        "stitched": stitched,
    })
    write_json(args.out_root / "candidate1_ar_run_report_v0.json", run_report)
    print(json.dumps(run_report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
