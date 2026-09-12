#!/usr/bin/env python3
"""Thin wrapper that reuses the PROVEN base dense generation core
(qxq_sample_base_dense_v0.py) to sample held-out val/test mutual ego windows,
feeding an EXISTING materialized clip_dir + a per-window dense-sequence-manifest
directly -- bypassing materialize_phase2a_clip_dir (which needs phase2a-only fields
the aligned-cache rows do not have).

Everything model-side (WanI2V two-expert build, adapter+LoRA injection, FULL
non-causal dense token injection patch, generate loop, mp4 writing) is identical to
qxq_sample_base_dense_v0.py -- we import its functions so the proven generation is
unchanged.

Inputs: a phase2a-source-manifest (produced by qxq_build_valtest_sampler_inputs_v0.py)
whose rows carry `clip_dir` + `phase2a_dense_sequence_manifest`.

Variants: true_dense / shuffled_dense / blank_dense / base.
Writes ONLY under --out-dir (caller must keep it under QXQ/output/).
"""
from __future__ import annotations
from runtime_paths import source_path

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

QXQ = Path(str(source_path('project', '')))
ZH = Path(str(source_path('scene', '')))


def add_path(p: Path) -> None:
    s = str(Path(p).resolve())
    if s in sys.path:
        sys.path.remove(s)
    sys.path.insert(0, s)


def choose_row(path: Path, *, clip_id: str | None, player_stem: str | None,
               window_id: str | None, clip_index: int) -> dict[str, Any]:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    if clip_id or player_stem or window_id:
        rows = [
            r for r in rows
            if (not clip_id or r.get("clip_id") == clip_id)
            and (not player_stem or r.get("player_stem") == player_stem)
            and (not window_id or r.get("phase2a_window_id") == window_id)
        ]
    if not rows:
        raise ValueError(f"no source rows match clip_id={clip_id!r} player_stem={player_stem!r} window_id={window_id!r}")
    if clip_index < 0 or clip_index >= len(rows):
        raise IndexError(f"clip_index {clip_index} outside row count {len(rows)}")
    return rows[clip_index]


def choose_shuffled_row(path: Path, *, current_clip_id: str) -> dict[str, Any]:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    for r in rows:
        if str(r.get("clip_id")) != str(current_clip_id) and r.get("phase2a_dense_sequence_manifest"):
            return r
    raise ValueError(f"no alternate row for shuffled_dense current_clip_id={current_clip_id!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-manifest", type=Path, required=True)
    ap.add_argument("--clip-id", default=None)
    ap.add_argument("--player-stem", default=None)
    ap.add_argument("--window-id", default=None)
    ap.add_argument("--clip-index", type=int, default=0)
    ap.add_argument("--adapter-checkpoint-low", type=Path, default=None)
    ap.add_argument("--adapter-checkpoint-high", type=Path, default=None)
    ap.add_argument("--lingbot-repo", type=Path, default=Path(str(source_path('lingbot', ''))))
    ap.add_argument("--ckpt-dir", type=Path, default=Path(str(source_path('assets', 'lingbot-world-base-cam'))))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--variant", choices=["base", "true_dense", "blank_dense", "shuffled_dense"], default="true_dense")
    ap.add_argument("--latent-frames", type=int, default=21)
    ap.add_argument("--chunk-size", type=int, default=3)
    ap.add_argument("--steps", type=int, default=70)
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--guide", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=20260613)
    ap.add_argument("--size", default="832*480")
    ap.add_argument("--target-height", type=int, default=480,
                    help="Force generation height; condition image is resized to this aspect "
                         "before generate() (WanI2V derives lat from the image aspect).")
    ap.add_argument("--target-width", type=int, default=832)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--offload-model", action=argparse.BooleanOptionalAction, default=False)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    add_path(ZH / "tools")
    add_path(QXQ / "tools")
    add_path(args.lingbot_repo)
    add_path(QXQ / "tools")  # keep qxq adapter modules ahead of older zhengqi copies

    # PROVEN generation core: import everything model-side from the base sampler.
    from qxq_sample_base_dense_v0 import (
        patch_base_forward_full_seq,
        CLEAN_NEG_PROMPT,
    )
    from sample_memory_dense_adapter_phase2a_v0 import (
        load_phase2a_dense_samples,
        dense_tokens_for_sequence,
        build_blank_release_samples,
        load_adapter_checkpoint,
        load_dense_path_sample,
        frame_num_for_latent_frames,
        effective_latent_frames_for_chunking,
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

    cfg = WAN_CONFIGS["i2v-A14B"]
    eff_lf = effective_latent_frames_for_chunking(args.latent_frames, args.chunk_size)
    eff_frame_num = frame_num_for_latent_frames(eff_lf)

    row = choose_row(args.source_manifest, clip_id=args.clip_id, player_stem=args.player_stem,
                     window_id=args.window_id, clip_index=args.clip_index)
    clip_dir = Path(row["clip_dir"])
    for need in ["image.jpg", "poses.npy", "intrinsics.npy"]:
        if not (clip_dir / need).exists():
            raise FileNotFoundError(f"existing clip_dir missing {need}: {clip_dir}")

    # dense samples come from THIS window's dense-sequence-manifest (release-resolved)
    samples, sample_ids = load_phase2a_dense_samples(Path(row["phase2a_dense_sequence_manifest"]), eff_lf)
    if len(samples) != eff_lf:
        raise ValueError(f"dense manifest has {len(samples)} rows but eff_lf={eff_lf}; clip {row.get('clip_id')}")
    load_dense_fn = load_dense_path_sample

    shuffled_source = None
    if args.variant == "shuffled_dense":
        srow = choose_shuffled_row(args.source_manifest, current_clip_id=str(row["clip_id"]))
        samples, sample_ids = load_phase2a_dense_samples(Path(srow["phase2a_dense_sequence_manifest"]), eff_lf)
        shuffled_source = {"clip_id": srow.get("clip_id"), "window_id": srow.get("phase2a_window_id")}
    if args.variant == "blank_dense":
        samples = build_blank_release_samples(samples, load_dense_fn)

    image = Image.open(clip_dir / "image.jpg").convert("RGB")
    # Same fix as qxq_sample_candidate1_ar_v0 (31884c7): WanI2V derives lat_h/lat_w from
    # the condition-image aspect, and MAX_AREA_CONFIGS["832*480"]=399360 floors 480->464
    # through //8 even for a correct-aspect image. Resize + max_area 399400 clears both.
    context_image_original_size = list(image.size)
    if image.size != (args.target_width, args.target_height):
        image = image.resize((args.target_width, args.target_height), Image.BICUBIC)
    gen_max_area = int(args.target_height * args.target_width * 1.0001) + 1
    prompt_path = clip_dir / "prompt.txt"
    prompt = (prompt_path.read_text(encoding="utf-8").strip() if prompt_path.exists()
              else (row.get("prompt") or "first-person gameplay video in Counter-Strike, de_dust2 map"))

    pipe = wan.WanI2V(
        config=cfg,
        checkpoint_dir=str(args.ckpt_dir),
        device_id=args.device_id,
        rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=False,
        convert_model_dtype=False,
    )

    def build_expert(which: str, ckpt_path: Path | None) -> dict[str, Any] | None:
        if ckpt_path is None or args.variant == "base":
            return None
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
        load_adapter_checkpoint(ckpt_path, expert, encoder)
        expert.to(pipe.device).eval()
        encoder.eval()
        patch_base_forward_full_seq(
            expert, encoder=encoder, samples=samples, load_dense_fn=load_dense_fn,
            dense_tokens_for_sequence=dense_tokens_for_sequence, dtype=pipe.param_dtype,
            patch_hw=pipe.patch_size[1], cond_key=config.cond_key,
        )
        return {"which": which, "ckpt": str(ckpt_path), "step": ckpt.get("step")}

    expert_reports = [
        build_expert("low_noise_model", args.adapter_checkpoint_low),
        build_expert("high_noise_model", args.adapter_checkpoint_high),
    ]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # F09: the Fast/distilled path samples at shift=10.0 while this unipc path
    # defaults to 3.0. Both stay inside the training sigma distribution, but the
    # two schedules route ~15 of 70 steps to a different expert (mean |dt| 151),
    # so results taken at different shifts must never share a median.
    if abs(float(args.shift) - 10.0) > 1e-6:
        print(json.dumps({
            "event": "base_solver_shift_warning",
            "shift": float(args.shift),
            "fast_path_sample_shift": 10.0,
            "note": "different shift => different expert routing; group every cross-tool comparison by shift",
        }, ensure_ascii=False), flush=True)
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
        offload_model=args.offload_model,
    )

    v = video.detach().cpu().float().numpy()
    v = np.transpose(v, (1, 2, 3, 0))
    v = np.clip((v + 1.0) / 2.0, 0, 1)
    frames = (v * 255.0 + 0.5).astype(np.uint8)
    if frames.shape[1] != args.target_height or frames.shape[2] != args.target_width:
        raise RuntimeError(
            f"generated resolution {frames.shape[1]}x{frames.shape[2]} != "
            f"target {args.target_height}x{args.target_width}; refusing to write output")
    import imageio
    mp4 = args.out_dir / f"{row['clip_id']}_{args.variant}_lf{eff_lf}_seed{args.seed}.mp4"
    writer = imageio.get_writer(str(mp4), fps=16, codec="libx264", quality=8,
                                macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"])
    for i in range(frames.shape[0]):
        writer.append_data(frames[i])
    writer.close()

    report = {
        "kind": "qxq_sample_valtest_v0",
        "status": "complete",
        "variant": args.variant,
        "clip_id": row["clip_id"],
        "player_stem": row.get("player_stem"),
        "split": row.get("split"),
        "window_id": row.get("phase2a_window_id"),
        "clip_dir": str(clip_dir),
        "dense_sequence_manifest": row.get("phase2a_dense_sequence_manifest"),
        "sample_ids": sample_ids,
        "experts": [e for e in expert_reports if e],
        "shuffled_source": shuffled_source,
        "frames": int(frames.shape[0]),
        "frame_shape": list(frames.shape),
        "eff_latent_frames": int(eff_lf),
        "steps": args.steps, "shift": args.shift, "guide": args.guide, "seed": args.seed,
        "target_height": args.target_height, "target_width": args.target_width,
        "gen_max_area": gen_max_area,
        "context_image_original_size": context_image_original_size,
        "context_image_fed_size": [args.target_width, args.target_height],
        "resolution_check": True,
        "mp4": str(mp4),
    }
    json.dump(report, open(args.out_dir / f"{row['clip_id']}_{args.variant}_report.json", "w"),
              ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print("DONE", mp4, frames.shape, flush=True)


if __name__ == "__main__":
    main()
