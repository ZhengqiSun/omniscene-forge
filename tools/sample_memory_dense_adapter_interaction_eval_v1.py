#!/usr/bin/env python3
"""M2 interaction-variant eval sampler for the taskB interaction_v1 run (base two-expert).

Thin wrapper over the PROVEN pieces of sample_memory_dense_adapter_state_v0.py:
model build, adapter wrap, LoRA injection, frozen-VAE dense encoder, state projector
and generate loop are all imported unchanged. This file adds ONLY:

1. --interaction-variant {true,blank,shuffled}: at inference the dense+state condition
   stream stays REAL in all three arms; only the 134-dim interaction token contribution
   switches:
     true     = this clip's own interaction features
     blank    = interaction contribution REMOVED entirely (no projector call; note this
                is "no interaction module", not zeros-through-projector, because a
                trained Linear bias would still inject a constant plane on zero input)
     shuffled = another val/test clip's interaction features (deterministic offset pick,
                same split, like the trainer's negative-shuffle-offset convention)
2. Loads memory_dense_interaction_projector from the checkpoint (key written by
   train_memory_dense_adapter_interaction_v1.py). An expert checkpoint without that key
   (e.g. the bigrun_120h_state train_HIGH warm-start sibling) simply contributes no
   interaction tokens; this is recorded per expert in the report.
3. A v2-tolerant lightweight map-manifest loader: the state_v0 loader hardcodes the v0
   dense shape [7,176,320]; here the expected shape is taken from the manifest header
   itself (channels order and geometry_backend_id are still enforced).

Eval-only tool: val/test splits only (enforced). Writes ONLY under --out-dir.
"""

from __future__ import annotations

from runtime_paths import ASSET_ROOT, LINGBOT_ROOT

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from sample_memory_dense_adapter_state_v0 import (  # noqa: E402
    CLEAN_NEG_PROMPT,
    COND_KEY,
    add_path,
    choose_record,
    dense_tokens_for_sequence,
    effective_latent_frames_for_chunking,
    frame_num_for_latent_frames,
    load_adapter_checkpoint,
    read_jsonl,
    resolve_clip_dir,
    state_tokens_for_sequence,
    write_json,
)


def load_v2_manifest_samples(
    manifest_path: Path,
    sample_ids: list[str],
    *,
    release_cls: Any,
    sample_from_manifest_fn: Any,
    channels: list[str],
    required_backend_id: str,
) -> tuple[list[Any], list[int]]:
    """Like state_v0.load_sampler_manifest_samples but the expected dense shape comes
    from the manifest header instead of the hardcoded v0 [7,176,320]."""
    manifest_path = manifest_path.resolve()
    readiness_path = manifest_path.parent / "training_readiness_v0.json"
    teacher_qa_path = (
        manifest_path.parent
        / "channel_teacher_qa_v0"
        / "memory_dense_channels_vs_teacher_v0.json"
    )
    # NOTE: always use streaming JSON parser — skips release_cls.load() which walks all
    # 1M+ samples and issues ~12M NFS stat() calls; streaming reads only the needed sample_ids.
    wanted = set(sample_ids)
    found: dict[str, Any] = {}
    decoder = json.JSONDecoder()
    chunk_size = 1024 * 1024
    with manifest_path.open("r", encoding="utf-8") as f:
        buffer = ""
        while '"samples"' not in buffer:
            chunk = f.read(chunk_size)
            if not chunk:
                raise ValueError(f"{manifest_path}: missing samples array")
            buffer += chunk
        key_pos = buffer.index('"samples"')
        array_pos = buffer.find("[", key_pos + len('"samples"'))
        while array_pos < 0:
            chunk = f.read(chunk_size)
            if not chunk:
                raise ValueError(f"{manifest_path}: unterminated samples field")
            buffer += chunk
            array_pos = buffer.find("[", key_pos + len('"samples"'))

        manifest_header = json.loads(buffer[:key_pos] + '"samples":[]}')
        expected_shape = manifest_header.get("shape")
        if not (isinstance(expected_shape, list) and len(expected_shape) == 3 and expected_shape[0] == len(channels)):
            raise ValueError(f"{manifest_path}: implausible manifest header shape {expected_shape}")
        if manifest_header.get("channels") != channels:
            raise ValueError(f"{manifest_path}: channel order mismatch")
        if manifest_header.get("geometry_backend_id") != required_backend_id:
            raise ValueError(
                f"{manifest_path}: geometry_backend_id "
                f"{manifest_header.get('geometry_backend_id')} != {required_backend_id}"
            )

        buffer = buffer[array_pos + 1 :]
        pos = 0
        while wanted - found.keys():
            while True:
                while pos < len(buffer) and buffer[pos] in " \t\r\n,":
                    pos += 1
                if pos < len(buffer) and buffer[pos] == "]":
                    missing = sorted(wanted - found.keys())
                    raise KeyError(f"{manifest_path}: missing requested sample ids first={missing[:5]}")
                try:
                    sample, end = decoder.raw_decode(buffer, pos)
                    pos = end
                    break
                except json.JSONDecodeError:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        raise ValueError(f"{manifest_path}: truncated samples array")
                    buffer = buffer[pos:] + chunk
                    pos = 0

            sid = str(sample.get("sample_id", ""))
            if sid not in wanted:
                if pos > chunk_size:
                    buffer = buffer[pos:]
                    pos = 0
                continue
            role = str(sample.get("selection_role", ""))
            if role not in {"positive", "context"}:
                raise ValueError(f"{manifest_path}: {sid}: invalid selection_role {role!r}")
            if sample.get("shape") != expected_shape:
                raise ValueError(f"{manifest_path}: {sid}: sample shape {sample.get('shape')} != header {expected_shape}")
            if sample.get("channels") != channels:
                raise ValueError(f"{manifest_path}: {sid}: sample channel order mismatch")
            if sample.get("geometry_backend_id") != required_backend_id:
                raise ValueError(f"{manifest_path}: {sid}: backend mismatch")
            if sid in found:
                raise ValueError(f"{manifest_path}: duplicate requested sample id {sid}")
            found[sid] = sample_from_manifest_fn(manifest_path, sample, sample)

    print(json.dumps({
        "event": "map_manifest_lightweight_loaded_v2",
        "manifest": str(manifest_path),
        "header_shape": expected_shape,
        "sample_ids": sample_ids,
    }, ensure_ascii=False), flush=True)
    return [found[sid] for sid in sample_ids], expected_shape


def choose_shuffled_interaction_clip(
    records: list[dict[str, Any]],
    state_rows: dict[str, dict[str, Any]],
    *,
    current_clip_id: str,
    split: str,
    offset: int,
) -> str:
    """Deterministic alternate-clip pick for the shuffled arm: sort the split's clip ids
    that have a sidecar row, walk offset positions past the current clip."""
    candidates = sorted(
        str(r.get("clip_id"))
        for r in records
        if str(r.get("map_memory_split", r.get("split", ""))) == split
        and str(r.get("clip_id")) in state_rows
    )
    if current_clip_id in candidates:
        base = candidates.index(current_clip_id)
    else:
        base = 0
    if len(candidates) < 2:
        raise ValueError(f"split={split!r}: fewer than 2 sidecar-covered clips; cannot build shuffled arm")
    pick = candidates[(base + max(1, int(offset))) % len(candidates)]
    if pick == current_clip_id:
        pick = candidates[(base + 1) % len(candidates)]
    if pick == current_clip_id:
        raise ValueError(f"could not find alternate clip for shuffled arm (clip_id={current_clip_id})")
    return pick


def sidecar_event_counts(row: dict[str, Any] | None) -> dict[str, int] | None:
    if not row:
        return None
    counts = (row.get("stats") or {}).get("interaction_event_counts") or {}
    return {k: int(counts.get(k, 0) or 0) for k in ("fire", "reload", "weapon_switch", "throw")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map-manifest", type=Path, required=True)
    ap.add_argument("--cache-manifest", type=Path, required=True)
    ap.add_argument("--state-cache-manifest", type=Path, required=True,
                    help="s6 interaction sidecar manifest (state_cache rows carry both state and interaction fields)")
    ap.add_argument("--adapter-checkpoint-low", type=Path, default=None,
                    help="optional; omit for a CLEAN BASE low expert (no adapter, no dense condition)")
    ap.add_argument("--adapter-checkpoint-high", type=Path, default=None,
                    help="optional; omit for a CLEAN BASE high expert (recorded in the report)")
    ap.add_argument("--lingbot-repo", type=Path, default=LINGBOT_ROOT)
    ap.add_argument("--ckpt-dir", type=Path, default=ASSET_ROOT / "lingbot-world-base-cam")
    ap.add_argument("--clip-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--clip-index", type=int, default=0)
    ap.add_argument("--clip-id", default=None)
    ap.add_argument("--interaction-variant", choices=["true", "blank", "shuffled"], default="true")
    ap.add_argument("--interaction-shuffled-offset", type=int, default=7)
    ap.add_argument("--boundary-override", type=float, default=None,
                    help="Override two-expert boundary sigma at inference (default: ckpt config, 0.947). "
                         ">=1.0 means LOW expert handles ALL steps.")
    ap.add_argument("--latent-frames", type=int, default=21)
    ap.add_argument("--chunk-size", type=int, default=3)
    ap.add_argument("--base-shift", type=float, default=3.0)
    ap.add_argument("--base-sampling-steps", type=int, default=70)
    ap.add_argument("--base-guide-scale", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--offload-model", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--allow-legacy-state-projection", action="store_true",
                    help="accept a state cache built with the pre-v2 projection contract "
                         "(pitch_sign=-1 / far=4096) instead of failing; only for deliberate "
                         "reruns of the old demo caches.")
    ap.add_argument("--target-height", type=int, default=480)
    ap.add_argument("--target-width", type=int, default=832)
    args = ap.parse_args()

    if args.target_height % 16 or args.target_width % 16:
        raise ValueError("--target-height/--target-width must be divisible by 16")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    add_path(Path.cwd() / "tools")
    add_path(TOOLS_DIR)
    if not args.lingbot_repo.exists():
        raise FileNotFoundError(f"lingbot repo not found: {args.lingbot_repo}")
    add_path(args.lingbot_repo)

    from map_memory_training_data_v0 import (
        CHANNELS,
        REQUIRED_BACKEND_ID,
        MapMemoryRelease,
        load_dense,
        sample_from_manifest,
    )
    from memory_dense_wan_adapter_v0 import (
        MemoryDenseAdapterConfig,
        MemoryDenseFrozenVAEWanTokenEncoder,
        inject_lora_into_wan_model_fast,
        wrap_wan_model_fast_with_memory_dense_adapter,
    )
    from memory_dense_state_adapter_v0 import (
        STATE_CHANNELS_V0,
        StateTokenProjector,
        load_state_manifest,
        resolve_state_cache_path,
        verify_state_projection_contract,
    )
    from memory_dense_interaction_adapter_v1 import (
        INTERACTION_CHANNELS_V1,
        InteractionTokenProjector,
        load_interaction_tensor,
    )
    import wan
    from wan.configs import WAN_CONFIGS
    from wan.utils.utils import save_video

    records = read_jsonl(args.cache_manifest)
    record = choose_record(records, split=args.split, clip_index=args.clip_index, clip_id=args.clip_id)
    clip_id = str(record["clip_id"])

    eff_lf = effective_latent_frames_for_chunking(args.latent_frames, args.chunk_size)
    eff_frame_num = frame_num_for_latent_frames(eff_lf)
    sample_ids = list(record["map_memory_sample_ids"])[:eff_lf]
    if len(sample_ids) < eff_lf:
        raise ValueError(f"{clip_id}: only {len(sample_ids)} map_memory_sample_ids, need {eff_lf}")

    samples, dense_header_shape = load_v2_manifest_samples(
        args.map_manifest,
        sample_ids,
        release_cls=MapMemoryRelease,
        sample_from_manifest_fn=sample_from_manifest,
        channels=CHANNELS,
        required_backend_id=REQUIRED_BACKEND_ID,
    )
    clip_dir = resolve_clip_dir(record, args.clip_root)
    for need in ["image.jpg", "poses.npy", "intrinsics.npy"]:
        if not (clip_dir / need).exists():
            raise FileNotFoundError(f"clip_dir missing {need}: {clip_dir}")

    state_rows = load_state_manifest(args.state_cache_manifest)
    verify_state_projection_contract(
        args.state_cache_manifest, state_rows,
        strict=not args.allow_legacy_state_projection)
    if clip_id not in state_rows:
        raise KeyError(f"{clip_id}: missing sidecar row in {args.state_cache_manifest}")
    state_row = dict(state_rows[clip_id])
    state_row["state_cache"] = str(resolve_state_cache_path(state_row, args.state_cache_manifest))

    interaction_row: dict[str, Any] | None = None
    interaction_source_clip_id: str | None = None
    if args.interaction_variant == "true":
        interaction_row = state_row
        interaction_source_clip_id = clip_id
    elif args.interaction_variant == "shuffled":
        interaction_source_clip_id = choose_shuffled_interaction_clip(
            records, state_rows,
            current_clip_id=clip_id, split=args.split,
            offset=args.interaction_shuffled_offset,
        )
        interaction_row = dict(state_rows[interaction_source_clip_id])
        interaction_row["state_cache"] = str(resolve_state_cache_path(interaction_row, args.state_cache_manifest))

    prompt_path = clip_dir / "prompt.txt"
    prompt = (prompt_path.read_text(encoding="utf-8").strip() if prompt_path.exists()
              else (record.get("prompt") or "first-person gameplay video in Counter-Strike, de_dust2 map"))
    image = Image.open(clip_dir / "image.jpg").convert("RGB")
    source_image_size = [int(image.width), int(image.height)]
    if image.size != (args.target_width, args.target_height):
        image = image.resize((args.target_width, args.target_height), Image.LANCZOS)
    # epsilon keeps 480 from flooring to 464 inside WanI2V's max_area->lat derivation
    gen_max_area = math.ceil(args.target_height * args.target_width * 1.0001)

    if not args.offload_model:
        # Stream each 14B expert to GPU inside WanModel.from_pretrained instead of
        # staging both on CPU: on the 92 GB PPU host, dual-expert CPU staging
        # (~81 GB anon) pushes MemAvailable under nohang's 3% soft threshold and
        # the loader gets SIGTERMed at the last shards. Final placement matches
        # the post-constructor early-move; CPU RSS peak drops to ~50 GB.
        from wan.modules.model import WanModel as _WanModel
        _orig_from_pretrained = _WanModel.from_pretrained.__func__

        def _from_pretrained_to_gpu(cls, *fp_args, **fp_kwargs):
            import gc as _gc
            model = _orig_from_pretrained(cls, *fp_args, **fp_kwargs)
            model = model.to(f"cuda:{args.device_id}")
            _gc.collect()
            torch.cuda.empty_cache()
            print(json.dumps({"event": "expert_streamed_to_gpu_after_load"}), flush=True)
            return model

        _WanModel.from_pretrained = classmethod(_from_pretrained_to_gpu)

    cfg = WAN_CONFIGS["i2v-A14B"]
    pipe = wan.WanI2V(
        config=cfg,
        checkpoint_dir=str(args.ckpt_dir),
        device_id=args.device_id,
        rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=False,
        convert_model_dtype=False,
    )
    if args.boundary_override is not None:
        pipe.boundary = float(args.boundary_override)
    if not args.offload_model:
        # Early-move both base experts to GPU to shed ~56 GB of CPU RSS during setup.
        # Matches the offload_model=False placement generate() would use anyway; the
        # login node cgroup (200 GB, shared with an 85 GB foreign /dev/shm) otherwise
        # OOM-kills this process at the dual-expert CPU staging peak (~87 GB RSS).
        import gc as _gc
        for _mname in ("low_noise_model", "high_noise_model"):
            _m = getattr(pipe, _mname, None)
            if _m is not None:
                setattr(pipe, _mname, _m.to(pipe.device))
        _gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps({"event": "experts_early_moved_to_gpu"}), flush=True)
    print(json.dumps({
        "event": "base_two_expert_loaded",
        "boundary_sigma": float(pipe.boundary),
    }, ensure_ascii=False), flush=True)

    def build_expert(expert_name: str, checkpoint_path: Path | None) -> dict[str, Any]:
        if checkpoint_path is None:
            return {"expert": expert_name, "mode": "clean_base"}
        ckpt = torch.load(checkpoint_path, map_location="cpu")
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
        )
        expert = getattr(pipe, expert_name)
        wrap_wan_model_fast_with_memory_dense_adapter(expert, config, freeze_base=True)
        lora_cfg = ckpt_config.get("lora_config") or {}
        if int(lora_cfg.get("rank", 0) or 0) > 0:
            inject_lora_into_wan_model_fast(
                expert,
                rank=int(lora_cfg["rank"]),
                targets=str(lora_cfg.get("targets", "attn,mlp")),
                alpha=lora_cfg.get("alpha"),
                dropout=float(lora_cfg.get("dropout", 0.0)),
            )
        dense_encoder_cfg = ckpt_config.get("dense_condition_encoder") or {}
        if dense_encoder_cfg.get("type") != "frozen_vae":
            raise ValueError(f"{checkpoint_path}: expected frozen_vae dense encoder, got {dense_encoder_cfg}")
        from wan.modules.vae2_1 import Wan2_1_VAE

        vae_pth = Path(dense_encoder_cfg.get("vae_pth") or (args.ckpt_dir / cfg.vae_checkpoint))
        vae = Wan2_1_VAE(vae_pth=str(vae_pth), device=pipe.device)
        # native grid follows the dense data actually fed (manifest header
        # shape // vae_stride), matching the trainers' dense_hw_from_args
        # derivation; pre-v2 ckpt configs record 22x40 or omit the key.
        ckpt_native_hw = dense_encoder_cfg.get("native_hw")
        derived_native_hw = (
            int(dense_header_shape[1]) // int(adapter_cfg.get("vae_stride", 8)),
            int(dense_header_shape[2]) // int(adapter_cfg.get("vae_stride", 8)),
        )
        if ckpt_native_hw and tuple(int(v) for v in ckpt_native_hw) != derived_native_hw:
            print(json.dumps({
                "event": "dense_encoder_native_hw_override",
                "expert": expert_name,
                "ckpt_native_hw": [int(v) for v in ckpt_native_hw],
                "derived_native_hw": list(derived_native_hw),
            }, ensure_ascii=False), flush=True)
        encoder = MemoryDenseFrozenVAEWanTokenEncoder(
            config,
            vae=vae,
            vae_pth=vae_pth,
            native_hw=derived_native_hw,
            packing=str(dense_encoder_cfg.get("packing", "img1_mask_img2_player_v0")),
        ).to(device=pipe.device, dtype=pipe.param_dtype)

        state_projector = StateTokenProjector(
            state_channels=len(STATE_CHANNELS_V0),
            cond_dim=int(adapter_cfg.get("cond_dim", 128)),
        ).to(device=pipe.device, dtype=pipe.param_dtype)
        load_adapter_checkpoint(checkpoint_path, expert, encoder, state_projector)

        interaction_state = (ckpt.get("state") or {}).get("memory_dense_interaction_projector") or {}
        interaction_projector = None
        if interaction_state:
            interaction_projector = InteractionTokenProjector(
                interaction_channels=len(INTERACTION_CHANNELS_V1),
                cond_dim=int(adapter_cfg.get("cond_dim", 128)),
            ).to(device=pipe.device, dtype=pipe.param_dtype)
            interaction_projector.load_state_dict(interaction_state)
            interaction_projector.eval()

        expert.to(pipe.device).eval()
        encoder.eval()
        state_projector.eval()

        original_forward = expert.forward
        cache: dict[tuple[int, int], torch.Tensor] = {}
        injection: dict[str, Any] = {
            "dense_token_shape": None,
            "state_token_shape": None,
            "interaction_token_shape": None,
        }

        def forward_with_condition(*fargs: Any, **kwargs: Any):
            dit_cond = dict(kwargs.get("dit_cond_dict") or {})
            x_arg = kwargs.get("x")
            if x_arg is None and fargs:
                x_arg = fargs[0]
            if x_arg is None:
                raise RuntimeError("cannot infer base latent for dense/state/interaction injection")
            current = x_arg[0] if isinstance(x_arg, list) else x_arg
            patch_hw = int(pipe.patch_size[1])
            token_hw = (int(current.shape[-2]) // patch_hw, int(current.shape[-1]) // patch_hw)
            if token_hw not in cache:
                dense_tokens, dense_hw = dense_tokens_for_sequence(
                    encoder=encoder,
                    samples=samples,
                    load_dense_fn=load_dense,
                    device=current.device,
                    dtype=pipe.param_dtype,
                    target_token_hw=token_hw,
                )
                if dense_hw != token_hw:
                    raise RuntimeError(f"dense token hw {dense_hw} != base target {token_hw}")
                injection["dense_token_shape"] = list(dense_tokens.shape)
                state_tokens = state_tokens_for_sequence(
                    state_projector=state_projector,
                    state_row=state_row,
                    latent_frames=eff_lf,
                    device=current.device,
                    dtype=pipe.param_dtype,
                    target_token_hw=token_hw,
                )
                if state_tokens.shape != dense_tokens.shape:
                    raise RuntimeError(
                        f"state token shape {tuple(state_tokens.shape)} != dense {tuple(dense_tokens.shape)}")
                injection["state_token_shape"] = list(state_tokens.shape)
                dense_tokens = dense_tokens + state_tokens
                if interaction_projector is not None and interaction_row is not None:
                    interaction = load_interaction_tensor(
                        interaction_row,
                        frame_indices=list(range(eff_lf)),
                        device=current.device,
                        dtype=pipe.param_dtype,
                    )
                    interaction_tokens = interaction_projector(
                        interaction, target_token_hw=token_hw,
                    ).to(pipe.param_dtype)
                    if interaction_tokens.shape != dense_tokens.shape:
                        raise RuntimeError(
                            f"interaction token shape {tuple(interaction_tokens.shape)} != dense {tuple(dense_tokens.shape)}")
                    injection["interaction_token_shape"] = list(interaction_tokens.shape)
                    dense_tokens = dense_tokens + interaction_tokens
                cache[token_hw] = dense_tokens.detach()
            dit_cond[config.cond_key] = cache[token_hw].to(device=current.device, dtype=current.dtype)
            kwargs["dit_cond_dict"] = dit_cond
            return original_forward(*fargs, **kwargs)

        expert.forward = forward_with_condition  # type: ignore[method-assign]
        report = {
            "expert": expert_name,
            "mode": "adapter",
            "checkpoint": str(checkpoint_path),
            "step": ckpt.get("step"),
            "state_projector_loaded": True,
            "interaction_projector_loaded": interaction_projector is not None,
            "interaction_contribution_active": (
                interaction_projector is not None and interaction_row is not None
            ),
            "injection": injection,
        }
        print(json.dumps({"event": "expert_built", **report}, ensure_ascii=False), flush=True)
        return report

    expert_reports = [
        build_expert("low_noise_model", args.adapter_checkpoint_low),
        build_expert("high_noise_model", args.adapter_checkpoint_high),
    ]
    low_report = expert_reports[0]
    if (low_report.get("mode") != "clean_base"
            and args.interaction_variant != "blank"
            and not low_report.get("interaction_projector_loaded")):
        raise ValueError(
            f"{args.adapter_checkpoint_low}: no memory_dense_interaction_projector in checkpoint; "
            f"cannot run interaction-variant={args.interaction_variant}"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # F09: the Fast/distilled path samples at shift=10.0 while this unipc path
    # defaults to 3.0. Both stay inside the training sigma distribution, but the
    # two schedules route ~15 of 70 steps to a different expert (mean |dt| 151),
    # so results taken at different shifts must never share a median.
    if abs(float(args.base_shift) - 10.0) > 1e-6:
        print(json.dumps({
            "event": "base_solver_shift_warning",
            "shift": float(args.base_shift),
            "fast_path_sample_shift": 10.0,
            "note": "different shift => different expert routing; group every cross-tool comparison by shift",
        }, ensure_ascii=False), flush=True)
    started = time.time()
    video = pipe.generate(
        prompt,
        image,
        action_path=str(clip_dir),
        allow_act2cam=False,
        action_string=None,
        vis_ui=False,
        max_area=gen_max_area,
        frame_num=eff_frame_num,
        shift=args.base_shift,
        sample_solver="unipc",
        sampling_steps=args.base_sampling_steps,
        guide_scale=(args.base_guide_scale, args.base_guide_scale),
        n_prompt=CLEAN_NEG_PROMPT,
        seed=args.seed,
        offload_model=args.offload_model,
    )
    gen_seconds = time.time() - started

    out_video = args.out_dir / f"{clip_id}_inter-{args.interaction_variant}_lf{eff_lf}_seed{args.seed}.mp4"
    save_video(video[None], save_file=str(out_video), fps=cfg.sample_fps, nrow=1, normalize=True, value_range=(-1, 1))

    frame_h = int(video.shape[-2])
    frame_w = int(video.shape[-1])
    if frame_h != args.target_height or frame_w != args.target_width:
        raise RuntimeError(
            f"generated resolution {frame_h}x{frame_w} != target "
            f"{args.target_height}x{args.target_width}; refusing to accept output")

    report = {
        "kind": "interaction_eval_sample_v1",
        "status": "complete",
        "interaction_variant": args.interaction_variant,
        "interaction_source_clip_id": interaction_source_clip_id,
        "interaction_shuffled_offset": args.interaction_shuffled_offset if args.interaction_variant == "shuffled" else None,
        "clip_id": clip_id,
        "split": args.split,
        "clip_dir": str(clip_dir),
        "clip_event_counts": sidecar_event_counts(state_row),
        "interaction_source_event_counts": sidecar_event_counts(interaction_row),
        "dense_header_shape": dense_header_shape,
        "sample_ids": sample_ids,
        "experts": expert_reports,
        "boundary_sigma": float(pipe.boundary),
        "prompt": prompt,
        "source_image_size": source_image_size,
        "target_height": args.target_height,
        "target_width": args.target_width,
        "gen_max_area": gen_max_area,
        "requested_latent_frames": args.latent_frames,
        "effective_latent_frames": eff_lf,
        "effective_frame_num": eff_frame_num,
        "chunk_size": args.chunk_size,
        "base_shift": args.base_shift,
        "base_sampling_steps": args.base_sampling_steps,
        "base_guide_scale": args.base_guide_scale,
        "seed": args.seed,
        "generation_seconds": round(gen_seconds, 1),
        "out_video": str(out_video),
        "state_cache_manifest": str(args.state_cache_manifest),
        "map_manifest": str(args.map_manifest),
        "cache_manifest": str(args.cache_manifest),
    }
    write_json(args.out_dir / f"{clip_id}_inter-{args.interaction_variant}_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
