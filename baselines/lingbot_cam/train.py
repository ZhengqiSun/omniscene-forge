from __future__ import annotations
from tools.runtime_paths import source_path

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .camera import add_lingbot_import, prepare_camera
from .checkpoint import base_identity, git_commit, load_checkpoint, save_checkpoint
from .lora import (
    assert_only_lora_gradients,
    assert_only_lora_trainable,
    enable_block_checkpointing,
    inject_lora,
    named_lora_parameters,
)
from .media import encode_initial_condition, encode_video_and_condition, load_initial_image, load_target_video
from .schema import LingBotSampleV1, load_manifest, manifest_sha256, validate_sample


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Camera-only LingBot Base Cam LoRA trainer")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--lingbot-repo", type=Path, default=Path(str(source_path('lingbot', ''))))
    parser.add_argument("--expert", choices=["low", "high"], required=True)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=float, default=64.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-targets", default="attn,mlp")
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug-allow-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _load_context(sample: LingBotSampleV1, *, checkpoint: Path, config, device: torch.device, cache: dict[str, torch.Tensor]) -> torch.Tensor:
    key = str(sample.text_cache) if sample.text_cache else f"prompt:{sample.prompt}"
    if key in cache:
        return cache[key].to(device)
    if sample.text_cache:
        obj = torch.load(sample.text_cache, map_location="cpu", weights_only=False)
        context = obj.get("context") if isinstance(obj, dict) else obj
    else:
        from wan.modules.t5 import T5EncoderModel
        encoder = T5EncoderModel(
            text_len=config.text_len, dtype=config.t5_dtype, device=torch.device("cpu"),
            checkpoint_path=str(checkpoint / config.t5_checkpoint),
            tokenizer_path=str(checkpoint / config.t5_tokenizer), shard_fn=None,
        )
        context = encoder([sample.prompt], torch.device("cpu"))[0]
        del encoder
    if tuple(context.shape)[-1] != 4096:
        raise RuntimeError(f"{sample.sample_id}: text context shape {tuple(context.shape)} is invalid")
    cache[key] = context.cpu()
    return context.to(device)


def _load_latents(sample: LingBotSampleV1, *, vae_holder: dict[str, Any], checkpoint: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if sample.target_latent:
        obj = torch.load(sample.target_latent, map_location="cpu", weights_only=False)
        x0 = obj.get("latent") if isinstance(obj, dict) else obj
        condition = obj.get("condition") if isinstance(obj, dict) else None
        if condition is not None:
            return x0.to(device), condition.to(device)
    else:
        x0 = None
        condition = None
    if "vae" not in vae_holder:
        from wan.modules.vae2_1 import Wan2_1_VAE
        vae_holder["vae"] = Wan2_1_VAE(
            vae_pth=str(checkpoint / "Wan2.1_VAE.pth"), dtype=torch.float32, device=device
        )
    vae = vae_holder["vae"]
    if x0 is None:
        video = load_target_video(sample)
        x0, condition = encode_video_and_condition(vae, video, device)
    elif condition is None:
        image = load_initial_image(sample)
        condition = encode_initial_condition(vae, image, device)
    return x0.to(device), condition.to(device)


def main() -> None:
    args = _parser().parse_args()
    if args.global_batch_size < 1:
        raise ValueError("global batch size must be positive")
    samples = load_manifest(args.manifest)
    train_samples = [sample for sample in samples if sample.split == "train"]
    if args.debug_allow_test:
        train_samples += [sample for sample in samples if sample.split == "test"]
    elif any(sample.split == "test" for sample in train_samples):
        raise RuntimeError("test split is forbidden")
    if not train_samples:
        raise RuntimeError("manifest contains no train rows")
    failures = [
        {"sample_id": sample.sample_id, "errors": validate_sample(sample, mode="train", deep=True)}
        for sample in train_samples
    ]
    failures = [item for item in failures if item["errors"]]
    if failures:
        raise RuntimeError(f"manifest validation failed: {json.dumps(failures[:5])}")
    identity = base_identity(args.base_checkpoint, args.expert, content_hash=False)
    lora_config = {
        "rank": args.lora_rank, "alpha": args.lora_alpha,
        "dropout": args.lora_dropout, "targets": args.lora_targets,
    }
    effective = {
        "schema_id": "lingbot_cam_train_config_v1", "camera_only": True,
        "use_dense": False, "use_state": False, "use_memory_adapter": False,
        "use_player_mask_loss": False, "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_sha256(args.manifest), "expert": args.expert,
        "base_identity": identity, "lora_config": lora_config,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "max_steps": args.max_steps, "global_batch_size": args.global_batch_size,
        "seed": args.seed, "precision": args.precision,
        "scheduler": "constant", "activation_checkpointing": args.activation_checkpointing,
        "lingbot_repo": str(args.lingbot_repo.resolve()),
        "lingbot_git_commit": git_commit(args.lingbot_repo),
        "project_git_commit": git_commit(Path(__file__).resolve().parents[2]),
        "output_dir": str(args.output_dir.resolve()),
        "resume_checkpoint": str(args.resume_checkpoint.resolve()) if args.resume_checkpoint else None,
    }
    if args.dry_run:
        print(json.dumps({**effective, "sample_count": len(train_samples), "accessed_fields": [
            "initial_image|raw_video", "target_video|target_latent", "poses", "intrinsics", "prompt|text_cache"
        ]}, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training")
    identity = base_identity(args.base_checkpoint, args.expert, content_hash=True)
    effective["base_identity"] = identity
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "effective_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if not args.resume_checkpoint:
            raise FileExistsError(f"run directory already contains a config: {config_path}")
        mutable = {"max_steps", "resume_checkpoint", "lora_report", "trainable_audit"}
        previous_fixed = {key: value for key, value in previous.items() if key not in mutable}
        effective_fixed = {key: value for key, value in effective.items() if key not in mutable}
        if previous_fixed != effective_fixed or args.max_steps < int(previous["max_steps"]):
            raise ValueError("resume config mismatch; only increasing max_steps and resume_checkpoint may change")
    config_path.write_text(json.dumps(effective, indent=2) + "\n", encoding="utf-8")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    add_lingbot_import(args.lingbot_repo)
    from wan.configs import WAN_CONFIGS
    from wan.modules.model import WanModel
    wan_config = WAN_CONFIGS["i2v-A14B"]
    expert_subdir = wan_config.low_noise_checkpoint if args.expert == "low" else wan_config.high_noise_checkpoint
    model = WanModel.from_pretrained(
        str(args.base_checkpoint), subfolder=expert_subdir, torch_dtype=dtype, control_type="cam"
    )
    model.requires_grad_(False)
    lora_report = inject_lora(
        model, rank=args.lora_rank, targets=args.lora_targets,
        alpha=args.lora_alpha, dropout=args.lora_dropout,
    )
    if args.activation_checkpointing:
        enable_block_checkpointing(model)
    model.to(device=device, dtype=dtype).train()
    trainable = assert_only_lora_trainable(model)
    effective["lora_report"] = lora_report
    effective["trainable_audit"] = trainable
    config_path.write_text(json.dumps(effective, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "trainable_parameters.txt").write_text("\n".join(trainable["names"]) + "\n", encoding="utf-8")
    params = [p for _, p in named_lora_parameters(model)]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    start_step = 0
    if args.resume_checkpoint:
        start_step = load_checkpoint(
            args.resume_checkpoint, model=model, optimizer=optimizer, scheduler=scheduler,
            expected=effective, restore_rng_state=True,
        )
    sigma_lo, sigma_hi = (0.0, float(wan_config.boundary)) if args.expert == "low" else (float(wan_config.boundary), 1.0)
    context_cache: dict[str, torch.Tensor] = {}
    latent_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    vae_holder: dict[str, Any] = {}
    log_path = args.output_dir / "train.jsonl"
    for step in range(start_step + 1, args.max_steps + 1):
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for micro in range(args.global_batch_size):
            sample = train_samples[((step - 1) * args.global_batch_size + micro) % len(train_samples)]
            if sample.sample_id not in latent_cache:
                x0, condition = _load_latents(sample, vae_holder=vae_holder, checkpoint=args.base_checkpoint, device=device)
                latent_cache[sample.sample_id] = (x0.cpu(), condition.cpu())
            x0, condition = (tensor.to(device=device, dtype=dtype) for tensor in latent_cache[sample.sample_id])
            context = _load_context(sample, checkpoint=args.base_checkpoint, config=wan_config, device=device, cache=context_cache).to(dtype)
            camera = prepare_camera(sample.poses, sample.intrinsics, device=device, dtype=dtype)
            noise = torch.randn_like(x0)
            sigma = torch.empty((), device=device, dtype=torch.float32).uniform_(sigma_lo, sigma_hi)
            xt = ((1.0 - sigma) * x0.float() + sigma * noise.float()).to(dtype)
            target = noise.float() - x0.float()
            timestep = (sigma * 1000.0).reshape(1)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
                prediction = model(
                    x=[xt], t=timestep, context=[context], seq_len=21 * 30 * 52,
                    y=[condition], dit_cond_dict={"c2ws_plucker_emb": camera.chunk(1, dim=0)},
                )[0]
                loss = F.mse_loss(prediction.float(), target) / args.global_batch_size
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            loss.backward()
            losses.append(float(loss.detach()) * args.global_batch_size)
        gradient_audit = assert_only_lora_gradients(model)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        optimizer.step(); scheduler.step()
        event = {
            "event": "train_step", "global_step": step, "expert": args.expert,
            "loss": sum(losses) / len(losses), "grad_norm": float(grad_norm),
            "lr": scheduler.get_last_lr()[0], "seconds": time.monotonic() - started,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(device),
            "gradient_tensor_count": gradient_audit["gradient_tensor_count"],
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)
        if step % args.save_every == 0 or step == args.max_steps:
            save_checkpoint(
                args.output_dir / "checkpoints" / f"{args.expert}_lora_step_{step:06d}.pt",
                model=model, optimizer=optimizer, scheduler=scheduler, step=step, config=effective,
            )


if __name__ == "__main__":
    main()
