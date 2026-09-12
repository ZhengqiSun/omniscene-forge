from __future__ import annotations
from tools.runtime_paths import source_path

import argparse
import json
import os
import platform
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .camera import add_lingbot_import
from .checkpoint import base_identity, git_commit, load_lora_for_inference, load_pair, sha256_file
from .lora import inject_lora
from .media import load_initial_image
from .schema import LingBotSampleV1, load_manifest, manifest_sha256, validate_sample


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Camera-only LingBot Base Cam standard inference")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Official base directory or lingbot_cam_lora_pair_v1.json")
    parser.add_argument("--lingbot-repo", type=Path, default=Path(str(source_path('lingbot', ''))))
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--sampling-steps", type=int, default=70)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _initial_pil(sample: LingBotSampleV1) -> Image.Image:
    tensor = load_initial_image(sample)
    array = tensor.add(1).mul(127.5).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def _video_contract(path: Path) -> dict[str, object]:
    cap = cv2.VideoCapture(str(path))
    result = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    if result["frames"] != 81 or abs(result["fps"] - 16.0) > 0.05 or (result["width"], result["height"]) != (832, 480):
        raise RuntimeError(f"generated video contract mismatch: {result}")
    return result


def main() -> None:
    args = _parser().parse_args()
    samples = load_manifest(args.manifest)
    selected = set(args.sample_id)
    samples = [sample for sample in samples if not selected or sample.sample_id in selected]
    if selected - {sample.sample_id for sample in samples}:
        raise ValueError(f"unknown sample IDs: {sorted(selected - {sample.sample_id for sample in samples})}")
    if args.limit is not None:
        samples = samples[:args.limit]
    if not samples:
        raise RuntimeError("no samples selected")
    failures = [
        {"sample_id": sample.sample_id, "errors": validate_sample(sample, mode="infer")}
        for sample in samples
    ]
    failures = [item for item in failures if item["errors"]]
    if failures:
        raise RuntimeError(f"manifest validation failed: {json.dumps(failures[:5])}")
    manifest_hash = manifest_sha256(args.manifest)
    pair = load_pair(args.checkpoint) if args.checkpoint.is_file() else None
    base_root = Path(pair["base_checkpoint"] if pair else args.checkpoint).resolve()
    dry = {
        "schema_id": "lingbot_cam_infer_plan_v1", "camera_only": True,
        "manifest_sha256": manifest_hash, "sample_ids": [sample.sample_id for sample in samples],
        "seeds": args.seeds, "sampling_steps": args.sampling_steps, "cfg": args.cfg,
        "base_checkpoint": str(base_root), "lora_pair": pair,
        "model_read_fields": ["initial_image|raw_video(frame raw_start only)", "prompt", "poses", "intrinsics"],
        "target_files_accessed": False, "output_contract": {"frames": 81, "width": 832, "height": 480, "fps": 16},
    }
    if args.dry_run:
        print(json.dumps(dry, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for inference")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    add_lingbot_import(args.lingbot_repo)
    from wan.configs import WAN_CONFIGS
    from wan.image2video import WanI2V
    from wan.utils.utils import save_video
    config = WAN_CONFIGS["i2v-A14B"]
    pipeline = WanI2V(
        config=config, checkpoint_dir=str(base_root), device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        init_on_cpu=True, convert_model_dtype=True,
    )
    base_identities = {
        "low": base_identity(base_root, "low", content_hash=True),
        "high": base_identity(base_root, "high", content_hash=True),
    }
    lora_meta = None
    if pair:
        low_cfg = inject_lora(pipeline.low_noise_model, **{
            "rank": int(pair["lora_config"]["rank"]), "targets": pair["lora_config"]["targets"],
            "alpha": float(pair["lora_config"]["alpha"]), "dropout": float(pair["lora_config"]["dropout"]),
        })
        high_cfg = inject_lora(pipeline.high_noise_model, **{
            "rank": int(pair["lora_config"]["rank"]), "targets": pair["lora_config"]["targets"],
            "alpha": float(pair["lora_config"]["alpha"]), "dropout": float(pair["lora_config"]["dropout"]),
        })
        low_meta = load_lora_for_inference(Path(pair["low_checkpoint"]), model=pipeline.low_noise_model, expert="low", base=base_identities["low"])
        high_meta = load_lora_for_inference(Path(pair["high_checkpoint"]), model=pipeline.high_noise_model, expert="high", base=base_identities["high"])
        if low_meta["lora_config"] != high_meta["lora_config"]:
            raise ValueError("LOW/HIGH LoRA configurations differ")
        lora_meta = {"low": low_meta, "high": high_meta, "injection": {"low": low_cfg, "high": high_cfg}}
    commits = {
        "lingbot": git_commit(args.lingbot_repo),
        "project": git_commit(Path(__file__).resolve().parents[2]),
    }
    for sample in samples:
        for seed in args.seeds:
            stem = f"{sample.sample_id}__seed_{seed}"
            video_path = args.output_dir / f"{stem}.mp4"
            metadata_path = args.output_dir / f"{stem}.json"
            expected_identity = {
                "sample_id": sample.sample_id, "manifest_sha256": manifest_hash,
                "seed": seed, "cfg": args.cfg, "sampling_steps": args.sampling_steps,
                "checkpoint": str(args.checkpoint.resolve()),
            }
            if video_path.exists() or metadata_path.exists():
                if not (args.skip_existing or args.resume):
                    raise FileExistsError(f"refusing to overwrite existing output {stem}")
                if video_path.is_file() and metadata_path.is_file():
                    existing = json.loads(metadata_path.read_text())
                    if all(existing.get(key) == value for key, value in expected_identity.items()):
                        _video_contract(video_path)
                        print(json.dumps({"event": "skip_existing", "sample_id": sample.sample_id, "seed": seed}))
                        continue
                raise RuntimeError(f"existing output metadata does not match requested run: {stem}")
            started = time.monotonic()
            with tempfile.TemporaryDirectory(prefix="lingbot_cam_", dir=args.output_dir) as temporary:
                action_dir = Path(temporary)
                os.symlink(sample.poses, action_dir / "poses.npy")
                os.symlink(sample.intrinsics, action_dir / "intrinsics.npy")
                video = pipeline.generate(
                    input_prompt=sample.prompt, img=_initial_pil(sample), action_path=str(action_dir),
                    vis_ui=False, max_area=480 * 832, frame_num=81, shift=float(config.sample_shift),
                    sample_solver="unipc", sampling_steps=args.sampling_steps,
                    guide_scale=args.cfg, seed=seed, offload_model=True,
                )
            if tuple(video.shape) != (3, 81, 480, 832):
                raise RuntimeError(f"model output shape {tuple(video.shape)} != (3,81,480,832)")
            save_video(tensor=video[None], save_file=str(video_path), fps=16, nrow=1, normalize=True, value_range=(-1, 1))
            contract = _video_contract(video_path)
            metadata = {
                **expected_identity, "schema_id": "lingbot_cam_inference_metadata_v1",
                "prompt": sample.prompt, "output_shape": [3, 81, 480, 832], "output_video": str(video_path),
                "video_contract": contract, "runtime_seconds": time.monotonic() - started,
                "git_commits": commits, "base_identity": base_identities,
                "pair_descriptor_sha256": sha256_file(args.checkpoint) if pair else None,
                "lora": lora_meta, "torch": torch.__version__, "python": platform.python_version(),
                "gpu": torch.cuda.get_device_name(0), "peak_gpu_bytes": torch.cuda.max_memory_allocated(0),
                "target_files_accessed": False,
            }
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"event": "inference_complete", "sample_id": sample.sample_id, "seed": seed, "video": str(video_path)}))


if __name__ == "__main__":
    main()
